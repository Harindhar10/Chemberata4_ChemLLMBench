import os
import re
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import evaluate
import Levenshtein
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

from utils import LiteLLMModel, HuggingFaceModel , strip_thinking , extract_json_answer
from .data import load_dataset, TASK_COL_MAP

# ── Supported tasks ─────────────────────────────────────────────────────────
SUPPORTED_TASKS = list(TASK_COL_MAP.keys())

# ── Metric loaders (cached so each is loaded once) ──────────────────────────
_METRIC_CACHE = {}


def _get_metric(name):
    if name not in _METRIC_CACHE:
        _METRIC_CACHE[name] = evaluate.load(name)
    return _METRIC_CACHE[name]


_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ── Parsing ─────────────────────────────────────────────────────────────────
def _parse_prediction(raw_text, task_name):
    if not raw_text:
        return ""

    text = raw_text.strip()
    text = strip_thinking(text)

    ans = extract_json_answer(text)
    if ans:
        return ans.strip()

    
    _, tgt_col = TASK_COL_MAP[task_name]
    match = None

    if task_name == "MOL_CAPTION":
        # captions can be multi-line -> capture everything after the label
        match = re.search(rf"{re.escape(tgt_col)}\s*:\s*(.+)", text,
                          re.IGNORECASE | re.DOTALL)
    elif task_name == "MOL_DESIGN":
        # SMILES contain no whitespace -> first token only
        match = re.search(rf"{re.escape(tgt_col)}\s*:\s*(\S+)", text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Fallback: model ignored the label -> first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.split()[0] if task_name == "MOL_DESIGN" else line
    return ""


# ── Chemistry helpers ───────────────────────────────────────────────────────
def _fp_similarities(mol_a, mol_b):
    """Tanimoto similarity of (MACCS, RDK, Morgan) fingerprints."""
    maccs = DataStructs.TanimotoSimilarity(
        MACCSkeys.GenMACCSKeys(mol_a), MACCSkeys.GenMACCSKeys(mol_b))
    rdk = DataStructs.TanimotoSimilarity(
        Chem.RDKFingerprint(mol_a), Chem.RDKFingerprint(mol_b))
    morgan = DataStructs.TanimotoSimilarity(
        _morgan_gen.GetFingerprint(mol_a), _morgan_gen.GetFingerprint(mol_b))
    return maccs, rdk, morgan


def _compute_fcd(valid_pred_smiles, true_smiles):
    """FCD between valid predictions and ground truth. NaN if unavailable."""
    try:
        from fcd import get_fcd , load_ref_model
        model = load_ref_model()
        if len(valid_pred_smiles) < 2 or len(true_smiles) < 2:
            return float("nan")
        return float(get_fcd(valid_pred_smiles, true_smiles,model))
    except Exception as e:
        print(f"[!] FCD could not be computed: {e}")
        return float("nan")


# ── Metrics ─────────────────────────────────────────────────────────────────
def _caption_metrics(y_true, y_pred):
    bleu = _get_metric("bleu")
    rouge = _get_metric("rouge")
    meteor = _get_metric("meteor")

    refs = [[t] for t in y_true]  # HF bleu expects a list of reference lists

    bleu2 = bleu.compute(predictions=y_pred, references=refs, max_order=2)["bleu"]
    bleu4 = bleu.compute(predictions=y_pred, references=refs, max_order=4)["bleu"]

    r = rouge.compute(predictions=y_pred, references=y_true, use_stemmer=True)
    m = meteor.compute(predictions=y_pred, references=y_true)["meteor"]

    return {
        "bleu2": float(bleu2),
        "bleu4": float(bleu4),
        "rouge1": float(r["rouge1"]),
        "rouge2": float(r["rouge2"]),
        "rougeL": float(r["rougeL"]),
        "meteor": float(m),
        "total": len(y_true),
    }


def _design_metrics(y_true, y_pred):
    n = len(y_true)

    # BLEU on character-level SMILES tokens
    bleu = _get_metric("bleu").compute(
        predictions=y_pred,
        references=[[t] for t in y_true],
        tokenizer=lambda s: list(s),
    )["bleu"]

    # Levenshtein over all samples (invalid/empty predictions still count)
    lev = float(np.mean([Levenshtein.distance(t, p) for t, p in zip(y_true, y_pred)]))

    exact, valid = 0, 0
    maccs_s, rdk_s, morgan_s = [], [], []
    valid_pred_can, true_can = [], []

    for t, p in zip(y_true, y_pred):
        mol_t = Chem.MolFromSmiles(t) if t else None
        mol_p = Chem.MolFromSmiles(p) if p else None
        if mol_t is not None:
            true_can.append(Chem.MolToSmiles(mol_t))
        if mol_p is None:
            continue
        valid += 1
        valid_pred_can.append(Chem.MolToSmiles(mol_p))
        if mol_t is None:
            continue
        if Chem.MolToSmiles(mol_t) == Chem.MolToSmiles(mol_p):
            exact += 1
        m, r, mo = _fp_similarities(mol_t, mol_p)
        maccs_s.append(m)
        rdk_s.append(r)
        morgan_s.append(mo)

    mean_or_zero = lambda xs: float(np.mean(xs)) if xs else 0.0

    return {
        "bleu": float(bleu),
        "exact": exact / n,
        "levenshtein": lev,
        "validity": valid / n,
        "maccs_fts": mean_or_zero(maccs_s),   # averaged over valid predictions only
        "rdk_fts": mean_or_zero(rdk_s),
        "morgan_fts": mean_or_zero(morgan_s),
        "fcd": _compute_fcd(valid_pred_can, true_can),
        "valid_predictions": valid,
        "total": n,
    }


def _compute_metrics(y_true, y_pred, task_name):
    """y_true: list of references, y_pred: list of parsed predictions."""
    if task_name == "MOL_CAPTION":
        return _caption_metrics(y_true, y_pred)
    elif task_name == "MOL_DESIGN":
        return _design_metrics(y_true, y_pred)
    raise ValueError(f"No metrics defined for task: {task_name!r}")


def _print_metrics(metrics, task_name, model_name, sample_strategy, k):
    print(f"\n{'═' * 55}")
    print(f"  Task    : {task_name}")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"{'─' * 55}")
    if task_name == "MOL_CAPTION":
        print(f"  BLEU-2   (↑): {metrics['bleu2']:.4f}")
        print(f"  BLEU-4   (↑): {metrics['bleu4']:.4f}")
        print(f"  ROUGE-1  (↑): {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2  (↑): {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L  (↑): {metrics['rougeL']:.4f}")
        print(f"  METEOR   (↑): {metrics['meteor']:.4f}")
    else:
        print(f"  BLEU        (↑): {metrics['bleu']:.4f}")
        print(f"  Exact       (↑): {metrics['exact']:.4f}")
        print(f"  Levenshtein (↓): {metrics['levenshtein']:.2f}")
        print(f"  Validity    (↑): {metrics['validity']:.4f}")
        print(f"  MACCS FTS   (↑): {metrics['maccs_fts']:.4f}")
        print(f"  RDK FTS     (↑): {metrics['rdk_fts']:.4f}")
        print(f"  Morgan FTS  (↑): {metrics['morgan_fts']:.4f}")
        print(f"  FCD         (↓): {metrics['fcd']:.4f}")
        print(f"  Valid        : {metrics['valid_predictions']}/{metrics['total']}")
    print(f"{'═' * 55}\n")


# ── Model ───────────────────────────────────────────────────────────────────
def _build_model(backend, model_name, temperature, max_new_tokens=256):
    if backend == "api":
        return LiteLLMModel(model_name=model_name, max_new_tokens=max_new_tokens)
    elif backend == "local":
        return HuggingFaceModel(
            temperature=temperature,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unknown backend: {backend!r}. Use 'api' or 'local'.")


# ── Benchmark ───────────────────────────────────────────────────────────────
def run_benchmark(
    task_name: str,
    model_name: str,
    backend: str = "api",
    sample_strategy: str = "random",
    k: int = 8,
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    output_dir: str = "results",
):
    """Run the benchmark on one task.

    Returns dict with keys: "metrics", "predictions", "raw_outputs"
    """
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unknown task: {task_name!r}. Choose from {SUPPORTED_TASKS}")

    # 1. Load data & prompts
    print(f"[*] Loading task: {task_name}  (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_dataset(
        name=task_name, sample_strategy=sample_strategy, k=k,
    )
    _, to_col = TASK_COL_MAP[task_name]
    y_true = df_test[to_col].astype(str).tolist()

    # 2. Build model
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = _build_model(backend, model_name, temperature, max_new_tokens)

    # 3. Inference
    print(f"[*] Running inference on {len(user_prompts)} samples …")
    all_outputs = model.generate_batch(user_prompts, system_prompts, n=n)
    raw_outputs = [outs[0] if outs else "" for outs in all_outputs]
    predictions = [_parse_prediction(t, task_name) for t in raw_outputs]

    # 4. Evaluate + print
    print("[*] Computing metrics …")
    metrics = _compute_metrics(y_true, predictions, task_name)
    _print_metrics(metrics, task_name, model_name, sample_strategy, k)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    run_tag = f"{task_name}_{safe_model}_{sample_strategy}_k{k}_{timestamp}"

    results_df = df_test.copy().reset_index(drop=True)
    results_df["raw_output"] = raw_outputs
    results_df["prediction"] = predictions
    results_df["ground_truth"] = y_true
    csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
    results_df.to_csv(csv_path, index=False)

    meta = {
        "task": task_name,
        "model": model_name,
        "backend": backend,
        "sample_strategy": sample_strategy,
        "k": k,
        "n": n,
        "temperature": temperature,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    json_path = os.path.join(output_dir, f"{run_tag}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[✓] Predictions saved to: {csv_path}")
    print(f"[✓] Metrics saved to:     {json_path}")

    return {"metrics": metrics, "predictions": predictions, "raw_outputs": raw_outputs}


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ChemLLMBench – Molecule Captioning / Design Benchmark"
    )
    parser.add_argument("--task", type=str, required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier (e.g. gpt-4, meta-llama/Llama-2-7b-chat-hf).")
    parser.add_argument("--backend", type=str, default="api", choices=["api", "local"])
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "scaffold"])
    parser.add_argument("--k", type=int, default=8, help="Few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1, help="Completions per sample.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(
        task_name=args.task,
        model_name=args.model,
        backend=args.backend,
        sample_strategy=args.strategy,
        k=args.k,
        n=args.n,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()