import os
import re
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger

from utils import LiteLLMModel, HuggingFaceModel, extract_json_answer, strip_thinking
from .data import (
    load_retro_dataset as load_dataset, INPUT_COL, TARGET_COL, INPUT_LABEL, TARGET_LABEL,
)

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings for invalid SMILES


# ── Parsing ─────────────────────────────────────────────────────────────────
def _clean(text):
    """Strip markdown fences, quotes and backticks."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    return text.strip().strip("`'\" \n")


def _parse_prediction(raw_text):
    """Extract the predicted reactant SMILES (a single whitespace-free token)."""
    if not raw_text:
        return ""
    text = _clean(raw_text)
    text = strip_thinking(text)
    if not text:
        return ""
    ans = extract_json_answer(text, key="answer")

    if ans:
        return ans.strip()

    # Preferred: text after the target label, e.g. "Reactants: CCO.CC(=O)O"
    match = re.search(rf"{re.escape(TARGET_LABEL)}\s*:\s*(\S+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip("`'\"")

    # Fallback: model ignored the label -> first token of first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.split()[0].strip("`'\"")
    return ""


# ── Validity / normalisation ────────────────────────────────────────────────
def _is_invalid_prediction(smiles):
    """True if the string is empty or any '.'-separated fragment fails to parse."""
    if not smiles:
        return True
    for frag in smiles.split("."):
        if not frag or Chem.MolFromSmiles(frag) is None:
            return True
    return False


def _normalise(smiles):
    """Canonicalise every fragment, sort them, and re-join with '.'.

    Returns "" if the string is empty or any fragment is invalid, so an invalid
    prediction can never match a reference.
    """
    if not smiles:
        return ""
    canon = []
    for frag in smiles.split("."):
        mol = Chem.MolFromSmiles(frag) if frag else None
        if mol is None:
            return ""
        canon.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canon))


# ── Metrics ─────────────────────────────────────────────────────────────────
def _compute_metrics(y_true, y_pred):
    """Top-1 accuracy and percentage of invalid predicted SMILES.

    y_true: list of reference reactant SMILES
    y_pred: list of parsed predicted reactant SMILES
    """
    n = len(y_true)
    num_invalid = sum(_is_invalid_prediction(p) for p in y_pred)

    norm_true = [_normalise(t) for t in y_true]
    norm_pred = [_normalise(p) for p in y_pred]
    num_correct = sum(1 for t, p in zip(norm_true, norm_pred) if p and p == t)

    return {
        "top1_accuracy": num_correct / n if n else 0.0,
        "invalid_smiles_percent": 100.0 * num_invalid / n if n else 0.0,
        "correct": num_correct,
        "invalid": num_invalid,
        "total": n,
    }


def _print_metrics(metrics, model_name, sample_strategy, k):
    print(f"\n{'═' * 55}")
    print("  Task    : RETROSYNTHESIS (top-1)")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"{'─' * 55}")
    print(f"  Top-1 Accuracy       (↑): {metrics['top1_accuracy']:.4f}"
          f"  ({metrics['correct']}/{metrics['total']})")
    print(f"  Invalid SMILES (%)   (↓): {metrics['invalid_smiles_percent']:.2f}"
          f"  ({metrics['invalid']}/{metrics['total']})")
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
    model_name: str,
    backend: str = "api",
    sample_strategy: str = "random",
    k: int = 8,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    output_dir: str = "results",
):
    """Run top-1 retrosynthesis benchmark.

    Returns dict with keys: "metrics", "predictions", "raw_outputs"
    """
    # 1. Load data & prompts
    print(f"[*] Loading retrosynthesis data (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_dataset(
        sample_strategy=sample_strategy, k=k,
    )
    y_true = df_test[TARGET_COL].astype(str).tolist()

    # 2. Build model
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = _build_model(backend, model_name, temperature, max_new_tokens)

    # 3. Inference (top-1 -> n=1, first completion only)
    print(f"[*] Running inference on {len(user_prompts)} samples …")
    all_outputs = model.generate_batch(user_prompts, system_prompts, n=1)
    raw_outputs = [outs[0] if outs else "" for outs in all_outputs]
    predictions = [_parse_prediction(t) for t in raw_outputs]

    if type(model) is LiteLLMModel:
        print(f"[✓] Total API cost: ${model.cost:.4f}")

        model.cost = 0.0  # Reset cost for next run

    # 4. Evaluate + print
    print("[*] Computing metrics …")
    metrics = _compute_metrics(y_true, predictions)
    _print_metrics(metrics, model_name, sample_strategy, k)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    run_tag = f"RETRO_{safe_model}_{sample_strategy}_k{k}_{timestamp}"

    results_df = df_test.copy().reset_index(drop=True)
    results_df["raw_output"] = raw_outputs
    results_df["prediction"] = predictions
    results_df["ground_truth"] = y_true
    results_df["is_correct"] = [
        bool(_normalise(p)) and _normalise(p) == _normalise(t)
        for p, t in zip(predictions, y_true)
    ]
    results_df["is_invalid"] = [_is_invalid_prediction(p) for p in predictions]
    csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
    results_df.to_csv(csv_path, index=False)

    meta = {
        "task": "RETROSYNTHESIS",
        "model": model_name,
        "backend": backend,
        "sample_strategy": sample_strategy,
        "k": k,
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
        description="ChemLLMBench – Retrosynthesis (top-1) Benchmark"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier (e.g. gpt-4, meta-llama/Llama-2-7b-chat-hf).")
    parser.add_argument("--backend", type=str, default="api", choices=["api", "local"])
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "scaffold"])
    parser.add_argument("--k", type=int, default=8, help="Few-shot examples (0 = zero-shot).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        backend=args.backend,
        sample_strategy=args.strategy,
        k=args.k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()