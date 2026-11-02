"""
Name Prediction Benchmark
============================
Modular benchmark for molecular naming and identifier prediction using LLMs.
Supports IUPAC name generation, SMILES prediction, and molecular formula tasks.

Usage (as a script):
    python -m Name_Prediction.name_pred \
        --task SMILES_TO_IUPAC --model gpt-4 --backend api \
        --strategy random --k 8 --n 1 --temperature 0.0

Usage (as a library):
    from ChemLLMBench.Name_Prediction.name_pred import run_benchmark
    results = run_benchmark(
        task_name="SMILES_TO_IUPAC", model_name="gpt-4", backend="api",
        sample_strategy="random", k=8,
    )
"""

import os
import re
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
from .data import load_dataset, TASK_COL_MAP
from utils import LiteLLMModel, HuggingFaceModel , extract_json_answer, strip_thinking

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings for invalid SMILES


# ── Supported tasks ─────────────────────────────────────────────────────────
SUPPORTED_TASKS = list(TASK_COL_MAP.keys())

# Column names / order used when reporting accuracy
TASK_SHORT_NAME = {
    "SMILES_TO_IUPAC":   "smiles2iupac",
    "IUPAC_TO_SMILES":   "iupac2smiles",
    "SMILES_TO_FORMULA": "smiles2formula",
    "IUPAC_TO_FORMULA":   "iupac2formula",
}
REPORT_COLUMNS = list(TASK_SHORT_NAME.values())

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


# ── Helper: parse the generated answer from raw LLM output ──────────────────
def _parse_prediction(raw_text, task_name):
    """Extract the predicted value from the raw LLM response.

    The model is prompted to reply in JSON format {"answer": "<value>"}.
    Falls back to `<col>: <value>` or the first non-empty line.

    Returns an empty string on parse failure.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    text = strip_thinking(text)
    if not text:
        return ""

    ans = extract_json_answer(text, key="answer")

    if ans:
        return ans.strip()

    _, to_col = TASK_COL_MAP[task_name]

    # Try structured pattern  "<to_col>: <answer>"
    pattern = re.compile(
        rf"{re.escape(to_col)}\s*:\s*(.+)", re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # Fallback: return the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line

    return ""


# ── Normalisers (one per target type) ───────────────────────────────────────
def _canonical(s):
    """Canonical SMILES, or "" if empty/invalid."""
    if not s:
        return ""
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol) if mol is not None else ""


def _formula_key(s):
    """Order-independent key for a molecular formula, e.g. 'H6C2O' == 'C2H6O'.

    Case-sensitive on purpose: 'CO' (carbon monoxide) is not 'Co' (cobalt).
    Falls back to the whitespace-stripped string if it can't be parsed
    (parentheses, charges, etc.).
    """
    s = re.sub(r"\s+", "", s)
    if not s:
        return ""
    tokens = _FORMULA_TOKEN.findall(s)
    if "".join(el + n for el, n in tokens) != s:
        return s
    counts = {}
    for el, n in tokens:
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return ",".join(f"{el}{c}" for el, c in sorted(counts.items()))


def _iupac_key(s):
    """Lower-cased, whitespace-collapsed IUPAC name."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _normaliser_for(to_col):
    return {"smiles": _canonical, "formula": _formula_key, "iupac": _iupac_key}[to_col]


# ── Helper: compute metrics (accuracy only) ─────────────────────────────────
def _compute_metrics(y_true, y_pred, task_name):
    """Accuracy: fraction of predictions equal to the reference after
    normalisation. Empty or invalid predictions never count as correct."""
    _, to_col = TASK_COL_MAP[task_name]
    norm = _normaliser_for(to_col)

    correct = 0
    for gt, pred in zip(y_true, y_pred):
        p = norm(str(pred).strip())
        t = norm(str(gt).strip())
        if p and p == t:
            correct += 1

    total = len(y_true)
    return {
        "task": TASK_SHORT_NAME[task_name],
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def format_accuracy_row(results):
    """results: {short_task_name: accuracy}. Returns (header_line, value_line)."""
    header = "  ".join(f"{c:>14}" for c in REPORT_COLUMNS)
    values = "  ".join(
        f"{results[c]:>14.4f}" if c in results else f"{'-':>14}" for c in REPORT_COLUMNS
    )
    return header, values


# ── Helper: build model ─────────────────────────────────────────────────────
def _build_model(backend, model_name, temperature, max_new_tokens=256):
    """Instantiate the right ModelInterface subclass.

    Uses a longer default *max_new_tokens* (256) than Property Prediction
    because IUPAC names can be lengthy.
    """
    if backend == "api":
        return LiteLLMModel(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
    elif backend == "local":
        return HuggingFaceModel(
            temperature=temperature,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'api' or 'local'.")


# ── Main benchmark runner ────────────────────────────────────────────────────
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
    """
    Run the name-prediction benchmark on one task.

    Returns
    -------
    dict  with keys: "metrics", "predictions", "raw_outputs"
    """
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task: {task_name!r}. Choose from {SUPPORTED_TASKS}"
        )

    # ── 1. Load data & prompts ───────────────────────────────────────────
    print(f"[*] Loading task: {task_name}  (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_dataset(
        name=task_name, sample_strategy=sample_strategy, k=k,
    )

    _, to_col = TASK_COL_MAP[task_name]
    y_true = df_test[to_col].astype(str).tolist()

    # ── 2. Build model ───────────────────────────────────────────────────
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = _build_model(
        backend=backend,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # ── 3. Run inference ─────────────────────────────────────────────────
    print(f"[*] Running inference on {len(user_prompts)} molecules …")
    all_outputs = model.generate_batch(user_prompts, system_prompts, n=n)
    raw_outputs = [outs[0] if outs else "" for outs in all_outputs]
    predictions = [_parse_prediction(t, task_name) for t in raw_outputs]

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    print("[*] Computing metrics …")
    metrics = _compute_metrics(y_true, predictions, task_name)

    print(f"\n{'═' * 70}")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"  Correct : {metrics['correct']}/{metrics['total']}")
    print(f"{'─' * 70}")
    header, values = format_accuracy_row({metrics["task"]: metrics["accuracy"]})
    print(header)
    print(values)
    print(f"{'═' * 70}\n")

    # ── 5. Save results ──────────────────────────────────────────────────
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


# ── CLI entry-point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ChemLLMBench – Molecular Name Prediction Benchmark"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=SUPPORTED_TASKS,
        help="Task name (e.g. SMILES_TO_IUPAC, IUPAC_TO_SMILES).",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model identifier (e.g. gpt-4, claude-3-opus-20240229, meta-llama/Llama-2-7b-chat-hf).",
    )
    parser.add_argument(
        "--backend", type=str, default="api", choices=["api", "local"],
        help="'api' for LiteLLM (OpenAI/Anthropic/etc.), 'local' for vLLM.",
    )
    parser.add_argument(
        "--strategy", type=str, default="random", choices=["random", "scaffold"],
        help="Few-shot example sampling strategy.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1, help="Number of completions per molecule.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")

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