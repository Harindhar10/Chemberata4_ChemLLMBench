import os
import re
import json
import argparse
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Import your custom data loader
try:
    from .data import load_function
except ImportError:
    from data import load_function
# Assuming utils is a local module you have that contains these classes
from utils import LiteLLMModel, HuggingFaceModel, extract_json_answer, strip_thinking

SUPPORTED_DATASETS = ["Buchwald-Hartwig", "Suzuki-Miyaura"]

def _parse_prediction(raw_text):
    """Extract the first Yes/No answer relevant to the target task.
    Returns 1 (Yes), 0 (No), or -1 (parse failure)."""
    if not raw_text:
        return -1

    text = raw_text.strip()
    text = strip_thinking(text)
    ans = extract_json_answer(text, key="answer")

    if ans:
        return ans.strip()

    # Try to find "yeild: Yes/No" or "yield: Yes/No" pattern first
    pattern = re.compile(
        r"y(?:ei|ie)ld\s*:\s*(Yes|No)", re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0

    # Fallback: first standalone Yes or No in the text
    fallback = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if fallback:
        return 1 if fallback.group(1).lower() == "yes" else 0

    return -1


def _compute_metrics(y_true, y_pred):
    valid = np.array(y_pred) != -1
    total = len(y_true)
    n_valid = int(valid.sum())
    n_invalid = total - n_valid

    # For metric computation, replace invalid predictions with wrong answer
    y_pred_clean = []
    for yt, yp in zip(y_true, y_pred):
        if yp == -1:
            y_pred_clean.append(1 - yt)  # guaranteed wrong
        else:
            y_pred_clean.append(yp)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_clean)

    metrics = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "total": total,
        "valid_predictions": n_valid,
        "invalid_predictions": n_invalid,
    }

    return metrics


def _build_model(backend, model_name, temperature, max_new_tokens=50):
    """Instantiate the right ModelInterface subclass."""
    if backend == "api":
        return LiteLLMModel(
            model_name=model_name,
            max_new_tokens=max_new_tokens
        )
    elif backend == "local":
        return HuggingFaceModel(
            temperature=temperature,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'api' or 'local'.")


def run_benchmark(
    dataset_name: str,
    model_name: str,
    backend: str = "api",
    sample_strategy: str = "random",
    k: int = 8,
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 50,
    output_dir: str = "results",
):
    """
    Run the yield-prediction benchmark on one dataset.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Choose from {SUPPORTED_DATASETS}"
        )

    # ── 1. Load data & prompts ───────────────────────────────────────────
    print(f"[*] Loading dataset: {dataset_name}  (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_function(
        name=dataset_name, sample_strategy=sample_strategy, k=k
    )

    # Map 'Yes' to 1 and 'No' to 0 for metric computation
    y_true = df_test["yeild"].map({"Yes": 1, "No": 0}).tolist()

    # ── 2. Build model ───────────────────────────────────────────────────
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = _build_model(
        backend=backend,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # ── 3. Run inference ─────────────────────────────────────────────────
    raw_outputs = []
    predictions = []

    print(f"[*] Running inference on {len(user_prompts)} molecules …")
    for idx in tqdm(range(len(user_prompts)), desc=dataset_name):
        sys_prompt = system_prompts[idx]
        usr_prompt = user_prompts[idx]

        # Generate model response
        outputs = model.generate(
            user_prompt=usr_prompt, sys_prompt=sys_prompt,
            n=n
        )

        raw_text = outputs[0] if outputs else ""
        raw_outputs.append(raw_text)

        pred = _parse_prediction(raw_text)
        predictions.append(pred)

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    print("[*] Computing metrics …")
    metrics = _compute_metrics(y_true, predictions)

    print(f"\n{'═' * 50}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"{'─' * 50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Valid    : {metrics['valid_predictions']}/{metrics['total']}")
    print(f"{'═' * 50}\n")

    # ── 5. Save results ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    run_tag = f"{dataset_name}_{safe_model}_{sample_strategy}_k{k}_{timestamp}"

    # Save predictions CSV
    results_df = df_test.copy().reset_index(drop=True)
    results_df["raw_output"] = raw_outputs
    results_df["prediction"] = predictions
    results_df["prediction_label"] = [
        "Yes" if p == 1 else ("No" if p == 0 else "INVALID") for p in predictions
    ]
    results_df["ground_truth_label"] = df_test["yeild"]
    csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
    results_df.to_csv(csv_path, index=False)

    # Save metrics JSON
    meta = {
        "dataset": dataset_name,
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
        description="ChemLLMBench – Reaction Yield Prediction Benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name (Buchwald-Hartwig or Suzuki-Miyaura).",
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
        "--strategy", type=str, default="random", choices=["random"],
        help="Few-shot example sampling strategy.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1, help="Number of completions per molecule.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens (local models).")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")

    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset,
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