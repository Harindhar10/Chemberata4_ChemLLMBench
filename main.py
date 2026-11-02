"""
ChemLLMBench – Main Entry Point
================================
Unified CLI to run one or more chemistry LLM benchmarks.

Quick start (run all tasks):
    python -m ChemLLMBench.main --model gpt-4 --backend api --tasks all

Run specific tasks:
    python -m ChemLLMBench.main --model gpt-4 --backend api \
        --tasks name_prediction property_prediction

Run the zero-shot reaction-component selection tasks:
    python -m ChemLLMBench.main --model gpt-4 --backend api \
        --tasks reaction_component_selection --data-dir path/to/csvs

Programmatic usage:
    from ChemLLMBench.main import run_all_benchmarks
    results = run_all_benchmarks(model_name="gpt-4", backend="api")
"""

import argparse
import datetime
import inspect
import json
import os


# ── Registry of available benchmark tasks ────────────────────────────────────
# Each entry maps a short task key to:
#   (display_name, module_import_path, run_function_name, extra_kwargs)
# extra_kwargs are merged into the common kwargs passed to run_benchmark.
# Common kwargs a function does not accept (e.g. `k` for zero-shot-only
# tasks) are dropped automatically, see _filter_kwargs.

# NOTE: adjust the module path of the selection tasks to wherever you put the
# reagent / ligand / solvent file.
SELECTION_MODULE = "Reaction_Selection.selection"

TASK_REGISTRY = {
    # ── Name Prediction (4 sub-tasks) ────────────────────────────────────
    "smiles_to_iupac": {
        "display": "Name Prediction – SMILES → IUPAC",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "SMILES_TO_IUPAC"},
    },
    "iupac_to_smiles": {
        "display": "Name Prediction – IUPAC → SMILES",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "IUPAC_TO_SMILES"},
    },
    "smiles_to_formula": {
        "display": "Name Prediction – SMILES → Formula",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "SMILES_TO_FORMULA"},
    },
    "iupac_to_formula": {
        "display": "Name Prediction – IUPAC → Formula",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "IUPAC_TO_FORMULA"},
    },

    # ── Property Prediction (5 datasets) ─────────────────────────────────
    "bbbp": {
        "display": "Property Prediction – BBBP",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "BBBP"},
    },
    "bace": {
        "display": "Property Prediction – BACE",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "BACE"},
    },
    "tox21": {
        "display": "Property Prediction – Tox21",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "Tox21"},
    },
    "hiv": {
        "display": "Property Prediction – HIV",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "HIV"},
    },
    "clintox": {
        "display": "Property Prediction – ClinTox",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "ClinTox"},
    },

    # ── Reaction Prediction ──────────────────────────────────────────────
    "reaction_prediction": {
        "display": "Forward Reaction Prediction",
        "module": "Reaction_prediction.reaction_pred",
        "function": "run_benchmark",
        "kwargs": {},
    },

    # ── Reaction Component Selection (3 sub-tasks, zero-shot only) ───────
    "reactant_selection": {
        "display": "Reaction Selection – Reactant (zero-shot)",
        "module": SELECTION_MODULE,
        "function": "run_benchmark",
        "kwargs": {"tasks": ["Reactant_selection"]},
    },
    "ligand_selection": {
        "display": "Reaction Selection – Ligand (zero-shot)",
        "module": SELECTION_MODULE,
        "function": "run_benchmark",
        "kwargs": {"tasks": ["Ligand_selection"]},
    },
    "solvent_selection": {
        "display": "Reaction Selection – Solvent (zero-shot)",
        "module": SELECTION_MODULE,
        "function": "run_benchmark",
        "kwargs": {"tasks": ["Solvent_selection"]},
    },

    # ── Retrosynthesis ───────────────────────────────────────────────────
    "retrosynthesis": {
        "display": "Retrosynthesis",
        "module": "Retrosynthesis.retro",
        "function": "run_benchmark",
        "kwargs": {},
    },

    # ── Molecule Design (2 sub-tasks) ────────────────────────────────────
    "mol_caption": {
        "display": "Molecule Caption (SMILES → Description)",
        "module": "Molecule_Design.Mol_design_and_mol_caption",
        "function": "run_benchmark",
        "kwargs": {"task_name": "MOL_CAPTION"},
    },
    "mol_design": {
        "display": "Molecule Design (Description → SMILES)",
        "module": "Molecule_Design.Mol_design_and_mol_caption",
        "function": "run_benchmark",
        "kwargs": {"task_name": "MOL_DESIGN"},
    },
    "Buchwald-Hartwig": {
        "display": "Reaction Yield Prediction – Buchwald-Hartwig",
        "module": "Yield_prediction.yield_pred",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "Buchwald-Hartwig"},
    },

    "Suzuki-Miyaura": {
        "display": "Reaction Yield Prediction – Suzuki-Miyaura",
        "module": "Yield_prediction.yield_pred",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "Suzuki-Miyaura"},
    },
}

# Convenience groups
TASK_GROUPS = {
    "all": list(TASK_REGISTRY.keys()),
    "name_prediction": [
        "smiles_to_iupac", "iupac_to_smiles",
        "smiles_to_formula", "formula_to_smiles",
    ],
    "property_prediction": ["bbbp", "bace", "tox21", "hiv", "clintox"],
    "reaction_prediction": ["reaction_prediction"],
    "reaction_component_selection": [
        "reagent_selection", "ligand_selection", "solvent_selection",
    ],
    "retrosynthesis": ["retrosynthesis"],
    "molecule_design": ["mol_caption", "mol_design"],

    "yield_prediction": ["Buchwald-Hartwig", "Suzuki-Miyaura"],
}

# Metric keys tried (in order) when printing the one-line summary
PRIMARY_METRIC_KEYS = ("accuracy", "exact_match", "top1_accuracy")
INVALID_METRIC_KEY = "invalid_smiles_percent"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _import_function(module_path: str, function_name: str):
    """Dynamically import a function from a dotted module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, function_name)


def _filter_kwargs(fn, kwargs):
    """Keep only the kwargs `fn` accepts (unless it takes **kwargs).

    Returns (accepted_kwargs, list_of_dropped_names).
    """
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = [k for k in kwargs if k not in params]
    return accepted, dropped


def _extract_metrics(result):
    """Get a flat metrics dict from a benchmark's return value.

    Handles both shapes:
      {"metrics": {...}, ...}                       (most tasks)
      {"Some_task": {"metrics": {...}, ...}}        (selection tasks)
    """
    if not isinstance(result, dict):
        return {}
    if isinstance(result.get("metrics"), dict):
        return result["metrics"]
    nested = [
        v["metrics"] for v in result.values()
        if isinstance(v, dict) and isinstance(v.get("metrics"), dict)
    ]
    return nested[0] if len(nested) == 1 else {}


def _primary_metric(metrics):
    """First available primary metric (0.0 is a valid value, not 'missing')."""
    for key in PRIMARY_METRIC_KEYS:
        if metrics.get(key) is not None:
            return metrics[key]
    return None


def _run_single_task(task_key, model_name, backend, sample_strategy, k,
                     temperature, max_new_tokens, output_dir, data_dir):
    """Run a single benchmark task and return its results dict."""
    entry = TASK_REGISTRY[task_key]
    print(f"  BENCHMARK: {entry['display']}")

    fn = _import_function(entry["module"], entry["function"])

    # Build kwargs: merge task-specific kwargs with common ones
    kwargs = dict(entry["kwargs"])
    kwargs["model_name"] = model_name
    kwargs["backend"] = backend
    kwargs["sample_strategy"] = sample_strategy
    kwargs["k"] = k
    kwargs["temperature"] = temperature
    kwargs["max_new_tokens"] = max_new_tokens
    kwargs["output_dir"] = output_dir
    kwargs["data_dir"] = data_dir

    # Drop anything this task's run_benchmark doesn't take
    # (e.g. k / sample_strategy for zero-shot-only tasks, data_dir for others)
    kwargs, dropped = _filter_kwargs(fn, kwargs)
    if dropped:
        print(f"  (ignored args for this task: {', '.join(dropped)})")

    try:
        result = fn(**kwargs)
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"\n[✗] Task {task_key} failed: {e}\n")
        return {"status": "error", "error": str(e)}


def run_all_benchmarks(
    model_name: str,
    backend: str = "api",
    tasks: list[str] | None = None,
    sample_strategy: str = "random",
    k: int = 8,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    output_dir: str = "results",
    data_dir: str = ".",
    thinking: str = "default",
    reasoning_effort: str | None = None,
):
    """
    Run one or more ChemLLMBench tasks.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. "gpt-4", "claude-3-opus-20240229").
    backend : str
        "api" for LiteLLM or "local" for vLLM.
    tasks : list[str] or None
        Task keys or group names (see TASK_REGISTRY / TASK_GROUPS).
        None or ["all"] runs every task.
    sample_strategy : str
        "random" or "scaffold" for few-shot example selection.
        Ignored by zero-shot-only tasks (reaction component selection).
    k : int
        Number of few-shot examples (0 for zero-shot).
        Ignored by zero-shot-only tasks (reaction component selection).
    temperature : float
        Sampling temperature.
    max_new_tokens : int
        Max tokens to generate.
    output_dir : str
        Directory to save per-task results.
    data_dir : str
        Directory containing the reagent/ligand/solvent *_sample.csv files.
        Only used by the reaction component selection tasks.
    thinking : str
        "on", "off" or "default" (keep the chat template's own behaviour).
        Local backend only.
    reasoning_effort : str or None
        Passed to the chat template as `reasoning_effort` (e.g. "low",
        "medium", "xhigh"). None keeps the template default. Local backend only.

    Returns
    -------
    dict[str, dict]
        Mapping from task_key to its results dict.
    """
    import utils
    if reasoning_effort is not None and thinking == "off":
        print("WARNING: --reasoning-effort has no effect with --thinking off")
    utils.set_thinking(thinking, reasoning_effort)
    utils.set_cache_dir(os.path.join(output_dir, "cache"))

    # Resolve task list
    if tasks is None:
        tasks = ["all"]

    task_keys = []
    for t in tasks:
        t_lower = t.lower()
        if t_lower in TASK_GROUPS:
            task_keys.extend(TASK_GROUPS[t_lower])
        elif t_lower in TASK_REGISTRY:
            task_keys.append(t_lower)
        else:
            raise ValueError(
                f"Unknown task or group: {t!r}. "
                f"Available tasks: {list(TASK_REGISTRY.keys())}. "
                f"Available groups: {list(TASK_GROUPS.keys())}."
            )

    # Deduplicate while preserving order
    seen = set()
    unique_keys = []
    for k_ in task_keys:
        if k_ not in seen:
            seen.add(k_)
            unique_keys.append(k_)
    task_keys = unique_keys

    print(f"{'═' * 70}")
    print(f"  ChemLLMBench — Running {len(task_keys)} benchmark(s)")
    print(f"  Model    : {model_name}")
    print(f"  Backend  : {backend}")
    print(f"  Strategy : {sample_strategy} (k={k})")
    print(f"  Thinking : {thinking} (effort={reasoning_effort or 'template default'})")
    print(f"  Data dir : {data_dir}")
    print(f"  Tasks    : {', '.join(task_keys)}")
    print(f"{'═' * 70}")

    all_results = {}
    for task_key in task_keys:
        result = _run_single_task(
            task_key=task_key,
            model_name=model_name,
            backend=backend,
            sample_strategy=sample_strategy,
            k=k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            output_dir=output_dir,
            data_dir=data_dir,
        )
        all_results[task_key] = result

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'─' * 70}")
    for task_key, res in all_results.items():
        display = TASK_REGISTRY[task_key]["display"]
        if res["status"] == "success":
            metrics = _extract_metrics(res["result"])
            primary = _primary_metric(metrics)
            metric_str = f"{primary:.4f}" if primary is not None else "done"
            invalid = metrics.get(INVALID_METRIC_KEY)
            invalid_str = f"  (invalid: {invalid:.2f}%)" if invalid is not None else ""
            print(f"  ✓ {display:<45s}  {metric_str}{invalid_str}")
        else:
            print(f"  ✗ {display:<45s}  FAILED: {res['error'][:40]}")
    print(f"{'═' * 70}\n")

    # Save combined summary
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = utils.run_tag("benchmark_summary", model_name, temperature, max_new_tokens, sample_strategy, k)
    summary_path = os.path.join(output_dir, f"{tag}.json")
    # Same config -> same file; keep task results from earlier runs of other tasks
    previous_tasks = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            previous_tasks = json.load(f).get("tasks", {})
    summary = {
        "model": model_name,
        "backend": backend,
        "sample_strategy": sample_strategy,
        "k": k,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "data_dir": data_dir,
        "timestamp": timestamp,
        "tasks": previous_tasks | {
            key: {
                "display": TASK_REGISTRY[key]["display"],
                "status": res["status"],
                "metrics": _extract_metrics(res["result"]) if res["status"] == "success" else None,
                "error": res.get("error"),
            }
            for key, res in all_results.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[✓] Summary saved to: {summary_path}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    available = sorted(list(TASK_REGISTRY.keys()) + list(TASK_GROUPS.keys()))

    parser = argparse.ArgumentParser(
        description="ChemLLMBench – Unified Chemistry LLM Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available tasks:\n"
            + "\n".join(f"  {k:<25s} {TASK_REGISTRY[k]['display']}" for k in TASK_REGISTRY)
            + "\n\nAvailable groups:\n"
            + "\n".join(f"  {k:<25s} → {', '.join(v)}" for k, v in TASK_GROUPS.items())
        ),
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model identifier (e.g. gpt-4, claude-3-opus-20240229).",
    )
    parser.add_argument(
        "--backend", type=str, default="api", choices=["api", "local"],
        help="'api' for LiteLLM (OpenAI/Anthropic/etc.), 'local' for vLLM.",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["all"],
        help=f"Task keys or group names to run. Choices: {available}",
    )
    parser.add_argument(
        "--strategy", type=str, default="random", choices=["random", "scaffold"],
        help="Few-shot example sampling strategy (not used by selection tasks).",
    )
    parser.add_argument("--k", type=int, default=8,
                        help="Number of few-shot examples (0 = zero-shot; not used by selection tasks).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory with reagent/ligand/solvent *_sample.csv files "
                             "(selection tasks only).")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="vLLM: max sequences decoded concurrently (lower it if the "
                             "KV/Mamba cache is too small).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="vLLM: fraction of GPU memory the engine may use.")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="vLLM: max context length (default: model's own limit).")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="vLLM: disable CUDA graphs (slower, uses less memory).")
    parser.add_argument("--api-concurrency", type=int, default=8,
                        help="API backend: number of concurrent requests.")
    parser.add_argument("--thinking", type=str, default="default",
                        choices=["default", "on", "off"],
                        help="Local models: enable/disable reasoning via the chat "
                             "template (default: template's own behaviour).")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "xhigh"],
                        help="Local models: reasoning_effort chat-template variable "
                             "(default: template's own default).")

    args = parser.parse_args()

    import utils
    utils.set_vllm_engine_args(
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    utils.set_api_concurrency(args.api_concurrency)

    run_all_benchmarks(
        model_name=args.model,
        backend=args.backend,
        tasks=args.tasks,
        sample_strategy=args.strategy,
        k=args.k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()