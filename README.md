# ChemLLMBench: Unified Chemistry Large Language Model Benchmark Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/chemistry-RDKit-green.svg)](https://www.rdkit.org/)
[![vLLM](https://img.shields.io/badge/inference-vLLM-orange.svg)](https://github.com/vllm-project/vllm)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-purple.svg)](https://github.com/BerriAI/litellm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ChemLLMBench** is an end-to-end, modular benchmarking framework designed to evaluate Large Language Models (LLMs) across a comprehensive spectrum of chemical discovery, synthesis, molecular naming, and property prediction tasks. 

It provides seamless support for **API-based models** (OpenAI GPT-4/o, Anthropic Claude, Mistral, Gemini, etc. via LiteLLM) and **local open-source models** (Llama, Mistral, Qwen, ChemLLM, ChemBERTa via vLLM), with advanced **scaffold-based in-context learning**, rigorous chemical validation with **RDKit**, and automated observability via **Langfuse**.

---

## Table of Contents

- [Overview & Key Features](#overview--key-features)
- [Repository Structure](#repository-structure)
- [Benchmark Tasks & Datasets](#benchmark-tasks--datasets)
  - [1. Molecular Name & Identifier Prediction](#1-molecular-name--identifier-prediction)
  - [2. Molecular Property Prediction](#2-molecular-property-prediction)
  - [3. Forward Reaction Prediction](#3-forward-reaction-prediction)
  - [4. Retrosynthesis](#4-retrosynthesis)
  - [5. Reaction Component Selection](#5-reaction-component-selection)
  - [6. Molecule Design & Captioning](#6-molecule-design--captioning)
  - [7. Reaction Yield Prediction](#7-reaction-yield-prediction)
- [Evaluation Metrics & Chemical Validation](#evaluation-metrics--chemical-validation)
- [Installation & Environment Setup](#installation--environment-setup)
- [Configuration (.env)](#configuration-env)
- [How to Run ChemLLMBench](#how-to-run-chemllmbench)
  - [1. Unified CLI (main.py)](#1-unified-cli-mainpy)
  - [2. Running Specific Task Groups](#2-running-specific-task-groups)
  - [3. Running Individual Task Modules](#3-running-individual-task-modules)
  - [4. Programmatic Python API](#4-programmatic-python-api)
- [CLI Options & Argument Reference](#cli-options--argument-reference)
- [Results & Output Formats](#results--output-formats)
- [LLM Observability & Cost Tracking](#llm-observability--cost-tracking)
- [Citation & Acknowledgments](#citation--acknowledgments)

---

## Overview & Key Features

- **Unified CLI & API**: Run 18+ chemistry tasks across 7 domains with a single command or import modular benchmarks as Python functions.
- **Dual Inference Backends**:
  - `api`: High-throughput API calls via [LiteLLM](https://github.com/BerriAI/litellm) (supporting OpenAI, Anthropic, DeepSeek, Bedrock, VertexAI, Ollama, etc.) with automated retries and cost tracking.
  - `local`: High-throughput local GPU inference via [vLLM](https://github.com/vllm-project/vllm) with HuggingFace chat templates.
- **Chemistry-Specific In-Context Learning**:
  - `random`: Uniformly sampled few-shot examples.
  - `scaffold`: Retrieval of few-shot demonstrations based on Bemis-Murcko scaffold similarity using Morgan fingerprints (ECFP4) and Tanimoto coefficient distance, testing out-of-scaffold generalization.
- **Robust Parsing & Canonicalisation**:
  - Primary JSON extraction (`{"answer": "<prediction>"}`) with multi-level regex fallback.
  - Molecular parsing and canonicalisation using RDKit (`Chem.MolToSmiles`), invalid SMILES rate tracking, and order-independent molecular formula normalization.
- **Full Observability**: Integrated with [Langfuse](https://langfuse.com/) to track token counts, per-call latencies, traces, and financial cost.

---

## Repository Structure

```text
Chemberata4_ChemLLMBench/
├── .env.example                     # Environment template for API keys & Langfuse
├── requirements.txt                 # Python dependencies
├── main.py                          # Unified CLI and orchestration entry point
├── utils.py                         # Model interfaces (LiteLLMModel, HuggingFaceModel)
│
├── Name_Prediction/                 # Molecular identifier prediction tasks
│   ├── name_pred.py                 # Runner for SMILES ↔ IUPAC & SMILES ↔ Formula
│   ├── data.py                      # Data loaders, prompts, and scaffold sampling
│   └── llm_test.csv                 # Evaluation split
│
├── Property_Prediction/             # Molecular property prediction benchmark
│   ├── Property_prediction.py       # Binary classification runner (BBBP, BACE, etc.)
│   └── data.py                      # Dataset loaders from DeepChem, prompt templates
│
├── Reaction_prediction/             # Forward reaction prediction
│   ├── reaction_pred.py             # Product prediction runner & validity checker
│   ├── data.py                      # USPTO reaction data loading & box caching
│   └── uspto_test.csv               # Test reactions
│
├── Retrosynthesis/                  # Retrosynthetic pathway prediction
│   ├── retro.py                     # Reactant prediction runner & canonicalisation
│   ├── data.py                      # USPTO-50k data loaders and prompt logic
│   └── uspto50k_retro_test.csv      # Test retrosynthesis set
│
├── Reaction_Selection/              # Zero-shot component optimization
│   ├── selection.py                 # Runner for Reagent/Ligand/Solvent selection
│   ├── data.py                      # Data loaders and yield-ground-truth generation
│   ├── ligand_sample.csv            # Ligand selection dataset
│   ├── reagent_sample.csv           # Reagent selection dataset
│   └── solvent_sample.csv           # Solvent selection dataset
│
├── Molecule_Design/                 # Text-guided generation & description
│   ├── Mol_design_and_mol_caption.py# Runner for molecule captioning & design
│   └── data.py                      # ChEBI-20 data loaders (MolT5 dataset)
│
└── Yeild_prediction/                # Reaction yield binary classification
    ├── yeild_pred.py                # Runner for Buchwald-Hartwig & Suzuki-Miyaura
    ├── data.py                      # .npz data loader and high-yield thresholding
    ├── BH_dataset .npz              # Buchwald-Hartwig dataset
    └── Suzuki.npz                   # Suzuki-Miyaura dataset
```

---

## Benchmark Tasks & Datasets

### 1. Molecular Name & Identifier Prediction
Translates between chemical structure representations and standardized naming formats:
- **`smiles_to_iupac`**: Predict systematic IUPAC name from SMILES.
- **`iupac_to_smiles`**: Generate canonical molecular SMILES from IUPAC name.
- **`smiles_to_formula`**: Infer the molecular formula from SMILES.
- **`formula_to_smiles`**: Generate valid SMILES matching the chemical formula.
- **Dataset**: `Name_Prediction/llm_test.csv`.

### 2. Molecular Property Prediction
Evaluates qualitative molecular property and bioactivity prediction across 5 benchmark datasets from MoleculeNet / DeepChem:
- **`bbbp`**: Blood-Brain Barrier Penetration (p/np).
- **`bace`**: Binding affinity / inhibition of human beta-secretase 1 (BACE-1).
- **`tox21`**: Toxicity prediction across 12 toxicological pathways.
- **`hiv`**: HIV replication inhibition (`HIV_active`).
- **`clintox`**: Clinical trial toxicity failure vs. FDA approval status.
- **Evaluation**: Binary classification (Yes / No). Metrics: Accuracy, F1-Score, Valid/Invalid prediction counts.

### 3. Forward Reaction Prediction
Given starting reactants and reagents separated by `.`, predict the main reaction product SMILES:
- **Task key**: `reaction_prediction`.
- **Dataset**: USPTO reaction dataset.
- **Metrics**: Top-1 Exact Match Accuracy (RDKit canonical SMILES), Invalid SMILES percentage.

### 4. Retrosynthesis
Given a target product molecule, predict the required reactants/precursors:
- **Task key**: `retrosynthesis`.
- **Dataset**: USPTO-50k dataset (`uspto50k_retro_test.csv`).
- **Metrics**: Top-1 Accuracy over multi-component reactant sets (each disconnected fragment canonicalised and sorted), Invalid SMILES percentage.

### 5. Reaction Component Selection
Evaluates the model's ability to select the optimal reaction component that maximizes reaction yield:
- **`reagent_selection`**: Predict optimal reactant/reagent.
- **`ligand_selection`**: Predict optimal ligand in Pd-catalyzed Suzuki-Miyaura reactions.
- **`solvent_selection`**: Predict optimal solvent.
- **Setting**: Zero-shot selection from candidates provided in the prompt.
- **Metrics**: Accuracy against highest-yielding experimental component, Macro Average Accuracy.

### 6. Molecule Design & Captioning
Bi-directional translation between natural language and molecular structures using the ChEBI-20 dataset:
- **`mol_caption`** (SMILES → Text): Generate detailed description of molecular function and classification.
  - *Metrics*: BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, METEOR.
- **`mol_design`** (Text → SMILES): Design a novel or exact molecular structure matching textual requirements.
  - *Metrics*: Exact Match, Character-level BLEU, Levenshtein Distance, Validity %, Fingerprint Tanimoto Similarity (MACCS, RDKit, Morgan/ECFP4), and Fréchet ChemNet Distance (FCD).

### 7. Reaction Yield Prediction
Predicts whether a chemical reaction achieves high yield (> 70% yield threshold):
- **`Buchwald-Hartwig`**: Pd-catalyzed C-N cross-couplings with additive variations.
- **`Suzuki-Miyaura`**: C-C bond cross-couplings.
- **Metrics**: Binary classification accuracy (Yes/No).

---

## Evaluation Metrics & Chemical Validation

| Task Category | Primary Metric | Chemical Validation & Secondary Metrics |
| :--- | :--- | :--- |
| **Name Prediction** | Accuracy | RDKit canonical SMILES equality, element-count formula invariant keys, lowercase IUPAC normalization |
| **Property Prediction** | Accuracy / F1 | Parsing validity rate; unparseable predictions scored as wrong |
| **Forward Reaction** | Top-1 Accuracy | RDKit canonicalisation; `% Invalid SMILES` tracked |
| **Retrosynthesis** | Top-1 Accuracy | Dot-separated fragment sorting & canonicalisation: `".".join(sorted(frags))` |
| **Reaction Selection** | Accuracy | Match against experimental `max(yield_details)` ground-truth |
| **Molecule Caption** | BLEU-4 / ROUGE-L | Standard NLP evaluation metrics via HuggingFace `evaluate` |
| **Molecule Design** | Exact Match / Validity | MACCS, RDK, Morgan Tanimoto similarity; Levenshtein distance; FCD |
| **Yield Prediction** | Accuracy | Binary classification threshold (> 70%) |

---

## Installation & Environment Setup

### 1. Prerequisites
- Linux / macOS / WSL2 (Windows Subsystem for Linux)
- Python 3.10 or higher
- NVIDIA GPU with CUDA 12+ (required only if running local inference via `vLLM`)

### 2. Clone and Create Virtual Environment

```bash
git clone https://github.com/your-username/ChemLLMBench.git
cd ChemLLMBench

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on RDKit and PyTorch**: RDKit is installed directly via `pip install rdkit`. For local GPU acceleration with vLLM, ensure your PyTorch version is compatible with your installed CUDA driver.

---

## Configuration (.env)

ChemLLMBench reads environment variables automatically using `python-dotenv`. Copy the example file and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# API Keys for API-based models
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Langfuse tracing & observability
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

---

## How to Run ChemLLMBench

### 1. Unified CLI (`main.py`)

The main entry point is `main.py`, which provides a standardized interface for running any task, group, or the full benchmark suite.

#### Run all benchmarks with OpenAI GPT-4 (API):
```bash
python main.py --model gpt-4 --backend api --tasks all
```

#### Run with a local open-source model using vLLM:
```bash
python main.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --backend local \
    --tasks all \
    --temperature 0.0
```

#### Run a large MoE model across 4 GPUs (vLLM engine args mirror `vllm serve` flags):
```bash
export VLLM_USE_RUST_FRONTEND=1
python main.py \
    --model Inferact/Qwen3.8-Flash-Next-NVFP4 \
    --backend local \
    --tasks all \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --moe-backend marlin \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --attention-config.indexer_kv_dtype fp8 \
    --thinking on \
    --max-new-tokens 16384
```
There is no in-process equivalent of `--reasoning-parser`; `<think>` blocks in the raw output are removed by `strip_thinking` before scoring.

---

### 2. Running Specific Task Groups

Use the predefined task groups to target specific domains:

```bash
# Run all Molecular Name Prediction tasks
python main.py --model gpt-4o --backend api --tasks name_prediction

# Run all Property Prediction benchmarks
python main.py --model gpt-4o --backend api --tasks property_prediction

# Run Reaction Component Selection
python main.py --model gpt-4o --backend api --tasks reaction_component_selection

# Run Reaction Prediction and Retrosynthesis
python main.py --model gpt-4o --backend api --tasks reaction_prediction retrosynthesis

# Run Molecule Design & Captioning
python main.py --model gpt-4o --backend api --tasks molecule_design
```

#### Available Task Groups:
- `all`: Executes all 16 registered benchmark tasks.
- `name_prediction`: `smiles_to_iupac`, `iupac_to_smiles`, `smiles_to_formula`, `formula_to_smiles`
- `property_prediction`: `bbbp`, `bace`, `tox21`, `hiv`, `clintox`
- `reaction_prediction`: `reaction_prediction`
- `reaction_component_selection`: `reagent_selection`, `ligand_selection`, `solvent_selection`
- `retrosynthesis`: `retrosynthesis`
- `molecule_design`: `mol_caption`, `mol_design`

---

### 3. Running Individual Task Modules

Each subfolder contains its own standalone script that can be executed directly for fine-grained debugging and testing.

#### Name Prediction:
```bash
python -m Name_Prediction.name_pred \
    --task SMILES_TO_IUPAC \
    --model gpt-4 \
    --backend api \
    --strategy scaffold \
    --k 8 \
    --temperature 0.0
```

#### Property Prediction:
```bash
python -m Property_Prediction.Property_prediction \
    --dataset BBBP \
    --model gpt-4 \
    --backend api \
    --strategy random \
    --k 8
```

#### Forward Reaction Prediction:
```bash
python -m Reaction_prediction.reaction_pred \
    --model gpt-4 \
    --backend api \
    --strategy scaffold \
    --k 8
```

#### Retrosynthesis:
```bash
python -m Retrosynthesis.retro \
    --model gpt-4 \
    --backend api \
    --strategy random \
    --k 8
```

#### Reaction Component Selection:
```bash
python -m Reaction_Selection.selection \
    --model gpt-4 \
    --backend api \
    --tasks Reagent_selection Ligand_selection Solvent_selection
```

#### Molecule Design & Captioning:
```bash
# Molecule Captioning (SMILES -> Description)
python -m Molecule_Design.Mol_design_and_mol_caption \
    --task MOL_CAPTION \
    --model gpt-4 \
    --backend api \
    --k 4

# Molecule Design (Description -> SMILES)
python -m Molecule_Design.Mol_design_and_mol_caption \
    --task MOL_DESIGN \
    --model gpt-4 \
    --backend api \
    --k 4
```

#### Reaction Yield Prediction:
```bash
python -m Yeild_prediction.yeild_pred \
    --dataset Buchwald-Hartwig \
    --model gpt-4 \
    --backend api \
    --strategy random \
    --k 8
```

---

### 4. Programmatic Python API

You can import and run benchmarks directly inside your Python scripts or Jupyter notebooks:

```python
from main import run_all_benchmarks

# Run multiple tasks programmatically
results = run_all_benchmarks(
    model_name="gpt-4o",
    backend="api",
    tasks=["bbbp", "smiles_to_iupac", "reaction_prediction"],
    sample_strategy="scaffold",
    k=8,
    temperature=0.0,
    output_dir="results",
)

# Inspect metrics
for task, res in results.items():
    print(f"Task {task}: {res['status']}")
    if res["status"] == "success":
        print("Metrics:", res["result"]["metrics"])
```

You can also import individual task runners:

```python
from Property_Prediction.Property_prediction import run_benchmark as run_property_bench

results = run_property_bench(
    dataset_name="BACE",
    model_name="gpt-4",
    backend="api",
    sample_strategy="scaffold",
    k=4,
)
print("Accuracy:", results["metrics"]["accuracy"])
```

---

## CLI Options & Argument Reference

### Options for `main.py`:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | *(Required)* | Model identifier (e.g. `gpt-4`, `claude-3-5-sonnet-20240620`, `meta-llama/Llama-2-7b-chat-hf`) |
| `--backend` | `str` | `api` | Inference engine: `api` (LiteLLM) or `local` (vLLM) |
| `--tasks` | `list` | `all` | One or more task keys or task group names |
| `--strategy`| `str` | `random` | Few-shot demonstration sampling: `random` or `scaffold` |
| `--k` | `int` | `8` | Number of few-shot in-context examples (`0` = zero-shot) |
| `--temperature` | `float` | `0.0` | Generation sampling temperature |
| `--max-new-tokens`| `int` | `256` | Maximum new tokens to generate |
| `--output-dir`| `str` | `results`| Directory to store result files and summaries |
| `--data-dir` | `str` | `.` | Directory containing `*_sample.csv` files (for reaction selection) |
| `--max-num-seqs` | `int` | `256` | vLLM: max sequences decoded concurrently |
| `--gpu-memory-utilization` | `float` | `0.90` | vLLM: fraction of GPU memory the engine may use |
| `--max-num-batched-tokens` | `int` | vLLM default | vLLM: max tokens scheduled per step |
| `--enable-prefix-caching` | flag | vLLM default | vLLM: reuse KV cache for shared prompt prefixes |
| `--tensor-parallel-size` | `int` | vLLM default | vLLM: number of GPUs to shard across |
| `--kv-cache-dtype` | `str` | vLLM default | vLLM: KV cache dtype (e.g. `fp8`) |
| `--moe-backend` | `str` | vLLM default | vLLM: MoE kernel backend (e.g. `marlin`) |
| `--attention-config.indexer_kv_dtype` | `str` | vLLM default | vLLM: `attention_config={"indexer_kv_dtype": ...}` |

---

## Results & Output Formats

All evaluation runs write structured outputs to the directory specified by `--output-dir` (default: `results/`):

1. **Per-Task Predictions (`.csv`)**:
   Contains sample inputs, ground truth references, model raw completions, parsed answers, and correctness/validity flags.
   - Example: `results/BBBP_gpt-4_scaffold_k8_20261028_120000_predictions.csv`

2. **Per-Task Metric Metadata (`.json`)**:
   Stores run configuration (model, backend, strategy, k, temperature) and final evaluated metrics.
   - Example: `results/BBBP_gpt-4_scaffold_k8_20261028_120000_metrics.json`

3. **Combined Benchmark Summary (`.json`)**:
   Generated automatically when using `main.py`, aggregating scores and statuses across all selected tasks:
   - Example: `results/benchmark_summary_20261028_120000.json`

A formatted summary table is also printed directly to your terminal at the conclusion of each run:

```text
══════════════════════════════════════════════════════════════════════
  BENCHMARK SUMMARY
──────────────────────────────────────────────────────────────────────
  ✓ Name Prediction – SMILES → IUPAC             0.4200
  ✓ Property Prediction – BBBP                   0.7800
  ✓ Forward Reaction Prediction                  0.6100  (invalid: 3.00%)
  ✓ Retrosynthesis                               0.4800  (invalid: 5.00%)
  ✓ Reaction Selection – Ligand (zero-shot)      0.5400
══════════════════════════════════════════════════════════════════════
[✓] Summary saved to: results/benchmark_summary_20261028_120000.json
```

---

## LLM Observability & Cost Tracking

ChemLLMBench integrates natively with [Langfuse](https://langfuse.com/) via LiteLLM callbacks. 

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are provided in your `.env` file:
- Every prompt and completion is automatically logged.
- Token counts, completion costs (USD), and response latencies are calculated and tracked in real-time.
- Full trace visualisations are available directly in your Langfuse dashboard.

If keys are omitted, ChemLLMBench automatically operates in offline mode without logging.

---
