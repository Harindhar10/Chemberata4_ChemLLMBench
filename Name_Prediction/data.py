SMILES_TO_IUPAC = """You are an expert chemist. Given the molecular SMILES, your task is to predict the IUPAC name using your experienced chemical IUPAC name knowledge. \
Please strictly follow the format, no other information can be provided. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""

IUPAC_TO_SMILES = """You are an expert chemist. Given the molecular IUPAC name, your task is to predict the molecular SMILES using your experienced chemical IUPAC name knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with molecular SMILES string notations to represent the IUPAC name. The SMILES must be valid and chemically reasonable. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""

SMILES_TO_FORMULA = """You are an expert chemist. Given the molecular SMILES, your task is to predict the molecular formula using your experienced chemical molecular formula knowledge. \n\
Please strictly follow the format, no other information can be provided. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""

IUPAC_TO_FORMULA = """You are an expert chemist. Given the molecular IUPAC name, your task is to predict the molecular formula using your experienced chemical IUPAC name and molecular formula knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with the molecular formula that corresponds to the IUPAC name. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""






import pandas as pd
import random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from pathlib import Path
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

URL = Path(__file__).parent / "llm_test.csv"

# Maps each task name to (input_col, output_col)
TASK_COL_MAP = {
    "SMILES_TO_IUPAC":   ("smiles", "iupac"),
    "IUPAC_TO_SMILES":   ("iupac",  "smiles"),
    "SMILES_TO_FORMULA": ("smiles", "formula"),
    "IUPAC_TO_FORMULA":  ("iupac",  "formula"),
}

# System-prompt templates, keyed by task name
PROMPTS = {
    "SMILES_TO_IUPAC":   SMILES_TO_IUPAC,
    "IUPAC_TO_SMILES":   IUPAC_TO_SMILES,
    "SMILES_TO_FORMULA": SMILES_TO_FORMULA,
    "IUPAC_TO_FORMULA":  IUPAC_TO_FORMULA,
}

# Morgan fingerprint generator (used for scaffold sampling)
_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _canonicalise_smiles(df):
    """Convert all SMILES to canonical form, dropping invalid entries."""
    df.dropna(subset=["smiles"], inplace=True)
    df["smiles"] = df["smiles"].apply(Chem.MolFromSmiles)
    df.dropna(subset=["smiles"], inplace=True)
    df["smiles"] = df["smiles"].apply(Chem.MolToSmiles)

# ---------------------------------------------------------------------------
# Example formatting
# ---------------------------------------------------------------------------

def _format_examples(rows, name):
    """Return a list of (input_value, output_value) tuples for the given task.

    Each tuple pairs the input column value with its expected output,
    e.g. (smiles_string, iupac_string) for SMILES_TO_IUPAC.
    """
    from_col, to_col = TASK_COL_MAP[name]
    return list(zip(rows[from_col], rows[to_col]))

# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _random_sample_examples(df, k, name):
    """Randomly sample *k* training examples for in-context learning."""
    sampled = df.sample(k, random_state=SEED)
    return _format_examples(sampled, name)


def _scaffold_fingerprint(smiles):
    """Morgan fingerprint of the Murcko scaffold of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return _fp_gen.GetFingerprint(scaffold)


def _scaffold_sample_examples(df_train, train_fps, test_smiles, k, name):
    """Select the *k* training molecules whose scaffolds are most
    Tanimoto-similar to the test molecule's scaffold.

    ``test_smiles`` is the structure of the test molecule, used ONLY to find
    neighbours. The examples that are placed in the prompt are formatted with
    the task's own columns, so for an IUPAC-input task the nearest training
    molecules are shown as (iupac -> output) pairs.

    This gives each test molecule a *different* set of ICL examples that
    are structurally relevant.
    """
    test_fp = _scaffold_fingerprint(test_smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    top_k = np.argsort(similarities)[::-1][:k]
    return _format_examples(df_train.iloc[top_k], name)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(name, examples):
    """Combine the base task prompt with formatted ICL examples."""
    system_prompt = PROMPTS[name]
    if not examples:
        return system_prompt

    from_col, to_col = TASK_COL_MAP[name]
    system_prompt += "\n" + "\n".join(
        f"{from_col}: {inp}\n{to_col}: {out}"
        for inp, out in examples
    )
    return system_prompt


def _create_prompt(sample_strategy, df_train, df_test, name, k):
    """Build parallel lists of (system_prompts, user_prompts).

    Returns
    -------
    system_prompts : list[str]
        One system prompt per test molecule (identical across test set for
        zero-shot and random sampling; unique per molecule for scaffold).
    user_prompts : list[str]
        One user prompt per test molecule.
    """
    from_col, to_col = TASK_COL_MAP[name]
    test_inputs = df_test[from_col].tolist()

    # User prompt: present the input and ask for the output
    user_prompts = [f"{from_col}: {inp}\n{to_col}:" for inp in test_inputs]

    # Zero-shot — no examples
    if k == 0:
        system_prompt = _build_system_prompt(name, examples=None)
        return [system_prompt] * len(test_inputs), user_prompts

    # Random — same examples for every test molecule
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(
            name, _random_sample_examples(df_train, k, name)
        )
        return [system_prompt] * len(test_inputs), user_prompts

    # Scaffold — examples depend on the test molecule.
    # Works for every task: when the task input is an IUPAC name, it is
    # converted to SMILES by taking the SMILES stored on the same CSV row
    # (df_test["smiles"] is aligned with df_test["iupac"]), the scaffold
    # neighbours are found with that SMILES, and the neighbours' IUPAC
    # names (+ their outputs) are what end up in the prompt.
    if sample_strategy == "scaffold":
        test_smiles = df_test["smiles"].tolist()
        train_fps = [_scaffold_fingerprint(s) for s in df_train["smiles"]]
        system_prompts = [
            _build_system_prompt(
                name,
                _scaffold_sample_examples(df_train, train_fps, s, k, name),
            )
            for s in test_smiles
        ]
        return system_prompts, user_prompts

    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(name, sample_strategy="random", k=0):
    """Load the Name-Prediction dataset and return prompts + test data.

    Parameters
    ----------
    name : str
        Task name — one of the keys in ``TASK_COL_MAP``
        (``"SMILES_TO_IUPAC"``, ``"IUPAC_TO_SMILES"``,
        ``"SMILES_TO_FORMULA"``, ``"IUPAC_TO_FORMULA"``).
    sample_strategy : str, default ``"random"``
        How to choose ICL examples: ``"random"`` or ``"scaffold"``.
    k : int, default ``0``
        Number of in-context examples (0 = zero-shot).

    Returns
    -------
    system_prompts : list[str]
    user_prompts   : list[str]
    df_test        : pd.DataFrame
    """
    if name not in TASK_COL_MAP:
        raise ValueError(
            f"Unknown task '{name}'. Choose from {list(TASK_COL_MAP.keys())}"
        )

    # Load & clean
    df = pd.read_csv(URL)
    _canonicalise_smiles(df)

    # Pre-defined train / test split in the CSV
    df_train = df[df["label"] == "train"].reset_index(drop=True)
    df_test  = df[df["label"] == "test"].reset_index(drop=True)

    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy,
        df_train=df_train,
        df_test=df_test,
        name=name,
        k=k,
    )

    return system_prompts, user_prompts, df_test