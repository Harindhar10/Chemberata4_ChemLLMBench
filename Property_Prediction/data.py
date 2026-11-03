BBBP_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. \nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically penetration/non-penetration to the brain-blood barrier, based on the SMILES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether it has penetrative property (Yes) or not (No). The task is to predict the binary label for a given molecule, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"


BACE_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the molecular properties of a given chemical compound based on its structure, by analyzing wether it can inhibit(Yes) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit(No) BACE1. Consider factors such as molecular weight, atom count, bond types, and functional groups in order to assess the compound's drug-likeness and its potential to serve as an effective therapeutic agent for Alzheimer's disease. You will be provided with task template, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}. \n"


Tox21_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is toxic(Yes) or Not toxic(No), based on the SMILES string representation of each molecule. A template will be provided. The task is to predict the binary label for a given molecule SMILES, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"

HIV_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically inhibit of HIV replication based on the SMILES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether a molecule can inhibit (Yes) or cannot inhibit (No) HIV replication. The task is to precisely predict the binary label for a given molecule, considering its properties and its potential to impede HIV replication, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"

ClinTox_prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule.. The FDA-approved status will specify if the drug is approved by the FDA for clinical trials(Yes) or Not approved by the FDA for clinical trials(No). You will be provided with task template. The task is to predict the binary label for a given molecule, please answer with only Yes or No. Wrap your response in JSON: {\"answer\": \"<Yes/No>\"}.\n"












import random
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

SEED = 42
SAMPLE_SIZE  = 100
random.seed(SEED)

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
Tox21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
HIV_URL =   "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
ClinTox_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"


DATASETS = {
    "BBBP": BBBP_URL,
    "BACE": BACE_URL,
    "Tox21": Tox21_URL,
    "HIV": HIV_URL,
    "ClinTox": ClinTox_URL
}

PROMPTS = {
    "BBBP":BBBP_prompt,
    "BACE":BACE_prompt,
    "Tox21":Tox21_prompt,
    "HIV":HIV_prompt,
    "ClinTox":ClinTox_prompt
}

TASK_DESCRIPTION = {
    "BBBP":"Penetration",
    "BACE":"BACE-1 Inhibit",
    "Tox21":"Toxic",
    "HIV":"Inhibit",
    "ClinTox":"Clinically-trail-toxic"
}

# Per-dataset primary label column
LABEL_COL_MAP = {
    "BBBP": "p_np",
    "BACE": "Class",
    "Tox21": "Toxic",          # aggregated column created in _preprocess
    "HIV": "HIV_active",
    "ClinTox": "CT_TOX",
}

# # Tox21 assay columns used for aggregation
TOX21_ASSAY_COLS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
                    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _preprocess(df, name):
    """Dataset-specific preprocessing. Creates a unified label column where needed."""
    if name == "Tox21":
        # If ANY assay flags toxic (1), the molecule is toxic
        df["Toxic"] = (df[TOX21_ASSAY_COLS].fillna(0).max(axis=1) >= 1).astype(int)
    return df


def _rename_mol_column(df):
    """Renames the 'mol' column to 'smiles' in a pandas DataFrame."""
    return df.rename(columns={"mol": "smiles"})

def _filter_smiles(df):
    """Drop rows with invalid SMILES in place and return the same df."""
    mask = np.fromiter(
        (isinstance(s, str) and Chem.MolFromSmiles(s) is not None
         for s in df["smiles"]),
        dtype=bool,
        count=len(df),
    )
    df.drop(index=df.index[~mask], inplace=True)
    return df


def _format_examples(rows, name):
    """Turn df rows into a list of example tuples.
    - ClinTox: (smiles, label, fda_label)
    - Others:  (smiles, label)
    HIV's activity class (CI/CM/CA) is deliberately left out: it defines HIV_active.
    """
    label_col = LABEL_COL_MAP[name]
    labels = ["Yes" if c == 1 else "No" for c in rows[label_col]]

    if name == "ClinTox":
        fda_labels = ["Yes" if c == 1 else "No" for c in rows["FDA_APPROVED"]]
        return list(zip(rows["smiles"], labels, fda_labels))
    else:
        return list(zip(rows["smiles"], labels))


def _random_sample_examples(df, k, name):
    """k/2 positive + k/2 negative random examples (same for every test molecule)."""
    label_col = LABEL_COL_MAP[name]
    positive_examples = df[df[label_col] == 1].sample(int(k/2), random_state=SEED)
    negative_examples = df[df[label_col] == 0].sample(int(k/2), random_state=SEED)

    if name.lower() == "hiv":
        positive_examples = df[df[label_col] == 1].sample(int(0.2*k), random_state=SEED)
        negative_examples = df[df[label_col] == 0].sample(int(0.8*k), random_state=SEED)

    return _format_examples(pd.concat([positive_examples, negative_examples]), name)


def _scaffold_fingerprint(smiles):
    """Morgan fingerprint of the Murcko scaffold of a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
    return _fp_gen.GetFingerprint(scaffold)


def _scaffold_sample_examples(df_train, train_fps, test_smiles, k, name):
    """k training molecules whose scaffolds are most similar to the test molecule
    (ChemLLMBench-style scaffold sampling). Different for every test molecule."""
    test_fp = _scaffold_fingerprint(test_smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    top_k = np.argsort(similarities)[::-1][:k]
    return _format_examples(df_train.iloc[top_k], name)


def _build_system_prompt(name, examples):
    """Base prompt, plus a dataset-specific examples block."""
    system_prompt = PROMPTS[name]
    if not examples:
        return system_prompt

    if name == "ClinTox":
        system_prompt += "\n" + "\n".join(
            f"SMILES: {smiles}\nFDA-approved: {fda}\n{TASK_DESCRIPTION[name]}: {label}"
            for smiles, label, fda in examples
        )
    else:
        system_prompt += "\n" + "\n".join(
            f"SMILES: {smiles}\n{TASK_DESCRIPTION[name]}: {label}"
            for smiles, label in examples
        )
    return system_prompt


def _create_prompt(sample_strategy, df_train, df_test, name, k):
    """Returns (system_prompts, user_prompts), one entry per test molecule."""
    test_smiles = df_test["smiles"].tolist()

    # Build dataset-specific user prompts
    if name == "ClinTox":
        fda_labels = ["Yes" if c == 1 else "No" for c in df_test["FDA_APPROVED"]]
        user_prompts = [
            f"SMILES: {s}\nFDA-approved: {fda}\n{TASK_DESCRIPTION[name]}:"
            for s, fda in zip(test_smiles, fda_labels)
        ]
    else:
        user_prompts = [f"SMILES: {s}\n{TASK_DESCRIPTION[name]}:" for s in test_smiles]

    # zero-shot: base prompt only
    if k == 0:
        system_prompt = _build_system_prompt(name, examples=None)
        return [system_prompt] * len(test_smiles), user_prompts

    # random: same examples for all test molecules -> one shared system prompt
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(name, _random_sample_examples(df_train, k, name))
        return [system_prompt] * len(test_smiles), user_prompts

    # scaffold: examples depend on the test molecule -> system prompt per molecule
    if sample_strategy == "scaffold":
        train_fps = [_scaffold_fingerprint(s) for s in df_train["smiles"]]  # computed once
        system_prompts = [
            _build_system_prompt(name, _scaffold_sample_examples(df_train, train_fps, s, k, name))
            for s in test_smiles
        ]
        return system_prompts, user_prompts

    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")


def load_dataset(name, sample_strategy, k):
    df = pd.read_csv(DATASETS[name])


    if name.lower() == "bace":
        df = _rename_mol_column(df)

    _preprocess(df, name)
    _filter_smiles(df)

    df_test = df.sample(SAMPLE_SIZE, random_state=SEED)

    df.drop(df_test.index, inplace=True)
    df.reset_index(drop=True, inplace=True)   # so iloc/positions line up with train_fps

    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy, df_train=df, df_test=df_test, name=name, k=k
    )

    return system_prompts, user_prompts, df_test


if __name__ == "__main__":
    sys_p, usr_p, df = load_dataset("HIV",sample_strategy="random",k=10)


    print(sys_p[0])
    print(usr_p[0])