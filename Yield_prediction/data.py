from warnings import warn
import pandas as pd
import random
import numpy as np
import os
from pathlib import Path
# ---------------------------------------------------------------------------
# Constants & Prompts
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

BH_PROMPT = """
You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical yield prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a Buchwald-Hartwig reaction, 
the task focuses on predicting reaction yield, specifically wether a reaction is High-yielding(Yes) or Not High-yielding(No),based on the SMILES string representation of each Buchwald-Hartwig reaction. 
The reactants are separated by '.' and product are separated by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. 
In comparison to the traditional Buchwald-Hartwig reaction, this study involved the addition of additives and aimed to evaluate the effects of isoxazole agment additives as Glorius fragment additives on the amination of various aryl and heteroaryl halides. 
You will be provided with several examples molecules. Please answer with only Yes or No. Wrap your response in JSON: {"answer": "<Yes/No>"}.
"""

SUZUKI_PROMPT = """
You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical yield prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a Suzuki-Miyaura reaction, 
the task focuses on predicting reaction yield, specifically wether a reaction is High-yielding(Yes) or Not High-yielding(No),based on the SMILES string representation of each Suzuki-Miyaura reaction. 
The reactants are separated by '.' and product are separated by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. 
You will be provided with several examples molecules. Please answer with only Yes or No. Wrap your response in JSON: {"answer": "<Yes/No>"}.
"""

# Local file paths (assumes the files are in the same folder as this script)

URL = Path(__file__).parent 

BH_TRAIN_FILE = URL/"BH_dataset .npz"
BH_TEST_FILE = URL/ "BH_sample_100_test.npz"

SUZUKI_TRAIN_FILE = URL/"Suzuki.npz"
SUZUKI_TEST_FILE = URL/"SU_sample_100_test.npz"

SUZUKI_MAP = {
    "train": SUZUKI_TRAIN_FILE,
    "test": SUZUKI_TEST_FILE,
    "prompt": SUZUKI_PROMPT
}

BH_MAP = {
    "train": BH_TRAIN_FILE,
    "test": BH_TEST_FILE,
    "prompt": BH_PROMPT
}

TASK_MAP = {
    "Buchwald-Hartwig": BH_MAP,
    "Suzuki-Miyaura": SUZUKI_MAP
}

# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------

def _extract_npz_to_df(file_path, split):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find '{file_path}'. Please make sure you downloaded it to the same directory as this script.")
        
    npz = np.load(file_path, allow_pickle=True)

    if split == "train":
        data = npz['data'][2]
        df = pd.DataFrame(data)
        df.rename(columns={"yld": "yeild", "rsmi": "reaction"}, inplace=True)
        cols = list(df.columns)
        cols[0], cols[1] = cols[1], cols[0]
        df = df[cols]
        df['yeild'] = np.where(df['yeild'] > 70, 'Yes', 'No')
        return df
    else:
        data = npz['data_df']
        df = pd.DataFrame({
            "reaction": data[:, 0],
            "yeild": data[:, 1]
        })
        return df

# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _format_examples(rows):
    """Return a list of (reaction, yeild) tuples."""
    return list(zip(rows["reaction"], rows["yeild"]))

def _random_sample_examples(df, k):
    """Randomly sample *k* training examples for in-context learning."""
    sampled = df.sample(k, random_state=SEED)
    return _format_examples(sampled)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(base_prompt, examples):
    """Combine the base task prompt with formatted ICL examples."""
    system_prompt = base_prompt.strip()
    if not examples:
        return system_prompt

    system_prompt += "\n\n" + "\n\n".join(
        f"reaction: {inp}\nHigh yeilding: {out}"
        for inp, out in examples
    )
    return system_prompt

def _create_prompt(sample_strategy, df_train, df_test, base_prompt, k):
    """Build parallel lists of (system_prompts, user_prompts)."""
    test_inputs = df_test["reaction"].tolist()

    # User prompt: present the input and ask for the output
    user_prompts = [f"reaction: {inp}\nyeild:" for inp in test_inputs]

    # Zero-shot — no examples
    if k == 0:
        system_prompt = _build_system_prompt(base_prompt, examples=None)
        return [system_prompt] * len(test_inputs), user_prompts

    # Random — same examples for every test molecule
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(
            base_prompt, _random_sample_examples(df_train, k)
        )
        return [system_prompt] * len(test_inputs), user_prompts
        
    else:
        warn(f"Only support random sampling strategy for now, falling back to random sampling")
        system_prompt = _build_system_prompt(
            base_prompt, _random_sample_examples(df_train, k)
        )
        return [system_prompt] * len(test_inputs), user_prompts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_function(name, sample_strategy="random", k=0):
    """
    Load the Yield-Prediction dataset and return prompts + test data.

    Parameters
    ----------
    name : str
        Task name — either 'Buchwald-Hartwig' or 'Suzuki-Miyaura'.
    sample_strategy : str, default "random"
        How to choose ICL examples. Only "random" is supported.
    k : int, default 0
        Number of in-context examples (0 = zero-shot).

    Returns
    -------
    system_prompts : list[str]
    user_prompts   : list[str]
    df_test        : pd.DataFrame
    """
    if name not in TASK_MAP:
        raise ValueError(
            f"Unknown task '{name}'. Choose from {list(TASK_MAP.keys())}"
        )

    task_config = TASK_MAP[name]

    # Load from local dataframes
    df_train = _extract_npz_to_df(task_config["train"], "train")
    df_test = _extract_npz_to_df(task_config["test"], "test")

    # Generate prompts
    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy,
        df_train=df_train,
        df_test=df_test,
        base_prompt=task_config["prompt"],
        k=k,
    )

    return system_prompts, user_prompts, df_test





if __name__ == "__main__":
    s_p , u_p , df_test = load_function("Buchwald-Hartwig", k = 10)
    print(s_p[0])
    print("-----------")
    print(u_p[0])
    print("i would say thats a Yes.")




