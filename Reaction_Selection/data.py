import ast
import pandas as pd


LIGAND_CSV = "ligand_sample.csv"
REAGENT_CSV = "reagent_sample.csv"
SOLVENT_CSV = "solvent_sample.csv"


TASK_MAP = {
    "Reactant_selection":REAGENT_CSV,
    "Ligand_selection":LIGAND_CSV,
    "Solvent_selection":SOLVENT_CSV
}

def _load_prompts(df):
    system_prompts = []
    user_prompts = []

    for task in df["task"]:
        # Everything before the first actual example
        split_marker = "Given the rest of reaction components:"

        system_prompt, user_prompt = task.split(split_marker, 1)

        system_prompts.append(system_prompt.strip() + " Wrap your response in JSON: {\"answer\": \"<your answer>\"}.")

        # Add the marker back because it is part of the actual task
        user_prompts.append(
            split_marker + user_prompt
        )

    return system_prompts, user_prompts


def _create_ground_truth(df):
    df["yield_details"] = df["yield_details"].apply(ast.literal_eval)

    df["ground_truth"] = df["yield_details"].apply(
        lambda x: max(x, key=lambda y: y[0][1])[0][0]
    )


def load_dataset(name):
    df = pd.read_csv(name)

    _create_ground_truth(df)

    system_prompts, user_prompts = _load_prompts(df)

    return system_prompts, user_prompts, df



