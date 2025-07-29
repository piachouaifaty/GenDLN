import os
import json
from genetic_dln.src.dln.gen_dln import GenDLN

def validate_individual(prompt_1: str, prompt_2: str, multi_label: bool = True):
    """
    Validates an individual by running its prompts through the DLN model.

    Args:
        prompt_1 (str)
        prompt_2 (str)
        multi_label (bool): Whether to use multi-label validation.

    Returns:
        dict: A dictionary containing the validation results (score_dict).
    """
    try:
        # Initialize the DLN model for validation
        gen_dln = GenDLN(multi_label=multi_label, validation=True)

        # Validate the prompts and get the score dictionary
        score_dict = gen_dln.predict(prompt_1, prompt_2, 0)
        return score_dict

    except Exception as e:
        print(f"Validation failed for prompts: {prompt_1}, {prompt_2}. Error: {e}")
        return {"error": str(e)}