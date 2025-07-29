import json
import os

from genetic_dln.src.constants import constants
from genetic_dln.src.dln.gen_dln import GenDLN


if __name__ == "__main__":
    run_id = 1
    score_dict_path = os.path.join(constants.root_folder, "genetic_dln", "results", f"score_dict_binary_run_{run_id}.json")
    os.makedirs(os.path.dirname(score_dict_path), exist_ok=True)

    prompt_01 = "Identify any possible areas of concern or controversy to grasp industry-specific implications."
    prompt_02 = "Is the original version of this sentence fair or unfair as it stands in relation to this industry or sector?"

    gen_dln = GenDLN(multi_label=True, validation=True)
    score_dict = gen_dln.predict(prompt_01, prompt_02, 0)

    with open(score_dict_path, 'w') as f:
        json.dump(score_dict, f, indent=4)

    print(f"Score dictionary saved to {score_dict_path}")
