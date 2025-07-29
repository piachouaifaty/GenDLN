
import json

from genetic_dln.src.dln.gen_dln import GenDLN

sentences_to_classify = {
    "sentence_1": {
        "label": 1,
        "text": "Give me a word from the list [fair, unfair]. Answer with exactly one word [fair/unfair]."
    },
    "sentence_2": {
        "label": 0,
        "text": "Give me a random word from the list [fair, unfair]. Answer with exactly one word [fair/unfair]."
    }
}

TRAN_FILE_PATH = "../../dataset/claudette/balanced_training_set_for_binary.json"

def test_print_scores():
    prompt = (
        "Explain the following statement in detail and provide arguments for and against whether the text is fair under EU rules. Think step by step. Statement: ",
        "Based on the explanation above, classify the following statement. Answer with exactly one word [fair/unfair]! Statement to classify: "
    )
    with open(TRAN_FILE_PATH, "r") as file:
        data = json.load(file)
    dln = GenDLN()
    result = dln.print_binary_stats(prompt, "../../../data/dataset/claudette_val_merged.tsv", 24)
    print(result)


test_print_scores()