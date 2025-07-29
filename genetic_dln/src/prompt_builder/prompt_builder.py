import json
from genetic_dln.src.prompt_builder.base_prompt_builder import BasePromptBuilder


class PromptBuilder(BasePromptBuilder):
    def build_prompt_layer_1(self, prompt: str, sentences_to_classify: list) -> list[dict]:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(sentences_to_classify)}
        ]

        return messages

    def build_prompt_layer_2(self, prompt: str, few_shots: dict, layer_1_output: str, sentences_to_classify: list) -> list[dict]:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": few_shots["input"]},
            {"role": "assistant", "content": few_shots["output"]},
            {"role": "user", "content": "previous_outputs:" + layer_1_output + "\n" +
                                        "sentences_to_classify" + json.dumps(sentences_to_classify, indent=4)}
        ]
        return messages

    def build_prompt(self, system_prompt, data):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data}
        ]
        return messages
