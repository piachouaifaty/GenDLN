from abc import ABC, abstractmethod


# ------------------------------------- TODO: Make a better description
# This class defines an interface for prompt builders
# A prompt builder should be able to build an input from a template or build a template from a pure prompt
# A pure prompt would look something like this: "This is a prompt"
# A template would look something like this: "This is a prompt <<data_text>>"
# We can build prompt builders that can use different templates.
# -------------------------------------

class BasePromptBuilder(ABC):
    @abstractmethod
    def build_prompt_layer_1(self, prompt: str, sentences_to_classify: dict) -> list[dict]:
        pass

    @abstractmethod
    def build_prompt_layer_2(self, prompt: str, few_shots, layer_1_output: str, sentences_to_classify: dict) -> list[dict]:
        pass
