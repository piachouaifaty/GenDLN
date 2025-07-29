from abc import ABC, abstractmethod


# ---------------------- TODO: Make a better description
# This class defines an interface for the frozen LLMs we use to predict
# ----------------------

class BaseGenDLNLLM(ABC):
    @abstractmethod
    def predict(self, prompt: list, temperature: float, index: int) -> str:
        pass
