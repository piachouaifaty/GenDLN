import os

from openai import RateLimitError, APIError

from tenacity import wait_random_exponential, stop_after_attempt, retry, wait_exponential

from genetic_dln.src.constants import constants
from genetic_dln.src.custom_logger.custom_logger import CustomLogger
from genetic_dln.src.models.base_gen_dln_llm import BaseGenDLNLLM

from mistralai import Mistral


class LLM(BaseGenDLNLLM):
    MAX_FAILURES = 40

    def __init__(self, number_of_workspaces=10):
        """
        Initializes the LLM instance with an optional logger.
        """
        log_dir = os.path.join(constants.root_folder, "logs")
        self.logger = CustomLogger(log_dir=log_dir, logger_name="LLM").logger

        self.number_of_workspaces = number_of_workspaces
        self.mistral_clients = [
            Mistral(api_key=constants.mistral_api_key_v1),
            Mistral(api_key=constants.mistral_api_key_v2),
            Mistral(api_key=constants.mistral_api_key_v3),
            Mistral(api_key=constants.mistral_api_key_v4),
            Mistral(api_key=constants.mistral_api_key_v5),
            Mistral(api_key=constants.mistral_api_key_v6),
            Mistral(api_key=constants.mistral_api_key_v7),
            Mistral(api_key=constants.mistral_api_key_v8),
            Mistral(api_key=constants.mistral_api_key_v9),
            Mistral(api_key=constants.mistral_api_key_v10)
        ]

        self.backup_clients = [
            Mistral(api_key=constants.mistral_api_key_v11),
            Mistral(api_key=constants.mistral_api_key_v12),
            Mistral(api_key=constants.mistral_api_key_v13),
            Mistral(api_key=constants.mistral_api_key_v14),
            Mistral(api_key=constants.mistral_api_key_v15),
            Mistral(api_key=constants.mistral_api_key_v16),
            Mistral(api_key=constants.mistral_api_key_v17),
            Mistral(api_key=constants.mistral_api_key_v18),
            Mistral(api_key=constants.mistral_api_key_v19),
            Mistral(api_key=constants.mistral_api_key_v20),
        ]

        self.failure_counts = [0] * self.number_of_workspaces
        self.used_backups = 0

        self.number_of_workspaces = number_of_workspaces

    def predict(self, messages: list[dict], temperature: float, index: int = 0):
        workspace_index = index % self.number_of_workspaces

        return self.generate_response_mistral(messages, temperature, workspace_index)

    def predict_ga(self, messages: list[dict], temperature: float):
        return self.generate_response_mistral(messages, temperature, self.number_of_workspaces)

    @retry(
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_attempt(100),
    )
    def generate_response_mistral(
            self, messages: list[dict], temperature, workspace_index
    ) -> str:
        """
        Generates a response using the AzureOpenAI client, retrying on failure.
        """

        client = self.mistral_clients[workspace_index]
        try:
            response = client.chat.complete(
                model="mistral-large-2411",
                messages=messages,
                temperature=temperature,
                max_tokens=16384,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                timeout_ms=120000,
            )
            self.logger.info("Response generated successfully")
            self.failure_counts[workspace_index] = 0
            return response.choices[0].message.content
        except (RateLimitError, APIError, Exception) as e:
            self.logger.error(f"{type(e).__name__} on workspace {workspace_index + 1}: {e}")
            self.failure_counts[workspace_index] += 1
            if self.failure_counts[workspace_index] >= self.MAX_FAILURES:
                self.replace_dead_client(workspace_index)
            raise

    def replace_dead_client(self, index: int):
        if self.used_backups >= len(self.backup_clients):
            self.logger.warning(f"No more backup Mistral clients available to replace workspace {index + 1}")
            return

        new_client = self.backup_clients[self.used_backups]
        self.logger.warning(f"Replacing Mistral client at index {index + 1} with backup client #{self.used_backups}")
        self.mistral_clients[index] = new_client
        self.failure_counts[index] = 0
        self.used_backups += 1
