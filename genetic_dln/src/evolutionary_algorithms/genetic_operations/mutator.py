import os.path
from typing import Any
import json
import re

from genetic_dln.src.constants import constants
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.models.llm import LLM
from genetic_dln.src.prompt_builder.prompt_builder import PromptBuilder


class Mutator:
    """
    A class to handle different types of mutations for text prompts using LLMs.
    """
    def __init__(self, logger=None):
        """
        Initialize the Mutator with a shared GALLM instance and an optional logger.

        Args:
            logger (MutationLogger): Optional logger for recording mutation events.
        """
        self.llm_client = LLM()
        self.prompt_builder = PromptBuilder()
        self.input_loader = InputLoader()
        self.prompts_file_path = os.path.join(constants.EVOLUTIONARY_ALGORITHMS_DIR, "data", "prompts_mutator.yaml")
        self.prompts = self.input_loader.load_prompt_template(self.prompts_file_path)
        self.logger = logger  # Add the logger instance

    @staticmethod
    def _extract_response_parts(response: str) -> tuple:
        """
        Extract the JSON object containing the mutated prompt and additional context.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            tuple: A tuple containing the mutated prompt and the remaining context.
        """
        mutated_prompt = None
        llm_context = response  # Default to entire response if no JSON is found

        # Regex to match valid JSON objects in the response
        json_pattern = r'\{(?:[^{}]|"[^"]*"|\'[^\']*\')*\}'
        json_matches = re.findall(json_pattern, response)

        for json_match in json_matches:
            try:
                # Attempt to parse the matched JSON object
                json_cleaned = json_match.replace("'", '"')  # Ensure valid JSON syntax
                json_obj = json.loads(json_cleaned)
                if "mutated_sentence" in json_obj:
                    mutated_prompt = json_obj.get("mutated_sentence", None)
                    # Extract context by removing the JSON from the response
                    llm_context = response.replace(json_match, "").strip()
                    break
            except json.JSONDecodeError:
                continue

        # If mutated_prompt is None, check if there is valid context
        if mutated_prompt is None and llm_context == response:
            print("Warning: No valid 'mutated_sentence' key found in the JSON object.")
            llm_context = "No mutated sentence could be extracted."

        return mutated_prompt, llm_context

    def mutate_prompt(self, prompt: str, mutation_type: str, temperature: float = 0.7, retries: int = 3, idx=0) -> dict[str, Any]:
        """
        Mutates a given prompt based on the mutation type, with retries in case of invalid responses.

        Args:
            prompt (str): The original text prompt to mutate.
            mutation_type (str): Type of mutation.
            temperature (float): Controls randomness of the output.
            retries (int): Number of retries in case of invalid response.

        Returns:
            dict: The mutation data with the mutated prompt or fallback values if retries fail.
        """

        # TODO load this from yaml
        # Base instruction
        base_instruction = self.prompts["system_role"]

        # Mutation instructions
        mutation_instructions = self.prompts["mutation_instructions"]
        if mutation_type not in mutation_instructions:
            raise ValueError(f"Invalid mutation type: {mutation_type}")

        # Construct the system instruction
        mutation_instruction = mutation_instructions.get(mutation_type, mutation_instructions[mutation_type])
        system_instruction = f"{base_instruction} Instruction: {mutation_instruction}"

        # Format the user input
        user_input = f"Sentence: {prompt}"
        messages = self.prompt_builder.build_prompt(system_instruction, user_input)

        for attempt in range(retries):
            # Log the system and user instructions for debugging
            print("Mutation instruction: " + mutation_instruction)
            print("User Input:")
            print(user_input)

            # Call the LLM to perform mutation
            response = self.llm_client.predict(
                messages=messages,
                temperature=temperature,
                index=idx
            )

            # Extract the JSON object and remaining text
            mutated_prompt, llm_context = self._extract_response_parts(response)

            if mutated_prompt: #Valid not None prompt successfully extracted
                mutation_data = {
                    "initial_prompt": prompt,
                    "mutation_type": mutation_type,
                    "mutation_instruction": mutation_instruction,
                    "mutated_prompt": mutated_prompt,
                    "system_prompt": system_instruction,
                    "LLM_context": llm_context,
                    "full_response": response,
                }

                # Log the mutation data if a logger is provided
                if self.logger:
                    self.logger.log_mutation(mutation_data)
                return mutation_data

            print(f"Attempt {attempt + 1}/{retries} failed. Retrying...")

        print("All retries failed. Returning default mutated prompt: empty string.")

        #FALLBACK VALUES

        mutation_data = {
            "initial_prompt": prompt,
            "mutation_type": mutation_type,
            "mutation_instruction": mutation_instructions[mutation_type],
            "mutated_prompt": "",  # Default empty string
            "system_prompt": system_instruction,
            "LLM_context": "WARNING: No mutated sentence could be extracted (JSON invalidity).",
            "full_response": "Invalid response after retries",
        }

        if self.logger:
            self.logger.log_mutation(mutation_data)
        return mutation_data

