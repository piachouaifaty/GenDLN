import json
import os
import re

from genetic_dln.src.constants import constants
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.models.llm import LLM
from genetic_dln.src.prompt_builder.prompt_builder import PromptBuilder


class Crossover:
    """
    A class to handle crossover operations for text prompts using LLMs.
    """

    def __init__(self, logger=None):
        """
        Initialize the Crossover class with an LLM client and optional logger.

        Args:
            logger (CrossoverLogger): Optional logger for recording crossover events.
        """
        self.llm_client = LLM()
        self.prompt_builder = PromptBuilder()
        self.input_loader = InputLoader()
        self.prompts_file_path = os.path.join(constants.EVOLUTIONARY_ALGORITHMS_DIR, "data", "prompts_crossover.yaml")
        self.prompts = self.input_loader.load_prompt_template(self.prompts_file_path)
        self.logger = logger

    @staticmethod
    def _extract_children_and_context(response: str) -> tuple:
        """
        Extract the JSON objects containing child prompts and additional context.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            tuple: A dictionary with child prompts ('child_1', 'child_2') and the remaining context.
        """
        child_1 = None
        child_2 = None
        llm_context = response  # Default to entire response if no JSONs are found

        # Regex to match valid JSON objects in the response
        json_pattern = r'\{(?:[^{}]|"[^"]*"|\'[^\']*\')*\}'
        json_matches = re.findall(json_pattern, response)

        # If there's a single JSON with both children
        for json_match in json_matches:
            try:
                # Attempt to parse the matched JSON object
                json_cleaned = json_match.replace("'", '"')  # Ensure valid JSON syntax
                json_obj = json.loads(json_cleaned)

                # Check if it contains both children
                if "child_1" in json_obj and "child_2" in json_obj:
                    child_1 = json_obj.get("child_1")
                    child_2 = json_obj.get("child_2")
                    # Remove the matched JSON object from the response to extract context
                    llm_context = response.replace(json_match, "").strip()
                    break  # Exit loop since both children are found
            except json.JSONDecodeError:
                continue

        # If individual JSONs exist for each child
        if not child_1 or not child_2:
            for json_match in json_matches:
                try:
                    # Attempt to parse each matched JSON object
                    json_cleaned = json_match.replace("'", '"')  # Ensure valid JSON syntax
                    json_obj = json.loads(json_cleaned)

                    # Assign child_1 or child_2 as found
                    if "child_1" in json_obj and not child_1:
                        child_1 = json_obj.get("child_1")
                    if "child_2" in json_obj and not child_2:
                        child_2 = json_obj.get("child_2")

                    # Remove the matched JSON object from the response to extract context
                    llm_context = response.replace(json_match, "").strip()
                except json.JSONDecodeError:
                    continue

        # If neither child is extracted, set context appropriately
        if not child_1 and not child_2 and llm_context == response:
            llm_context = "No valid children could be extracted."

        return {"child_1": child_1, "child_2": child_2}, llm_context

    def perform_crossover(self, parent_1: str, parent_2: str, crossover_type: str, temperature: float = 0.7, retries: int = 3, idx=0) -> dict:
        """
        Perform crossover between two parent prompts, with retries for invalid responses.

        Args:
            parent_1 (str): The first parent prompt.
            parent_2 (str): The second parent prompt.
            crossover_type (str): The type of crossover to perform.
            temperature (float): Controls LLM randomness.
            retries (int): Number of retries in case of invalid response.

        Returns:
            dict: A dictionary containing parents, children, and crossover details.
        """
        base_system_prompt = self.prompts["system_role"]

        # Map crossover types to instructions
        crossover_instructions = self.prompts["crossover_instructions"]

        # Validate the crossover type
        if crossover_type not in crossover_instructions:
            raise ValueError(f"Invalid crossover type: {crossover_type}")

        # Construct the full system instruction
        system_instruction = f"{base_system_prompt} Instruction: {crossover_instructions[crossover_type]}"

        # Format the user input
        user_input = f"Parent 1: {parent_1}\nParent 2: {parent_2}"

        messages = self.prompt_builder.build_prompt(system_instruction, user_input)

        for attempt in range(retries):
            print("Crossover instruction:", crossover_instructions[crossover_type])
            print("User Input:")
            print(user_input)

            # Call the LLM
            response = self.llm_client.predict(
                messages=messages,
                temperature=temperature,
                index=idx
            )

            # Parse the response to extract children and context
            children, llm_context = self._extract_children_and_context(response)

            if children.get("child_1") and children.get("child_2"):  # Valid children extracted
                result = {
                    "parent_1": parent_1,
                    "parent_2": parent_2,
                    "crossover_type": crossover_type,
                    "child_1": children.get("child_1"),
                    "child_2": children.get("child_2"),
                    "LLM_context": llm_context,
                    "full_response": response,
                }

                if self.logger:
                    self.logger.log_crossover(result)
                return result

            print(f"Attempt {attempt + 1}/{retries} failed. Retrying...")

        print("All retries failed. Returning default children: empty strings.")

        # FALLBACK VALUES
        result = {
            "parent_1": parent_1,
            "parent_2": parent_2,
            "crossover_type": crossover_type,
            "child_1": "",
            "child_2": "",
            "LLM_context": "WARNING: No valid children could be extracted (JSON invalidity).",
            "full_response": "Invalid response after retries.",
        }

        # Log the result if a logger is provided
        if self.logger:
            self.logger.log_crossover(result)

        return result
