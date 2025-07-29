import json
import os
import random
import re

from genetic_dln.src.constants import constants
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.models.llm import LLM
from genetic_dln.src.prompt_builder.prompt_builder import PromptBuilder


class PopulationInitializer:
    """
    A class to initialize a population for a genetic algorithm, with optional LLM-based augmentation.
    """

    def __init__(self, layer1_file, layer2_file, population_size, augment_with_llm=False, logger=None,
                 temperature=0.7):
        """
        Initialize the PopulationInitializer.

        Args:
            layer1_file (str): Path to the JSON file containing layer 1 base prompts (prompt_1).
            layer2_file (str): Path to the JSON file containing layer 2 base prompts (prompt_2).
            population_size (int): Desired size of the population.
            augment_with_llm (bool): Whether to augment the population using an LLM if needed.
        """
        self.layer1_file = layer1_file
        self.layer2_file = layer2_file
        self.population_size = population_size
        self.augment_with_llm = augment_with_llm
        self.llm_client = LLM()
        self.logger = logger  # Reference to the logger
        self.temperature = temperature
        self.population = []
        self.input_loader = InputLoader()
        self.prompts_file_path = os.path.join(constants.EVOLUTIONARY_ALGORITHMS_DIR, "data",
                                              "prompts_population_initializations.yaml")
        self.prompts = self.input_loader.load_prompt_template(self.prompts_file_path)
        self.prompt_builder = PromptBuilder()

    def _load_prompts(self, file_path):
        """
        Load prompts from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            list: List of prompts loaded from the file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
        return data.get("prompts", [])

    def initialize_population(self):
        """
        Initialize the population by loading prompts and optionally augmenting.

        Returns:
            list: Initialized population as a list of dictionaries.

        Raises:
            ValueError: If population size exceeds available individuals and augmentation is disabled.
        """
        # Load prompts from files
        layer1_prompts = self._load_prompts(self.layer1_file)
        layer2_prompts = self._load_prompts(self.layer2_file)

        # Combine prompts into individuals
        base_individuals = [{"prompt_1": p1, "prompt_2": p2, "source": "base"} for p1, p2 in
                            zip(layer1_prompts, layer2_prompts)]
        print(f"Loaded base individuals: {len(base_individuals)}")

        augmentation_temperature = None  #defaults to none

        # Handle population size
        if self.population_size <= len(base_individuals):
            # Randomly sample without augmentation (ensures unique individuals)
            self.population = random.sample(base_individuals, self.population_size)
        else:
            # Use all base individuals and calculate how many more are needed
            self.population = base_individuals
            if self.augment_with_llm and self.llm_client:
                total_needed = self.population_size - len(base_individuals)
                print(f"Augmentation required: {total_needed} individuals")
                augmentation_temperature = self.temperature
                print(f"Augmentation temperature:  {augmentation_temperature}")

                # Pass the number of required individuals to the augmentation function
                additional_individuals = self._augment_population(base_individuals, total_needed)

                # Add only the required number of unique new individuals
                unique_additional = []
                for individual in additional_individuals:
                    if individual not in self.population:
                        unique_additional.append(individual)
                    if len(unique_additional) == total_needed:
                        break

                if len(unique_additional) < total_needed:
                    raise ValueError(
                        "Not enough unique individuals could be generated. Consider revising the augmentation logic."
                    )
                    # Wrap augmented individuals with metadata

                augmented_individuals = [{"prompt_1": p1, "prompt_2": p2, "source": "LLM-augmented"} for p1, p2 in
                                         unique_additional]

                self.population.extend(augmented_individuals)


            else:
                raise ValueError(
                    "Population size exceeds available individuals, and augmentation is disabled. "
                    "Provide a valid augmentation prompt or reduce the population size."
                )

        print(f"Total population after augmentation: {len(self.population)}")

        # Generate IDs for the population
        individual_ids = self._generate_individual_ids(len(self.population))
        print(f"Generated individual IDs: {individual_ids}")

        # Combine IDs with individuals
        self.population = [
            {
                "id": individual_id,
                "prompt_1": individual["prompt_1"],
                "prompt_2": individual["prompt_2"],
                # Assign source based on whether the individual is augmented
                "source": individual["source"]
            }
            for individual_id, individual in zip(individual_ids, self.population)
        ]

        # Debugging population sources
        base_count = sum(1 for individual in self.population if individual["source"] == "base")
        augmented_count = sum(1 for individual in self.population if individual["source"] == "LLM-augmented")
        print(f"Base individuals: {base_count}, Augmented individuals: {augmented_count}")

        # Log each individual
        if self.logger:
            for individual in self.population:
                self.logger.log_individual(individual)

        # Save the logs
        if self.logger:
            self.logger.save_logs()

        return self.population, augmentation_temperature

    def _augment_population(self, base_individuals, total_needed, retries: int = 3):
        """
        Augment the population using an LLM to generate additional individuals, with retries.

        Args:
            base_individuals (list): List of existing individuals to inform the LLM.
            total_needed (int): Number of additional individuals needed.
            retries (int): Number of retries in case of invalid response.

        Returns:
            list: List of augmented individuals as (prompt_1, prompt_2) tuples.

        Raises:
            ValueError: If no valid individuals are generated after all retries.

        """
        # Construct the system instruction
        system_instruction = self.prompts["system_role"]

        # user input with task-specific details and examples
        # TODO: needs to be refactored as a parameter

        user_input = self.prompts["user_input"]
        for individual in base_individuals:
            user_input += f"- Prompt 1: {individual['prompt_1']}\n  Prompt 2: {individual['prompt_2']}\n"

        user_input += (
            f"\nGenerate {total_needed} additional pairs of prompts."
            "Ensure all new pairs are distinct from the examples."
        )

        messages = self.prompt_builder.build_prompt(system_instruction, user_input)

        for attempt in range(retries):
            print(f"Attempt {attempt + 1}/{retries}: Generating {total_needed} additional individuals.")
            response = self.llm_client.predict(messages=messages, temperature=self.temperature, index=0)
            try:
                new_individuals = self._parse_llm_response(response, total_needed)
                return new_individuals  # Valid response
            except ValueError as e:
                print(f"Failed to parse LLM response: {str(e)}. Retrying...")

            # If all retries fail, raise an error
        error_message = (
            f"Failed to augment population: No valid individuals could be extracted after {retries} retries. "
            f"Ensure the LLM is configured properly and the prompt templates are valid."
        )
        print(error_message)
        raise ValueError(error_message)

    def _parse_llm_response(self, response, total_needed):
        """
        Parse the LLM's response to extract new individuals.

        Args:
            response (str): Raw response from the LLM.
            total_needed (int): Number of new individuals needed.

        Returns:
            list: List of new individuals as (prompt_1, prompt_2) tuples.
        """
        # Regex pattern to extract JSON objects with "prompt_1" and "prompt_2"
        json_pattern = r'\{(?:[^{}]|"[^"]*"|\'[^\']*\')*\}'
        json_matches = re.findall(json_pattern, response)

        new_individuals = []
        for json_match in json_matches:
            try:
                # Parse the JSON object
                json_obj = json.loads(json_match)

                # Ensure the JSON object contains both "prompt_1" and "prompt_2"
                if "prompt_1" in json_obj and "prompt_2" in json_obj:
                    new_individuals.append((json_obj["prompt_1"], json_obj["prompt_2"]))

                # Stop parsing if the required number of individuals has been reached
                if len(new_individuals) >= total_needed:
                    break
            except json.JSONDecodeError:
                continue  # Skip malformed JSON objects

        # Raise an error if fewer than the required number of individuals are extracted
        if len(new_individuals) < total_needed:
            raise ValueError(
                f"Only {len(new_individuals)} valid individuals could be parsed from the LLM response. "
                f"Expected at least {total_needed}."
            )

        return new_individuals

    def _generate_individual_ids(self, count: int):
        """
        Generate unique IDs for individuals in the population.

        Args:
            count (int): Number of IDs to generate.

        Returns:
            list: List of unique IDs for the population.
        """
        return [f"I{i + 1:02d}" for i in range(count)]
