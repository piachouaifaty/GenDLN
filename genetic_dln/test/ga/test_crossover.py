import os
import random
from dotenv import load_dotenv
from pathlib import Path

from genetic_dln.src.evolutionary_algorithms.genetic_operations.crossover import Crossover
from genetic_dln.src.evolutionary_algorithms.loggers.crossover_logger import CrossoverLogger
from genetic_dln.src.models.llm import LLM

# Load environment variables
root_folder = Path().resolve().parent
env_file = os.path.join(root_folder, ".env")
load_dotenv(env_file)


def test_crossover():
    """
    Test the Crossover class with various crossover types and randomly selected parents.
    """
    # Initialize the LLM client and the Crossover class
    crossover_logger = CrossoverLogger(log_dir="../../src/evolutionary_algorithms/logs")
    crossover = Crossover(logger=crossover_logger)

    # Example population of individuals (each individual is a tuple of prompts)
    population = [
        ("Summarize the following text.", "Is the above text fair or unfair?"),
        ("Explain the key points of this document.", "Based on the details above, is this fair?"),
        ("Provide a concise overview of this topic.", "Determine whether the topic is just or unjust."),
        ("Highlight the main ideas from the text.", "Assess the fairness of the statements above."),
    ]

    # Define crossover types
    crossover_types = [
        "single_point",
        "two_point",
        "semantic_blending",
        "phrase_swapping",
        "token_level",
    ]

    # Randomly select two individuals as parents
    #THIS NEEDS TO BE CHANGED, AND MAKE SURE NOT TO SELECT THE SAME PARENT TWICE
    parent_1 = random.choice(population)
    parent_2 = random.choice(population)

    print(f"Selected Parents:\nParent 1: {parent_1}\nParent 2: {parent_2}")

    # Perform crossover for each crossover type
    for crossover_type in crossover_types:
        print(f"\n--- Crossover Type: {crossover_type} ---")

        # Crossover for prompt_1
        print("\nCROSSOVER FOR PROMPT 1:")
        result_1 = crossover.perform_crossover(
            parent_1=parent_1[0],  # prompt_1 of parent 1
            parent_2=parent_2[0],  # prompt_1 of parent 2
            crossover_type=crossover_type,
        )
        result_1["prompt_type"] = "prompt_1"
        for key, value in result_1.items():
            if key != "full_response":
                print(f"{key}: {value}")
        crossover_logger.log_crossover(result_1)

        # Crossover for prompt_2
        print("\nCROSSOVER FOR PROMPT 2:")
        result_2 = crossover.perform_crossover(
            parent_1=parent_1[1],  # prompt_2 of parent 1
            parent_2=parent_2[1],  # prompt_2 of parent 2
            crossover_type=crossover_type,
        )
        result_2["prompt_type"] = "prompt_2"
        for key, value in result_2.items():
            if key != "full_response":
                print(f"{key}: {value}")
        crossover_logger.log_crossover(result_1)

    crossover_logger.save_logs()


if __name__ == "__main__":
    test_crossover()
