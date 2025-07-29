import os

from genetic_dln.src.constants import constants
from genetic_dln.src.evolutionary_algorithms.genetic_operations.population_initialization import PopulationInitializer
from genetic_dln.src.evolutionary_algorithms.loggers.population_initialization_logger import \
    PopulationInitializationLogger
from genetic_dln.src.models.llm import LLM


def test_population_initializer_real():
    """
    Test the entire PopulationInitializer workflow with a real LLM, including
    population initialization, augmentation, and parsing, with logging.
    """

    # Initialize the LLM client
    llm_client = LLM()

    # Desired population size
    population_size = 15  # Adjust as necessary

    # Initialize the logger
    logger = PopulationInitializationLogger(log_dir="../../src/evolutionary_algorithms/logs")

    # Initialize the PopulationInitializer with the logger
    initializer = PopulationInitializer(
        layer1_file=constants.layer1_file,
        layer2_file=constants.layer2_file,
        population_size=population_size,
        augment_with_llm=True,
        llm_client=llm_client,
        logger=logger  # Pass the logger
    )

    print("=== Starting Population Initialization ===")
    try:
        # Load prompts from files
        print("\n--- Loading Base Prompts ---")
        layer1_prompts = initializer._load_prompts(constants.layer1_file)
        layer2_prompts = initializer._load_prompts(constants.layer2_file)
        print("Layer 1 Prompts:", layer1_prompts)
        print("Layer 2 Prompts:", layer2_prompts)

        # Generate the full population
        print("\n--- Initializing Population ---")
        population, augmentation_temperature = initializer.initialize_population()

        print("\n--- Population Initialization Completed ---")
        print(f"Base Individuals Used: {len(layer1_prompts)}")
        print(f"Desired Population Size: {population_size}")
        print(f"Total Individuals Generated: {len(population)}")

        # Print any individuals added by augmentation
        if len(population) > len(layer1_prompts):
            print("\n--- Added Individuals from LLM Augmentation ---")
            new_individuals = population[len(layer1_prompts):]
            for idx, individual in enumerate(new_individuals, start=1):
                print(f"Added {idx}. ID: {individual['id']}")
                print(f"         Prompt 1: {individual['prompt_1']}")
                print(f"         Prompt 2: {individual['prompt_2']}")
                print(f"         Source: {individual['source']}")

            # Log the augmentation temperature
            print(f"\nAugmentation Temperature Used: {augmentation_temperature}")


        # Print the final population
        print("\n--- Final Population ---")
        for idx, individual in enumerate(population, start=1):
            print(f"{idx}. ID: {individual['id']}")
            print(f"   Prompt 1: {individual['prompt_1']}")
            print(f"   Prompt 2: {individual['prompt_2']}")
            print(f"   Source: {individual['source']}")

        # Indicate the location of the log file
        print(f"\nPopulation initialization logs saved to {logger.log_file}")

    except ValueError as e:
        print(f"Error during initialization: {e}")

    # Debugging logger contents
    if logger:
        logs = logger.get_logs()
        print("\n--- Logged Data ---")
        for entry in logs:
            print(entry)

    print("\n=== Test Completed ===")


if __name__ == "__main__":
    test_population_initializer_real()
