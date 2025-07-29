import os
import json

from genetic_dln.src.constants import constants
from genetic_dln.src.evolutionary_algorithms.ga_engine import GAEngine
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.task.task import Task

TASK = Task(
        layer_1_system_prompt_path="",
        layer_2_system_prompt_path="",
        layer_2_few_shots_path="",
        layer_1_initial_prompts_path="",
        layer_2_initial_prompts_path="",
        train_dataset_path="",
        val_dataset_path="",
)

def run_ga():
    """
    Test the full genetic algorithm run with detailed configurations and logging.
    """
    # Verbose output to console
    print("=== Genetic Algorithm Runner ===")
    print("Loading configurations and setting up GA engine...")

    input_loader = InputLoader()
    # GA Configurations
    config = input_loader.read_hyperparameters(constants.HYPERPARAMETERS_PATH)["run_config"]

    print("Configuration: ")
    print(json.dumps(config, indent=4))

    # Initialize GA engine
    ga_engine = GAEngine(config=config, task=TASK)

    print("GA Engine initialized successfully.")
    print("Starting the genetic algorithm...")

    # Run the genetic algorithm
    try:
        ga_engine.run()
        print("Genetic algorithm run completed successfully.")
    except Exception as e:
        print(f"An error occurred during the GA run: {str(e)}")
        raise

    # Load and display the final logs for verification
    ga_log_path = ga_engine.ga_logger.log_file
    print(f"GA log file saved at: {ga_log_path}")

    if os.path.exists(ga_log_path):
        with open(ga_log_path, "r") as log_file:
            ga_logs = json.load(log_file)
            print("\n=== GA Logs Summary ===")
            print(json.dumps(ga_logs, indent=4))
    else:
        print("Warning: GA log file not found after execution.")


if __name__ == "__main__":
    run_ga()
