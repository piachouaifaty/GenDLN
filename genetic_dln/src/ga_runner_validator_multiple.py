import os
import json
from datetime import datetime
from genetic_dln.src.constants import constants
from genetic_dln.src.evolutionary_algorithms.ga_engine import GAEngine
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.task.task import Task
from genetic_dln.src.validation.ga_run_summary_validator import validate_run_summary

# Configurations for multiple runs
RUN_CONFIGS = [
    # MULTI
    {"multi_label": True, "selection_strategy": "roulette", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "tournament", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "rank", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "sus", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "steady-state", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "roulette", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "tournament", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "rank", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "sus", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "steady-state", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "roulette", "mutation_type": "syntactic",
     "crossover_type": "token_level", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "tournament", "mutation_type": "syntactic",
     "crossover_type": "token_level", "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "rank", "mutation_type": "syntactic", "crossover_type": "token_level",
     "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "sus", "mutation_type": "syntactic", "crossover_type": "token_level",
     "cache_file": "multi_mistral_large_temp_0"},

    {"multi_label": True, "selection_strategy": "steady-state", "mutation_type": "syntactic",
     "crossover_type": "token_level", "cache_file": "multi_mistral_large_temp_0"},

    # BINARY
    {"multi_label": False, "selection_strategy": "roulette", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "tournament", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "rank", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "sus", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "steady-state", "mutation_type": "semantic",
     "crossover_type": "semantic_blending", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "roulette", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "tournament", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "rank", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "sus", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "steady-state", "mutation_type": "insertion",
     "crossover_type": "phrase_swapping", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "roulette", "mutation_type": "syntactic",
     "crossover_type": "token_level", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "tournament", "mutation_type": "syntactic",
     "crossover_type": "token_level", "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "rank", "mutation_type": "syntactic", "crossover_type": "token_level",
     "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "sus", "mutation_type": "syntactic", "crossover_type": "token_level",
     "cache_file": "binary_mistral_large_temp_0"},

    {"multi_label": False, "selection_strategy": "steady-state", "mutation_type": "syntactic",
     "crossover_type": "token_level", "cache_file": "binary_mistral_large_temp_0"},
]

TASK = Task(
        layer_1_system_prompt_path="",
        layer_2_system_prompt_path="",
        layer_2_few_shots_path="",
        layer_1_initial_prompts_path="",
        layer_2_initial_prompts_path="",
        train_dataset_path="",
        val_dataset_path="",
)


# MIGHT NEED TO CHANGE THESE PATHS TO WORK WITH YOUR ENV
current_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory of this script

#WHERE THE GA_LOGS will be saved
logs_dir = os.path.join(current_dir, "..", "..", "logs")
#Where the summary and validation will be saved
results_dir = os.path.join(current_dir, "..", "..", "ga_results")  # Results directory for GA runs

# summary location


def run_ga_and_validate():
    """
    Runs the genetic algorithm with multiple configurations, generates a summary, gets the best individuals
    from ga_logs and adds them to the summary, and validates the best individuals in the summary.
    """
    print("=== Genetic Algorithm Runner and Validator (Multiple Configurations) ===")

    print("Log dir:", logs_dir)
    print("Results dir:", results_dir)

    os.makedirs(results_dir, exist_ok=True)

    # Load base configuration
    input_loader = InputLoader()
    base_config = input_loader.read_hyperparameters(constants.HYPERPARAMETERS_PATH)["run_config"]

    # STEP 1: Run GA and generate summary
    log_summary = []
    for idx, custom_config in enumerate(RUN_CONFIGS, start=1):
        run_id = f"Run_{idx:02d}"
        print(f"\n=== Starting GA RUN {run_id} ===")

        # Merge base configuration with the custom parameters for this run
        run_config = base_config.copy()
        run_config.update(custom_config)

        # Initialize GA Engine
        ga_engine = GAEngine(config=run_config, task=TASK)

        try:
            # Run the GA
            ga_engine.run()
            print(f"=== {run_id} completed successfully ===")

            # Save summary
            log_summary.append({
                "run_id": run_id,
                "config": run_config,
                "log_file": ga_engine.ga_logger.log_file
            })

        except Exception as e:
            print(f"=== {run_id} failed: {str(e)} ===")
            log_summary.append({
                "run_id": run_id,
                "config": run_config,
                "error": str(e)
            })

    # Save the GA runs summary
    summary_file = f"ga_runs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path = os.path.join(results_dir, summary_file)

    with open(summary_path, "w") as f:
        json.dump(log_summary, f, indent=4)

    print("GA runs completed")
    print(f"GA runs summary saved to: {summary_path}")
    print("-----------------------")


    # STEP 2: Validate best individuals from the summary

    print("-----------------------")
    print("Beginning VALIDATION")
    print("-----------------------")

    validation_dir = os.path.join(results_dir, "val")  # Validation results directory
    os.makedirs(validation_dir, exist_ok=True)
    print("Validation dir:", validation_dir)

    try:
        print("\n=== Validating Best Individuals from Summary ===")
        validate_run_summary(summary_path, logs_dir, validation_dir)
        print(f"Validation process completed successfully. Progress saved incrementally.")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
        run_ga_and_validate()