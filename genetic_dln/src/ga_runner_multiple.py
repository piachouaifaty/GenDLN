import os
import json
from datetime import datetime
from genetic_dln.src.constants import constants
from genetic_dln.src.evolutionary_algorithms.ga_engine import GAEngine
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.task.task import Task

# Configurations for multiple runs


RUN_CONFIGS = [
    {
        "multi_label": False,
        "selection_strategy": "sus",
        "mutation_type": "syntactic",
        "crossover_type": "token_level",
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "roulette",
        "mutation_type": "insertion",
        "crossover_type": "phrase_swapping",
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "steady-state",
        "mutation_type": "deletion",
        "crossover_type": "semantic_blending",
        "crossover_rate": 0.85,
        "mutation_rate": 0.2,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "rank",
        "mutation_type": "insertion",
        "crossover_type": "semantic_blending",
        "crossover_rate": 0.85,
        "mutation_rate": 0.3,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "sus",
        "mutation_type": "inversion",
        "crossover_type": "semantic_blending",
        "crossover_rate": 0.85,
        "mutation_rate": 0.2,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "steady-state",
        "mutation_type": "semantic",
        "crossover_type": "phrase_swapping",
        "crossover_rate": 0.8,
        "mutation_rate": 0.3,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "steady-state",
        "mutation_type": "scramble",
        "crossover_type": "two_point",
        "crossover_rate": 0.85,
        "mutation_rate": 0.2,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "tournament",
        "selection_params": {"tournament_size": 3},
        "mutation_type": "insertion",
        "crossover_type": "token_level",
        "crossover_rate": 0.85,
        "mutation_rate": 0.2,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    },
    {
        "multi_label": False,
        "selection_strategy": "sus",
        "mutation_type": "random",
        "crossover_type": "single_point",
        "crossover_rate": 0.8,
        "mutation_rate": 0.3,
        "mutate_elites": False,
        "population_size": 30,
        "generations": 30,
        "elitism_k": 1,
        "early_stopping": {"max_stagnant_generations": 10, "fitness_goal": 0.99},
        "cache_file": "binary_mistral_large_temp_0"
    }
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

# GA Runner for multiple runs
def run_ga_multiple():
    """
    Run multiple configurations of the GA with logging for each run.
    """
    print("=== Genetic Algorithm Runner (Multiple Configurations) ===")

    # Input loader for shared hyperparameters
    input_loader = InputLoader()
    base_config = input_loader.read_hyperparameters(constants.HYPERPARAMETERS_PATH)["run_config"]

    results_dir = "ga_results"
    os.makedirs(results_dir, exist_ok=True)
    log_summary = []

    for idx, custom_config in enumerate(RUN_CONFIGS, start=1):
        run_id = f"Run_{idx:02d}"
        print(f"\n=== Starting RUN {run_id} ===")

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

    # Save a summary of all runs
    summary_file = os.path.join(results_dir, f"ga_runs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w") as f:
        json.dump(log_summary, f, indent=4)
    print(f"Run summary saved to {summary_file}")

if __name__ == "__main__":
    run_ga_multiple()