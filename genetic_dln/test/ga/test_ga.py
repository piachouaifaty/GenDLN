import os
from dotenv import load_dotenv
import json

from genetic_dln.src.constants import constants
from genetic_dln.src.evolutionary_algorithms.ga_engine import GAEngine
from genetic_dln.src.input_loader.input_loader import InputLoader


def test_ga_engine():
    """
    Test the GA engine with real-world configurations and logging.
    """
    # Configurations
    input_loader = InputLoader()
    # GA Configurations
    config = input_loader.read_hyperparameters(constants.HYPERPARAMETERS_PATH)["run_config"]

    # Initialize GA engine
    ga_engine = GAEngine(config=config)

    # Run the engine
    #ga_engine.run()

    ga_engine.initialize_population()

    print("POPULATION IS (AFTER INIT):")
    print(ga_engine.population)

    # Validate GA logger for population initialization
    ga_logger = ga_engine.ga_logger
    assert len(ga_logger.data["generations"]) == 1, "GA log should have one entry for generation 0."
    assert "initial_population" in ga_logger.data["generations"][0], "Initial population data missing."
    assert ga_logger.data["generations"][0]["initial_population"], "Initial population log should not be empty."
    print("GA logger contents after initialization:")
    print(json.dumps(ga_logger.data, indent=4))

    x = '''# TEST CACHE MANUALLY // REMOVE WHEN ENTIRE LOOP IS IMPLEMENTED

    # Create a Fitness Cache and Populate It
    fitness_cache = {}
    precomputed_individuals = ga_engine.population[:2]  # Precompute metrics for 2 individuals

    for individual in precomputed_individuals:
        key = (individual["prompt_1"], individual["prompt_2"])
        fitness_cache[key] = {
            "accuracy": 0.85,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "macro_f1": 0.84,
            "weighted_f1": 0.86,
            "micro_f1": 0.87,
            "auc_roc": 0.88,
            "semantic": 0.8,
            "diversity": 0.75,
            "perplexity": 25,
            "response_length": 30,
        }

    # Assign the cache to the GA engine
    ga_engine.fitness_cache = fitness_cache

    # Step 2: Test fitness evaluation
    print("\n=== Testing Fitness Evaluation ===")
    ga_engine.evaluate_fitness()  # This should compute fitness for generation 0

    # Validate Cache Logger
    print("\n=== Fitness Cache Logger ===")
    cache_logger = ga_engine.cache_logger
    cache_logs = cache_logger.get_logs()

    # Remove duplicate entries from hits and misses
    cache_hits = {tuple((log["prompt_1"], log["prompt_2"])) for log in cache_logs["cache_hits"]}
    cache_misses = {tuple((log["prompt_1"], log["prompt_2"])) for log in cache_logs["cache_misses"]}

    # Check for overlapping entries
    overlapping_entries = cache_hits & cache_misses
    assert not overlapping_entries, f"Overlap found between cache hits and misses: {overlapping_entries}"

    # Print logs for verification
    print("Cache Logs:")
    print(json.dumps(cache_logs, indent=4))

    # Assert cache hits and misses
    assert len(cache_logs["cache_hits"]) == len(precomputed_individuals), \
        "Number of cache hits does not match precomputed individuals."

    # Save cache logs
    cache_logger.save_logs()

    # END CACHE VALIDATION'''

    # Step 2: Test fitness evaluation
    print("\n=== Testing Fitness Evaluation ===")
    ga_engine.evaluate_fitness()  # This should compute fitness for generation 0

    # Validate GA logger for fitness data
    assert "fitness_data" in ga_logger.data["generations"][0], "Fitness data missing from GA logger."
    fitness_summary = ga_logger.data["generations"][0]["fitness_data"]
    assert "average_fitness" in fitness_summary, "Fitness summary missing average fitness."
    print("GA logger contents after fitness evaluation:")
    print(json.dumps(ga_logger.data, indent=4))

    # Step 3: Validate that FitnessLogger was reset (indirectly)
    fitness_logs = ga_engine.fitness_logger.get_logs()
    assert not fitness_logs, "Fitness logger should be empty after reset."

    print("Fitness evaluation test completed successfully.")

    # Step 4: Test selection process
    print("\n=== Testing Selection ===")
    selected_population = ga_engine.perform_selection()  # Perform selection

    print("POPULATION IS (AFTER SELECTION):")
    print(ga_engine.population)

    # Validate GA logger for selection data
    assert "selection_data" in ga_logger.data["generations"][0], "Selection data missing from GA logger."
    selection_logs = ga_logger.data["generations"][0]["selection_data"]
    assert selection_logs, "Selection logger should have logged data."
    print("GA logger contents after selection:")
    print(json.dumps(ga_logger.data, indent=4))

    # Validate that the selected population is a subset of the original population
    selected_ids = {ind["id"] for ind in selected_population}
    original_ids = {ind["id"] for ind in ga_engine.population}
    assert selected_ids.issubset(original_ids), "Selected population contains unknown individuals."

    # Validate GA logger for final selected population
    assert "population_after_selection" in ga_logger.data["generations"][
        0], "Final selected population missing from GA logger."
    population_after_selection = ga_logger.data["generations"][0]["population_after_selection"]
    assert population_after_selection, "Final selected population log should not be empty."
    print("Final selected population (logged in GA Logger):")
    print(json.dumps(population_after_selection, indent=4))

    # Validate that the final selected population matches the structure of `self.population`
    assert len(population_after_selection) == len(
        population_after_selection), "Mismatch in final selected population size."
    for ind in population_after_selection:
        assert "id" in ind and "fitness_score" in ind, "Missing keys in final selected population log."
        assert "prompt_1" in ind and "prompt_2" in ind, "Missing prompts in final selected population log."

    print("Selection test completed successfully.")

    # Step 4: Test crossover process
    print("\n=== Testing Crossover ===")
    ga_engine.perform_crossover()  # Perform crossover

    print("POPULATION IS (AFTER CROSSOVER):")
    print(ga_engine.population)

    # Validate offspring in the population
    offspring = [ind for ind in ga_engine.population if ind["source"] == "offspring"]
    assert offspring, "No offspring generated during crossover."
    print(f"Number of offspring generated: {len(offspring)}")

    # Validate GA logger for crossover data
    assert "crossover_data" in ga_logger.data["generations"][0], "Crossover data missing from GA logger."
    crossover_logs = ga_logger.data["generations"][0]["crossover_data"]
    assert crossover_logs, "Crossover logger should have logged data."
    print("GA logger contents after crossover:")
    print(json.dumps(ga_logger.data, indent=4))

    # Validate the structure of offspring
    for ind in offspring:
        assert "id" in ind and "fitness_score" in ind, "Missing keys in offspring log."
        assert "prompt_1" in ind and "prompt_2" in ind, "Missing prompts in offspring log."

    # Validate GA logger for population after crossover
    assert "population_after_crossover" in ga_logger.data["generations"][
        0], "Population after crossover missing from GA logger."
    crossover_population_log = ga_logger.data["generations"][0]["population_after_crossover"]
    assert crossover_population_log, "Population after crossover log should not be empty."
    print("Population after crossover (logged in GA Logger):")
    print(json.dumps(crossover_population_log, indent=4))

    print("Crossover test completed successfully.")

    # Step 5: Test mutation process
    print("\n=== Testing Mutation ===")
    ga_engine.perform_mutation()  # Perform mutation

    print("POPULATION IS (AFTER MUTATION):")
    print(ga_engine.population)

    # Validate mutation in the population
    mutated_individuals = [
        ind for ind in ga_engine.population if ind["source"] != "elite"
    ]  # Mutation affects non-elites unless configured otherwise
    assert mutated_individuals, "No individuals were mutated during the mutation process."
    print(f"Number of individuals potentially mutated: {len(mutated_individuals)}")

    # Validate GA logger for mutation data
    assert "mutation_data" in ga_logger.data["generations"][0], "Mutation data missing from GA logger."
    mutation_logs = ga_logger.data["generations"][0]["mutation_data"]
    assert mutation_logs, "Mutation logger should have logged data."
    print("GA logger contents after mutation:")
    print(json.dumps(ga_logger.data, indent=4))

    # Validate "mutated" flag in the population
    mutated_flags = [ind["mutated"] for ind in ga_engine.population]
    assert all(isinstance(flag, bool) for flag in mutated_flags), "Each individual must have a 'mutated' flag set."
    print("Mutated flags in population:", mutated_flags)

    # Validate GA logger for population after mutation
    assert "population_after_mutation" in ga_logger.data["generations"][
        0], "Population after mutation missing from GA logger."
    mutation_population_log = ga_logger.data["generations"][0]["population_after_mutation"]
    assert mutation_population_log, "Population after mutation log should not be empty."
    print("Population after mutation (logged in GA Logger):")
    print(json.dumps(mutation_population_log, indent=4))

    # Validate the structure of mutated population
    for ind in ga_engine.population:
        assert "id" in ind and "fitness_score" in ind, "Missing keys in population after mutation."
        assert "prompt_1" in ind and "prompt_2" in ind, "Missing prompts in population after mutation."

    print("Mutation test completed successfully.")


if __name__ == "__main__":
    test_ga_engine()
