from genetic_dln.src.evolutionary_algorithms.genetic_operations.replacement import replacement
from genetic_dln.src.evolutionary_algorithms.loggers.replacement_logger import ReplacementLogger


def test_replacement():
    """
    Tests the replacement function with a logger and various scenarios.
    """
    # Example population
    population_with_fitness = [
        {"id": "I01", "individual": "ind1", "fitness": 0.9},
        {"id": "I02", "individual": "ind2", "fitness": 0.6},
        {"id": "I03", "individual": "ind3", "fitness": 0.8},
        {"id": "I04", "individual": "ind4", "fitness": 0.7},
        {"id": "I05", "individual": "ind5", "fitness": 0.5},
    ]
    target_size = 3

    # Initialize the logger
    logger = ReplacementLogger(log_dir="../../src/evolutionary_algorithms/logs")

    # Perform replacement
    print("Testing Replacement Function...")
    new_population = replacement(population_with_fitness, target_size, logger=logger)

    # Print results
    print("\nOriginal Population:")
    for individual in population_with_fitness:
        print(f"ID: {individual['id']}, Fitness: {individual['fitness']}")

    print("\nNew Population:")
    for individual in new_population:
        print(f"ID: {individual['id']}, Fitness: {individual['fitness']}")

    # Save logs
    logger.save_logs()


if __name__ == "__main__":
    test_replacement()
