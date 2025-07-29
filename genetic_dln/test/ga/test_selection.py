from genetic_dln.src.evolutionary_algorithms.genetic_operations.selection import Selection
from genetic_dln.src.evolutionary_algorithms.loggers.selection_logger import SelectionLogger


def test_selection(population, fitness_scores, elitism_k):
    """
    Test the Selection class with various strategies and edge cases.
    Includes detailed prints for clarity.
    """
    print("\n=== Testing Selection Class ===\n")

    # Initialize the Selection class
    selector = Selection(elitism_k=elitism_k)

    print(f"Initial Population: {population}")
    print(f"Fitness Scores (0-1 range): {fitness_scores}")
    print(f"Elitism (k): {elitism_k}\n")

    # Test each strategy
    strategies = ["roulette", "tournament", "rank", "sus", "random", "steady-state"]
    for strategy in strategies:
        print(f"\n--- Testing Strategy: {strategy} ---")
        print(f"Parameters: elitism_k={elitism_k}")

        if strategy == "tournament":
            kwargs = {"tournament_size": 3}  # Custom parameter for tournament
            print("Tournament Size:", kwargs["tournament_size"])
        else:
            kwargs = {}

        # Perform selection
        selected_population = selector.select(population, fitness_scores, strategy=strategy, **kwargs)

        # Print results
        print("\nSelected Population:")
        for idx, entry in enumerate(selected_population):
            print(f"{idx + 1}. ID: {entry['id']}, Individual: {entry['individual']}, Source: {entry['source']}")


def test_selection_edge():
    print("\n--- Additional Tests for Edge Cases ---")

    # Common population and fitness scores for edge cases
    population = [{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(5)]
    fitness_scores = [0.1, 0.2, 0.2, 0.4, 0.5]

    # Edge Case: No Elitism
    print("\n--- Edge Case: No Elitism (k=0) ---")
    selector_no_elite = Selection(elitism_k=0)
    selected_population_no_elite = selector_no_elite.select(population, fitness_scores, strategy="rank")
    print("\nSelected Population (No Elitism):")
    for idx, entry in enumerate(selected_population_no_elite):
        print(f"{idx + 1}. ID: {entry['id']}, Individual: {entry['individual']}, Source: {entry['source']}")

    # Edge Case: All Equal Fitness
    print("\n--- Edge Case: All Equal Fitness ---")
    equal_fitness_scores = [0.2, 0.2, 0.2, 0.2, 0.2]
    selector = Selection(elitism_k=2)  # Re-initialize selector
    selected_population_equal_fitness = selector.select(population, equal_fitness_scores, strategy="roulette")
    print("Fitness Scores (Equal):", equal_fitness_scores)
    print("\nSelected Population (Equal Fitness):")
    for idx, entry in enumerate(selected_population_equal_fitness):
        print(f"{idx + 1}. ID: {entry['id']}, Individual: {entry['individual']}, Source: {entry['source']}")

    # Edge Case: Small Population
    print("\n--- Edge Case: Small Population ---")
    small_population = [{"id": "I01", "individual": "ind1"}, {"id": "I02", "individual": "ind2"}]
    small_fitness_scores = [0.5, 1.0]
    selected_population_small = selector.select(small_population, small_fitness_scores, strategy="tournament")
    print("Small Population:", small_population)
    print("Small Fitness Scores:", small_fitness_scores)
    print("\nSelected Population (Small Population):")
    for idx, entry in enumerate(selected_population_small):
        print(f"{idx + 1}. ID: {entry['id']}, Individual: {entry['individual']}, Source: {entry['source']}")

    # Edge Case: Steady-State with Ties
    print("\n--- Edge Case: Steady-State with Ties ---")
    tied_fitness_scores = [0.2, 0.2, 0.2, 0.4, 0.5]
    selected_population_steady_state = selector.select(population, tied_fitness_scores, strategy="steady-state")
    print("Fitness Scores (With Ties):", tied_fitness_scores)
    print("\nSelected Population (Steady-State with Ties):")
    for idx, entry in enumerate(selected_population_steady_state):
        print(f"{idx + 1}. Individual: {entry['individual']}, Source: {entry['source']}")

    # Edge Case: Elitism > Population
    print("\n--- Edge Case: Elitism Greater Than Population Size ---")
    selector_high_elite = Selection(elitism_k=10)
    selected_population_high_elite = selector_high_elite.select(population, fitness_scores, strategy="rank")
    print("\nSelected Population (High Elitism):")
    for idx, entry in enumerate(selected_population_high_elite):
        print(f"{idx + 1}. Individual: {entry['individual']}, Source: {entry['source']}")

    print("\n=== Selection Tests Completed ===")


def test_selection_with_logging():
    """
    Test the Selection class with logging integration.
    """
    population = [{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(5)]
    fitness_scores = [0.1, 0.2, 0.2, 0.4, 0.5]
    elitism_k = 2

    # Initialize the logger
    logger = SelectionLogger(log_dir="../../src/evolutionary_algorithms/logs")

    # Initialize the Selection class with the logger
    selector = Selection(elitism_k=elitism_k, logger=logger)

    print("\nTesting Selection with Logging")
    strategies = ["roulette", "tournament", "rank", "random"]
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        selected_population = selector.select(population, fitness_scores, strategy=strategy)
        print(f"Selected Population: {[entry['individual'] for entry in selected_population]}")

    # Save logs
    logger.save_logs()


if __name__ == "__main__":
    # Test Cases
    test_cases = [
        # Test Case 1: Standard population with varying fitness and elitism
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(5)], [0.1, 0.2, 0.2, 0.4, 0.5], 2),

        # Test Case 2: Small population with higher fitness variability
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(3)], [0.3, 0.6, 0.1], 1),

        # Test Case 3: Equal fitness for all individuals
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(4)], [0.25, 0.25, 0.25, 0.25], 0),

        # Test Case 4: Larger population with some individuals having very low fitness
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(6)], [0.5, 0.1, 0.2, 0.1, 0.05, 0.05], 3),

        # Test Case 5: Single individual in population (edge case)
        ([{"id": "I01", "individual": "ind1"}], [0.9], 1),

        # Test Case 6: Population size greater than elitism count
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(5)], [0.2, 0.1, 0.3, 0.4, 0.5], 5),

        # Test Case 7: Elitism count is zero, ensuring no elitism is applied
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(4)], [0.3, 0.3, 0.2, 0.2], 0),

        # Test Case 8: Highly skewed fitness scores with elitism
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(5)], [0.9, 0.05, 0.02, 0.01, 0.02], 2),

        # Test Case 9: Population size equal to elitism count
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(5)], [0.5, 0.4, 0.3, 0.2, 0.1], 5),

        # Test Case 10: Population with fitness scores summing to less than 1
        ([{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(4)], [0.1, 0.1, 0.1, 0.1], 1),

        # Test Case 11: Large population with uniform fitness scores
        (
            [{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(20)],
            [0.05] * 20,
            5,
        ),

        # Test Case 12: Large population with varied fitness scores
        (
            [{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(20)],
            [i / 100 for i in range(1, 21)],
            3,
        ),

        # Test Case 13: Population with non-normalized fitness scores
        (
            [{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(5)],
            [10, 20, 30, 40, 50],
            2,
        ),

        # Test Case 14: Edge case where elitism is larger than population size
        (
            [{"id": f"I0{i + 1}", "individual": f"ind{i + 1}"} for i in range(3)],
            [0.3, 0.6, 0.1],
            5,
        ),
    ]

    for i, (pop, fitness, elite) in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1}: Population Size {len(pop)}, Elitism k={elite} ---")
        print(f"Initial Population: {pop}")
        print(f"Fitness Scores: {fitness}")
        print(f"Elitism k: {elite}")
        test_selection(pop, fitness, elite)

    test_selection_edge()

    test_selection_with_logging()
