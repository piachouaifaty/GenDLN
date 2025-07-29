from genetic_dln.src.evolutionary_algorithms.genetic_operations.fitness import FitnessEvaluator
from genetic_dln.src.evolutionary_algorithms.loggers.fitness_cache_logger import FitnessCacheLogger
from genetic_dln.src.evolutionary_algorithms.loggers.fitness_logger import FitnessLogger

# Test configuration
generation_id = "G01"


def test_fitness():
    """
    Test the FitnessEvaluator and FitnessLogger integration.
    """
    # Initialize the FitnessEvaluator
    evaluator = FitnessEvaluator()

    # Initialize the FitnessLogger and FitnessCacheLogger
    logger = FitnessLogger(log_dir="../../src/evolutionary_algorithms/logs")
    cache_logger = FitnessCacheLogger(log_dir="../../src/evolutionary_algorithms/logs")

    # Manually create a fitness cache with precomputed metrics for 2 individuals
    cache = {
        ("Summarize the following text.", "Is the above text fair or unfair?"): {
            "accuracy": 0.9,
            "precision": 0.85,
            "recall": 0.88,
            "f1": 0.86,
            "macro_f1": 0.87,
            "weighted_f1": 0.84,
            "micro_f1": 0.85,
            "auc_roc": 0.9,
            "semantic": 0.8,
            "diversity": 0.75,
            "perplexity": 30.0,
            "response_length": 40.0,
        },
        ("Explain the key points of this document.", "Based on the details above, is this fair?"): {
            "accuracy": 0.8,
            "precision": 0.78,
            "recall": 0.82,
            "f1": 0.80,
            "macro_f1": 0.79,
            "weighted_f1": 0.81,
            "micro_f1": 0.82,
            "auc_roc": 0.85,
            "semantic": 0.77,
            "diversity": 0.73,
            "perplexity": 25.0,
            "response_length": 35.0,
        },
    }

    # Example population of individuals (each individual is a tuple of prompts)
    population = [
        ("Summarize the following text.", "Is the above text fair or unfair?"),
        ("Explain the key points of this document.", "Based on the details above, is this fair?"),
        ("Provide a concise overview of this topic.", "Determine whether the topic is just or unjust."),
        ("Highlight the main ideas from the text.", "Assess the fairness of the statements above."),
    ]

    # Assign unique IDs to individuals
    individual_ids = [f"I{i:02d}" for i in range(1, len(population) + 1)]

    print(f"Testing Fitness Evaluation for Generation: {generation_id}")
    print("===================================================")

    print("PRE RUN CACHE: ")
    print(cache)

    # Evaluate fitness for each individual in the population
    for individual, individual_id in zip(population, individual_ids):
        print(f"Evaluating Individual ID: {individual_id}")
        fitness = evaluator.calculate_fitness(
            individual=individual,
            generation_id=generation_id,
            individual_id=individual_id,
            logger=logger,  # Log fitness for each individual
            cache=cache,
            cache_logger=cache_logger,
        )
        print(f"Fitness Score: {fitness}")
        print("---------------------------------------------------")

    # Summarize the generation
    print("\nSummarizing Generation:")
    generation_summary = logger.summarize_generation(generation_id=generation_id)

    # Save the logs to file
    logger.save_logs(generation_summary)

    # Print summary details
    print("\nGeneration Summary:")
    print(f"Generation ID: {generation_summary['generation_id']}")
    print(f"Average Fitness: {generation_summary['average_fitness']:.2f}")
    print("Best Individual:")
    print(generation_summary["best_individual"])
    print("Worst Individual:")
    print(generation_summary["worst_individual"])

    print("POST RUN CACHE: ")
    print(cache)

    cache_logger.save_logs()


if __name__ == "__main__":
    test_fitness()
