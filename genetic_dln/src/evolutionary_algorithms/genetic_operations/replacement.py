def replacement(population_with_fitness, target_size, logger=None):
    """
    Replaces the current population with the top individuals based on fitness.

    Args:
        population_with_fitness (list): A list of dictionaries with "id", "individual", and "fitness" keys.
                                        Example: [{"id": "I01", "individual": <ind1>, "fitness": 0.9}, ...]
        target_size (int): The desired size of the population after replacement.
        logger (ReplacementLogger, optional): Logger instance to log the replacement process.

    Returns:
        list: The new population, a list of the top 'target_size' individuals.
    """
    # Sort the population by fitness in descending order
    sorted_population = sorted(
        population_with_fitness, key=lambda x: x["fitness"], reverse=True
    )

    # Select the top 'target_size' individuals
    new_population = sorted_population[:target_size]

    # Log the process if a logger is provided
    if logger:
        logger.log(
            original_population=population_with_fitness,
            sorted_population=sorted_population,
            new_population=new_population,
        )

    return new_population
