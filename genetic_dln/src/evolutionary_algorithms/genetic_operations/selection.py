import random
import numpy as np


class Selection:
    """
    A class to perform selection for the GA using various strategies.
    Supports elitism and customizable parameters for different selection methods.
    """

    def __init__(self, elitism_k=0, logger = None):
        """
        Initialize the selection process.

        Args:
            elitism_k (int): Number of top individuals to carry over directly to the next generation.
                             These individuals are preserved without modification.
        """
        self.elitism_k = elitism_k
        self.logger = logger

    def select(self, population, fitness_scores, strategy="roulette", **kwargs):
        """
        Perform selection based on the given strategy.

        Args:
            population (list[dict]): List of individuals, where each individual has an "id" and "individual" key.
            fitness_scores (list): Fitness scores corresponding to the population.
            strategy (str): The selection strategy to use. Options include:
                            - "roulette": Roulette Wheel Selection.
                            - "tournament": Tournament Selection.
                            - "rank": Rank-Based Selection.
                            - "sus": Stochastic Universal Sampling (SUS).
                            - "random": Random Selection.
                            - "steady-state": Steady-State Selection.
            **kwargs: Additional parameters specific to the selection strategy.

        Returns:
            list[dict]: List of selected individuals with metadata. Each entry is a dictionary:
                        {"id": "I0x", "individual": <individual_object>, "source": "elite" | "selected"}

        Raises:
            ValueError: If an unknown selection strategy is provided.

        Note:
            - Helper methods (_random_selection, _steady_state_selection, etc.) return plain lists of individuals.
            - Metadata ("source": "elite" or "selected") is added here in the `select` method.

        Elitism ties:
            Randomly broken by reshuffling the population before picking top k elites.
        """
        #1: Elitism - Directly preserve top-k individuals
        shuffled_indices = np.random.permutation(len(fitness_scores))  # Shuffle indices to break ties randomly
        sorted_indices = np.argsort(np.take(fitness_scores, shuffled_indices))[-self.elitism_k:][::-1]  # Sort shuffled indices
        elite_indices = [shuffled_indices[i] for i in sorted_indices]  # Map back to original indices
        elites = [{"id": population[i]["id"], "individual": population[i]["individual"], "source": "elite"} for i in elite_indices]

        #2: Prepare the remaining population for selection
        remaining_population = [population[i] for i in range(len(population)) if i not in elite_indices]
        remaining_fitness_scores = [fitness_scores[i] for i in range(len(population)) if i not in elite_indices]

        # Handle edge case: No remaining population after elitism
        if not remaining_population:
            return elites  # Return only elites as the selected population

        #3: Apply the specified selection strategy
        if strategy == "roulette":
            selected = self._roulette_wheel_selection(remaining_population, remaining_fitness_scores, **kwargs)
        elif strategy == "tournament":
            selected = self._tournament_selection(remaining_population, remaining_fitness_scores, **kwargs)
        elif strategy == "rank":
            selected = self._rank_selection(remaining_population, remaining_fitness_scores, **kwargs)
        elif strategy == "sus":
            selected = self._sus_selection(remaining_population, remaining_fitness_scores, **kwargs)
        elif strategy == "random":
            selected = self._random_selection(remaining_population)
        elif strategy == "steady-state":
            selected = self._steady_state_selection(remaining_population, remaining_fitness_scores, **kwargs)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

        #4: Tag remaining selected individuals
        selected_with_metadata = [{"id": ind["id"], "individual": ind["individual"], "source": "selected"} for ind in selected]

        #5: Combine elites and other selected individuals
        total_population = elites + selected_with_metadata[: len(population) - len(elites)]

        if self.logger:
            # Log the selection event
            self.logger.log_selection(
                strategy={"type": strategy, "params": kwargs},  # Record the strategy and its parameters
                elites=elites,  # Elites are already computed in the code
                selected_population=selected_with_metadata[: len(population) - len(elites)],
            )


        return total_population

    def _roulette_wheel_selection(self, population, fitness_scores, **kwargs):
        """
        Roulette Wheel Selection: Individuals are selected probabilistically based on their fitness.

        Probability of selection for each individual is proportional to its fitness.

        Args:
            population (list): The current population of individuals.
            fitness_scores (list): Fitness scores corresponding to the population.

        Returns:
            list: Selected individuals for the next generation.

        Ties:
            Probabilities are proportional to fitness scores.
            If multiple individuals have the same fitness score, they will have identical selection probabilities.
            The np.random.choice function will break ties randomly based on these probabilities.
        """
        probabilities = np.array(fitness_scores) / sum(fitness_scores)  # Normalize fitness scores into probabilities
        selected_indices = np.random.choice(len(population), len(population), p=probabilities)
        return [population[i] for i in selected_indices]

    def _tournament_selection(self, population, fitness_scores, tournament_size=3, **kwargs):
        """
        Tournament Selection: Groups of individuals (tournaments) are formed randomly,
        and the fittest individual in each group is selected.

        Args:
            population (list): The current population of individuals.
            fitness_scores (list): Fitness scores corresponding to the population.
            tournament_size (int): Number of individuals in each tournament.

        Returns:
            list: Selected individuals for the next generation.

        Ties:
            intra-tournament ties are handled by max (picks first occurring max). shuffling is performed to avoid order-bias
        """
        selected = []

        # Ensure tournament size does not exceed population size
        adjusted_tournament_size = min(tournament_size, len(population))
        if adjusted_tournament_size < tournament_size:
            print(
                f"WARNING: Tournament size ({tournament_size}) exceeds remaining population size "
                f"({len(population)}). Adjusted tournament size: {adjusted_tournament_size}."
            )
        elif adjusted_tournament_size == len(population):
            print(
                f"NOTICE: Tournament size equals the remaining population size "
                f"({len(population)}). This is equivalent to selecting the fittest remaining individual every single time and will lead to lack of diversity."
            )

        #perform tournament selection
        for _ in range(len(population)):
            contenders = random.sample(range(len(population)), adjusted_tournament_size)  # Randomly pick contenders
            random.shuffle(contenders)  # Break ties randomly
            #shuffling the tournament subset before comparison to ensure no ordering bias - since max will select the first occurring highest
            winner = max(contenders, key=lambda idx: fitness_scores[idx])  # Select the fittest among contenders
            selected.append(population[winner])
        return selected

    def _rank_selection(self, population, fitness_scores, **kwargs):
        """
        Rank-Based Selection: Individuals are ranked by fitness, and probabilities of selection
        are proportional to their rank (not their raw fitness).

        Args:
            population (list): The current population of individuals.
            fitness_scores (list): Fitness scores corresponding to the population.

        Returns:
            list: Selected individuals for the next generation.

        Ties:
            Individuals with the same fitness will have the same rank, and identical selection probabilities.
            Handled implicitly and fairly by the probabilistic nature of the selection.
        """
        ranks = np.argsort(np.argsort(fitness_scores))  # Calculate ranks (0 = lowest fitness, n-1 = highest fitness)

        # Convert ranks to probabilities and ensure normalization
        probabilities = (ranks + 1) / sum(ranks + 1)  # Initial calculation
        probabilities = np.array(probabilities, dtype=np.float64)  # Cast to float64 for precision

        # Normalize and handle floating-point errors
        probabilities /= probabilities.sum()  # Ensure the sum is exactly 1

        probabilities[-1] += 1.0 - probabilities.sum()  # Adjust last probability if needed (fix precision)

        # Debug print for validation
        print(f"Rank probabilities (sum={probabilities.sum()}): {probabilities}")

        # Perform selection
        selected_indices = np.random.choice(len(population), len(population), replace=False, p=probabilities)

        return [population[i] for i in selected_indices]

    def _sus_selection(self, population, fitness_scores, **kwargs):
        """
        Stochastic Universal Sampling (SUS): Distributes selection points evenly across the
        fitness-proportionate range, ensuring a diverse selection.

        Args:
            population (list): The current population of individuals.
            fitness_scores (list): Fitness scores corresponding to the population.

        Returns:
            list: Selected individuals for the next generation.

        Ties:
            NA, handled implicitly
        """
        total_fitness = sum(fitness_scores)
        point_distance = total_fitness / len(population)  # Distance between selection points
        start_point = random.uniform(0, point_distance)  # First point is chosen randomly
        points = [start_point + i * point_distance for i in range(len(population))]  # Distribute points

        selected = []
        fitness_accumulator = 0
        i = 0
        for point in points:
            while fitness_accumulator < point:
                fitness_accumulator += fitness_scores[i]
                i += 1
            selected.append(population[i - 1])  # Select the individual corresponding to the point
        return selected

    def _random_selection(self, population):
        """
        Perform random selection of individuals.

        Args:
            population (list): The population to select from.

        Returns:
            list: Selected individuals (without replacement).
        """
        return random.sample(population, len(population))

    def _steady_state_selection(self, population, fitness_scores, **kwargs):
        """
        Perform steady-state selection, (also known as non-generational) preserving elites and replacing a fraction of the population.

        Args:
            population (list): The population to select from.
            fitness_scores (list): Fitness scores corresponding to the population.
            **kwargs: Additional parameters, such as:
                      - k (int): Number of elites to preserve (should match elitism_k).

        Returns:
            list: Selected individuals for the next generation.
        """
        k = kwargs.get("k", self.elitism_k)

        # Preserve the top-k individuals
        elite_indices = np.argsort(fitness_scores)[-k:][::-1]
        elites = [population[i] for i in elite_indices]

        # Select the remaining individuals randomly without replacement
        remaining_population = [population[i] for i in range(len(population)) if i not in elite_indices]
        num_to_select = len(population) - k
        selected = random.sample(remaining_population, num_to_select)

        return elites + selected


