from genetic_dln.src.task.task import Task
from genetic_dln.src.constants import constants
from genetic_dln.src.dln.gen_dln import GenDLN
from genetic_dln.src.input_loader.input_loader import InputLoader


class FitnessEvaluator:
    """
    A class to evaluate the fitness of individuals based on multiple criteria.
    Default fitness metric is Accuracy.
    Value of fitness will be between 0 and 1.
    """

    def __init__(self, multi_label: bool = False, task: Task = None):
        """
            This class calculates the fitness for the batch.
        """
        self.input_loader = InputLoader()
        self.multi_label = multi_label
        self.task = task
        self.hyperparameters = self.input_loader.read_hyperparameters(constants.HYPERPARAMETERS_PATH)
        self.binary_weights = self.hyperparameters["binary_weights"]
        self.multi_label_weights = self.hyperparameters["multi_label_weights"]
        self.number_of_workspaces = self.hyperparameters["run_config"]["number_of_workspaces"]

        self.weights = self.binary_weights
        if self.multi_label:
            self.weights = self.multi_label_weights

        self.gen_dln = GenDLN(
            multi_label=self.multi_label,
            number_of_workspaces=self.number_of_workspaces,
            task=self.task
        )

    def _validate_weights(self):
        """
        Ensure the weights sum to 1. If not, raise an error.
        """
        total_weight = sum(self.weights.values())
        if not 0.99 <= total_weight <= 1.01:  # Allow a small margin for floating-point precision
            raise ValueError(
                f"Weights must sum to 1. Current sum: {total_weight:.2f}. Provided weights: {self.weights}")
        print("Weights validation passed.")

    def evaluate_with_dln(self, individual: tuple, index) -> dict:
        """
        Placeholder method to simulate passing an individual through the DLN.

        This function will eventually use the DLN framework to compute real metrics for accuracy, F1, etc.

        Args:
            individual (tuple): A tuple of (prompt_1, prompt_2).
            index (int): The index of the individual - determines which mistral workspace to use later on.

        Returns:
            dict: A dictionary with simulated metrics (accuracy, F1, etc.).
        """

        prompt_01, prompt_02 = individual
        scores = self.gen_dln.predict(prompt_01, prompt_02, index)
        return scores

    def recursive_weighted_sum(self, scores, weights):
        fitness = 0.0
        for key, value in weights.items():
            if key in scores:
                if isinstance(value, dict) and isinstance(scores[key], dict):
                    # Recurse into nested dictionaries
                    fitness += self.recursive_weighted_sum(scores[key], value)
                else:
                    # Apply weight to scalar values
                    fitness += value * scores[key]
        return fitness

    def calculate_fitness(self, individual, generation_id, individual_id, logger=None, cache=None, cache_logger=None, index=0):
        """
        Calculate the fitness of an individual based on the weighted metrics.

        Fitness is computed as a weighted sum of normalized metrics. Accuracy is used by default if no weights are provided.

        Args:
        individual (tuple): A tuple of (prompt_1, prompt_2).
        generation_id (str): ID of the generation.
        individual_id (str): Unique ID for the individual.
        logger (FitnessLogger): Optional logger for recording fitness.
        cache: Optional cache for recording metrics of tuples and avoiding re-computation (rerunning through DLN).
            cache = {
                <individual>: {
                "accuracy": 0.85,
                "precision": 0.9,
                "recall": 0.87,
                # Other metrics...}

        cache_logger (FitnessCacheLogger): Optional logger for recording fitness cache hits and misses.

        Returns:
            float: The calculated fitness score.
        """

        # Check for empty prompts
        prompt_1, prompt_2 = individual
        if not prompt_1.strip() or not prompt_2.strip():
            print(f"Invalid Individual Detected: Empty prompt(s) for {individual}. Returning fitness = -1.")
            # Log the fitness if a logger is provided
            if logger:
                logger.log_individual_fitness(
                    generation_id=generation_id,
                    individual_id=individual_id,
                    individual=individual,
                    raw_metrics={"error": "Empty prompt(s) detected"},
                    fitness=-1,
                    weights=self.weights,
                )
            return -1  # Return fitness of -1 for empty prompts

        # Check if metrics are already cached
        if cache and individual in cache:
            # Cache hit: Retrieve metrics from cache
            scores = cache[individual]

            # Log the cache hit
            if cache_logger:
                cache_logger.log_hit(individual)

            # Print cache hit for visibility
            print(f"Cache HIT for individual: {individual}")

        else:
            # Cache miss: Evaluate and add to cache
            scores = self.evaluate_with_dln(individual, index)

            # Log the cache miss
            if cache_logger:
                cache_logger.log_miss(individual)

            # Print cache miss for visibility
            print(f"Cache MISS for individual: {individual}")

            # Add metrics to the cache
            if cache is not None:
                cache[individual] = scores

        fitness = self.recursive_weighted_sum(scores=scores, weights=self.weights)

        # Log fitness components for debugging
        print(f"Evaluating Individual: {individual}")
        print(f"Scores: {scores}")
        print(f"Fitness Score: {fitness}")

        # Log the fitness if a logger is provided
        # Log raw and normalized metrics for traceability if a logger is provided
        if logger:
            logger.log_individual_fitness(
                generation_id=generation_id,
                individual_id=individual_id,
                individual=individual,
                raw_metrics=scores,
                fitness=fitness,
                weights=self.weights,
            )

        return fitness
