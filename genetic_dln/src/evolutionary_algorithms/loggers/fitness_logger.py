import os
import json
from datetime import datetime

class FitnessLogger:
    """
    Logger class to record fitness evaluations for individuals and generations.
    """

    def __init__(self, log_dir="logs", log_file_prefix="fitness_log"):
        """
        Initialize the FitnessLogger.

        Args:
            log_dir (str): Directory where log files will be stored.
            log_file_prefix (str): Prefix for log filenames.
        """
        self.log_dir = log_dir
        self.log_file_prefix = log_file_prefix
        self.log_file = self._generate_log_filename()

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        self.generation_data = []  # Holds all fitness data for a generation

    def _generate_log_filename(self):
        """
        Generate a unique log filename using a timestamp.

        Returns:
            str: Full path to the log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"{self.log_file_prefix}_{timestamp}.log")

    def log_individual_fitness(self, generation_id, individual_id, individual, raw_metrics, fitness, weights):
        """
        Log the fitness evaluation for a single individual.

        Args:
            generation_id (str): ID of the generation (e.g., "G01").
            individual_id (str): Unique ID for the individual.
            individual (tuple): Tuple of (prompt_1, prompt_2).
            raw_metrics (dict): Raw metric values from evaluation.
            fitness_score (float): Final fitness score.
            weights (dict): Weights used to calculate fitness.
        """
        log_entry = {
            "generation": generation_id,
            "individual_id": individual_id,
            "individual": {"prompt_1": individual[0], "prompt_2": individual[1]},
            "raw_metrics": raw_metrics,
            "fitness_score": fitness,
            "weights": weights,
            "timestamp": datetime.now().isoformat(),
        }
        self.generation_data.append(log_entry)

    def summarize_generation(self, generation_id):
        """
        Summarize the fitness data for a generation.

        Args:
            generation_id (int): ID of the generation (e.g., "G01").

        Returns:
            dict: Summary of the generation.
        """
        fitness_scores = [entry["fitness_score"] for entry in self.generation_data]
        best_individual = max(self.generation_data, key=lambda x: x["fitness_score"])
        worst_individual = min(self.generation_data, key=lambda x: x["fitness_score"])
        average_fitness = sum(fitness_scores) / len(fitness_scores)

        summary = {
            "generation_id": generation_id,
            "best_individual": best_individual,
            "worst_individual": worst_individual,
            "average_fitness": average_fitness,
            "fitness_summary": self.generation_data,
        }
        return summary

    def save_logs(self, summary):
        """
        Save the fitness logs to a file.

        Args:
            summary (dict): Summary of the generation-level fitness data.
        """
        with open(self.log_file, "w") as file:
            json.dump(summary, file, indent=4)
        print(f"Fitness logs saved to {self.log_file}")

    def get_logs(self):
        """
        Retrieve all logged fitness data for the current generation.

        Returns:
            list: List of logged fitness evaluations.
        """
        return self.generation_data

    def reset(self):
        """
        Clear all logged fitness data for the current generation.
        """
        self.generation_data = []

