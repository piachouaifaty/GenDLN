import os
import json
from datetime import datetime


class GALogger:
    """
    Centralized logger for the GA process.
    """

    def __init__(self, log_dir="logs", log_file_prefix="ga_log"):
        """
        Initialize the GA Logger.

        Args:
            log_dir (str): Directory where log files will be stored.
            log_file_prefix (str): Prefix for log filenames.
        """
        self.log_dir = log_dir
        self.log_file_prefix = log_file_prefix
        self.log_file = self._generate_log_filename()
        os.makedirs(self.log_dir, exist_ok=True)
        self.data = {"generations": []}  # Centralized log structure

    def _generate_log_filename(self):
        """
        Generate a unique log filename using a timestamp.

        Returns:
            str: Full path to the log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"{self.log_file_prefix}_{timestamp}.log")

    def log_population_initialization(self, generation_id, population_data, augmentation_temperature=None):
        """
        Log population initialization details for a generation.

        Args:
            generation_id (int): Current generation ID.
            population_data (list): List of initialized population details.
        """
        entry = {
            "generation": generation_id,
            "initial_population": population_data,
            "augmentation_temperature": augmentation_temperature,
            "selection_data": None,
            "population_after_selection": None,
            "crossover_data": None,
            "population_after_crossover": None,
            "mutation_data": None,
            "population_after_mutation": None,
            "fitness_data": None,
            "replacement_data": None
        }

        if augmentation_temperature is not None:
            entry["augmentation_temperature"] = augmentation_temperature


        self.data["generations"].append(entry)

    def log_generation(self, generation_id, population, fitness_data=None, selection_data=None,
                       population_after_selection=None, crossover_data=None, population_after_crossover=None,
                       mutation_data=None, population_after_mutation=None, replacement_data=None):
        """
        Log details of a single generation.

        Args:
            generation_id (int): Current generation ID.
            population (list): Starting population of the generation.
            fitness_data (dict, optional): Fitness evaluation details.
            selection_data (list, optional): Selection process logs.
            population_after_selection (list, optional): Population after selection.
            crossover_data (list, optional): Crossover process logs.
            population_after_crossover (list, optional): Population after crossover.
            mutation_data (list, optional): Mutation process logs.
            population_after_mutation (list, optional): Population after mutation.
            replacement_data (dict, optional): Replacement process logs.
        """
        entry = {
            "generation": generation_id,
            "population": population,
            "selection_data": selection_data,
            "population_after_selection": population_after_selection,
            "crossover_data": crossover_data,
            "population_after_crossover": population_after_crossover,
            "mutation_data": mutation_data,
            "population_after_mutation": population_after_mutation,
            "fitness_data": fitness_data,
            "replacement_data": replacement_data
        }
        self.data["generations"].append(entry)

    def save_logs(self):
        """
        Save all logged data to a JSON-like log file.
        """
        with open(self.log_file, "w") as file:
            json.dump(self.data, file, indent=4)
        print(f"GA logs saved to {self.log_file}")
