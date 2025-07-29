import os
import json
from datetime import datetime

class PopulationInitializationLogger:
    """
    Logger for population initialization events.
    """

    def __init__(self, log_dir="logs", log_file_prefix="population_init_log"):
        """
        Initialize the PopulationInitializationLogger.

        Args:
            log_dir (str): Directory where log files will be stored.
            log_file_prefix (str): Prefix for log filenames.
        """
        self.log_dir = log_dir
        self.log_file_prefix = log_file_prefix
        self.log_file = self._generate_log_filename()

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        self.data = []  # Stores logged individuals

    def _generate_log_filename(self):
        """
        Generate a unique log filename using a timestamp.

        Returns:
            str: Full path to the log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"{self.log_file_prefix}_{timestamp}.log")

    def log_individual(self, individual_data: dict):
        """
        Log an individual's initialization details.

        Args:
            individual_data (dict): Data about the individual.
        """
        # Add a timestamp to the individual's data
        individual_data["timestamp"] = datetime.now().isoformat()
        self.data.append(individual_data)

    def save_logs(self):
        """
        Save the logged data to a JSON file.
        """
        with open(self.log_file, "w") as file:
            json.dump(self.data, file, indent=4)
        print(f"Population initialization logs saved to {self.log_file}")

    def get_logs(self):
        """
        Retrieve all logged population initialization data.

        Returns:
            list: List of logged population initialization events.
        """
        return self.data

    def reset(self):
        """
        Clear all logged population initialization data.
        """
        self.data = []