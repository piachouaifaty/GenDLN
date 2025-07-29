import os
import json
from datetime import datetime


class CrossoverLogger:
    """
    Logger class to record crossover events for analysis.
    """

    def __init__(self, log_dir="logs", log_file_prefix="crossover_log"):
        """
        Initialize the CrossoverLogger.

        Args:
            log_dir (str): Directory where log files will be stored.
            log_file_prefix (str): Prefix for log filenames.
        """
        self.log_dir = log_dir
        self.log_file_prefix = log_file_prefix
        self.log_file = self._generate_log_filename()

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        self.data = []
        # Similar to MutationLogger: logs are appended to `self.data` for bulk saving later.

    def _generate_log_filename(self):
        """
        Generate a unique log filename using a timestamp.

        Returns:
            str: Full path to the log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"{self.log_file_prefix}_{timestamp}.log")

    def log_crossover(self, crossover_data: dict):
        """
        Log a crossover operation to the log file.

        Args:
            crossover_data (dict): Dictionary containing crossover details.
        """
        # Add a timestamp to the crossover data
        crossover_data["timestamp"] = datetime.now().isoformat()
        self.data.append(crossover_data)

    def save_logs(self):
        """
        Writes the collected log data to a .log file.
        Actual internal structure: PYTHON LIST OF JSONS
        [{crossover 1}, {crossover 2}, {crossover 3}...]
        """
        with open(self.log_file, "w") as file:
            json.dump(self.data, file, indent=4)
        print(f"Logs saved to {self.log_file}")

    def get_logs(self):
        """
        Retrieve all logged crossover data.

        Returns:
            list: List of logged crossover events.
        """
        return self.data

    def reset(self):
        """
        Clear all logged crossover data.
        """
        self.data = []
