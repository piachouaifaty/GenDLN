import os
import json
from datetime import datetime

class ReplacementLogger:
    def __init__(self, log_dir="logs"):
        """
        Initializes the ReplacementLogger.

        Args:
            log_dir (str): Directory to save logs. Default is 'logs'.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
        self.logs = []

    def log(self, original_population, sorted_population, new_population):
        """
        Logs the details of the replacement process.

        Args:
            original_population (list): The original population with fitness.
            sorted_population (list): The population sorted by fitness.
            new_population (list): The final selected population.
        """
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_population": original_population,
            "sorted_population": sorted_population,
            "new_population": new_population,
        }
        self.logs.append(log_entry)

    def save_logs(self):
        """
        Saves all logs to a file.
        """
        log_file = os.path.join(self.log_dir, f"replacement_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_file, "w") as f:
            json.dump(self.logs, f, indent=4)
        print(f"Replacement logs saved to {log_file}")

    def get_logs(self):
        """
        Retrieve all logged replacement data.

        Returns:
            list: List of logged replacement events.
        """
        return self.logs

    def reset(self):
        """
        Clear all logged replacement data.
        """
        self.logs = []