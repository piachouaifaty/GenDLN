import os
import json
from datetime import datetime

class FitnessCacheLogger:
    """
    Logger for tracking cache hits and misses during fitness evaluation.
    """

    def __init__(self, log_dir="logs", log_file_prefix="fitness_cache_log"):
        """
        Initialize the FitnessCacheLogger.

        Args:
            log_dir (str): Directory where log files will be stored.
            log_file_prefix (str): Prefix for log filenames.
        """
        self.log_dir = log_dir
        self.log_file_prefix = log_file_prefix
        self.log_file = self._generate_log_filename()

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        self.data = {
            "cache_hits": [],
            "cache_misses": []
        }  # Separate lists to track hits and misses

    def _generate_log_filename(self):
        """
        Generate a unique log filename using a timestamp.

        Returns:
            str: Full path to the log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"{self.log_file_prefix}_{timestamp}.log")

    def log_hit(self, prompt_pair):
        """
        Log a cache hit.

        Args:
            prompt_pair (tuple): The prompt pair that resulted in a cache hit.
        """
        self.data["cache_hits"].append({
            "prompt_1": prompt_pair[0],
            "prompt_2": prompt_pair[1],
            "timestamp": datetime.now().isoformat()
        })

    def log_miss(self, prompt_pair):
        """
        Log a cache miss.

        Args:
            prompt_pair (tuple): The prompt pair that resulted in a cache miss.
        """
        self.data["cache_misses"].append({
            "prompt_1": prompt_pair[0],
            "prompt_2": prompt_pair[1],
            "timestamp": datetime.now().isoformat()
        })


    def save_logs(self):
        """
        Save the logged data to a JSON file.
        """
        with open(self.log_file, "w") as file:
            json.dump(self.data, file, indent=4)
        print(f"Cache logs saved to {self.log_file}")

    def reset(self):
        """
        Clear all logged data.
        """
        self.data = {
            "cache_hits": [],
            "cache_misses": []
        }

    def get_logs(self):
        """
        Retrieve all logged cache events.

        Returns:
            dict: A dictionary containing cache hits and misses.
        """
        return self.data