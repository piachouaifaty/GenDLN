import os
import json
from datetime import datetime


class SelectionLogger:
    """
    Logger class to record selection events for analysis.
    """

    def __init__(self, log_dir="logs", log_file_prefix="selection_log"):
        """
        Initialize the SelectionLogger.

        Args:
            log_dir (str): Directory where log files will be stored.
            log_file_prefix (str): Prefix for log filenames.
        """
        self.log_dir = log_dir
        self.log_file_prefix = log_file_prefix
        self.log_file = self._generate_log_filename()

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        self.data = []  # Holds selection log entries

    def _generate_log_filename(self):
        """
        Generate a unique log filename using a timestamp.

        Returns:
            str: Full path to the log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"{self.log_file_prefix}_{timestamp}.log")

    def log_selection(self, strategy, elites, selected_population):
        """
        Log a selection event.

        Args:
            strategy (dict): Details of the selection strategy (e.g., type, parameters).
            elites (list[dict]): List of elite individuals (e.g., [{"id": "I01", "fitness": 0.95}]).
            selected_population (list[dict]): List of selected individuals (e.g., [{"id": "I02", "individual": <individual>, "source": "selected"}]).
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,  # e.g., {"type": "tournament", "params": {"size": 3}}
            "elites": elites,  # [{"id": "I01", "individual": <ind>...}, ...]
            "selected_population": selected_population,  # [{"id": "I02", "individual": <ind>, "source": "selected"}, ...]
        }

        self.data.append(log_entry)

    def save_logs(self):
        """
        Save all logged selection events to a JSON file.
        """
        with open(self.log_file, "w") as file:
            json.dump(self.data, file, indent=4)
        print(f"Selection logs saved to {self.log_file}")

    def get_logs(self):
        """
        Retrieve all logged selection data.

        Returns:
            list: List of logged selection events.
        """
        return self.data

    def reset(self):
        """
        Clear all logged selection data.
        """
        self.data = []