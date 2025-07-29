import os
import json
from datetime import datetime


class MutationLogger:
    """
    Logger class to record mutation events for analysis.
    """

    def __init__(self, log_dir="logs", log_file_prefix="mutation_log"):
        """
        Initialize the MutationLogger.

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
        #in case a single mutation logger is instantiated and we are performing several mutations with the same LLM as in (only valid for test scenario):
        #mutation_logger = MutationLogger(log_dir="logs")
        #llm_client = GALLM(api_key=API_KEY, endpoint=ENDPOINT)
        #mutator = Mutator(llm_client=llm_client, logger=mutation_logger)
            #perform several mutations
            #logger self.data will add all these mutations to a list as they are performed
        #then calling mutation_logger.save_logs() at the end will save them to the SAME file (concatenated)
        #not technically correct json, but this is simply for testing - better to visualize in one place when assessing mutation prompts

        #alternative to save each mutation to a new file (too much IO, will probably never do this)
        #for every mutation, reinstantiate mutation logger as follows:
            #Re-instantiate MutationLogger for each mutation
            #mutation_logger = MutationLogger(log_dir="logs")
            #Initialize Mutator with a NEW logger for each mutation
            #mutator = Mutator(llm_client=llm_client, logger=mutation_logger)
            #mutate
            #result = mutator.mutate_prompt(prompt, mutation_type)
            #save log for each mutation
            #mutation_logger.save_logs()



    def _generate_log_filename(self):
        """
        Generate a unique log filename using a timestamp.

        Returns:
            str: Full path to the log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #the timestamp for the log file
        return os.path.join(self.log_dir, f"{self.log_file_prefix}_{timestamp}.log")

    def log_mutation(self, mutation_data: dict):
        """
        Log a mutation operation to the log file.

        Args:
            data (dict): Dictionary containing mutation details.
        """
        # Add a timestamp to the mutation data
        mutation_data["timestamp"] = datetime.now().isoformat()
        self.data.append(mutation_data)

    def save_logs(self):
        """
        Writes the collected log data to a .log file.
        Actual internal structure: PYTHON LIST OF JSONS
        [{mutation 1}, {mutation 2}, {mutation 3}, {mutation 4}, {mutation 5}...] for testing in a loop
        Full GA context [{mutation}]
        """
        with open(self.log_file, "w") as file:
            json.dump(self.data, file, indent=4)
        print(f"Logs saved to {self.log_file}")

    def get_logs(self):
        """
        Retrieve all logged mutation data.

        Returns:
            list: List of logged mutation events.
        """
        return self.data

    def reset(self):
        """
        Clear all logged mutation data.
        """
        self.data = []