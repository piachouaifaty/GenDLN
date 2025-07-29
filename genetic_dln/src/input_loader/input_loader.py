import json
import os
from typing import Any

import pandas as pd
import yaml
from pandas import DataFrame
from sklearn.utils import resample

from genetic_dln.src.constants import constants
from genetic_dln.src.custom_logger.custom_logger import CustomLogger


# ------------------------------------- TODO: Make a better description
# Here are the functions to read the standard files we use as configuration
# -------------------------------------

class InputLoader:
    def __init__(self):
        log_dir = os.path.join(constants.root_folder, "logs")
        self.logger = CustomLogger(log_dir=log_dir, logger_name="InputLoader").logger

    def read_hyperparameters(self, hyperparameter_path: str):
        with open(hyperparameter_path, "r") as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file: {exc}")
                return None

    def load_prompt_template(self, prompt_template_path: str):
        with open(prompt_template_path, "r", encoding='utf-8') as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file: {exc}")
                return None

    def load_few_shots(self, few_shots_path):
        with open(few_shots_path, "r") as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file: {exc}")
                return None

    def read_initial_prompts(self, prompts_path: str) -> list[str]:
        """
        Reads initial prompts from a JSON file.

        Args:
            prompts_path (str): Path to the JSON file containing prompts.

        Returns:
            list[str]: A list of prompts.
        """
        with open(prompts_path, "r") as f:
            data = json.load(f)  # Use json.load instead of json.loads
        return data["prompts"]

    def read_data(self, data_path: str) -> DataFrame:
        return pd.read_csv(data_path)

    def load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        return data


    def load_balanced_subset(self, dataset_path: str, sample_size: int=100) -> dict:
        """
        Load a balanced subset of the Claudette dataset with equal representation of "fair" and "unfair" labels,
        returning the data as a JSON object.
        """
        self.logger.info("Loading dataset from %s", dataset_path)
        data = pd.read_csv(dataset_path, sep='\t')
        total_rows = len(data)
        self.logger.info("Dataset loaded with %d entries", total_rows)

        # Check if dataset has 'text' and 'label' columns
        if not all(col in data.columns for col in ['text', 'label']):
            self.logger.error("Dataset must contain 'text' and 'label' columns.")
            return {}

        # Separate data into fair (label=0) and unfair (label=1)
        fair_data = data[data['label'] == 0]
        unfair_data = data[data['label'] == 1]
        samples_per_label = sample_size // 2

        # Resample to get balanced fair and unfair samples
        fair_sample = resample(fair_data, n_samples=samples_per_label, random_state=42, replace=True)
        unfair_sample = resample(unfair_data, n_samples=samples_per_label, random_state=42, replace=True)
        final_data = pd.concat([fair_sample, unfair_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

        json_output = {}
        for idx, row in final_data.iterrows():
            try:
                sentence_id = f"sentence_{idx}"
                json_output[sentence_id] = {
                    "text": row['text'],
                    "label": row['label']
                }
            except Exception as e:
                self.logger.warning("Error processing row %d: %s", idx, e)

        self.logger.info("Balanced dataset of %d samples prepared", sample_size)
        return json_output

    def load_balanced_binary_subset(self, dataset_path, sample_size: int = 100, batch_size=10):
        """
        Load the entire Claudette dataset and yield it in batches, converting each batch
        into the JSON input format for the LLM.

        Args:
            dataset_path (str): Path to the dataset file.
            sample_size (int): Size of balanced subset.
            batch_size (int): Number of data points to include in each batch.

        Yields:
            dict: A dictionary in JSON input format for each batch.
        """
        data = pd.read_csv(dataset_path, sep='\t')

        # Check if dataset has 'text' and 'label' columns
        if not all(col in data.columns for col in ['text', 'label']):
            self.logger.error("Dataset must contain 'text' and 'label' columns.")
            return {}

        # Separate data into fair (label=0) and unfair (label=1)
        fair_data = data[data['label'] == 0]
        unfair_data = data[data['label'] == 1]
        samples_per_label = sample_size // 2

        # Resample to get balanced fair and unfair samples
        fair_sample = resample(fair_data, n_samples=samples_per_label, random_state=42, replace=True)
        unfair_sample = resample(unfair_data, n_samples=samples_per_label, random_state=42, replace=True)
        final_data = pd.concat([fair_sample, unfair_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

        for i in range(0, len(final_data), batch_size):
            batch = final_data.iloc[i:i + batch_size]
            json_input = {}

            for idx, row in batch.iterrows():
                try:
                    sentence_id = f"sentence_{idx}"
                    json_input[sentence_id] = {
                        "text": row['text'],
                        "label": row['label']
                    }
                except Exception as e:
                    self.logger.info(f"Error processing row {idx}: {e}")

            yield json_input
            self.logger.info("Processed batch %d of %d", (i // batch_size) + 1, (len(final_data) // batch_size) + 1)

