import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import accuracy_score, classification_report

import constants
from custom_logger import CustomLogger
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.models.llm import LLM
from genetic_dln.src.models.rate_limiter import RateLimiter
from genetic_dln.src.post_processor.post_processor import PostProcessor
from genetic_dln.src.prompt_builder.base_prompt_builder import BasePromptBuilder
from genetic_dln.src.prompt_builder.prompt_builder import PromptBuilder


class OPROPrompterBinary:
    def __init__(self, validation=False):
        log_dir = os.path.join(constants.root_folder, "logs")
        self.logger = CustomLogger(log_dir=log_dir, logger_name="OPRO_Prompter").logger
        self.llm = LLM(number_of_workspaces=1)

        self.input_loader = InputLoader()
        self.prompt_builder: BasePromptBuilder = PromptBuilder()
        self.post_processor = PostProcessor()

        self.optimizer_prompt_template_path = os.path.join(constants.root_folder, "baselines", "opro", "data",
                                                           "mrpc",
                                                           "opro_prompt_optimizer.yaml")
        self.solver_prompt_template_path = os.path.join(constants.root_folder, "baselines", "opro", "data",
                                                        "mrpc",
                                                        "opro_prompt_solver.yaml")
        self.few_shots_path = os.path.join(constants.root_folder, "baselines", "opro", "data", "mrpc", "few_shots.yaml")
        self.optimization_trajectory_path = os.path.join(constants.root_folder, "baselines", "opro", "data",
                                                         "mrpc",
                                                         "optimization_trajectory.json")
        self.optimizer_prompt = self.input_loader.load_prompt_template(self.optimizer_prompt_template_path)[
            "system_role"]
        self.solver_prompt = self.input_loader.load_prompt_template(self.solver_prompt_template_path)["system_role"]
        self.few_shots = self.input_loader.load_few_shots(self.few_shots_path)

        if validation:
            self.validation_set = self.input_loader.load_json(constants.MRPC_VAL_DATA_PATH)
            self.data_batches = self.divide_json_into_subsets(self.validation_set, num_subsets=100)
        else:
            self.binary_train_set = self.input_loader.load_json(constants.MRPC_TRAIN_DATA_PATH)
            self.data_batches = self.divide_json_into_subsets(self.binary_train_set, 10)

    def optimize_prompt_opro_style(self):
        for i in range(0, 100):
            self.logger.info(f"Starting prompt optimization in OPRO style. Iteration {i} out of 100.")
            optimized_prompt = self.generate_optimized_prompt()
            results = self.classify_and_collect_results(optimized_prompt, 20)
            results_for_eval = []
            for _, result_list in results:
                for sentence_classification in result_list:
                    results_for_eval.append(sentence_classification)

            accuracy_score = self.evaluate_results(results_for_eval)

            optimization_trajectory = self.input_loader.load_json(self.optimization_trajectory_path)
            optimization_trajectory.append(
                {
                    "prompt": optimized_prompt,
                    "accuracy": accuracy_score
                }
            )

            optimization_trajectory = self.prune_and_sort_optimization_trajectory(optimization_trajectory)

            try:
                with open(self.optimization_trajectory_path, "w") as f:
                    json.dump(optimization_trajectory, f, indent=4)
                self.logger.info("Optimization trajectory updated successfully.")
            except Exception as e:
                self.logger.error("Failed to write optimization trajectory: %s", e)

    def validate_prompt(self, prompt):
        results = self.classify_and_collect_results(prompt, 20)
        results_for_eval = []
        for _, result_list in results:
            for sentence_classification in result_list:
                results_for_eval.append(sentence_classification)

        eval = self.evaluate_results_binary(results_for_eval)
        return eval

    def classify_and_collect_results(self, prompt, max_threads=20, index=0):
        """
        Perform classification by prompting the LLM and collect results with ground-truth labels using concurrency.

        Args:
            prompt (str): The prompt.
            max_threads (int): Maximum number of concurrent threads for processing.

        Returns:
            list: A list of tuples where each tuple contains the data batch and the batch results.
        """
        # Load previously saved intermediate results if available
        error_batches = []
        all_results = []

        rate_limiter = RateLimiter(rate_per_second=1)

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {}
            for batch_index, data_batch in enumerate(self.data_batches, 1):
                # Enforce rate limiting before submitting the task
                rate_limiter.wait()
                self.logger.info("Submitting batch %d for processing", batch_index)

                # Prepare batch for classification
                data_batch_without_labels = {
                    key: {
                        "sentence1": value["sentence1"],
                        "sentence2": value["sentence2"]
                    } for key, value in data_batch.items()}

                messages = self.create_solver_prompt(prompt, data_batch_without_labels)
                # Submit task to thread pool
                future = executor.submit(self._process_batch, messages, data_batch, index)
                futures[future] = (batch_index, data_batch)

            # Collect results from remaining futures
            for future in as_completed(futures):
                batch_index, batch_data = futures[future]
                try:
                    batch_results = future.result()
                    if not batch_results:
                        self.logger.warning("Batch %d returned an empty result, adding to error_batches", batch_index)
                        error_batches.append(batch_data)
                    else:
                        all_results.append((batch_data, batch_results))
                except Exception as e:
                    self.logger.error("Error processing batch %d: %s", batch_index, e)
                    error_batches.append(batch_data)

        # Reprocess error batches
        i = 0
        while error_batches and i < 5:
            i += 1
            self.logger.info("Reprocessing %d error batches", len(error_batches))
            new_error_batches = []
            for batch_index, data_batch in enumerate(error_batches, 1):
                try:
                    self.logger.info("Reprocessing batch %d", batch_index)
                    data_batch_without_labels = {
                        key: {
                            "sentence1": value["sentence1"],
                            "sentence2": value["sentence2"]
                        } for key, value in data_batch.items()}

                    messages = self.create_solver_prompt(prompt, data_batch_without_labels)
                    batch_results = self._process_batch(messages, data_batch, 0)

                    if batch_results:
                        all_results.append((data_batch, batch_results))
                    else:
                        self.logger.error("Reprocessing batch %d failed again, skipping", batch_index)
                        new_error_batches.append(data_batch)
                except Exception as e:
                    self.logger.error("Error reprocessing batch %d: %s", batch_index, e)

            error_batches = new_error_batches
        self.logger.info("Classification completed with %d entries", len(all_results))
        return all_results

    def _process_batch(self, messages: list, data_batch: dict, index: int):
        """
        Helper function to process a single batch.

        Args:
            messages (list): The input prompt messages.
            data_batch (dict): The batch of data being processed.

        Returns:
            list: Results for the processed batch.
        """
        response_text = self.llm.predict(messages, temperature=0.0, index=0)
        model_outputs = self.post_processor.extract_json_objects(response_text, '{', '}')

        batch_results = []
        for sentence_id, row in data_batch.items():
            for classified_sentence in model_outputs:
                for classified_sentence_id, value in classified_sentence.items():
                    if sentence_id == classified_sentence_id:
                        gold_label = row["label"]
                        model_classification = value.get("classification", [""])

                        result = {
                            "model_classification": model_classification,
                            "gold_label": gold_label,
                        }
                        batch_results.append(result)

        return batch_results

    def prune_and_sort_optimization_trajectory(self, optimization_trajectory):
        """
        Reads the optimization trajectory, orders prompts by accuracy, and keeps only the top 20.

        Args:
            optimization_trajectory (dict): dict containing the optimization trajectory.

        Returns:
            list: A list of the 20 best prompts ordered by accuracy.
        """
        # Sort the trajectory by accuracy in descending order to get highest accuracies first
        sorted_trajectory = sorted(optimization_trajectory, key=lambda x: x["accuracy"], reverse=False)

        # Keep only the top 20 prompts
        pruned_trajectory = sorted_trajectory[-20:]

        self.logger.info(f"Pruned optimization trajectory saved with {len(pruned_trajectory)} entries.")
        return pruned_trajectory

    def create_optimizer_prompt(self) -> list:
        self.logger.info("Creating optimizer prompt.")
        optimization_trajectory = self.input_loader.load_json(self.optimization_trajectory_path)
        system_prompt = str.replace(self.optimizer_prompt, "[OPTIMIZATION_TRAJECTORY]",
                                    json.dumps(optimization_trajectory))

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.logger.info("Optimizer prompt created successfully.")
        return messages

    def create_solver_prompt(self, prompt: str, data: json) -> list[dict]:
        self.logger.info("Creating solver prompt.")
        system_prompt = str.replace(self.solver_prompt, "<INS>", prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.few_shots["input"]},
            {"role": "assistant", "content": self.few_shots["output"]},
            {"role": "user", "content": json.dumps(data)}
        ]
        self.logger.info("Solver prompt created successfully.")
        return messages

    def generate_optimized_prompt(self):
        self.logger.info("Generating optimized prompt.")

        optimizer_prompt = self.create_optimizer_prompt()
        optimized_prompt = self.llm.predict(optimizer_prompt, temperature=1, index=0)
        self.logger.info("Optimized prompt generated.")
        return optimized_prompt

    def evaluate_results_binary(self, results):
        """
        Evaluate the results by comparing model predictions with the gold labels and save them to files.

        Args:
            results (list): List of dictionaries containing text, model classification, and gold label.

        Returns:
            dict: A dictionary mapping each individual evaluation metric to its corresponding name.
        """
        self.logger.info("Starting evaluation of results")

        model_predictions = [result.get("model_classification", 0) for result in results]
        gold_labels = [result.get("gold_label", 0) for result in results]

        accuracy = accuracy_score(gold_labels, model_predictions)
        report_dict = classification_report(gold_labels, model_predictions, output_dict=True)

        # Map individual metrics to their names
        evaluation_metrics = {"accuracy": accuracy}
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):  # Only process detailed metrics
                for metric_name, metric_value in metrics.items():
                    evaluation_metrics[f"{label}_{metric_name}"] = metric_value

        return evaluation_metrics

    def divide_json_into_subsets(self, input_json, num_subsets=10):
        """
        Divides a JSON object into a specified number of subsets.

        :param input_json: The JSON object to divide.
        :param num_subsets: The number of subsets to divide the JSON into.
        :return: A list of dictionaries, each containing a subset of the input JSON.
        """
        # Convert the JSON object into a list of key-value pairs
        items = list(input_json.items())

        # Calculate the chunk size for each subset
        chunk_size = max(1, (len(items) // num_subsets))

        # Divide the items into subsets
        subsets = [
            dict(items[i:i + chunk_size])
            for i in range(0, len(items), chunk_size)
        ]

        return subsets

    def evaluate_results(self, results):
        self.logger.info("Starting evaluation of results.")
        model_predictions = [result.get("model_classification", 0) for result in results]
        gold_labels = [result.get("gold_label", 0) for result in results]

        accuracy = accuracy_score(gold_labels, model_predictions)
        self.logger.info("Evaluation completed with accuracy: %f", accuracy)
        return accuracy


if __name__ == "__main__":
    # opro_prompter = OPROPrompterBinary()

    # opro_prompter.optimize_prompt_opro_style()
    opro_prompter = OPROPrompterBinary(validation=True)
    prompt = str("Immerse yourself in the nuances of language and become a master of discerning semantic essence. "
                 "Your task is to meticulously evaluate the provided sentence pairs and determine if they convey"
                 " the same fundamental information and intent. Disregard surface-level differences in phrasing, "
                 "word order, or minor details. Focus exclusively on the core factual content and purpose. "
                 "Classify the pairs as 1 ('equivalent') if they share the same crucial meaning, or 0 "
                 "('not equivalent') if they differ in any significant way. Provide a crisp rationale "
                 "highlighting the pivotal semantic similarities or discrepancies that supported your decision.")
    eval_results = opro_prompter.validate_prompt(prompt=prompt)
    opro_results_path = os.path.join(constants.root_folder, "results", "mrpc", "opro_validation_results.json")
    with open(opro_results_path, "w") as f:
        json.dump(eval_results, f)
