import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import accuracy_score, classification_report

from genetic_dln.src.constants import constants
from genetic_dln.src.custom_logger.custom_logger import CustomLogger
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.models.llm import LLM
from genetic_dln.src.models.rate_limiter import RateLimiter
from genetic_dln.src.post_processor.post_processor import PostProcessor
from genetic_dln.src.prompt_builder.base_prompt_builder import BasePromptBuilder
from genetic_dln.src.prompt_builder.prompt_builder import PromptBuilder
from genetic_dln.src.task.task import Task


class GenDLN:
    def __init__(
            self,
            multi_label: bool = False,
            number_of_workspaces: int = 8,
            task: Task = None,
            validation: bool = False
    ) -> None:
        self.task = task
        if not self.task:
            raise ValueError("Task must not be none!")

        self.llm = LLM(number_of_workspaces=number_of_workspaces)
        self.input_loader = InputLoader()
        self.prompt_builder: BasePromptBuilder = PromptBuilder()
        self.post_processor = PostProcessor()
        log_dir = os.path.join(constants.root_folder, "logs")
        self.logger = CustomLogger(log_dir=log_dir, logger_name="GenDLN").logger
        self.multi_label = multi_label

        self.prompt_01_template_path = self.task.layer_1_system_prompt_path
        self.prompt_02_template_path = self.task.layer_2_system_prompt_path
        self.prompt_02_few_shots_path = self.task.layer_2_few_shots_path
        self.train_dataset_path = self.task.train_dataset_path
        self.val_dataset_path = self.task.val_dataset_path

        self.prompt_01_template = self.input_loader.load_prompt_template(self.prompt_01_template_path)
        self.prompt_02_template = self.input_loader.load_prompt_template(self.prompt_02_template_path)
        self.prompt_02_few_shots = self.input_loader.load_few_shots(self.prompt_02_few_shots_path)
        if validation:
            self.validation_set = self.input_loader.load_json(self.val_dataset_path)
            self.data_batches = self.divide_json_into_subsets(self.validation_set, num_subsets=60)
        else:
            self.train_dataset = self.input_loader.load_json(self.train_dataset_path)
            self.data_batches = self.divide_json_into_subsets(self.train_dataset, num_subsets=10)

    def predict(self, prompt_01: str, prompt_02: str, index: int) -> dict:
        layer_1_outputs = self.classify_and_collect_results_first_layer(prompt_01, max_threads=50, index=index)
        results = self.classify_and_collect_results_second_layer(prompt_02, layer_1_outputs, max_threads=50, index=index)
        results_for_eval = []
        for _, result_list in results:
            for sentence_classification in result_list:
                results_for_eval.append(sentence_classification)

        if self.multi_label:
            score_dict = self.evaluate_results_multi_label(results_for_eval)
        else:
            score_dict = self.evaluate_results_binary(results_for_eval)
        return score_dict

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

    def evaluate_results_multi_label(self, results):
        """
        Evaluate multi-label classification results, including per-class metrics, macro-average, and weighted-average.

        Args:
            results (list): A list of dictionaries containing model predictions and gold labels.

        Returns:
            dict: A dictionary with overall accuracy, per-class metrics, macro-average, and weighted-average.
        """
        self.logger.info("Starting evaluation of multi-label classification results.")

        # Extract predictions and gold labels
        model_predictions = [set(result.get("model_classification", [])) for result in results]
        gold_labels = [set(result.get("gold_label", [])) for result in results]

        # Compute overall accuracy
        accuracy = self._calculate_accuracy(model_predictions, gold_labels)

        # Compute per-class metrics
        class_metrics = self._calculate_per_class_metrics(model_predictions, gold_labels)

        # Compute macro-average and weighted-average metrics
        macro_avg, weighted_avg = self._calculate_average_metrics(class_metrics, len(results))

        # Log results
        self.logger.info("Evaluation completed.")
        self.logger.info("Accuracy: %.4f", accuracy)
        self.logger.info("Macro-Average Precision: %.4f, Recall: %.4f, F1-Score: %.4f",
                         macro_avg["precision"], macro_avg["recall"], macro_avg["f1_score"])
        self.logger.info("Weighted-Average Precision: %.4f, Recall: %.4f, F1-Score: %.4f",
                         weighted_avg["precision"], weighted_avg["recall"], weighted_avg["f1_score"])

        return {
            "accuracy": accuracy,
            "class_metrics": class_metrics,
            "macro_avg": macro_avg,
            "weighted_avg": weighted_avg
        }

    def _calculate_accuracy(self, model_predictions, gold_labels):
        """
        Calculate overall accuracy: a prediction is correct if all predicted labels for a sample
        match exactly the gold labels for that sample.

        Args:
            model_predictions (list of set): Predicted label sets for each sample.
            gold_labels (list of set): Gold label sets for each sample.

        Returns:
            float: Accuracy score.
        """
        correct_predictions = [
            1 if predictions == gold else 0
            for predictions, gold in zip(model_predictions, gold_labels)
        ]
        accuracy = sum(correct_predictions) / len(correct_predictions)
        return accuracy

    def _calculate_per_class_metrics(self, model_predictions, gold_labels):
        """
        Calculate precision, recall, and F1-score for each class based on exact matches.

        Args:
            model_predictions (list of set): Predicted label sets for each sample.
            gold_labels (list of set): Gold label sets for each sample.

        Returns:
            dict: Per-class metrics (precision, recall, F1-score, support).
        """
        # Initialize containers for per-class metrics
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        # Calculate TP, FP, FN for each class
        for predictions, gold in zip(model_predictions, gold_labels):
            for cls in predictions:
                if cls in gold:
                    true_positives[cls] += 1
                else:
                    false_positives[cls] += 1
            for cls in gold:
                if cls not in predictions:
                    false_negatives[cls] += 1

        # Calculate precision, recall, and F1-score for each class
        class_metrics = {}
        for cls in set(true_positives.keys()).union(false_positives.keys()).union(false_negatives.keys()):
            tp = true_positives[cls]
            fp = false_positives[cls]
            fn = false_negatives[cls]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics[cls] = {"precision": precision, "recall": recall, "f1_score": f1, "support": tp + fn}

        return class_metrics

    def _calculate_average_metrics(self, class_metrics, total_samples):
        """
        Calculate macro-average and weighted-average metrics for precision, recall, and F1-score.

        Args:
            class_metrics (dict): Per-class metrics including precision, recall, and F1-score.
            total_samples (int): Total number of samples in the dataset.

        Returns:
            tuple: (macro_avg, weighted_avg), each being a dictionary with precision, recall, and F1-score.
        """
        # Initialize accumulators
        macro_precision, macro_recall, macro_f1 = 0, 0, 0
        weighted_precision, weighted_recall, weighted_f1 = 0, 0, 0

        # Accumulate metrics for each class
        total_support = sum(metrics["support"] for metrics in class_metrics.values())
        for cls, metrics in class_metrics.items():
            support = metrics["support"]
            macro_precision += metrics["precision"]
            macro_recall += metrics["recall"]
            macro_f1 += metrics["f1_score"]

            weighted_precision += metrics["precision"] * support
            weighted_recall += metrics["recall"] * support
            weighted_f1 += metrics["f1_score"] * support

        num_classes = len(class_metrics)
        macro_avg = {
            "precision": macro_precision / num_classes,
            "recall": macro_recall / num_classes,
            "f1_score": macro_f1 / num_classes
        }
        weighted_avg = {
            "precision": weighted_precision / total_support,
            "recall": weighted_recall / total_support,
            "f1_score": weighted_f1 / total_support
        }

        return macro_avg, weighted_avg


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
        chunk_size = max(1, (len(items) // num_subsets) + 1)

        # Divide the items into subsets
        subsets = [
            dict(items[i:i + chunk_size])
            for i in range(0, len(items), chunk_size)
        ]

        return subsets

    def classify_and_collect_results_first_layer(self, prompt_01, max_threads=20, index=0):
        """
        Perform classification by prompting the LLM and collect results with ground-truth labels using concurrency.

        Args:
            prompt_01 (str): The prompt for the first layer.
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
                data_batch_without_labels = {key: value["text"] for key, value in data_batch.items()}
                full_prompt_01 = str.replace(self.prompt_01_template["system_role"], "<Prompt_01_Placeholder>",
                                             prompt_01)
                messages = self.prompt_builder.build_prompt_layer_1(full_prompt_01, data_batch_without_labels)

                # Submit task to thread pool
                future = executor.submit(self._process_batch_layer_1, messages, index)
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
                    data_batch_without_labels = {key: value["text"] for key, value in data_batch.items()}
                    messages = self.prompt_builder.build_prompt_layer_1(prompt_01, data_batch_without_labels)
                    batch_results = self._process_batch_layer_1(messages, index)

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

    def classify_and_collect_results_second_layer(self, prompt_02, layer_1_results, max_threads=20, index=0):
        """
        Perform classification by prompting the LLM and collect results with ground-truth labels using concurrency.

        Args:
            prompt_02 (str): The prompt for the second layer.
            layer_1_results (list): A list containing tuples of format (data_batch, layer_1_output).
            max_threads (int): Maximum number of concurrent threads for processing.

        Returns:
            list: A list of dictionaries containing the original text, model's classification, and gold label.
        """
        # Load previously saved intermediate results if available
        error_batches = []
        all_results = []
        rate_limiter = RateLimiter(rate_per_second=1)  # 3 request per second

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {}
            for batch_index, (data_batch, layer_1_output) in enumerate(layer_1_results, 1):
                # Enforce rate limiting before submitting the task
                rate_limiter.wait()
                self.logger.info("Submitting batch %d for processing", batch_index)

                # Prepare batch for classification
                data_batch_without_labels = {key: value["text"] for key, value in data_batch.items()}
                full_prompt_02 = str.replace(self.prompt_02_template["system_role"], "<Prompt_02_Placeholder>",
                                             prompt_02)
                messages = self.prompt_builder.build_prompt_layer_2(
                    prompt=full_prompt_02,
                    few_shots=self.prompt_02_few_shots,
                    layer_1_output=layer_1_output,
                    sentences_to_classify=data_batch_without_labels
                )

                # Submit task to thread pool
                future = executor.submit(self._process_batch, messages, data_batch, index)
                futures[future] = (batch_index, data_batch, layer_1_output)

            # Collect results from remaining futures
            for future in as_completed(futures):
                batch_index, batch_data, layer_1_output = futures[future]
                try:
                    batch_results = future.result()
                    if not batch_results:
                        self.logger.warning("Batch %d returned an empty result, adding to error_batches", batch_index)
                        error_batches.append((batch_data, layer_1_output))
                    else:
                        all_results.append((batch_data, batch_results))
                except Exception as e:
                    self.logger.error("Error processing batch %d: %s", batch_index, e)
                    error_batches.append((batch_data, layer_1_output))

        # Reprocess error batches
        i = 0
        while error_batches and i < 5:
            i += 1
            self.logger.info("Reprocessing %d error batches", len(error_batches))
            new_error_batches = []
            for batch_index, (data_batch, layer_1_output) in enumerate(error_batches, 1):
                try:
                    self.logger.info("Reprocessing batch %d", batch_index)
                    data_batch_without_labels = {key: value["text"] for key, value in data_batch.items()}
                    full_prompt_02 = str.replace(self.prompt_02_template["system_role"], "<Prompt_02_Placeholder>",
                                                 prompt_02)
                    messages = self.prompt_builder.build_prompt_layer_2(
                        prompt=full_prompt_02,
                        few_shots=self.prompt_02_few_shots,
                        layer_1_output=layer_1_output,
                        sentences_to_classify=data_batch_without_labels
                    )

                    batch_results = self._process_batch(messages, data_batch, index)

                    if batch_results:
                        all_results.append((data_batch, batch_results))
                    else:
                        self.logger.error("Reprocessing batch %d failed again, skipping", batch_index)
                        new_error_batches.append((data_batch, layer_1_output))
                except Exception as e:
                    self.logger.error("Error reprocessing batch %d: %s", batch_index, e)

            error_batches = new_error_batches
        self.logger.info("Classification completed with %d entries", len(all_results))
        return all_results

    def _process_batch_layer_1(self, messages: list, index: int):
        response_text = self.llm.predict(messages, temperature=0.0, index=index)
        return response_text

    def _process_batch(self, messages: list, data_batch: dict, index: int):
        """
        Helper function to process a single batch.

        Args:
            messages (list): The input prompt messages.
            data_batch (dict): The batch of data being processed.

        Returns:
            list: Results for the processed batch.
        """
        response_text = self.llm.predict(messages, temperature=0.0, index=index)
        model_outputs = self.post_processor.extract_json_objects(response_text, '{', '}')

        batch_results = []
        for sentence_id, row in data_batch.items():
            for classified_sentence in model_outputs:
                for classified_sentence_id, value in classified_sentence.items():
                    if sentence_id == classified_sentence_id:
                        original_text = row["text"]

                        gold_label = row["classes"]
                        if not self.multi_label:
                            gold_label = row["label"]

                        model_classification = value.get("classification", [""])

                        if not self.multi_label:
                            model_classification = 0 if value.get("classification", "") == "fair" else 1

                        result = {
                            "text": original_text,
                            "model_classification": model_classification,
                            "gold_label": gold_label,
                        }
                        batch_results.append(result)

        return batch_results
