import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from baselines.opro import constants
from baselines.opro.custom_logger import CustomLogger


class SvmMultiLabelClassifier:
    def __init__(self):
        log_dir = os.path.join(constants.root_folder, "logs")
        self.logger = CustomLogger(log_dir=log_dir, logger_name="OPRO_SvmMultiLabelClassifier").logger

    def load_data_from_json(self, dataset_path):
        """
        Load the dataset from a JSON file.

        Args:
            dataset_path (str): Path to the JSON dataset.

        Returns:
            tuple: A tuple (X, Y) where X is a list of sentences, and Y is a list of corresponding label sets.
        """
        self.logger.info(f"Loading dataset from {dataset_path}.")
        with open(dataset_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        X, Y = [], []
        for sentence_id, content in data.items():
            X.append(content["text"])
            Y.append(content["classes"])

        self.logger.info(f"Loaded {len(X)} sentences and labels.")
        return X, Y

    def apply_tf_idf_to_sentences(self, X_train, X_test):
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def train_model(self, X_train_tfidf, Y_train_binarized):
        """
        Train an SVM model for multi-label classification using OneVsRestClassifier.

        Args:
            X_train_tfidf: TF-IDF transformed training data.
            Y_train_binarized: MultiLabelBinarizer-transformed labels.

        Returns:
            Trained OneVsRestClassifier model.
        """
        self.logger.info("Training SVM model with OneVsRestClassifier for multi-label classification.")

        # Wrap SVC in OneVsRestClassifier
        ovr_svm_model = OneVsRestClassifier(SVC(kernel='linear', verbose=True, probability=True))
        ovr_svm_model.fit(X_train_tfidf, Y_train_binarized)

        self.logger.info("Model training completed.")
        return ovr_svm_model

    def evaluate_model_on_test_set(self, X_test_tfidf, Y_test_binarized, model, mlb):
        """
        Evaluate the model's performance on the test set and save results to files.

        Args:
            X_test_tfidf: TF-IDF transformed test data.
            Y_test_binarized: MultiLabelBinarizer-transformed test labels.
            model: Trained OneVsRestClassifier model.
            mlb: MultiLabelBinarizer instance.

        Saves:
            Accuracy, classification report, and macro metrics to files in the results folder.
        """
        self.logger.info("Evaluating model on the test set.")

        # Predict probabilities for each class
        y_pred_prob = model.predict_proba(X_test_tfidf)

        # Convert probabilities to binary predictions
        y_pred_binarized = (y_pred_prob > 0.5).astype(int)

        # Convert predictions back to label lists for readability
        y_pred = mlb.inverse_transform(y_pred_binarized)
        Y_test = mlb.inverse_transform(Y_test_binarized)

        # Compute metrics
        accuracy = accuracy_score(Y_test_binarized, y_pred_binarized)
        classification_rep = classification_report(Y_test_binarized, y_pred_binarized, target_names=mlb.classes_)
        precision, recall, f1, _ = precision_recall_fscore_support(Y_test_binarized, y_pred_binarized, average="macro")

        self.logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, Macro F1-Score: {f1:.4f}")

        # Save results to files
        results_dir = os.path.join(constants.root_folder, "results")
        os.makedirs(results_dir, exist_ok=True)

        accuracy_path = os.path.join(results_dir, "accuracy.txt")
        report_path = os.path.join(results_dir, "classification_report.txt")
        metrics_path = os.path.join(results_dir, "macro_metrics.txt")

        with open(accuracy_path, "w") as acc_file:
            acc_file.write(f"Accuracy: {accuracy:.4f}\n")

        with open(report_path, "w") as report_file:
            report_file.write("Classification Report:\n")
            report_file.write(classification_rep)

        with open(metrics_path, "w") as metrics_file:
            metrics_file.write(f"Macro Precision: {precision:.4f}\n")
            metrics_file.write(f"Macro Recall: {recall:.4f}\n")
            metrics_file.write(f"Macro F1-Score: {f1:.4f}\n")

        self.logger.info(f"Saved accuracy to {accuracy_path}")
        self.logger.info(f"Saved classification report to {report_path}")
        self.logger.info(f"Saved macro metrics to {metrics_path}")

    def classify_sentences_multi_label(self):
        train_dataset_path = os.path.join(constants.root_folder, "dataset", "claudette", "unfair_sentences_train.json")
        test_dataset_path = os.path.join(constants.root_folder, "dataset", "claudette", "unfair_sentences_test.json")

        # Load train and test data
        X_train, Y_train = self.load_data_from_json(train_dataset_path)
        X_test, Y_test = self.load_data_from_json(test_dataset_path)

        # Binarize the labels
        mlb = MultiLabelBinarizer()
        Y_train_binarized = mlb.fit_transform(Y_train)
        Y_test_binarized = mlb.transform(Y_test)

        # Apply TF-IDF to both train and test sets
        X_train_tfidf, X_test_tfidf = self.apply_tf_idf_to_sentences(X_train, X_test)

        # Train the model
        trained_svm_model = self.train_model(X_train_tfidf, Y_train_binarized)

        # Evaluate the model
        self.evaluate_model_on_test_set(X_test_tfidf, Y_test_binarized, trained_svm_model, mlb)


if __name__ == "__main__":
    classifier = SvmMultiLabelClassifier()
    classifier.classify_sentences_multi_label()
