import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class SvmBinaryClassifierMRPC:
    def load_dataset(self, json_path):
        # Load JSON and convert to DataFrame
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        # Convert to DataFrame
        data = pd.DataFrame.from_dict(data_dict, orient='index')

        # Combine sentence1 and sentence2 into one string for TF-IDF
        combined_sentences = data['sentence1'] + " [SEP] " + data['sentence2']
        labels = data['label']
        return combined_sentences, labels

    def apply_tf_idf_to_sentences(self, X_train, X_test):
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def train_model(self, X_train_tfidf, Y_train):
        svm_model = SVC(kernel='linear', verbose=True)
        svm_model.fit(X_train_tfidf, Y_train)
        return svm_model

    def evaluate_model_on_test_set(self, X_test_tfidf, Y_test, model):
        y_pred = model.predict(X_test_tfidf)
        print("Accuracy:", accuracy_score(Y_test, y_pred))
        print("Classification Report:\n", classification_report(Y_test, y_pred))

    def classify_sentences_binary(self):
        train_path = os.path.join("../../dataset", "mrpc_full_baselines", "mrpc_train_70.json")
        test_path = os.path.join("../../dataset", "mrpc_full_baselines", "mrpc_test_10.json")

        X_train, Y_train = self.load_dataset(train_path)
        X_test, Y_test = self.load_dataset(test_path)

        X_train_tfidf, X_test_tfidf = self.apply_tf_idf_to_sentences(X_train, X_test)
        trained_model = self.train_model(X_train_tfidf, Y_train)
        self.evaluate_model_on_test_set(X_test_tfidf, Y_test, trained_model)


if __name__ == "__main__":
    classifier = SvmBinaryClassifierMRPC()
    classifier.classify_sentences_binary()
