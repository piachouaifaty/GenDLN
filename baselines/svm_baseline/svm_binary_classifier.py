import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class SvmBinaryClassifier:
    def load_train_data(self, train_dataset_path):
        train_data = pd.read_csv(train_dataset_path, sep='\t')
        # Extract features and labels
        X_train = train_data['text']
        Y_train = train_data['label']
        return X_train, Y_train


    def load_test_data(self, test_dataset_path):
        # Load the TSV file
        test_data = pd.read_csv(test_dataset_path, sep='\t')
        X_test = test_data['text']
        Y_test = test_data['label']
        return X_test, Y_test


    def apply_tf_idf_to_sentences(self, X_train, X_test):
        tfidf_vectorizer = TfidfVectorizer()
        # Fit on the training data and transform both training and test data
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf


    def train_model(self, X_train_tfidf, Y_train):
        svm_model = SVC(kernel='linear', verbose=True)
        # Train the model on the TF-IDF transformed data
        svm_model.fit(X_train_tfidf, Y_train)
        return svm_model


    def evaluate_model_on_test_set(self, X_test_tfidf, Y_test, model):
        # Predict on the test set
        y_pred = model.predict(X_test_tfidf)
        # Print accuracy and classification report
        print("Accuracy:", accuracy_score(Y_test, y_pred))
        print("Classification Report:\n", classification_report(Y_test, y_pred))

    def classify_sentences_binary(self):
        train_dataset_path = os.path.join("../../dataset", "claudette", "claudette_train_merged.tsv")
        test_dataset_path = os.path.join("../../dataset", "claudette", "claudette_test_merged.tsv")

        X_train, Y_train = self.load_train_data(train_dataset_path)
        X_test, Y_test = self.load_test_data(test_dataset_path)

        # Apply TF-IDF to both train and test sets
        X_train_tfidf, X_test_tfidf = self.apply_tf_idf_to_sentences(X_train, X_test)

        # Train the model
        trained_svm_model = self.train_model(X_train_tfidf, Y_train)

        # Evaluate the model
        self.evaluate_model_on_test_set(X_test_tfidf, Y_test, trained_svm_model)


if __name__ == "__main__":
    svm_binary_classifier = SvmBinaryClassifier()
    svm_binary_classifier.classify_sentences_binary()
