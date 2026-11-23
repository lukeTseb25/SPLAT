#Train a KGBoost on the data streamed from a deque
#Define a class that trains a KGBoost model on the streamed data and writes the model to disk

import os
import joblib
from collections import deque
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class BPTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.features = deque()
        self.labels = deque()
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def add_data(self, features, labels):
        self.features.append(np.asarray(features, dtype=float).ravel())
        self.labels.append(int(labels))

    def train(self, test_size=0.2, random_state=42):
        if not self.features or not self.labels:
            raise ValueError("No data available for training")

        X = np.asarray(list(self.features), dtype=float)
        y = np.asarray(list(self.labels), dtype=int).reshape(-1)

        allowed = np.array([0, 1, 2, 3], dtype=int)
        unique = np.unique(y)
        bad = unique[~np.isin(unique, allowed)]
        if len(bad) > 0:
            raise ValueError(f"Label values must all be in {{0,1,2,3}}; got {sorted(unique.tolist())} and invalid labels {sorted(bad.tolist())}")

        if test_size:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = X, None, y, None

        # Print the number of samples for each label in the training set and the test set (if applicable)
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print(f"Training set label distribution: {dict(zip(unique_train.tolist(), counts_train.tolist()))}")
        if X_test is not None and y_test is not None:
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            print(f"Test set label distribution: {dict(zip(unique_test.tolist(), counts_test.tolist()))}")

        # The model must be configured for 4-class training; if a subset of 0..3 is missing,
        # we fail early instead of silencing the issue and letting XGBoost infer a binary model.
        self.model = XGBClassifier(
            objective="multi:softmax",
            num_class=unique.size,
            eval_metric="mlogloss",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_trained = True

    def evaluate(self, X_test=None, y_test=None):
        if not self.is_trained or self.model is None:
            raise ValueError("The model must be trained before evaluation")

        if X_test is None or y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        if X_test is None or y_test is None:
            raise ValueError("No test data is available for evaluation")

        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=int).reshape(-1)
        y_pred = self.model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))

    def save_model(self):
        if self.model is None:
            raise ValueError("No trained model to save")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)