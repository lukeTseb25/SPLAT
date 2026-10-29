#Train a KGBoost on the data streamed from a deque
#Define a class that trains a KGBoost model on the streamed data and writes the model to disk

import os
import joblib
from collections import deque
import numpy as np
from xgboost import XGBClassifier


class BPTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.features = deque()
        self.labels = deque()

    def add_data(self, features, labels):
        self.features.append(np.asarray(features, dtype=float).ravel())
        self.labels.append(int(labels))

    def train(self):
        if not self.features or not self.labels:
            raise ValueError("No data available for training")

        X = np.asarray(list(self.features), dtype=float)
        y = np.asarray(list(self.labels), dtype=int).reshape(-1)
        if y.ndim != 1:
            y = y.ravel()
        unique = np.unique(y)
        allowed = np.array([0, 1, 2, 3], dtype=int)
        bad = unique[~np.isin(unique, allowed)]
        if len(bad) > 0:
            raise ValueError(f"Label values must all be in {{0,1,2,3}}; got {sorted(unique.tolist())} and invalid labels {sorted(bad.tolist())}")

        num_classes = int(np.max(unique)) + 1 if len(unique) else 1
        self.model = XGBClassifier(
            objective="multi:softmax",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        self.model.fit(X, y)

    def save_model(self):
        if self.model is None:
            raise ValueError("No trained model to save")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)