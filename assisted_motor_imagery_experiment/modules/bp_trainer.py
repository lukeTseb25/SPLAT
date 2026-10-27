#Train a KGBoost on the data streamed from a deque
#Define a class that trains a KGBoost model on the streamed data and writes the model to disk

import os
import joblib
from collections import deque
import threading
import numpy as np
from xgboost import XGBClassifier

class BPTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = XGBClassifier()
        self.features = deque()
        self.labels = deque()

    def add_data(self, features, labels):
        self.features.append(features)
        self.labels.append(labels)

    def train(self):
        if not self.features or not self.labels:
            raise ValueError("No data available for training")

        X = np.array(self.features)
        y = np.array(self.labels)
        self.model.fit(X, y)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)