import csv
import time
from collections import deque
import threading
import os

from modules.band_extractor import TrialExtractor
from modules.bp_trainer import BPTrainer
from modules.config import *


CSV_FILE = os.path.join(RAW_DATA_DIR, "SaraReach3sorted_MI_EEG_20261027_185615.csv")
OUTPUT_FEATURES = os.path.join("data", "processed", "ct_SaraReach3sorted_MI_EEG_20261027_185615.csv")
OUTPUT_LABELS = os.path.join("data", "processed", "ct_SaraReach3sorted_MI_EEG_20261027_185615.csv")
MODEL_PATH = os.path.join("data", "processed", "bp_model.joblib")


def build_trial_row(row):
    """CSV rows are timestamp, ch1..ch8, marker; TrialExtractor expects marker last."""
    try:
        ts = float(row[0])
    except (TypeError, ValueError, IndexError):
        return None

    channels = []
    for idx in range(1, 1 + NUM_CHANNELS):
        try:
            channels.append(float(row[idx]))
        except (TypeError, ValueError, IndexError):
            channels.append(0.0)

    marker = row[1 + NUM_CHANNELS] if len(row) > 1 + NUM_CHANNELS else ""
    return [ts] + channels + [marker]


def main():
    inp = deque()
    stop_event = threading.Event()

    extractor = TrialExtractor(inp, stop_event, max_output_len=None)
    extractor.start()

    with open(CSV_FILE, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            trial_row = build_trial_row(row)
            if trial_row is not None:
                inp.append(trial_row)

    while inp:
        time.sleep(0.01)

    stop_event.set()
    extractor.join(timeout=5.0)

    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    with open(OUTPUT_FEATURES, 'w', newline='') as fx, open(OUTPUT_LABELS, 'w', newline='') as fy:
        writer_fx = csv.writer(fx)
        writer_fy = csv.writer(fy)
        for entry in extractor.extracted_bands:
            label = entry.get('label')
            if label is None:
                continue
            mu = entry.get('mu', [])
            beta = entry.get('beta', [])
            writer_fx.writerow(list(mu) + list(beta))
            writer_fy.writerow([label])

    print(f"Wrote features to {OUTPUT_FEATURES} and labels to {OUTPUT_LABELS}")

    trainer = BPTrainer(MODEL_PATH)
    features = []
    labels = []
    with open(OUTPUT_FEATURES, newline='') as fx, open(OUTPUT_LABELS, newline='') as fy:
        feature_reader = csv.reader(fx)
        label_reader = csv.reader(fy)
        for feature_row, label_row in zip(feature_reader, label_reader):
            if not feature_row or not label_row:
                continue
            try:
                features.append([float(value) for value in feature_row])
                labels.append(int(label_row[0]))
            except ValueError:
                continue

    if not features:
        raise ValueError("No extracted features were found for training")

    for feature_row, label in zip(features, labels):
        trainer.add_data(feature_row, label)

    trainer.train()
    trainer.save_model()
    print(f"Trained model saved to {MODEL_PATH}")
