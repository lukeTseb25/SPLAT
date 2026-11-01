import csv
import os
import threading
import time
from collections import deque

import numpy as np
import pandas as pd

from modules.band_extractor import TrialExtractor
from modules.bp_trainer import BPTrainer
from modules.config import *
from modules.csv_stream_writer import CSVStreamWriter


DO_EXTRACTION = True
DO_TRAIN = True

CSV_FILE = os.path.join(RAW_DATA_DIR, "SaraReach3sorted_MI_EEG_20261027_185615.csv")
OUTPUT_DATA = os.path.join(PROCESSED_DATA_DIR, "extracted_SaraReach3sorted_MI_EEG_20261027_185615.csv")
MODEL_PATH = os.path.join(PROCESSED_DATA_DIR, "bp_model.joblib")

def build_header(channel_idxs=None):
    columns = ["timestamp"]
    for channel_idx in (channel_idxs or BAND_CHANNELS):
        channel_name = f"CH{channel_idx + 1}"
        columns.extend([f"{channel_name}_mu", f"{channel_name}_beta"])
    columns.append("label")
    return columns


CSV_COLUMNS = build_header(BAND_CHANNELS)


def preview_csv(path, max_rows=10):
    if not os.path.exists(path):
        print(f"CSV preview unavailable: {path} does not exist")
        return
    with open(path, newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"CSV is empty: {path}")
            return
        print(f"Header: {header}")
        for idx, row in enumerate(reader, start=1):
            print(f"Row {idx}: {row[:10]}")
            if idx >= max_rows:
                break


def build_trial_row(row):
    """CSV rows are timestamp, ch1..ch8, marker; TrialExtractor expects the marker last."""
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


def read_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError(f"Dataset {path} is missing the 'label' column")
    return df


def prepare_dataset():
    if DO_EXTRACTION:
        if not os.path.exists(CSV_FILE):
            raise FileNotFoundError(
                f"Input CSV not found: {CSV_FILE}. Put the CSV in the raw data folder or set CSV_FILE to the correct file."
            )

        preview_csv(CSV_FILE)

        inp = deque(maxlen=4096)
        stop_event = threading.Event()
        extractor = TrialExtractor(inp, stop_event, max_output_len=4096, channel_idxs=BAND_CHANNELS)
        extractor.start()

        output_queue = deque(maxlen=4096)
        writer = CSVStreamWriter(output_queue, OUTPUT_DATA, header=CSV_COLUMNS, stop_event=stop_event)
        writer.start()

        try:
            with open(CSV_FILE, newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    trial_row = build_trial_row(row)
                    if trial_row is not None:
                        inp.append(trial_row)

            while inp:
                time.sleep(0.01)

            while extractor.is_alive() or extractor.extracted_bands:
                while extractor.extracted_bands:
                    try:
                        output_queue.append(extractor.extracted_bands.popleft())
                    except Exception:
                        break
                if not extractor.is_alive() and not extractor.extracted_bands:
                    break
                time.sleep(0.01)

            stop_event.set()
            extractor.join(timeout=5.0)
            while output_queue:
                time.sleep(0.01)
            writer.join(timeout=5.0)

        finally:
            stop_event.set()
            extractor.join(timeout=5.0)
            writer.join(timeout=5.0)

        if not os.path.exists(OUTPUT_DATA):
            raise ValueError(
                "TrialExtractor produced no rows. This usually means the CSV is missing, empty, header-only, or the marker/row format does not match the expected [timestamp, ch1..ch8, marker] layout."
            )

        df = pd.read_csv(OUTPUT_DATA)
        if df.empty:
            raise ValueError(f"Dataset {OUTPUT_DATA} is empty")
        print(f"Wrote dataset to {OUTPUT_DATA}")
        return df

    df = read_dataset(OUTPUT_DATA)
    if df.empty:
        raise ValueError(f"Dataset {OUTPUT_DATA} is empty")
    print(f"Loaded dataset from {OUTPUT_DATA}")
    return df


def main():
    df = prepare_dataset()

    if not DO_TRAIN:
        return

    label_col = 'label'
    feature_cols = [col for col in df.columns if col not in {'timestamp', label_col}]
    X = df[feature_cols]
    y = df[label_col].astype(int).to_numpy().reshape(-1)
    if y.ndim != 1:
        y = y.ravel()
    allowed = {0, 1, 2, 3}
    bad = sorted({int(v) for v in np.unique(y) if int(v) not in allowed})
    if bad:
        raise ValueError(f"Training labels must be one integer label per trial in {{0,1,2,3}}; found invalid labels {bad}")

    trainer = BPTrainer(MODEL_PATH)
    for features, label in zip(X.to_numpy().tolist(), y.tolist()):
        trainer.add_data(features, int(label))

    trainer.train()
    trainer.save_model()
    print(f"Trained model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()