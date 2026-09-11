import csv
import time
from collections import deque
import threading
import os
import sys
import numpy as np
from modules.band_extractor import BandExtractor
from modules.config import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


CSV_FILE = os.path.join(RAW_DATA_DIR, "sorted_MI_EEG_20260503_PranatiLast.csv")
OUTPUT_FEATURES = os.path.join("data", "processed", "output_sorted_MI_EEG_20260503_PranatiLast.csv")
OUTPUT_LABELS = os.path.join("data", "processed", "labels_sorted_MI_EEG_20260503_PranatiLast.csv")


def marker_to_label(marker_str):
    if marker_str is None:
        return None
    m = str(marker_str).strip()
    if m == "2":
        return 0
    if m == "1":
        return 1
    if m == "3":
        return 2
    return None


def main():
    inp = deque()
    stop_event = threading.Event()

    extractor = BandExtractor(inp, stop_event, max_output_len=100000)
    extractor.start()

    # feed CSV rows (skip header)
    with open(CSV_FILE, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            # Each row already in format: timestamp, ch1..ch8, marker
            # Convert numeric fields where possible
            try:
                ts = float(row[0])
            except Exception:
                ts = None
            # keep channel values as floats or empty
            channels = []
            for i in range(1, 1 + 8):
                try:
                    channels.append(float(row[i]))
                except Exception:
                    channels.append(0.0)
            marker = row[1 + 8] if len(row) > 1 + 8 else ""
            # push as list matching expected parser: [ts, ch1..ch8, marker]
            inp.append([ts] + channels + [marker])

    # signal no more incoming data
    # allow extractor to finish processing buffered items
    while inp:
        time.sleep(0.01)

    stop_event.set()
    extractor.join(timeout=5.0)

    # collect extracted bands and write features/labels
    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    with open(OUTPUT_FEATURES, 'w', newline='') as fx, open(OUTPUT_LABELS, 'w', newline='') as fy:
        writer_fx = csv.writer(fx)
        writer_fy = csv.writer(fy)
        for entry in extractor.extracted_bands:
            marker = entry.get('marker', '')
            label = marker_to_label(marker)
            if label is None:
                continue
            mu = entry.get('mu', [])
            beta = entry.get('beta', [])
            row = list(mu) + list(beta)
            writer_fx.writerow(row)
            writer_fy.writerow([label])

    print(f"Wrote features to {OUTPUT_FEATURES} and labels to {OUTPUT_LABELS}")

    # Run classifier module and capture results for printing/figures
    import bp_classifier
    try:
        pipeline, results = bp_classifier.main()
    except Exception as e:
        print("bp_classifier.main() raised an exception:", e)
        return

    # results: (X_test, y_test, y_pred)
    if results is None:
        return
    X_test, y_test, y_pred = results

    print("\n=== Classification Test Summary ===")
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # Save confusion matrix figure for quick comparison
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.colormaps.get_cmap('Blues'))
    plt.colorbar()
    tick_marks = range(len(np.unique(y_test)))
    plt.xticks(tick_marks, [str(x) for x in tick_marks])
    plt.yticks(tick_marks, [str(x) for x in tick_marks])
    plt.title('Confusion Matrix (classification_test)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    out_fig = os.path.join('confusion_matrix_classification_test.png')
    plt.savefig(out_fig, dpi=300)
    print(f"Saved confusion matrix to {out_fig}")


if __name__ == '__main__':
    main()
