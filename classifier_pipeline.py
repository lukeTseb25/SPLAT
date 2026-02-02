import numpy as np
import pandas as pd
import scipy.signal
import os
import math
import sys
import threading
import time
from collections import deque
import pickle

# Parameters
FS = 250
NUM_CHANNELS = 8

FREQ_LOW = 8
FREQ_HIGH = 30
FFT_WINDOW_SIZE_SEC = 0.25
FFT_WINDOW_SIZE_SAMPLES = int(FFT_WINDOW_SIZE_SEC * FS)
FFT_STEP_SIZE_SEC = 1.0/FS
FFT_STEP_SIZE_SAMPLES = int(FFT_STEP_SIZE_SEC * FS)
NFFT = 256
CLASSIFICATION_WINDOW_SEC = 0.5
CLASSIFICATION_WINDOW_SAMPLES = int(CLASSIFICATION_WINDOW_SEC * FS)
CLASSIFICATION_WINDOW_STEP_SEC = 1.0/FS
# CLASSIFICATION_WINDOW_STEP_SEC = 1.0
CLASSIFICATION_WINDOW_STEP_SAMPLES = int(CLASSIFICATION_WINDOW_STEP_SEC * FS)
CLASSIFICATION_MARKER_TIME_THRESHOLD_SEC = 0.5
CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES = int(CLASSIFICATION_MARKER_TIME_THRESHOLD_SEC * FS)
CLASSIFICATION_FILTERING_PADDING_SEC = 0.2
CLASSIFICATION_FILTERING_PADDING_SAMPLES = int(CLASSIFICATION_FILTERING_PADDING_SEC * FS)

def read_eeg_csv(filepath, times, eeg_data, markers):
    """
    Reads CSV where:
      - first column = time (s)
      - next NUM_CHANNELS columns = EEG channels
      - last column = marker (1–4)
    """
    df = pd.read_csv(filepath)
    times_list = df.iloc[:, 0].values
    eeg_data_list = df.iloc[:, 1:NUM_CHANNELS+1].values
    eeg_data_list = eeg_data_list[~np.isnan(eeg_data_list).any(axis=1)]
    markers_list = [int(num) if num is not None and not math.isnan(num) else 0 for num in df.iloc[:, -1]]

    #Convert to deques for thread-safe appending/popping
    times.extend(times_list)
    eeg_data.extend(eeg_data_list)
    markers.extend(markers_list)

    return times, eeg_data, markers

def main():
    os.makedirs("./data/raw", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)

    filename = "MI_EEG_20251005_171205_Session1LS.csv"

    if len(sys.argv) < 2:
        print("This script expects a filename.")
        #sys.exit(1)
    else:
        filename = sys.argv[1]

    filepath = os.path.abspath(os.path.join("data", "raw", filename))
    times, eeg_data, markers = deque(), deque(), deque()

    read_thread = threading.Thread(target=read_eeg_csv, args=(filepath, times, eeg_data, markers))
    read_thread.start()

    preproccess_stop_event = threading.Event()
    preprocess_return_values = list()
    preprocess_thread = threading.Thread(target=preprocess_data, args=(preproccess_stop_event, times, eeg_data, markers, preprocess_return_values))
    preprocess_thread.start()

    #TODO: Add saving thread

    #read_eeg_csv(filepath, times, eeg_data, markers)

    time.sleep(10)
    
    preproccess_stop_event.set()
    preprocess_thread.join()
    
    trials = preprocess_return_values[0]
    freqs = preprocess_return_values[1]

    # Save trials to a pickle file
    output_filepath = os.path.abspath(os.path.join("data", "processed", f"processed_{filename}.pkl"))
    with open(output_filepath, 'wb') as f:
        pickle.dump({
            'trials': trials,
            'freqs': freqs
        }, f)