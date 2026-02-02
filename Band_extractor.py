import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import os
import math
import sklearn
import csv

# Event markers
MARKER_RIGHT = 1.0  # right arm imagery
MARKER_LEFT = 2.0   # left arm imagery
MARKER_LEG = 3.0   # leg imagery
MARKER_STOP = 4.0   # end of imagery period
MARKER_FAIL = 5.0   # failed to add marker

# Epoching parameters
EPOCH_START = -2.0
EPOCH_END = 4.0
BASELINE_START = -2.0
BASELINE_END = 0.0

# --------------------
# ----- FUNCTIONS -----
# --------------------

def read_eeg_csv(filepath):
   df = pd.read_csv(filepath)
   time = df.iloc[:, 0].values
   eeg_data = df.iloc[:, 1:9].values  # Channels 1-8
   markers = [float(num) if not math.isnan(num) else 0 for num in df.iloc[:, -1]]
   return time, eeg_data, markers


def bandpass_filter(data, fs, lowcut, highcut, order=4):
   data = np.nan_to_num(data)
   nyquist = 0.5 * fs
   low = lowcut / nyquist
   high = highcut / nyquist
   b, a = scipy.signal.butter(order, [low, high], btype='band')
   return scipy.signal.filtfilt(b, a, data, axis=0)


def extract_epochs(signal_data, timestamps, events, fs, t_start, t_end):
   epochs_dict = {}
   epoch_len = int((t_end - t_start) * fs)


   for (evt_time, evt_label) in events:
       evt_sample = np.argmin(np.abs(timestamps - evt_time))
       start_sample = max(evt_sample + int(t_start * fs), 0)  # Prevent negative indexing
       end_sample = start_sample + epoch_len
       if end_sample > signal_data.shape[0]:
           continue  # Skip epochs that exceed data length
       epoch_data = signal_data[start_sample:end_sample, :]


       if evt_label not in epochs_dict:
           epochs_dict[evt_label] = []
       epochs_dict[evt_label].append(epoch_data)


   for label in epochs_dict:
       epochs_dict[label] = np.array(epochs_dict[label])


   return epochs_dict

def compute_bandpower(epoch, fs, band):
   low, high = band
   freqs, psd = scipy.signal.welch(epoch, fs=fs, nperseg=int(fs*1.0), axis=0)
   mask = (freqs >= low) & (freqs <= high)
   return np.trapz(psd[mask], freqs[mask], axis=0)

def compute_erd_ers(epoch, fs, band, baseline_samples):
   baseline = epoch[:baseline_samples, :]
   task = epoch[baseline_samples:, :]
   bp_base = compute_bandpower(baseline, fs, band)
   bp_task = compute_bandpower(task, fs, band)
   erd = (bp_task - bp_base) / bp_base
   return erd

# -------------------------
# ---- MAIN PIPELINE  -----
# -------------------------
filename = "sorted_MI_EEG_20251005_171205_Session1LS.csv"
filepath = os.path.abspath(os.path.join("data", "raw", filename))


# Load data
time, eeg_data, markers = read_eeg_csv(filepath)
timestamps = time - time[0]
fs = 250


# Identify events robustly
events = []
for i, marker_val in enumerate(markers):
    if marker_val == MARKER_RIGHT:
       events.append((timestamps[i], "Right"))
    elif marker_val == MARKER_LEFT:
       events.append((timestamps[i], "Left"))
    elif marker_val == MARKER_LEG:
       events.append((timestamps[i], "Leg"))


print(f"Found {len(events)} events")


# Filter data in MI band
filt_eeg = bandpass_filter(eeg_data, fs, 8, 30)


# Extract epochs
epochs_dict = extract_epochs(filt_eeg, timestamps, events,
                            fs=fs,
                            t_start=EPOCH_START,
                            t_end=EPOCH_END)


print("Epoch keys available:", list(epochs_dict.keys()))

# ----------------------------
# ---- BANDPOWER FEATURES -----
# ----------------------------

bands = {
    #"theta": (4, 7), # Theta is ignored and bandpower analysis is only focused on mu and beta 
   "mu": (8, 13),
   "beta": (13, 30)
}

baseline_samples = int(abs(EPOCH_START) * fs)

X = []
y = []


for label in ["Left", "Right", "Leg"]:
   if label not in epochs_dict:
       print(f"No epochs for label {label}, skipping")
       continue
   class_id = 0 if label == "Left" else 1 if label == "Right" else 2 # 0 is left arm, 1 is right arm, 2 is leg 
   for epoch in epochs_dict[label]: 
       features = []
       # Bandpower for each band
       for band_name, band_range in bands.items():
           bp = compute_bandpower(epoch, fs, band_range)
           features.extend(bp)
       # ERD/ERS in mu and beta
       erd_ers_mu = compute_erd_ers(epoch, fs, bands["mu"], baseline_samples)
       erd_ers_beta = compute_erd_ers(epoch, fs, bands["beta"], baseline_samples)
       features.extend(erd_ers_mu)
       features.extend(erd_ers_beta)
       X.append(features)
       y.append(class_id)

X = np.array(X) 
y = np.array(y)

with open('output.csv', mode='w', newline='') as fileX:
    writer = csv.writer(fileX)
    writer.writerows(X) 

with open('labels.csv', mode='w', newline='') as fileY:
    writer = csv.writer(fileY)
    for label in y:
        writer.writerow([label])

label_map = {0: "Left", 1: "Right", 2: "Leg"}
for i in range(len(X)):
    mu_bp = np.array(X[i][0:8])
    beta_bp = np.array(X[i][8:16])

    class_name = label_map[int(y[i])] if i < len(y) else "Unknown"

    channels = np.arange(1, 9)
    width = 0.35

    plt.figure()
    plt.bar(channels - width/2, mu_bp, width, label='Mu (8–13 Hz)', color='tab:blue')
    plt.bar(channels + width/2, beta_bp, width, label='Beta (13–30 Hz)', color='tab:orange')
    plt.xlabel('Channel')
    plt.xticks(channels)
    plt.ylabel('Bandpower')
    plt.title(f"Trial {i+1} - {class_name} - Per-channel Mu and Beta Bandpower")
    plt.legend()
    plt.tight_layout()
    plt.show()
print("Feature extraction completed.")
