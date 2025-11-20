import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import os
import math
import sys

# Parameters
FS = 250
NUM_CHANNELS = 8

FREQ_LOW = 8
FREQ_HIGH = 30
WINDOW_SIZE_SEC = 0.25
WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_SEC * FS)
STEP_SIZE_SEC = 1.0/FS
STEP_SIZE_SAMPLES = int(STEP_SIZE_SEC * FS)
NFFT = 256

def get_frequency_mask(freq_low, freq_high, nfft, fs):
    freqs = np.fft.rfftfreq(nfft, d=1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    return freqs, freq_mask

def fft_channel(signal, window_size, nfft, freq_mask):
    window = scipy.signal.windows.hann(window_size)
    if len(signal) < window_size:
        signal_padded = np.pad(signal, (0, window_size - len(signal)), mode='constant')
    else:
        signal_padded = signal[:window_size]
    windowed_signal = signal_padded * window
    fft_result = np.fft.rfft(windowed_signal, n=nfft)
    magnitude = np.abs(fft_result)
    return magnitude[freq_mask]

def compute_time_frequency_tensor(eeg_segment, window_size, step_size, nfft, freq_mask):
    segment_length = eeg_segment.shape[0]
    num_channels = eeg_segment.shape[1]
    num_frames = max(1, (segment_length - window_size) // step_size + 1)
    freq_bins_in_band = np.sum(freq_mask)
    spectrogram = np.zeros((num_frames, freq_bins_in_band, num_channels))
    for frame_idx in range(num_frames):
        start_idx = frame_idx * step_size
        end_idx = start_idx + window_size
        if end_idx <= segment_length:
            for channel_idx in range(num_channels):
                channel_data = eeg_segment[start_idx:end_idx, channel_idx]
                spectrogram[frame_idx, :, channel_idx] = fft_channel(
                    channel_data, window_size, nfft, freq_mask)
    return spectrogram

def read_eeg_csv(filepath):
    """
    Reads CSV where:
      - first column = time (s)
      - next NUM_CHANNELS columns = EEG channels
      - last column = marker (1–4)
    """
    df = pd.read_csv(filepath)
    time = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:NUM_CHANNELS+1].values
    eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
    markers = [int(num) if num is not None and not math.isnan(num) else 0 for num in df.iloc[:, -1]]
    return time, eeg_data, markers

def extract_trials(time, eeg_data, markers):
    """
    Detects trials based on marker logic:
      - 1–3 mark start (different trial types)
      - 4 marks end
    Returns list of tuples:
      (trial_type, time_series_array, spectrogram_tensor)
    """
    trials = []
    freqs, freq_mask = get_frequency_mask(FREQ_LOW, FREQ_HIGH, NFFT, FS)
    indices = []
    for i in range(1, len(markers)):
        if markers[i] in [1,2,3,4]:
            indices.append(i)
    for i in range(len(indices)-1):
        start_idx = indices[i]
        end_idx = indices[i+1]
        start_marker = markers[start_idx]
        segment = eeg_data[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]

        #Convert all the elements of segment from microvolts to volts
        segment = segment * 1e-6

        #Perform bandpass using butterworth filter
        N, Wn = scipy.signal.buttord([FREQ_LOW, FREQ_HIGH], [FREQ_LOW-2, FREQ_HIGH+2], 3, 40, fs=FS)
        sos = scipy.signal.butter(N, Wn, btype='bandpass', output='sos', fs=FS)
        for ch in range(segment.shape[1]):
            segment[:, ch] = scipy.signal.sosfiltfilt(sos, segment[:, ch])
        
        spectrogram = compute_time_frequency_tensor(segment,
                                                    WINDOW_SIZE_SAMPLES,
                                                    STEP_SIZE_SAMPLES,
                                                    NFFT,
                                                    freq_mask)
        
        #Convert from volts to decibels
        spectrogram = 20 * np.log10(np.abs(spectrogram) + 1e-12)

        #Make time relative to start of trial
        time_segment = time_segment - time_segment[0]

        trials.append((start_marker, time_segment, segment, spectrogram))
    return trials, freqs[freq_mask]

filename = "MI_EEG_20251005_171205_Session1LS.csv"

if len(sys.argv) < 1:
        print("This script expects a filename.")
        sys.exit(1)
else:
    filename = sys.argv[1]

filepath = os.path.abspath(os.path.join("data", "raw", filename))
time, eeg_data, markers = read_eeg_csv(filepath)
trials, freqs = extract_trials(time, eeg_data, markers)