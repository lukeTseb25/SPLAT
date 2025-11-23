import numpy as np
import pandas as pd
import scipy.signal
import os
import math
import sys
import threading
import time
from collections import deque

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
# CLASSIFICATION_WINDOW_STEP_SEC = 1.0/FS
CLASSIFICATION_WINDOW_STEP_SEC = 1.0
CLASSIFICATION_WINDOW_STEP_SAMPLES = int(CLASSIFICATION_WINDOW_STEP_SEC * FS)
CLASSIFICATION_MARKER_TIME_THRESHOLD_SEC = 0.5
CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES = int(CLASSIFICATION_MARKER_TIME_THRESHOLD_SEC * FS)
CLASSIFICATION_FILTERING_PADDING_SEC = 0.2
CLASSIFICATION_FILTERING_PADDING_SAMPLES = int(CLASSIFICATION_FILTERING_PADDING_SEC * FS)

POLL_SLEEP = 0.001

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

    times_list = times_list[:6000]
    eeg_data_list = eeg_data_list[:6000]
    markers_list = markers_list[:6000]

    #Convert to deques for thread-safe appending/popping
    times.extend(times_list)
    eeg_data.extend(eeg_data_list)
    markers.extend(markers_list)

    return times, eeg_data, markers

def preprocess_data(stop_event, times:deque, eeg_data:deque, markers:deque=None, return_values:list=None):
    """
    Creates sprectrograms of data segments of a given window size and labels it with the marker before if given.
    """
    
    segments = []
    freqs, freq_mask = get_frequency_mask(FREQ_LOW, FREQ_HIGH, NFFT, FS)
    i=0

    #Design Butterworth bandpass filter
    N, Wn = scipy.signal.buttord([FREQ_LOW, FREQ_HIGH], [FREQ_LOW-2, FREQ_HIGH+2], 3, 40, fs=FS)
    sos = scipy.signal.butter(N, Wn, btype='bandpass', output='sos', fs=FS)
    
    #Unmarked/live data
    if markers is None:
        while not stop_event.is_set() or (
            len(eeg_data) >= CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES*2 and len(times) >= CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES*2
            ):
            if len(eeg_data) >= CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES*2 and len(times) >= CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES*2:
                #Get segment including padding
                segment = np.array(eeg_data)[:CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES*2]

                #Perform bandpass using butterworth filter
                for ch in range(segment.shape[1]):
                    segment[:, ch] = scipy.signal.sosfiltfilt(sos, segment[:, ch], padtype=None, padlen=0)
                
                windowed_segment = segment[CLASSIFICATION_FILTERING_PADDING_SAMPLES:-CLASSIFICATION_FILTERING_PADDING_SAMPLES]
                times_segment = np.array(times)[CLASSIFICATION_FILTERING_PADDING_SAMPLES:CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES]

                #Get spectrogram without padding
                spectrogram = compute_time_frequency_tensor(windowed_segment,
                    FFT_WINDOW_SIZE_SAMPLES,
                    FFT_STEP_SIZE_SAMPLES,
                    NFFT,
                    freq_mask)
                
                #Convert from volts to decibels
                spectrogram = 20 * np.log10(np.abs(spectrogram) + 1e-12)

                segments.append((None, times_segment, windowed_segment, spectrogram))

                #Pop a step size from the front of the data
                for _ in range(CLASSIFICATION_WINDOW_STEP_SAMPLES):
                    eeg_data.popleft()
                    times.popleft()

            else:
                time.sleep(POLL_SLEEP)

    #Marked data
    else:
        collected_marker=None
        required_buffer_size=max(CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES*2,CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES)
        rel_last_marker_pos = -CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES - 1
        while not stop_event.is_set() or (
            len(eeg_data) >= required_buffer_size and len(markers) >= required_buffer_size and len(times) >= required_buffer_size
            ):
            if len(eeg_data) >= required_buffer_size and len(markers) >= required_buffer_size and len(times) >= required_buffer_size:
                #Collect a marker if it is in the area of interest
                for i in range(required_buffer_size):
                    if markers[i] is not None and markers[i] != 0:
                        collected_marker=markers[i]
                        rel_last_marker_pos=i

                #If marker is outside threshold segment data as usual
                if rel_last_marker_pos < -CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES or rel_last_marker_pos > required_buffer_size:
                    #Get segment including padding
                    segment = np.array(eeg_data)[:CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES*2]

                    #Perform bandpass using butterworth filter
                    for ch in range(segment.shape[1]):
                        segment[:, ch] = scipy.signal.sosfiltfilt(sos, segment[:, ch], padtype=None, padlen=0)
                
                    windowed_segment = segment[CLASSIFICATION_FILTERING_PADDING_SAMPLES:-CLASSIFICATION_FILTERING_PADDING_SAMPLES]
                    times_segment = np.array(times)[CLASSIFICATION_FILTERING_PADDING_SAMPLES:CLASSIFICATION_WINDOW_SAMPLES+CLASSIFICATION_FILTERING_PADDING_SAMPLES]

                    #Get spectrogram without padding
                    spectrogram = compute_time_frequency_tensor(windowed_segment,
                        FFT_WINDOW_SIZE_SAMPLES,
                        FFT_STEP_SIZE_SAMPLES,
                        NFFT,
                        freq_mask)
                
                    #Convert from volts to decibels
                    spectrogram = 20 * np.log10(np.abs(spectrogram) + 1e-12)

                    segments.append((collected_marker, times_segment, windowed_segment, spectrogram))

                #Pop a step size from the front of the data
                for _ in range(CLASSIFICATION_WINDOW_STEP_SAMPLES):
                    eeg_data.popleft()
                    markers.popleft()
                    times.popleft()
                rel_last_marker_pos -= CLASSIFICATION_WINDOW_STEP_SAMPLES

            else:
                time.sleep(POLL_SLEEP)

    if return_values is not None:
        return_values.append(segments)
        return_values.append(freqs[freq_mask])
    
    return segments, freqs[freq_mask]

def plot_trial(trials, trial_number, channel, freqs, window=(0.0,0.0)):
    """
    Plots time-series and time-frequency for a given trial and channel.
    """

    import matplotlib.pyplot as plt

    if trial_number < 0 or trial_number >= len(trials):
        print(f"Trial {trial_number} out of range.")
        return

    start_marker, time_segment, segment, spectrogram = trials[trial_number]

    #Grab a section of the trial if window is specified
    if window != (0.0,0.0):
        start_time, end_time = window
        mask = (time_segment >= start_time) & (time_segment <= end_time)
        time_segment = time_segment[mask]
        segment = segment[mask, :]
        spectrogram = spectrogram[np.where((time_segment >= start_time) & (time_segment <= end_time))[0], :, :]
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Trial {trial_number} | Marker {start_marker} | Channel {channel}")

    # --- Time series ---
    channel_idx = channel - 1  # zero-based index
    axs[0].plot(time_segment, segment[:, channel_idx])
    axs[0].set_title("Time Series")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Voltage (V)")

    # --- Time-frequency ---
    tf_data = spectrogram[:, :, channel_idx].T  # shape: (freq_bins, time_frames)
    im = axs[1].imshow(tf_data,
                       aspect='auto',
                       origin='lower',
                       extent=[time_segment[0], time_segment[-1], freqs[0], freqs[-1]])
    axs[1].set_title("Time-Frequency (Spectrogram)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axs[1], label="Magnitude (VdB)")

    plt.tight_layout()
    plt.show()

def main():
    filename = "MI_EEG_20251005_171205_Session1LS.csv"

    if len(sys.argv) < 2:
        print("This script expects a filename.")
        #sys.exit(1)
    else:
        filename = sys.argv[1]

    filepath = os.path.abspath(os.path.join("data", "raw", filename))
    times, eeg_data, markers = deque(), deque(), deque()
    
    stop_event = threading.Event()
    return_values = list()
    thread = threading.Thread(target=preprocess_data, args=(stop_event, times, eeg_data, markers, return_values))
    thread.start()

    read_eeg_csv(filepath, times, eeg_data, markers)

    time.sleep(10)
    
    stop_event.set()
    thread.join()

    trials = return_values[0]
    freqs = return_values[1]

    for trial in range(len(trials)):
        plot_trial(trials, trial, 1, freqs)

if __name__ == "__main__":
    main()