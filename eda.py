import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

# ======================
# --- Existing params ---
# ======================
FS = 250
NUM_CHANNELS = 8

FREQ_LOW = 8
FREQ_HIGH = 30
WINDOW_SIZE_SEC = 0.25
WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_SEC * FS)
STEP_SIZE_SEC = 1.0/FS
STEP_SIZE_SAMPLES = int(STEP_SIZE_SEC * FS)
NFFT = 256

# --- Your existing helper functions (unchanged) ---
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

# ====================================
# --- New: File reader + trial maker ---
# ====================================
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
    markers = df.iloc[:, -1].values.astype(int)
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
    indices = np.where(markers != '')[0]
    for i in range(len(indices)):
        start_marker = markers[idx]
        # find nearest following 4 marker
        
        if len(end_candidates) == 0:
            continue
        end_idx = start_idx + end_candidates[0]
        segment = eeg_data[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]
        spectrogram = compute_time_frequency_tensor(segment,
                                                    WINDOW_SIZE_SAMPLES,
                                                    STEP_SIZE_SAMPLES,
                                                    NFFT,
                                                    freq_mask)
        trials.append((start_marker, time_segment, segment, spectrogram))
    return trials, freqs[freq_mask]

# =======================
# --- Plotting helpers ---
# =======================
def plot_trial(trials, trial_number, channel_idx, freqs):
    """
    Plots time-series and time-frequency for a given trial and channel.
    """
    if trial_number < 0 or trial_number >= len(trials):
        print(f"Trial {trial_number} out of range.")
        return

    start_marker, time_segment, segment, spectrogram = trials[trial_number]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Trial {trial_number} | Marker {start_marker} | Channel {channel_idx}")

    # --- Time series ---
    axs[0].plot(time_segment, segment[:, channel_idx])
    axs[0].set_title("Time Series")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Voltage (µV)")

    # --- Time-frequency ---
    tf_data = spectrogram[:, :, channel_idx].T  # shape: (freq_bins, time_frames)
    im = axs[1].imshow(tf_data,
                       aspect='auto',
                       origin='lower',
                       extent=[time_segment[0], time_segment[-1], freqs[0], freqs[-1]])
    axs[1].set_title("Time-Frequency (Spectrogram)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axs[1], label="Magnitude")

    plt.tight_layout()
    plt.show()

# =====================
# --- Example usage ---
# =====================
# time, eeg_data, markers = read_eeg_csv("your_file.csv")
# trials, freqs = extract_trials(time, eeg_data, markers)
# plot_trial(trials, trial_number=0, channel_idx=5, freqs=freqs)

filepath = "..\\data\\raw\\MI_EEG_20251005_171205_Session1LS.csv"
time, eeg_data, markers = read_eeg_csv("your_file.csv")
trials, freqs = extract_trials(time, eeg_data, markers)
plot_trial(trials, trial_number=0, channel_idx=5, freqs=freqs)