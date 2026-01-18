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

    #Convert to deques for thread-safe appending/popping
    times.extend(times_list)
    eeg_data.extend(eeg_data_list)
    markers.extend(markers_list)

    return times, eeg_data, markers

def preprocess_data(stop_event, times:deque, eeg_data:deque, markers:deque=None, return_values:list=None):
    """
    Creates sprectrograms of data segments of a given window size and labels it with the marker before if given.

    Optimized: incremental filtering + cached FFT frames keyed by absolute start index so FFT work for
    overlapping frames is reused instead of recomputing identical windows.
    
    Buffers are kept sorted chronologically by times at each iteration.
    """
    
    segments = []
    freqs, freq_mask = get_frequency_mask(FREQ_LOW, FREQ_HIGH, NFFT, FS)

    # Design Butterworth bandpass filter (sos) once
    N, Wn = scipy.signal.buttord([FREQ_LOW, FREQ_HIGH], [FREQ_LOW-2, FREQ_HIGH+2], 3, 40, fs=FS)
    sos = scipy.signal.butter(N, Wn, btype='bandpass', output='sos', fs=FS)

    # Prepare incremental filtered buffer and per-channel filter states (zi)
    filtered_buffer = deque()  # stores filtered samples as rows [num_samples, num_channels]
    num_channels = NUM_CHANNELS
    zi_per_channel = [scipy.signal.sosfilt_zi(sos) * 0.0 for _ in range(num_channels)]

    # Cache for FFT frames: maps absolute_start_sample_index -> magnitude array shape (freq_bins, num_channels)
    fft_cache = {}
    # Track how many samples have been popped from the left (absolute offset of filtered_buffer[0])
    sample_offset = 0

    import itertools

    def _sort_buffers(k):
        """
        Sorts times, eeg_data, and markers (if present) chronologically based on times.
        Performs the sort on just the first k indices or, if there are less than k, all the indices.
        """

        k = min(k, min(len(times), len(eeg_data), len(markers) if markers is not None else len(times)))
        
        if markers is not None:
            sorted_times = list()
            sorted_eeg = list()
            sorted_markers = list()

            for _ in range(k):
                sorted_times.append(times.popleft())
                sorted_eeg.append(eeg_data.popleft())
                sorted_markers.append(markers.popleft())
            
            sorted_indices = np.argsort(sorted_times)
            for idx in reversed(sorted_indices):
                times.appendleft(sorted_times[idx])
                eeg_data.appendleft(sorted_eeg[idx])
                markers.appendleft(sorted_markers[idx])

        else:
            sorted_times = list()
            sorted_eeg = list()

            for _ in range(k):
                sorted_times.append(times.popleft())
                sorted_eeg.append(eeg_data.popleft())
            
            sorted_indices = np.argsort(sorted_times)
            for idx in reversed(sorted_indices):
                times.appendleft(sorted_times[idx])
                eeg_data.appendleft(sorted_eeg[idx])

    def _append_new_raw_and_filter():
        """
        Consume any new samples appended to eeg_data since last call, filter them incrementally,
        and append results to filtered_buffer. Updates zi_per_channel in place.
        """
        nonlocal zi_per_channel

        raw_len = len(eeg_data)
        filt_len = len(filtered_buffer)
        if raw_len <= filt_len:
            return  # no new samples

        # Take only the delta
        # Convert to numpy once
        raw_arr = np.array(eeg_data)
        new_raw = raw_arr[filt_len:raw_len]  # shape (n_new, num_channels)
        if new_raw.size == 0:
            return

        # Ensure zi initialisation uses first raw sample if zi was zeros
        for ch in range(num_channels):
            # Apply sosfilt on the new_raw column preserving zi
            col = new_raw[:, ch].astype(np.float64)
            # if zi is all zeros we can initialize it to sosfilt_zi * first_value to reduce start transients
            if np.allclose(zi_per_channel[ch], 0.0) and col.size > 0:
                zi_per_channel[ch] = scipy.signal.sosfilt_zi(sos) * col[0]
            y, zf = scipy.signal.sosfilt(sos, col, zi=zi_per_channel[ch])
            zi_per_channel[ch] = zf
            # append filtered column values into filtered_buffer (we'll build rows next)
            if ch == 0:
                filtered_cols = y.reshape(-1, 1)
            else:
                filtered_cols = np.hstack((filtered_cols, y.reshape(-1, 1)))

        # extend filtered_buffer with new rows
        for row in filtered_cols:
            filtered_buffer.append(row)

    def _compute_frame_at_abs_start(abs_start_local, window_size, nfft, freq_mask):
        """
        Compute FFT magnitude for a frame that starts at local index abs_start_local (absolute index on original stream).
        Returns magnitude for all channels (freq_bins, num_channels).
        """
        # local index in current filtered_buffer = abs_start_local - sample_offset
        local_start = abs_start_local - sample_offset
        # If local_start is negative or beyond current buffer, cannot compute
        if local_start < 0 or local_start + window_size > len(filtered_buffer):
            return None
        # Extract window for all channels
        # Build a numpy array slice
        rows = list(itertools.islice(filtered_buffer, local_start, local_start + window_size))
        window_arr = np.vstack(rows)  # shape (window_size, num_channels)
        # For each channel compute fft magnitude and return stacked
        magnitudes = []
        for ch in range(window_arr.shape[1]):
            magn = fft_channel(window_arr[:, ch], window_size, nfft, freq_mask)
            magnitudes.append(magn)
        # resulting shape (num_channels, freq_bins) -> transpose to (freq_bins, num_channels)
        return np.stack(magnitudes, axis=1)  # (window_freq_bins, num_channels)

    def _produce_segments_from_current_buffer(collected_marker=None):
        nonlocal sample_offset, fft_cache
        # Ensure filtered_buffer is long enough
        total_needed = CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES * 2
        if len(filtered_buffer) < total_needed or len(times) < total_needed:
            return False  # not ready
        
        # Get full (with padding) segment from filtered_buffer (do not pop yet)
        # We will compute frames for the windowed_segment (no padding)
        windowed_start = CLASSIFICATION_FILTERING_PADDING_SAMPLES
        windowed_end = windowed_start + CLASSIFICATION_WINDOW_SAMPLES
        windowed_length = CLASSIFICATION_WINDOW_SAMPLES

        # Frame parameters
        window_size = FFT_WINDOW_SIZE_SAMPLES
        step_size = FFT_STEP_SIZE_SAMPLES
        # number of frames for the windowed_segment
        num_frames = max(1, (windowed_length - window_size) // step_size + 1)

        # For each frame compute absolute start index (absolute on original stream)
        # The first frame's absolute_start = sample_offset + windowed_start + frame_idx*step_size
        frames = []
        freq_bins_in_band = np.sum(freq_mask)

        for frame_idx in range(num_frames):
            abs_start = sample_offset + windowed_start + frame_idx * step_size
            if abs_start in fft_cache:
                frame_magn = fft_cache[abs_start]
            else:
                # compute and store
                frame_magn = _compute_frame_at_abs_start(abs_start, window_size, NFFT, freq_mask)
                if frame_magn is None:
                    # this can happen if buffer shrank - bail out
                    return False
                fft_cache[abs_start] = frame_magn
            frames.append(frame_magn)  # each is (freq_bins, num_channels)

        # Stack frames into spectrogram shape (time_frames, freq_bins, num_channels)
        spectrogram = np.stack(frames, axis=0)

        # Prepare times and windowed_segment to return
        times_arr = np.array(times)
        times_segment = times_arr[windowed_start:windowed_end]
        # windowed_segment from filtered_buffer: convert slice to numpy array
        rows = list(itertools.islice(filtered_buffer, windowed_start, windowed_end))
        windowed_segment = np.vstack(rows)

        # Convert to decibels
        spectrogram_db = 20 * np.log10(np.abs(spectrogram) + 1e-12)

        segments.append((collected_marker, times_segment, windowed_segment, spectrogram_db))
        return True

    # Main loops: unmarked vs marked
    if markers is None:
        # Unmarked / live data
        while not stop_event.is_set() or (
            len(eeg_data) >= CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES*2 and len(times) >= CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES*2
        ):
            if len(eeg_data)%250 == 0:
                print(len(eeg_data), "samples in buffer")
            # Sort buffers chronologically
            _sort_buffers(CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES*2)
            
            # Add any newly appended raw samples to filtered_buffer
            _append_new_raw_and_filter()

            if len(filtered_buffer) >= CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES*2 and len(times) >= CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES*2:
                # produce a segment using current filtered buffer (does not modify deques yet)
                produced = _produce_segments_from_current_buffer(collected_marker=None)
                if not produced:
                    # if unable, wait for more data
                    time.sleep(POLL_SLEEP)
                    continue

                # Pop a step size from the front of the data (both raw deques and filtered buffer)
                for _ in range(CLASSIFICATION_WINDOW_STEP_SAMPLES):
                    if len(eeg_data) > 0:
                        eeg_data.popleft()
                    if len(times) > 0:
                        times.popleft()
                    if len(filtered_buffer) > 0:
                        filtered_buffer.popleft()
                    sample_offset += 1

                # Remove stale FFT cache entries (whose absolute start < sample_offset)
                stale_keys = [k for k in fft_cache.keys() if k < sample_offset]
                for k in stale_keys:
                    del fft_cache[k]

            else:
                time.sleep(POLL_SLEEP)

    else:
        # Marked data flow
        collected_marker = None
        required_buffer_size = max(CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES*2,
                                   CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES)
        rel_last_marker_pos = -CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES - 1

        while not stop_event.is_set() or (
            len(eeg_data) >= required_buffer_size and len(markers) >= required_buffer_size and len(times) >= required_buffer_size
        ):
            if len(eeg_data)%250 == 0:
                print(len(eeg_data), "samples in buffer")
            # Sort buffers chronologically
            _sort_buffers(required_buffer_size)
            
            # fill filtered buffer from any new raw samples
            _append_new_raw_and_filter()

            if len(filtered_buffer) >= required_buffer_size and len(markers) >= required_buffer_size and len(times) >= required_buffer_size:
                # Collect a marker if it is in the area of interest
                for i in range(required_buffer_size):
                    if markers[i] is not None and markers[i] != 0:
                        collected_marker = markers[i]
                        rel_last_marker_pos = i

                # If marker is outside threshold and data is marked segment data as usual
                if collected_marker is not None and (rel_last_marker_pos < -CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES or rel_last_marker_pos > required_buffer_size):
                    produced = _produce_segments_from_current_buffer(collected_marker=collected_marker)
                    if not produced:
                        time.sleep(POLL_SLEEP)
                        continue

                # Pop a step size from the front of the data (raw deques, markers, filtered buffer)
                for _ in range(CLASSIFICATION_WINDOW_STEP_SAMPLES):
                    if len(eeg_data) > 0:
                        eeg_data.popleft()
                    if len(markers) > 0:
                        markers.popleft()
                    if len(times) > 0:
                        times.popleft()
                    if len(filtered_buffer) > 0:
                        filtered_buffer.popleft()
                    sample_offset += 1
                rel_last_marker_pos -= CLASSIFICATION_WINDOW_STEP_SAMPLES

                # Remove stale FFT cache entries
                stale_keys = [k for k in fft_cache.keys() if k < sample_offset]
                for k in stale_keys:
                    del fft_cache[k]

            else:
                time.sleep(POLL_SLEEP)

    # return values if requested
    if return_values is not None:
        return_values.append(segments) #TODO: Change to update return values regularly for oncurrent saving
        #Add emptey segments list and freq list to return values at start, append to them regularly
        return_values.append(freqs[freq_mask])

    return segments, freqs[freq_mask]

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

if __name__ == "__main__":
    main()