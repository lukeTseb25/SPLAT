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
import itertools

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

def fft_channel(signal, window, nfft, freq_mask):
    """
    Computes FFT magnitude for a single channel.
    window: precomputed Hann window
    """
    if len(signal) < len(window):
        signal_padded = np.pad(signal, (0, len(window) - len(signal)), mode='constant')
    else:
        signal_padded = signal[:len(window)]
    windowed_signal = signal_padded * window
    fft_result = np.fft.rfft(windowed_signal, n=nfft)
    magnitude = np.abs(fft_result)
    return magnitude[freq_mask]

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

def sort_and_validate_thread(stop_event, raw_times, raw_eeg_data, raw_markers,
                             sorted_times, sorted_eeg_data, sorted_markers):
    """
    Thread that sorts and validates timestamp ordering of raw EEG data.
    Ensures data is chronologically ordered before passing to filter stage.
    
    Buffers data and only releases when confident no earlier data will arrive (live data safety).
    Works for both file-based and live implementations.
    """
    # Buffer for sorting - similar to optimized_classifier_prep.py
    buffer_times = deque()
    buffer_eeg = deque()
    buffer_markers = deque()
    
    # Minimum buffer size before we consider a window "final"
    # For live data, this ensures we have enough lookahead for out-of-order data
    min_buffer_for_release = CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES * 2
    
    samples_processed = 0
    out_of_order_count = 0
    
    def _sort_and_release_window(k):
        """
        Sort first k samples chronologically and release them.
        Returns count of samples released.
        """
        nonlocal out_of_order_count
        
        k = min(k, len(buffer_times))
        if k == 0:
            return 0
        
        # Extract first k items
        temp_times = [buffer_times.popleft() for _ in range(k)]
        temp_eeg = [buffer_eeg.popleft() for _ in range(k)]
        temp_markers = [buffer_markers.popleft() for _ in range(k)]
        
        # Check and log out-of-order data
        for i in range(1, len(temp_times)):
            if temp_times[i] < temp_times[i-1]:
                out_of_order_count += 1
        
        # Sort by timestamp
        sorted_indices = np.argsort(temp_times)
        
        # Release sorted data
        for idx in sorted_indices:
            sorted_times.append(temp_times[idx])
            sorted_eeg_data.append(temp_eeg[idx])
            sorted_markers.append(temp_markers[idx])
        
        return k
    
    while not stop_event.is_set() or len(raw_eeg_data) > 0:
        # Accumulate raw data into buffer
        while len(raw_eeg_data) > 0 and len(raw_times) > 0 and len(raw_markers) > 0:
            buffer_eeg.append(raw_eeg_data.popleft())
            buffer_times.append(raw_times.popleft())
            buffer_markers.append(raw_markers.popleft())
        
        # Release data when buffer exceeds threshold or stream is ending
        if len(buffer_times) >= min_buffer_for_release or (stop_event.is_set() and len(buffer_times) > 0):
            # Determine how much to release
            if stop_event.is_set():
                # Final release: sort all remaining data
                release_count = len(buffer_times)
            else:
                # Normal release: keep min_buffer_for_release as lookahead, release the rest
                release_count = len(buffer_times) - min_buffer_for_release
            
            released = _sort_and_release_window(release_count)
            samples_processed += released
            
            if samples_processed % 250 == 0:
                print(f"[Sort] Processed {samples_processed} samples, "
                      f"out-of-order detections: {out_of_order_count}, "
                      f"buffer: {len(buffer_times)}, output: {len(sorted_times)}")
        
        time.sleep(POLL_SLEEP)
    
    # Final report
    if out_of_order_count > 0:
        print(f"[Sort] WARNING: Found {out_of_order_count} out-of-order timestamp transitions")
    else:
        print(f"[Sort] All {samples_processed} samples passed chronological validation")

def bandpass_filter_thread(stop_event, sorted_eeg_data, sorted_times, sorted_markers, 
                           filtered_eeg_data, filtered_times, filtered_markers):
    """
    Thread that applies bandpass filtering to sorted EEG data.
    Takes data from sorted_* deques and outputs to filtered_* deques.
    """
    # Design Butterworth bandpass filter (sos)
    N, Wn = scipy.signal.buttord([FREQ_LOW, FREQ_HIGH], [FREQ_LOW-2, FREQ_HIGH+2], 3, 40, fs=FS)
    sos = scipy.signal.butter(N, Wn, btype='bandpass', output='sos', fs=FS)
    
    # Per-channel filter states
    zi_per_channel = [scipy.signal.sosfilt_zi(sos) * 0.0 for _ in range(NUM_CHANNELS)]
    
    processed_count = 0
    
    while not stop_event.is_set() or len(sorted_eeg_data) > 0:
        # Check all three deques have data before popping
        if len(sorted_eeg_data) == 0 or len(sorted_times) == 0 or len(sorted_markers) == 0:
            time.sleep(POLL_SLEEP)
            continue
        
        # Process a single sample
        sorted_sample = sorted_eeg_data.popleft()
        sorted_time = sorted_times.popleft()
        sorted_marker = sorted_markers.popleft()
        
        # Filter sample through each channel
        filtered_sample = np.zeros(NUM_CHANNELS)
        for ch in range(NUM_CHANNELS):
            col = np.array([sorted_sample[ch]], dtype=np.float64)
            if np.allclose(zi_per_channel[ch], 0.0):
                zi_per_channel[ch] = scipy.signal.sosfilt_zi(sos) * col[0]
            y, zf = scipy.signal.sosfilt(sos, col, zi=zi_per_channel[ch])
            zi_per_channel[ch] = zf
            filtered_sample[ch] = y[0]
        
        # Add to filtered deques
        filtered_eeg_data.append(filtered_sample)
        filtered_times.append(sorted_time)
        filtered_markers.append(sorted_marker)
        
        processed_count += 1
        if processed_count % 250 == 0:
            print(f"[Filter] Processed {processed_count} samples, buffered: {len(filtered_eeg_data)}, "
                  f"times: {len(filtered_times)}, markers: {len(filtered_markers)}")

def fft_thread(stop_event, filtered_eeg_data, filtered_times, filtered_markers,
               spectrogram_data, spectrogram_times, spectrogram_markers):
    """
    Optimized thread that computes FFT spectrograms from filtered EEG data.
    Key optimizations:
    - Precomputed Hann window
    - Vectorized channel processing
    - Numpy array buffering for fast slicing
    - Batch frame processing
    """
    freqs, freq_mask = get_frequency_mask(FREQ_LOW, FREQ_HIGH, NFFT, FS)
    freq_bins_in_band = np.sum(freq_mask)
    
    # Precompute window once
    hann_window = scipy.signal.windows.hann(FFT_WINDOW_SIZE_SAMPLES)
    
    # Use numpy array buffer for efficient slicing
    # Start with reasonable capacity, grow if needed
    buffer_capacity = 10000
    buffer = np.zeros((buffer_capacity, NUM_CHANNELS), dtype=np.float32)
    buffer_times = np.zeros(buffer_capacity, dtype=np.float32)
    buffer_markers = np.zeros(buffer_capacity, dtype=np.int32)
    buffer_len = 0
    
    frame_count = 0
    
    while not stop_event.is_set() or len(filtered_eeg_data) > 0:
        # Add newly available samples to buffer (check all three deques)
        while len(filtered_eeg_data) > 0 and len(filtered_times) > 0 and len(filtered_markers) > 0 and buffer_len < buffer_capacity:
            buffer[buffer_len] = filtered_eeg_data.popleft()
            buffer_times[buffer_len] = filtered_times.popleft()
            buffer_markers[buffer_len] = filtered_markers.popleft()
            buffer_len += 1
        
        # Grow buffer if needed
        if buffer_len >= buffer_capacity - FFT_WINDOW_SIZE_SAMPLES:
            new_capacity = buffer_capacity * 2
            new_buffer = np.zeros((new_capacity, NUM_CHANNELS), dtype=np.float32)
            new_buffer_times = np.zeros(new_capacity, dtype=np.float32)
            new_buffer_markers = np.zeros(new_capacity, dtype=np.int32)
            
            new_buffer[:buffer_len] = buffer[:buffer_len]
            new_buffer_times[:buffer_len] = buffer_times[:buffer_len]
            new_buffer_markers[:buffer_len] = buffer_markers[:buffer_len]
            
            buffer = new_buffer
            buffer_times = new_buffer_times
            buffer_markers = new_buffer_markers
            buffer_capacity = new_capacity
        
        # Compute frames if we have enough samples
        if buffer_len >= FFT_WINDOW_SIZE_SAMPLES:
            # Compute all possible frames from current buffer
            num_possible_frames = (buffer_len - FFT_WINDOW_SIZE_SAMPLES) // FFT_STEP_SIZE_SAMPLES + 1
            
            for frame_idx in range(num_possible_frames):
                start_idx = frame_idx * FFT_STEP_SIZE_SAMPLES
                end_idx = start_idx + FFT_WINDOW_SIZE_SAMPLES
                
                # Extract window for all channels at once (vectorized)
                window_data = buffer[start_idx:end_idx]  # shape (window_size, num_channels)
                
                # Apply window to all channels at once
                windowed_data = window_data * hann_window[:, np.newaxis]
                
                # Compute FFT for all channels (vectorized)
                fft_result = np.fft.rfft(windowed_data, n=NFFT, axis=0)
                magnitude = np.abs(fft_result)
                
                # Apply frequency mask to all channels
                frame_magnitude = magnitude[freq_mask, :]  # shape (freq_bins, num_channels)
                
                # Convert to decibels
                frame_db = 20 * np.log10(frame_magnitude + 1e-12)
                
                # Use the center time and marker of the frame
                center_idx = start_idx + FFT_WINDOW_SIZE_SAMPLES // 2
                frame_time = buffer_times[center_idx]
                frame_marker = int(buffer_markers[center_idx])
                
                spectrogram_data.append(frame_db)
                spectrogram_times.append(frame_time)
                spectrogram_markers.append(frame_marker)
                
                frame_count += 1
            
            # Shift buffer: remove processed samples, keep overlap
            pop_count = num_possible_frames * FFT_STEP_SIZE_SAMPLES
            buffer[:buffer_len - pop_count] = buffer[pop_count:buffer_len]
            buffer_times[:buffer_len - pop_count] = buffer_times[pop_count:buffer_len]
            buffer_markers[:buffer_len - pop_count] = buffer_markers[pop_count:buffer_len]
            buffer_len -= pop_count
            
            if frame_count % 100 == 0:
                print(f"[FFT] Computed {frame_count} frames, buffered: {len(spectrogram_data)}, "
                      f"buffer_len: {buffer_len}, input queue: {len(filtered_eeg_data)}")
        
        time.sleep(POLL_SLEEP)

def trial_separation_and_write_thread(stop_event, spectrogram_data, spectrogram_times, spectrogram_markers,
                                      output_filepath):
    """
    Thread that separates spectrograms into trials based on markers and writes to file.
    """
    # Buffer for building trials
    trial_buffer = deque()
    trial_times_buffer = deque()
    trial_markers_buffer = deque()
    
    trials = []
    current_marker = None
    marker_collection_window = deque()
    marker_position_in_window = -CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES - 1
    
    required_buffer_size = max(CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_FILTERING_PADDING_SAMPLES * 2,
                               CLASSIFICATION_WINDOW_SAMPLES + CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES)
    
    trial_count = 0
    
    while not stop_event.is_set() or len(spectrogram_data) > 0:
        # Add newly available spectrograms to trial buffer
        while len(spectrogram_data) > 0:
            trial_buffer.append(spectrogram_data.popleft())
            trial_times_buffer.append(spectrogram_times.popleft())
            trial_markers_buffer.append(spectrogram_markers.popleft())
        
        # Check if we have enough data to form trials
        if len(trial_buffer) >= required_buffer_size:
            # Look for markers in the buffer
            for i in range(required_buffer_size):
                if trial_markers_buffer[i] is not None and trial_markers_buffer[i] != 0:
                    current_marker = trial_markers_buffer[i]
                    marker_position_in_window = i
            
            # If marker is properly positioned outside threshold, create trial
            if (current_marker is not None and 
                (marker_position_in_window < -CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES or 
                 marker_position_in_window > required_buffer_size)):
                
                # Extract trial window with padding
                window_start = CLASSIFICATION_FILTERING_PADDING_SAMPLES
                window_end = window_start + CLASSIFICATION_WINDOW_SAMPLES
                
                trial_spectrogram = list(itertools.islice(trial_buffer, window_start, window_end))
                trial_times = list(itertools.islice(trial_times_buffer, window_start, window_end))
                
                if len(trial_spectrogram) == CLASSIFICATION_WINDOW_SAMPLES:
                    # Stack spectrograms into (time_frames, freq_bins, num_channels)
                    trial_spectrogram = np.stack(trial_spectrogram, axis=0)
                    trial_times = np.array(trial_times)
                    
                    trials.append({
                        'marker': current_marker,
                        'times': trial_times,
                        'spectrogram': trial_spectrogram
                    })
                    
                    trial_count += 1
                    if trial_count % 10 == 0:
                        print(f"[Writer] Created {trial_count} trials, input queue: {len(spectrogram_data)}")
            
            # Pop one classification window step from front
            pop_count = min(CLASSIFICATION_WINDOW_STEP_SAMPLES, len(trial_buffer))
            for _ in range(pop_count):
                if len(trial_buffer) > 0:
                    trial_buffer.popleft()
                    trial_times_buffer.popleft()
                    trial_markers_buffer.popleft()
            
            marker_position_in_window -= pop_count
        
        time.sleep(POLL_SLEEP)
    
    # Save trials to pickle file
    freqs, freq_mask = get_frequency_mask(FREQ_LOW, FREQ_HIGH, NFFT, FS)
    
    with open(output_filepath, 'wb') as f:
        pickle.dump({
            'trials': trials,
            'freqs': freqs[freq_mask],
            'metadata': {
                'classification_window_samples': CLASSIFICATION_WINDOW_SAMPLES,
                'sampling_rate': FS,
                'num_channels': NUM_CHANNELS,
                'freq_low': FREQ_LOW,
                'freq_high': FREQ_HIGH
            }
        }, f)
    
    print(f"[Writer] Saved {trial_count} trials to {output_filepath}")

def main():
    os.makedirs("./data/raw", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)

    filename = "MI_EEG_20251005_171205_Session1LS.csv"

    if len(sys.argv) < 2:
        print("This script expects a filename.")
    else:
        filename = sys.argv[1]

    filepath = os.path.abspath(os.path.join("data", "raw", filename))
    
    # Shared deques between threads
    raw_times, raw_eeg_data, raw_markers = deque(), deque(), deque()
    sorted_times, sorted_eeg_data, sorted_markers = deque(), deque(), deque()
    filtered_eeg_data, filtered_times, filtered_markers = deque(), deque(), deque()
    spectrogram_data, spectrogram_times, spectrogram_markers = deque(), deque(), deque()
    
    # Stop events
    read_stop_event = threading.Event()
    sort_stop_event = threading.Event()
    filter_stop_event = threading.Event()
    fft_stop_event = threading.Event()
    
    # Start read thread
    read_thread = threading.Thread(target=read_eeg_csv, args=(filepath, raw_times, raw_eeg_data, raw_markers))
    read_thread.start()
    
    # Start sort/validation thread
    sort_thread = threading.Thread(target=sort_and_validate_thread,
                                   args=(read_stop_event, raw_times, raw_eeg_data, raw_markers,
                                         sorted_times, sorted_eeg_data, sorted_markers))
    sort_thread.start()
    
    # Start filter thread
    filter_thread = threading.Thread(target=bandpass_filter_thread, 
                                     args=(sort_stop_event, sorted_eeg_data, sorted_times, sorted_markers,
                                           filtered_eeg_data, filtered_times, filtered_markers))
    filter_thread.start()
    
    # Start FFT thread
    fft_thread_obj = threading.Thread(target=fft_thread,
                                      args=(filter_stop_event, filtered_eeg_data, filtered_times, filtered_markers,
                                            spectrogram_data, spectrogram_times, spectrogram_markers))
    fft_thread_obj.start()
    
    # Prepare output filepath
    output_filepath = os.path.abspath(os.path.join("data", "processed", f"processed_{filename}.pkl"))
    
    # Start trial separation and write thread
    write_thread = threading.Thread(target=trial_separation_and_write_thread,
                                    args=(fft_stop_event, spectrogram_data, spectrogram_times, spectrogram_markers,
                                          output_filepath))
    write_thread.start()
    
    # Wait for read to complete
    read_thread.join()
    print("[Main] Read thread completed")
    
    # Signal sort thread to stop after processing remaining data
    read_stop_event.set()
    sort_thread.join()
    print("[Main] Sort thread completed")
    
    # Signal filter thread to stop after processing remaining data
    sort_stop_event.set()
    filter_thread.join()
    print("[Main] Filter thread completed")
    
    # Signal FFT thread to stop after processing remaining data
    filter_stop_event.set()
    fft_thread_obj.join()
    print("[Main] FFT thread completed")
    
    # Signal write thread to stop and save
    fft_stop_event.set()
    write_thread.join()
    print("[Main] Write thread completed")

if __name__ == "__main__":
    main()