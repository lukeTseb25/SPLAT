from collections import deque
import csv
import logging
import threading
import datetime
import os
import sys
from modules.config import *

class LSLDataCollector(threading.Thread):
    """
    Background thread that collects and synchronizes EEG and marker data.
    """
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event
        self.eeg_buffer = deque()
        self.marker_buffer = deque()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        self.output_csv = os.path.join(RAW_DATA_DIR, f"MI_EEG_{timestamp_str}.csv")
        self.eeg_inlet = None
        self.marker_inlet = None
        self.clock_offset = 0.0

    def resolve_streams(self):
        try:
            logging.info("Resolving EEG LSL stream...")
            from pylsl import resolve_byprop, StreamInlet
            eeg_streams = resolve_byprop("name", EEG_STREAM_NAME, timeout=10)
            if not eeg_streams:
                logging.error(f"No EEG stream found with name '{EEG_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            self.clock_offset = self.eeg_inlet.time_correction()
            logging.info(f"Computed clock offset: {self.clock_offset:.6f} seconds")

            logging.info("Resolving Marker LSL stream...")
            marker_streams = resolve_byprop("name", MARKER_STREAM_NAME, timeout=10)
            if not marker_streams:
                logging.error(f"No Marker stream found with name '{MARKER_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.marker_inlet = StreamInlet(marker_streams[0], max_buflen=360)

            logging.info("LSL streams resolved. Starting data collection...")
        except Exception as e:
            logging.exception("Exception during stream resolution:")
            sys.exit(1)

    def flush_remaining(self, writer):
        """Flush any remaining buffered data to the CSV."""
        while self.eeg_buffer or self.marker_buffer:
            if self.eeg_buffer and self.marker_buffer:
                ts_eeg, eeg_data = self.eeg_buffer[0]
                ts_marker, marker = self.marker_buffer[0]
                if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                    row = [ts_eeg] + eeg_data + [marker]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
                    self.marker_buffer.popleft()
                elif ts_marker < ts_eeg:
                    row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                    writer.writerow(row)
                    self.marker_buffer.popleft()
                else:
                    row = [ts_eeg] + eeg_data + [""]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
            elif self.eeg_buffer:
                ts_eeg, eeg_data = self.eeg_buffer.popleft()
                row = [ts_eeg] + eeg_data + [""]
                writer.writerow(row)
            elif self.marker_buffer:
                ts_marker, marker = self.marker_buffer.popleft()
                row = [ts_marker] + ([""] * len(EEG_CHANNELS)) + [marker]
                writer.writerow(row)