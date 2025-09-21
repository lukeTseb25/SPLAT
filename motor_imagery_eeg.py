#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor Imagery EEG Data Collection Script

This script implements a motor imagery experiment using PsychoPy and LSL integration.
It presents instructions for left/right arm and right leg motor imagery and records EEG data
with precise timing markers.

Requirements:
  pip install pylsl psychopy pandas numpy
"""

import threading
import time
import random
import csv
import os
import sys
from collections import deque
from datetime import datetime
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Configuration Parameters ---
EEG_STREAM_NAME = "OpenBCI_EEG"
MARKER_STREAM_NAME = "MI_MarkerStream"

SAMPLE_RATE = 250  # Hz
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

# Motor Imagery Experiment Parameters
NUM_TRIALS = 6  # Configurable number of trials
INSTRUCTION_DURATION = 2.0  # seconds to show left/right instruction
IMAGERY_DURATION = 3.0  # seconds for motor imagery
INTER_TRIAL_INTERVAL = 3.0  # seconds between trials

# Marker values
MARKER_RIGHT = "1"  # right arm imagery
MARKER_LEFT = "2"   # left arm imagery
MARKER_LEG = "3"   # leg imagery
MARKER_STOP = "4"   # end of imagery period
MARKER_FAIL = "5"   # end of imagery period

# File/directories
RAW_DATA_DIR = "./data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Data collection parameters
MERGE_THRESHOLD = 0.002  # seconds threshold for aligning EEG and marker timestamps
POLL_SLEEP = 0.001      # sleep time between polls in collector loop

###############################################################################
#                         DATA COLLECTOR
###############################################################################
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

    def run(self):
        from pylsl import resolve_byprop, StreamInlet
        self.resolve_streams()
        try:
            with open(self.output_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = ["lsl_timestamp"] + EEG_CHANNELS + ["marker"]
                writer.writerow(header)
                while not self.stop_event.is_set():
                    try:
                        sample_eeg, ts_eeg = self.eeg_inlet.pull_sample(timeout=0.0)
                        if sample_eeg is not None and ts_eeg is not None:
                            self.eeg_buffer.append((ts_eeg, sample_eeg))
                    except Exception as e:
                        logging.exception("Error pulling EEG sample:")
                    try:
                        sample_marker, ts_marker = self.marker_inlet.pull_sample(timeout=0.0)
                        if sample_marker is not None and ts_marker is not None:
                            adjusted_ts_marker = ts_marker - self.clock_offset
                            marker_val = sample_marker[0]
                            self.marker_buffer.append((adjusted_ts_marker, marker_val))
                    except Exception as e:
                        logging.exception("Error pulling Marker sample:")

                    # Merge data from both buffers
                    while self.eeg_buffer and self.marker_buffer:
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
                    time.sleep(POLL_SLEEP)
                logging.info("Stop event set. Flushing remaining data...")
                self.flush_remaining(writer)
        except Exception as e:
            logging.exception("Exception in data collector run loop:")
        finally:
            logging.info(f"Data collection stopped. Data saved to {self.output_csv}")

###############################################################################
#                        EXPERIMENT LOGIC
###############################################################################
def run_motor_imagery_experiment():
    from pylsl import StreamInfo, StreamOutlet
    # Create LSL Marker stream
    try:
        marker_info = StreamInfo(MARKER_STREAM_NAME, "Markers", 1, 0, "string", "marker_id")
        marker_outlet = StreamOutlet(marker_info)
        logging.info("Marker stream created.")
    except Exception as e:
        logging.exception("Failed to create LSL Marker stream:")
        sys.exit(1)

    # PsychoPy setup
    try:
        from psychopy import visual, core, event
        win = visual.Window(
            size=(1024, 768),
            color="black",
            units="norm",
            fullscr=False
        )
        logging.info("PsychoPy window created.")
    except Exception as e:
        logging.exception("Error setting up PsychoPy window:")
        sys.exit(1)

    # Create visual stimuli
    instruction_text = visual.TextStim(win, text="", height=0.15, color="white")
    ready_text = visual.TextStim(win, text="Get Ready", height=0.15, color="white")

    # Generate randomized trial list
    trials = ["right arm", "left arm", "right leg"] * (NUM_TRIALS // 3)
    random.shuffle(trials)
    
    # Show initial instructions
    instruction_text.text = "Motor Imagery Experiment\n\nImagine moving your limb when instructed\n\nPress SPACE to begin"
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    # Main experiment loop
    for trial_num, limb in enumerate(trials, 1):
        # Display get ready message
        ready_text.draw()
        win.flip()
        core.wait(1.0)
        
        # Show instruction (limb)
        instruction_text.text = f"{limb.upper()}"
        instruction_text.draw()
        win.flip()
        core.wait(INSTRUCTION_DURATION)
        
        # Show START cue and send marker
        instruction_text.text = "START"
        instruction_text.draw()
        # Schedule marker to be sent on next flip
        if limb == "right arm": marker_val = MARKER_RIGHT 
        elif limb == "left arm": marker_val = MARKER_LEFT
        elif limb == "right leg": marker_val = MARKER_LEG
        else: marker_val = MARKER_FAIL
        win.callOnFlip(lambda m=marker_val: marker_outlet.push_sample([m]))
        win.flip()
        core.wait(0.25)

        # Show HOLD during imagery period
        instruction_text.text = "HOLD"
        instruction_text.draw()
        win.flip()
        core.wait(IMAGERY_DURATION)
        
        # Show STOP and send stop marker
        instruction_text.text = "STOP"
        instruction_text.draw()
        win.callOnFlip(lambda: marker_outlet.push_sample([MARKER_STOP]))
        win.flip()
        core.wait(1.0)
        
        # Inter-trial interval
        win.flip()  # clear screen
        core.wait(INTER_TRIAL_INTERVAL)
        
        logging.info(f"Completed trial {trial_num}/{len(trials)}: {limb}")
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            break

    # Cleanup
    win.close()
    logging.info("Motor imagery experiment finished.")

###############################################################################
#                                   MAIN
###############################################################################
def main():
    logging.info("Starting Motor Imagery experiment script")
    # Start data collector thread
    stop_event = threading.Event()
    collector = LSLDataCollector(stop_event)
    collector.start()

    # Run the experiment
    run_motor_imagery_experiment()

    # Stop the collector thread gracefully
    logging.info("Experiment complete. Stopping data collector thread...")
    stop_event.set()
    collector.join()

    logging.info("All processes complete. Exiting script.")

if __name__ == "__main__":
    main() 
