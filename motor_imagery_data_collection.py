#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ================= FORCE STABLE PSYCHOPY BACKEND (MAC FIX) =================
# pygame backend is broken in newer PsychoPy (abstract methods unimplemented).
# glfw is the correct stable backend for macOS.
# Install with: pip install glfw
# Must be set BEFORE importing psychopy.visual.
from psychopy import prefs
prefs.general['winType'] = 'glfw'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Configuration Parameters ---
EEG_STREAM_NAME = "OpenBCI_EEG"
MARKER_STREAM_NAME = "MI_MarkerStream"

SAMPLE_RATE = 250
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

NUM_TRIALS = 51
READY_DURATION = 1.0
CUE_DURATION = 0.25
INSTRUCTION_DURATION = 2.0
IMAGERY_DURATION = 3.0
INTER_TRIAL_INTERVAL = 3.0

MARKER_RIGHT = "1"
MARKER_LEFT = "2"
MARKER_LEG = "3"
MARKER_STOP = "4"
MARKER_FAIL = "5"

RAW_DATA_DIR = "./data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

MERGE_THRESHOLD = 0.002
POLL_SLEEP = 0.001


# =============================================================================
#                         DATA COLLECTOR
# =============================================================================
class LSLDataCollector(threading.Thread):
    def __init__(self, stop_event):
        super().__init__()
        self.daemon = True
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
                logging.error("No EEG stream found.")
                sys.exit(1)

            self.eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            self.clock_offset = self.eeg_inlet.time_correction()

            logging.info("Resolving Marker LSL stream...")
            marker_streams = resolve_byprop("name", MARKER_STREAM_NAME, timeout=10)
            if not marker_streams:
                logging.error("No Marker stream found.")
                sys.exit(1)

            self.marker_inlet = StreamInlet(marker_streams[0], max_buflen=360)
            logging.info("LSL streams ready.")

        except Exception:
            logging.exception("Stream resolution failed")
            sys.exit(1)

    def flush_remaining(self, writer):
        while self.eeg_buffer or self.marker_buffer:
            if self.eeg_buffer and self.marker_buffer:
                ts_eeg, eeg_data = self.eeg_buffer[0]
                ts_marker, marker = self.marker_buffer[0]

                if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                    writer.writerow([ts_eeg] + eeg_data + [marker])
                    self.eeg_buffer.popleft()
                    self.marker_buffer.popleft()
                elif ts_marker < ts_eeg:
                    writer.writerow([ts_marker] + ([""] * len(EEG_CHANNELS)) + [marker])
                    self.marker_buffer.popleft()
                else:
                    writer.writerow([ts_eeg] + eeg_data + [""])
                    self.eeg_buffer.popleft()

            elif self.eeg_buffer:
                ts, eeg = self.eeg_buffer.popleft()
                writer.writerow([ts] + eeg + [""])

            elif self.marker_buffer:
                ts, m = self.marker_buffer.popleft()
                writer.writerow([ts] + ([""] * len(EEG_CHANNELS)) + [m])

    def run(self):
        self.resolve_streams()

        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lsl_timestamp"] + EEG_CHANNELS + ["marker"])

            while not self.stop_event.is_set():
                try:
                    sample_eeg, ts_eeg = self.eeg_inlet.pull_sample(timeout=0.0)
                    if sample_eeg is not None:
                        self.eeg_buffer.append((ts_eeg, sample_eeg))
                except:
                    pass

                try:
                    sample_marker, ts_marker = self.marker_inlet.pull_sample(timeout=0.0)
                    if sample_marker is not None:
                        ts_marker -= self.clock_offset
                        self.marker_buffer.append((ts_marker, sample_marker[0]))
                except:
                    pass

                while self.eeg_buffer and self.marker_buffer:
                    ts_eeg, eeg = self.eeg_buffer[0]
                    ts_m, m = self.marker_buffer[0]

                    if abs(ts_m - ts_eeg) < MERGE_THRESHOLD:
                        writer.writerow([ts_eeg] + eeg + [m])
                        self.eeg_buffer.popleft()
                        self.marker_buffer.popleft()
                    elif ts_m < ts_eeg:
                        writer.writerow([ts_m] + ([""] * len(EEG_CHANNELS)) + [m])
                        self.marker_buffer.popleft()
                    else:
                        writer.writerow([ts_eeg] + eeg + [""])
                        self.eeg_buffer.popleft()

                time.sleep(POLL_SLEEP)

            self.flush_remaining(writer)


# =============================================================================
#                         EXPERIMENT LOGIC
# =============================================================================
def run_motor_imagery_experiment():
    from pylsl import StreamInfo, StreamOutlet

    marker_info = StreamInfo(MARKER_STREAM_NAME, "Markers", 1, 0, "string", "marker_id")
    marker_outlet = StreamOutlet(marker_info)

    # prefs.general['winType'] = 'glfw' was set at module top — safe to import now
    from psychopy import visual, event

    win = visual.Window(
        size=(1024, 768),
        color="black",
        units="norm",
        fullscr=False,
        allowGUI=True,
        winType='glfw'
    )

    instruction_text = visual.TextStim(win, text="", height=0.15, color="white")
    ready_text = visual.TextStim(win, text="Get Ready", height=0.15, color="white")

    trials = ["right arm", "left arm", "right leg"] * (NUM_TRIALS // 3)
    random.shuffle(trials)

    instruction_text.text = "Press SPACE to begin"
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    for trial_num, limb in enumerate(trials, 1):

        ready_text.draw()
        win.flip()
        time.sleep(READY_DURATION)

        instruction_text.text = limb.upper()
        instruction_text.draw()
        win.flip()
        time.sleep(INSTRUCTION_DURATION)

        instruction_text.text = "START"
        instruction_text.draw()

        if limb == "right arm":
            marker_val = MARKER_RIGHT
        elif limb == "left arm":
            marker_val = MARKER_LEFT
        elif limb == "right leg":
            marker_val = MARKER_LEG
        else:
            marker_val = MARKER_FAIL

        win.callOnFlip(lambda m=marker_val: marker_outlet.push_sample([m]))
        win.flip()
        time.sleep(CUE_DURATION)

        instruction_text.text = "HOLD"
        instruction_text.draw()
        win.flip()
        time.sleep(IMAGERY_DURATION - CUE_DURATION)

        instruction_text.text = "STOP"
        instruction_text.draw()
        win.callOnFlip(lambda: marker_outlet.push_sample([MARKER_STOP]))
        win.flip()
        time.sleep(CUE_DURATION)

        win.flip()
        time.sleep(INTER_TRIAL_INTERVAL - CUE_DURATION)

        logging.info(f"Trial {trial_num} done: {limb}")

        if event.getKeys(["escape"]):
            break

    win.close()


# =============================================================================
#                                   MAIN
# =============================================================================
def main():
    stop_event = threading.Event()

    collector = LSLDataCollector(stop_event)
    collector.start()

    run_motor_imagery_experiment()

    stop_event.set()
    collector.join()

    logging.info("Finished cleanly")


if __name__ == "__main__":
    main()
