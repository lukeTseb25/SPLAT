import threading
import time
from time import perf_counter
import random
import csv
import os
import sys
from collections import deque
from datetime import datetime
import numpy as np
import logging
from modules.config import *

def run_motor_imagery_experiment():
	from pylsl import StreamInfo, StreamOutlet
	# Create LSL Marker stream
	try:
		marker_info = StreamInfo(MARKER_STREAM_NAME, "Markers", 1, 0, "string", "marker_id") # type: ignore
		marker_outlet = StreamOutlet(marker_info)
		logging.info("Marker stream created.")
	except Exception as e:
		logging.exception("Failed to create LSL Marker stream:")
		sys.exit(1)

	# PsychoPy setup
	try:
		from psychopy import visual, event
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
		# Display get ready message with wall clock timing
		ready_start = perf_counter()
		ready_text.draw()
		win.flip()
		ready_elapsed = perf_counter() - ready_start
		while ready_elapsed < READY_DURATION:
			ready_elapsed = perf_counter() - ready_start
			if ready_elapsed >= READY_DURATION:
				break
			time.sleep((READY_DURATION - ready_elapsed)*0.95)

		# Display instruction
		instr_start = perf_counter()
		instruction_text.text = f"{limb.upper()}"
		instruction_text.draw()
		win.flip()
		instr_elapsed = perf_counter() - instr_start
		while instr_elapsed < INSTRUCTION_DURATION:
			instr_elapsed = perf_counter() - instr_start
			if instr_elapsed >= INSTRUCTION_DURATION:
				break
			time.sleep((INSTRUCTION_DURATION - instr_elapsed)*0.95)

		# Display START cue and send marker with precise timing
		start_start = perf_counter()
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

		start_elapsed = perf_counter() - start_start
		while start_elapsed < CUE_DURATION:
			start_elapsed = perf_counter() - start_start
			if start_elapsed >= CUE_DURATION:
				break
			time.sleep((CUE_DURATION - start_elapsed)*0.95)

		# Show HOLD for the rest of the imagery duration minus the time already spent on START cue
		hold_start = perf_counter()
		instruction_text.text = "HOLD"
		instruction_text.draw()
		win.flip()
		
		hold_elapsed = perf_counter() - hold_start
		while hold_elapsed < (IMAGERY_DURATION - CUE_DURATION):
			hold_elapsed = perf_counter() - hold_start
			if hold_elapsed >= (IMAGERY_DURATION - CUE_DURATION):
				break
			time.sleep((IMAGERY_DURATION - CUE_DURATION - hold_elapsed)*0.95)
		
		# Show STOP cue and send stop marker; STOP counts toward INTER_TRIAL_INTERVAL
		stop_start = perf_counter()
		instruction_text.text = "STOP"
		instruction_text.draw()
		win.callOnFlip(lambda: marker_outlet.push_sample([MARKER_STOP]))
		win.flip()
		
		stop_elapsed = perf_counter() - stop_start
		while stop_elapsed < CUE_DURATION:
			stop_elapsed = perf_counter() - stop_start
			if stop_elapsed >= CUE_DURATION:
				break
			time.sleep((CUE_DURATION - stop_elapsed)*0.95)

		# Show wait for the rest of the inter-trial interval minus the time already spent on STOP cue
		instr_start = perf_counter()
		win.flip()
		instr_elapsed = perf_counter() - instr_start
		while instr_elapsed < (INTER_TRIAL_INTERVAL - CUE_DURATION):
			instr_elapsed = perf_counter() - instr_start
			if instr_elapsed >= (INTER_TRIAL_INTERVAL - CUE_DURATION):
				break
			time.sleep((INTER_TRIAL_INTERVAL - CUE_DURATION - instr_elapsed)*0.95)

		logging.info(f"Completed trial {trial_num}/{len(trials)}: {limb}")

		# Check for quit
		if event.getKeys(keyList=["escape"]):
			break

	# Cleanup
	win.close()
	logging.info("Motor imagery experiment finished.")
