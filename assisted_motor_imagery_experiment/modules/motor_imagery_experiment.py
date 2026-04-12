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

def run_motor_imagery_experiment(neurofeedback_processor=None):
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
		from psychopy import visual, event, core
		win = visual.Window(
			size=(1024, 768),
			color="black",
			units="norm",
			fullscr=False,
			allowGUI=False,  # Disable GUI for better performance
			waitBlanking=False  # Disable waiting for screen refresh for lower latency
		)
		win.recordFrameIntervals = True  # Track frame timing
		logging.info("PsychoPy window created.")
	except Exception as e:
		logging.exception("Error setting up PsychoPy window:")
		sys.exit(1)

	# Create visual stimuli
	instruction_text = visual.TextStim(win, text="", height=0.15, color="white")
	ready_text = visual.TextStim(win, text="Get Ready", height=0.15, color="white")
	
	# Create channel indicator (only needed if using neurofeedback)
	channel_text = None
	if neurofeedback_processor:
		channel_text = visual.TextStim(win, text="", height=0.05, color="white", pos=(0, -0.7))
	
	# Create feedback bars if neurofeedback processor is provided
	mu_meter_bg = None
	mu_meter_fill = None
	mu_meter_text = None
	beta_meter_bg = None
	beta_meter_fill = None
	beta_meter_text = None
	feedback_bar_bg = None
	feedback_bar = None
	
	if neurofeedback_processor:
		# Create Mu band feedback meter (right side)
		meter_width = 0.2
		meter_height = 0.6
		mu_meter_pos = (0.7, 0)  # Position on right side of screen
		mu_meter_bg = visual.Rect(
			win=win, 
			width=meter_width, 
			height=meter_height, 
			pos=mu_meter_pos, 
			fillColor='gray', 
			lineColor='white',
			autoLog=False
		)
		mu_meter_fill = visual.Rect(
			win=win, 
			width=meter_width, 
			height=0, 
			pos=mu_meter_pos, 
			fillColor='green',
			autoLog=False
		)
		mu_meter_text = visual.TextStim(
			win=win, 
			text="Mu Band Power", 
			height=0.05,
			pos=(mu_meter_pos[0], mu_meter_pos[1] + meter_height/2 + 0.1),
			autoLog=False
		)
		
		# Create Beta band feedback meter (left side)
		beta_meter_pos = (-0.7, 0)  # Position on left side of screen
		beta_meter_bg = visual.Rect(
			win=win, 
			width=meter_width, 
			height=meter_height, 
			pos=beta_meter_pos, 
			fillColor='gray', 
			lineColor='white',
			autoLog=False
		)
		beta_meter_fill = visual.Rect(
			win=win, 
			width=meter_width, 
			height=0, 
			pos=beta_meter_pos, 
			fillColor='blue',
			autoLog=False
		)
		beta_meter_text = visual.TextStim(
			win=win, 
			text="Beta Band Power", 
			height=0.05,
			pos=(beta_meter_pos[0], beta_meter_pos[1] + meter_height/2 + 0.1),
			autoLog=False
		)
		
		# Create central feedback bar
		feedback_bar_bg = visual.Rect(
			win=win,
			width=FEEDBACK_BAR_MAX_WIDTH,
			height=0.2,
			fillColor="gray",
			lineColor="white",
			pos=(0, -0.5),
			autoLog=False
		)
		
		feedback_bar = visual.Rect(
			win=win,
			width=0,  # Will be updated based on ERD
			height=0.15,
			fillColor="green",
			lineColor=None,
			pos=(-FEEDBACK_BAR_MAX_WIDTH/2, -0.5),  # Centered position
			anchor="left",  # Anchor to left for width updates
			autoLog=False
		)

	# Generate randomized trial list
	trials = ["right arm", "left arm", "right leg"] * (NUM_TRIALS // 3)
	random.shuffle(trials)
	
	# Show initial instructions
	if neurofeedback_processor:
		instruction_text.text = "Motor Imagery Experiment with Neurofeedback\n\nImagine moving your limb when instructed\n\nThe feedback bars will show your brain activity\n\nPress SPACE to begin"
	else:
		instruction_text.text = "Motor Imagery Experiment\n\nImagine moving your limb when instructed\n\nPress SPACE to begin"
	instruction_text.draw()
	win.flip()
	event.waitKeys(keyList=["space"])
	
	# Initial baseline collection if using neurofeedback
	if neurofeedback_processor:
		instruction_text.text = "Collecting baseline\n\nPlease relax and remain still"
		neurofeedback_processor.start_initial_baseline_collection()
		
		from psychopy import core
		baseline_clock = core.Clock()
		baseline_clock.reset()
		countdown_text = visual.TextStim(win, text="", height=0.2, color="white", pos=(0, 0.2))
		
		# Countdown for initial baseline
		while baseline_clock.getTime() < INITIAL_BASELINE_DURATION:
			remaining = int(INITIAL_BASELINE_DURATION - baseline_clock.getTime())
			instruction_text.draw()
			countdown_text.text = f"{remaining}s"
			countdown_text.draw()
			win.flip()
			
			# Check for quit
			if event.getKeys(keyList=["escape"]):
				win.close()
				neurofeedback_processor.stop_initial_baseline_collection()
				return
		
		neurofeedback_processor.stop_initial_baseline_collection()
		
		# Indicate baseline completion
		instruction_text.text = "Baseline Completed\n\nThe experiment will begin shortly"
		instruction_text.draw()
		win.flip()
		core.wait(2.0)

	# Main experiment loop
	for trial_num, limb in enumerate(trials, 1):
		# Set active hand and channel for neurofeedback
		if neurofeedback_processor:
			if limb == "right arm":
				neurofeedback_processor.set_active_hand("right")
				channel_text.text = "Using CH3 (Contralateral Hemisphere)"
			elif limb == "left arm":
				neurofeedback_processor.set_active_hand("left")
				channel_text.text = "Using CH6 (Contralateral Hemisphere)"
			elif limb == "right leg":
				neurofeedback_processor.set_active_hand("right")
				channel_text.text = "Using CH3 (Contralateral Hemisphere)"
		
		# Display get ready message with wall clock timing
		ready_start = perf_counter()
		ready_text.draw()
		if neurofeedback_processor and channel_text:
			channel_text.draw()
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
		if neurofeedback_processor and channel_text:
			channel_text.draw()
		win.flip()
		instr_elapsed = perf_counter() - instr_start
		while instr_elapsed < INSTRUCTION_DURATION:
			instr_elapsed = perf_counter() - instr_start
			if instr_elapsed >= INSTRUCTION_DURATION:
				break
			time.sleep((INSTRUCTION_DURATION - instr_elapsed)*0.95)
		
		# Start collecting baseline data before START cue for neurofeedback
		if neurofeedback_processor:
			from psychopy import core
			neurofeedback_processor.start_baseline_collection()
			baseline_clock = core.Clock()
			baseline_clock.reset()
			
			# Show instruction during baseline collection (2 seconds)
			while baseline_clock.getTime() < abs(BASELINE_START - BASELINE_END):  # 2 seconds
				instruction_text.draw()
				if channel_text:
					channel_text.draw()
				win.flip()
				
				# Check for quit
				if event.getKeys(keyList=["escape"]):
					win.close()
					return
			
			neurofeedback_processor.stop_baseline_collection()

		# Display START cue and send marker with precise timing
		start_start = perf_counter()
		instruction_text.text = "START"
		instruction_text.draw()
		if neurofeedback_processor and channel_text:
			channel_text.draw()

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
		
		# Start real-time processing after START cue for neurofeedback
		if neurofeedback_processor:
			neurofeedback_processor.start_processing()

		start_elapsed = perf_counter() - start_start
		while start_elapsed < CUE_DURATION:
			start_elapsed = perf_counter() - start_start
			if start_elapsed >= CUE_DURATION:
				break
			time.sleep((CUE_DURATION - start_elapsed)*0.95)

		# Show HOLD for the rest of the imagery duration minus the time already spent on START cue
		hold_start = perf_counter()
		if neurofeedback_processor:
			from psychopy import core
			display_timer = core.Clock()
			display_timer.reset()
			meter_width = 0.2
			meter_height = 0.6
			mu_meter_pos = (0.7, 0)
			beta_meter_pos = (-0.7, 0)
		
		instruction_text.text = "HOLD"
		hold_elapsed = perf_counter() - hold_start
		while hold_elapsed < (IMAGERY_DURATION - CUE_DURATION):
			hold_elapsed = perf_counter() - hold_start
			
			if neurofeedback_processor:
				# Update feedback bars at regular intervals
				if display_timer.getTime() >= DISPLAY_UPDATE_INTERVAL:
					display_timer.reset()
					
					# Get current ERD values
					smoothed_mu = neurofeedback_processor.get_smoothed_erd_mu()
					smoothed_beta = neurofeedback_processor.get_smoothed_erd_beta()
					
					# Update central feedback bar
					bar_width = min(smoothed_mu * FEEDBACK_BAR_MAX_WIDTH, FEEDBACK_BAR_MAX_WIDTH)
					feedback_bar.width = bar_width
					
					# Update mu meter height
					mu_meter_fill.height = smoothed_mu * meter_height
					mu_meter_fill.pos = (mu_meter_pos[0], mu_meter_pos[1] - (meter_height/2) + (smoothed_mu * meter_height/2))
					
					# Update beta meter height
					beta_meter_fill.height = smoothed_beta * meter_height
					beta_meter_fill.pos = (beta_meter_pos[0], beta_meter_pos[1] - (meter_height/2) + (smoothed_beta * meter_height/2))
					
					# Update central bar color based on mu ERD value
					if smoothed_mu < 0.2:
						feedback_bar.fillColor = "red"
					elif smoothed_mu < 0.6:
						feedback_bar.fillColor = "yellow"
					else:
						feedback_bar.fillColor = "green"
					
					# Draw instruction and all elements
					instruction_text.text = "HOLD"
					instruction_text.draw()
					
					# Draw meters
					mu_meter_bg.draw()
					mu_meter_fill.draw()
					mu_meter_text.draw()
					beta_meter_bg.draw()
					beta_meter_fill.draw()
					beta_meter_text.draw()
					
					# Draw central feedback bar
					feedback_bar_bg.draw()
					feedback_bar.draw()
					
					# Draw channel indicator
					channel_text.draw()
					
					win.flip()
				
				# Check for quit during feedback update
				if event.getKeys(keyList=["escape"]):
					win.close()
					return
			else:
				# Without neurofeedback, just display HOLD text
				instruction_text.draw()
				win.flip()
				
				# Check for quit
				if event.getKeys(keyList=["escape"]):
					win.close()
					return
			
			if hold_elapsed >= (IMAGERY_DURATION - CUE_DURATION):
				break
			time.sleep((IMAGERY_DURATION - CUE_DURATION - hold_elapsed)*0.95)
		
		# Stop real-time processing for neurofeedback
		if neurofeedback_processor:
			neurofeedback_processor.stop_processing()
		
		# Show STOP cue and send stop marker; STOP counts toward INTER_TRIAL_INTERVAL
		stop_start = perf_counter()
		instruction_text.text = "STOP"
		instruction_text.draw()
		if neurofeedback_processor and channel_text:
			channel_text.draw()
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
		
		# Report frame timing stats if using neurofeedback
		if neurofeedback_processor and win.recordFrameIntervals:
			frame_times = win.frameIntervals
			if frame_times:
				mean_frame_time = np.mean(frame_times)
				std_frame_time = np.std(frame_times)
				logging.info(f"Frame timing: Mean={mean_frame_time*1000:.1f}ms, Std={std_frame_time*1000:.1f}ms")
				win.frameIntervals = []  # Reset for next trial

		# Check for quit
		if event.getKeys(keyList=["escape"]):
			break

	# Cleanup
	win.close()
	logging.info("Motor imagery experiment finished.")
