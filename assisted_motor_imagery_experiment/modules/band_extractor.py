import threading
import time
import logging
from collections import deque
import numpy as np

from modules.config import *
from modules.calculate_band_power import calculate_band_power


class BandExtractor(threading.Thread):
	"""Threaded band extractor.

	- Consumes rows from `input_deque`. Rows may be either:
	  * tuple (timestamp, sample_array) or
	  * list-like where first element is timestamp and rest are channel values
	  * or just a sequence of channel values (no timestamp)
	- Extracts bandpower (mu and beta) for channels in `BAND_CHANNELS` using
	  a sliding window defined by `BAND_WINDOW_SIZE_SAMPLES` and
	  `BAND_WINDOW_STEP_SAMPLES` from `config.py`.
	- Appends dicts to `self.extracted_bands` deque containing timestamp, mu and beta
	  per-channel arrays.
	- If input deque is empty, waits until new data is added or `stop_event` is set.
	"""

	def __init__(self, input_deque: deque, stop_event: threading.Event, fs: int = FS, max_output_len: int | None = None):
		super().__init__(daemon=True)
		self.input_deque = input_deque
		self.stop_event = stop_event
		self.fs = fs

		self.window_size = BAND_WINDOW_SIZE_SAMPLES
		self.step_size = max(1, BAND_WINDOW_STEP_SAMPLES)
		self.channel_idxs = BAND_CHANNELS

		# buffer holds tuples (timestamp, [selected_channel_values], marker_value)
		self.buffer = deque()
		# last seen marker and its timestamp
		self.last_marker = None
		self.last_marker_ts = None
		self.extracted_bands = deque(maxlen=max_output_len)

		self._log = logging.getLogger(self.__class__.__name__)

	def _parse_row(self, row):
		"""Return (timestamp, sample_list, marker) or (None, sample_list, "")"""
		try:
			# tuple-like (ts, sample_array) - no marker included
			if isinstance(row, tuple) and len(row) == 2:
				ts, sample = row
				return ts, list(sample), ""
			# list-like where first element is timestamp and then channels and optional marker
			if isinstance(row, (list, deque)) and len(row) >= len(EEG_CHANNELS) + 1:
				ts = row[0]
				channels = list(row[1:1+len(EEG_CHANNELS)])
				marker = ""
				# marker exists if there is an extra column after channels
				if len(row) >= 1 + len(EEG_CHANNELS) + 1:
					marker = row[1+len(EEG_CHANNELS)]
				return ts, channels, marker
			# otherwise assume the row itself is just channel samples
			if isinstance(row, (list, tuple, np.ndarray)):
				return None, list(row), ""
		except Exception:
			pass
		return None, [], ""

	def _compute_and_store(self, timestamp_for_window, attach_marker):
		# buffer contains tuples (ts, channels, marker)
		if len(self.buffer) < self.window_size:
			return
		arr = np.array([item[1] for item in list(self.buffer)[: self.window_size]])
		# arr shape: (window_size, n_selected_channels)
		# compute band power per channel for mu and beta bands
		mu_powers = []
		beta_powers = []
		for ch_idx in range(arr.shape[1]):
			signal = arr[:, ch_idx]
			mu_bp, _, _ = calculate_band_power(signal, self.fs, MU_BAND)
			beta_bp, _, _ = calculate_band_power(signal, self.fs, BETA_BAND)
			mu_powers.append(mu_bp)
			beta_powers.append(beta_bp)

		entry = {
			"timestamp": timestamp_for_window,
			"mu": mu_powers,
			"beta": beta_powers,
			"marker": attach_marker,
		}
		self.extracted_bands.append(entry)

	def run(self):
		last_ts = None
		while not (self.stop_event.is_set() and not self.input_deque):
			# consume all available input
			consumed = False
			while self.input_deque:
				consumed = True
				row = None
				try:
					row = self.input_deque.popleft()
				except Exception:
					break
				ts, sample, marker = self._parse_row(row)
				if not sample:
					continue
				# select channels
				try:
					selected = [float(sample[i]) for i in self.channel_idxs]
				except Exception:
					continue
				# update last seen marker if marker present
				if marker is not None and marker != "":
					try:
						mstr = str(marker)
						self.last_marker = mstr
						self.last_marker_ts = ts
					except Exception:
						pass
				# append tuple with marker
				self.buffer.append((ts, selected, marker))
				last_ts = ts

				# compute while we have at least one window
				while len(self.buffer) >= self.window_size:
					# compute window indices and timestamps
					window_items = list(self.buffer)[: self.window_size]
					window_start_ts = window_items[0][0]
					window_end_ts = window_items[-1][0]
					# If the window contains any marker (new trial boundary), drop and skip to after that marker
					marker_in_window_idx = None
					for idx, item in enumerate(window_items):
						m = item[2]
						if m is not None and m != "":
							marker_in_window_idx = idx
							break
					if marker_in_window_idx is not None:
						# update last marker info from the marker inside the window
						self.last_marker = str(window_items[marker_in_window_idx][2])
						self.last_marker_ts = window_items[marker_in_window_idx][0]
						# drop this window and skip buffer to just after the marker sample
						for _ in range(marker_in_window_idx + 1):
							if self.buffer:
								self.buffer.popleft()
						# after skipping, wait for more samples if needed
						break
					# If window starts too close after the start marker, drop it
					if (
						self.last_marker_ts is not None
						and self.last_marker is not None
						and self.last_marker in {MARKER_RIGHT, MARKER_LEFT, MARKER_LEG, MARKER_NO_MOVEMENT}
						and 0 <= (window_start_ts - self.last_marker_ts) < BAND_CHANNEL_PADDING_SEC
					):
						# drop this window by advancing step_size samples
						for _ in range(self.step_size):
							if self.buffer:
								self.buffer.popleft()
						break
					# Otherwise compute and store, attaching last seen marker
					timestamp_for_window = window_end_ts
					self._compute_and_store(timestamp_for_window, self.last_marker)
					# advance by step_size
					for _ in range(self.step_size):
						if self.buffer:
							self.buffer.popleft()
			if not consumed:
				# nothing to consume right now
				if self.stop_event.is_set():
					break
				time.sleep(POLL_SLEEP)

		self._log.info("BandExtractor stopped")


if __name__ == "__main__":
	pass