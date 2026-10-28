import threading
import time
import logging
from collections import deque
import numpy as np

from modules.config import *
from modules.calculate_band_power import calculate_band_power


class BandExtractor(threading.Thread):
	"""Threaded band extractor.

	- Consumes rows from `input_deque` as a list: [timestamp, channel_1, channel_2, ...].
	- Extracts bandpower (mu and beta) for channels in `BAND_CHANNELS` using
	  a sliding window defined by `BAND_WINDOW_SIZE_SAMPLES` and
	  `BAND_WINDOW_STEP_SAMPLES` from `config.py`.
	- Appends dicts to `self.extracted_bands` with timestamp and per-channel band data.
	- Marker handling is intentionally excluded here; use `TrialExtractor` for marker-aware trial labeling.
	"""

	def __init__(self, input_deque: deque, stop_event: threading.Event, fs: int = FS, max_output_len: int | None = None):
		super().__init__(daemon=True)
		self.input_deque = input_deque
		self.stop_event = stop_event
		self.fs = fs

		self.window_size = BAND_WINDOW_SIZE_SAMPLES
		self.step_size = max(1, BAND_WINDOW_STEP_SAMPLES)
		self.channel_idxs = BAND_CHANNELS

		self.buffer = deque()
		self.extracted_bands = deque(maxlen=max_output_len)
		self._log = logging.getLogger(self.__class__.__name__)

	def _parse_row(self, row):
		"""Return (timestamp, sample_list) for a list of the form [timestamp, channel_1, ...]."""
		if not isinstance(row, (list, tuple, np.ndarray)):
			return None, []
		values = list(row)
		if not values:
			return None, []
		try:
			timestamp = float(values[0])
		except (TypeError, ValueError):
			return None, []
		channels = []
		for value in values[1:]:
			try:
				channels.append(float(value))
			except (TypeError, ValueError):
				# Ignore trailing marker values for compatibility with older data streams.
				continue
		return timestamp, channels

	def _compute_and_store(self, timestamp_for_window):
		if len(self.buffer) < self.window_size:
			return
		arr = np.array([item[1] for item in list(self.buffer)[: self.window_size]])
		if arr.size == 0 or arr.ndim != 2:
			return
		mu_powers = []
		beta_powers = []
		for ch_idx in range(arr.shape[1]):
			signal = arr[:, ch_idx]
			mu_bp, _, _ = calculate_band_power(signal, self.fs, MU_BAND)
			beta_bp, _, _ = calculate_band_power(signal, self.fs, BETA_BAND)
			mu_powers.append(mu_bp)
			beta_powers.append(beta_bp)

		row = [timestamp_for_window]
		for mu_bp, beta_bp in zip(mu_powers, beta_powers):
			row.extend([mu_bp, beta_bp])
		self.extracted_bands.append(row)

	def run(self):
		while not (self.stop_event.is_set() and not self.input_deque):
			consumed = False
			while self.input_deque:
				consumed = True
				try:
					row = self.input_deque.popleft()
				except Exception:
					break
				ts, sample = self._parse_row(row)
				if not sample:
					continue
				try:
					selected = [float(sample[i]) for i in self.channel_idxs if i < len(sample)]
				except Exception:
					continue
				self.buffer.append((ts, selected))
				while len(self.buffer) >= self.window_size:
					window_items = list(self.buffer)[: self.window_size]
					window_end_ts = window_items[-1][0]
					self._compute_and_store(window_end_ts)
					for _ in range(self.step_size):
						if self.buffer:
							self.buffer.popleft()
			if not consumed:
				if self.stop_event.is_set():
					break
				time.sleep(POLL_SLEEP)

		self._log.info("BandExtractor stopped")


def marker_to_label(marker_str):
	if marker_str is None:
		return None
	m = str(marker_str).strip()
	if m == "4":
		return 0
	try:
		return int(m)
	except Exception:
		return None


class TrialExtractor(threading.Thread):
	"""Threaded trial extractor.

	Consumes rows shaped like [timestamp, channel_1, ..., marker] and emits labeled bands
	with the form {"timestamp": ts, "bands": {"mu": [...], "beta": [...]}, "marker": marker, "label": label}.
	"""

	def __init__(self, input_deque: deque, stop_event: threading.Event, fs: int = FS, max_output_len: int | None = None):
		super().__init__(daemon=True)
		self.input_deque = input_deque
		self.stop_event = stop_event
		self.fs = fs
		self.window_size = BAND_WINDOW_SIZE_SAMPLES
		self.step_size = max(1, BAND_WINDOW_STEP_SAMPLES)
		self.channel_idxs = BAND_CHANNELS
		self.buffer = deque()
		self.last_marker = None
		self.last_marker_ts = None
		self.extracted_bands = deque(maxlen=max_output_len)
		self._log = logging.getLogger(self.__class__.__name__)

	def _parse_row(self, row):
		if not isinstance(row, (list, tuple, np.ndarray)):
			return None, [], ""
		values = list(row)
		if not values:
			return None, [], ""
		try:
			timestamp = float(values[0])
		except (TypeError, ValueError):
			return None, [], ""
		channels = []
		marker = ""
		for value in values[1:]:
			try:
				channels.append(float(value))
			except (TypeError, ValueError):
				if marker == "":
					marker = str(value)
				continue
		return timestamp, channels, marker

	def _normalize_marker(self, marker):
		if marker is None:
			return None
		marker_str = str(marker).strip()
		return marker_str if marker_str != "" else None

	def _advance_past_marker(self, marker_idx):
		for _ in range(marker_idx + 1):
			if self.buffer:
				self.buffer.popleft()

	def _compute_and_store(self, timestamp_for_window, attach_marker):
		if len(self.buffer) < self.window_size:
			return
		arr = np.array([item[1] for item in list(self.buffer)[: self.window_size]])
		if arr.size == 0 or arr.ndim != 2:
			return
		mu_powers = []
		beta_powers = []
		for ch_idx in range(arr.shape[1]):
			signal = arr[:, ch_idx]
			mu_bp, _, _ = calculate_band_power(signal, self.fs, MU_BAND)
			beta_bp, _, _ = calculate_band_power(signal, self.fs, BETA_BAND)
			mu_powers.append(mu_bp)
			beta_powers.append(beta_bp)

		label = marker_to_label(attach_marker)
		if label is None:
			return
		row = [timestamp_for_window]
		for mu_bp, beta_bp in zip(mu_powers, beta_powers):
			row.extend([mu_bp, beta_bp])
		row.append(label)
		self.extracted_bands.append(row)

	def run(self):
		while not (self.stop_event.is_set() and not self.input_deque):
			consumed = False
			while self.input_deque:
				consumed = True
				try:
					row = self.input_deque.popleft()
				except Exception:
					break
				ts, sample, marker = self._parse_row(row)
				if not sample:
					continue
				try:
					selected = [float(sample[i]) for i in self.channel_idxs if i < len(sample)]
				except Exception:
					continue
				marker = self._normalize_marker(marker)
				if marker is not None:
					self.last_marker = marker
					self.last_marker_ts = ts
				self.buffer.append((ts, selected, marker))
				while len(self.buffer) >= self.window_size:
					window_items = list(self.buffer)[: self.window_size]
					window_start_ts = window_items[0][0]
					window_end_ts = window_items[-1][0]
					marker_in_window_idx = None
					marker_in_window = None
					for idx, item in enumerate(window_items):
						m = self._normalize_marker(item[2])
						if m is not None:
							marker_in_window_idx = idx
							marker_in_window = m
							break
					if marker_in_window_idx is not None:
						self.last_marker = marker_in_window
						self.last_marker_ts = window_items[marker_in_window_idx][0]
						self._advance_past_marker(marker_in_window_idx)
						break
					if (
						self.last_marker is not None
						and self.last_marker_ts is not None
						and self.last_marker in {MARKER_RIGHT, MARKER_LEFT, MARKER_LEG, MARKER_NO_MOVEMENT}
						and 0 <= (window_start_ts - self.last_marker_ts) < BAND_CHANNEL_PADDING_SEC
					):
						for _ in range(self.step_size):
							if self.buffer:
								self.buffer.popleft()
						break
					self._compute_and_store(window_end_ts, self.last_marker)
					for _ in range(self.step_size):
						if self.buffer:
							self.buffer.popleft()
			if not consumed:
				if self.stop_event.is_set():
					break
				time.sleep(POLL_SLEEP)

		self._log.info("TrialExtractor stopped")


if __name__ == "__main__":
	pass