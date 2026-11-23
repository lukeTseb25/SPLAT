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

	def __init__(self, input_deque: deque, stop_event: threading.Event, fs: int = FS, max_output_len: int | None = None, channel_idxs=None):
		super().__init__(daemon=True)
		self.input_deque = input_deque
		self.stop_event = stop_event
		self.fs = fs

		self.window_size = BAND_WINDOW_SIZE_SAMPLES
		self.step_size = max(1, BAND_WINDOW_STEP_SAMPLES)
		self.channel_idxs = list(channel_idxs) if channel_idxs is not None else None

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
					if self.channel_idxs is None:
						selected = [float(value) for value in sample]
					else:
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


# Do not change this fuction's mapping or I will tweak tf out
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
	with the form {"timestamp": ts, "bands": {"mu": [...], "beta": [...]}, "label": label}.
	"""

	def __init__(self, input_deque: deque, stop_event: threading.Event, fs: int = FS, max_output_len: int | None = None, channel_idxs=None):
		super().__init__(daemon=True)
		self.input_deque = input_deque
		self.stop_event = stop_event
		self.fs = fs
		self.window_size = BAND_WINDOW_SIZE_SAMPLES
		self.step_size = max(1, BAND_WINDOW_STEP_SAMPLES)
		self.channel_idxs = list(channel_idxs) if channel_idxs is not None else None
		self.buffer = deque()
		self.last_marker = None
		# Dov fix: Track EEG sample positions instead of a separate marker timestamp.
		self.last_marker_sample = None
		self.sample_count = 0
		self.buffer_start_sample = 0
		self.samples_to_skip = 0
		self.extracted_bands = deque(maxlen=max_output_len)
		self._log = logging.getLogger(self.__class__.__name__)
		self.start_timestamp = None
		self.last_progress_second = -1

	def _format_progress(self, elapsed_seconds):
		elapsed_seconds = max(0, int(elapsed_seconds))
		minutes, seconds = divmod(elapsed_seconds, 60)
		return f"{minutes:02d}:{seconds:02d}"

	def _update_progress(self, timestamp):
		if self.start_timestamp is None:
			self.start_timestamp = timestamp
			self.last_progress_second = -1

		elapsed_seconds = int(max(0, timestamp - self.start_timestamp))
		if elapsed_seconds > self.last_progress_second:
			self.last_progress_second = elapsed_seconds
			print(
				f"\rTrialExtractor processed {self._format_progress(elapsed_seconds)} "
				f"| buffer {len(self.buffer)}/{self.window_size}",
				end="",
				flush=True,
			)

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
		if len(values) == 1:
			return timestamp, [], ""
		marker = values[-1]
		marker = "" if marker is None else str(marker).strip()
		channels = []
		for value in values[1:-1]:
			try:
				channels.append(float(value))
			except (TypeError, ValueError):
				continue
		return timestamp, channels, marker

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

	def _process_available_windows(self):
		"""Process every complete window remaining in the buffer."""
		while len(self.buffer) >= self.window_size:
			padding_samples = max(0, int(round(BAND_CHANNEL_PADDING_SEC * self.fs)))
			if (
				self.last_marker is not None
				and self.last_marker_sample is not None
				and self.buffer_start_sample >= self.last_marker_sample + padding_samples
			):
				window_end_ts = self.buffer[self.window_size - 1][0]
				self._compute_and_store(window_end_ts, self.last_marker)
			popped = min(self.step_size, len(self.buffer))
			for _ in range(popped):
				self.buffer.popleft()
				self.buffer_start_sample += 1
			self.samples_to_skip += self.step_size - popped

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

				# Hrishi Edited: Safety check but unlikely to happen
				if ts is None:
					continue

				self._update_progress(ts)

				# Hrishi Edited: Record the boundary first, so marker-only rows still count.
				label = marker_to_label(marker)
				if label is not None:
					self.last_marker = label
					self.last_marker_sample = self.sample_count
					self._log.info("marker %r -> label %s at ts=%.4f (sample %d)",
					               marker, label, ts, self.sample_count)

				# Hrishi Edited: After recording boundaries
				if not sample:
					continue
				self.sample_count += 1
				if self.samples_to_skip:
					self.samples_to_skip -= 1
					self.buffer_start_sample += 1
					continue
				try:
					if self.channel_idxs is None:
						selected = [float(value) for value in sample]
					else:
						selected = [float(sample[i]) for i in self.channel_idxs if i < len(sample)]
				except Exception:
					continue

				self.buffer.append((ts, selected, label))
				self._process_available_windows()

			if not consumed:
				if self.stop_event.is_set():
					break
				time.sleep(POLL_SLEEP)

		# A stop signal means no more input is coming: drain every complete window
		# already in the buffer instead of waiting for additional samples.
		self._process_available_windows()
		if self.start_timestamp is not None:
			print()
		self._log.info("TrialExtractor stopped")


if __name__ == "__main__":
	pass
