import csv
import os
import threading
import time
from collections import deque


class CSVStreamWriter(threading.Thread):
    """Consumes rows from an input deque and writes them to disk as a CSV stream."""

    def __init__(self, input_deque: deque, csv_path: str, header=None, stop_event: threading.Event | None = None, flush_interval: float = 0.01):
        super().__init__(daemon=True)
        self.input_deque = input_deque
        self.csv_path = csv_path
        self.header = header
        self.stop_event = stop_event or threading.Event()
        self.flush_interval = flush_interval
        self._file = None
        self._writer = None

    def _open_file(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        if self.header is not None:
            self._writer.writerow(self.header)

    def run(self):
        self._open_file()
        try:
            while not (self.stop_event.is_set() and not self.input_deque):
                while self.input_deque:
                    try:
                        row = self.input_deque.popleft()
                    except Exception:
                        break
                    if row is None:
                        continue
                    if isinstance(row, dict):
                        if self.header is not None:
                            row = [row.get(column) for column in self.header]
                        else:
                            row = list(row.values())
                    self._writer.writerow(row)
                    self._file.flush()
                if self.stop_event.is_set() and not self.input_deque:
                    break
                time.sleep(self.flush_interval)
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
                self._writer = None
