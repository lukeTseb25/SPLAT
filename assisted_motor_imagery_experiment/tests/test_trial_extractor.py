import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import ModuleType
from threading import Event
from unittest.mock import patch


MODULES_DIR = Path(__file__).resolve().parents[1] / "modules"
sys.path.insert(0, str(MODULES_DIR.parent))

spec = importlib.util.spec_from_file_location(
    "band_extractor", MODULES_DIR / "band_extractor.py"
)
band_extractor = importlib.util.module_from_spec(spec)
band_power_module = ModuleType("modules.calculate_band_power")
band_power_module.calculate_band_power = lambda *_args: (0.0, None, None)
with patch.dict(sys.modules, {"modules.calculate_band_power": band_power_module}):
    spec.loader.exec_module(band_extractor)


def _sample(index, marker=""):
    return [float(index), float(index), marker]


def _run_extractor(rows, window_size, step_size, padding_seconds=0.0):
    assert len(rows) <= 4000
    with (
        patch.object(band_extractor, "BAND_WINDOW_SIZE_SAMPLES", window_size),
        patch.object(band_extractor, "BAND_WINDOW_STEP_SAMPLES", step_size),
        patch.object(band_extractor, "BAND_CHANNEL_PADDING_SEC", padding_seconds),
        patch.object(
            band_extractor,
            "calculate_band_power",
            side_effect=lambda signal, *_args: (float(signal[0]), None, None),
        ),
    ):
        extractor = band_extractor.TrialExtractor(
            deque(rows), Event(), channel_idxs=[0]
        )
        extractor.stop_event.set()
        extractor.run()
        return list(extractor.extracted_bands)


def test_trial_extractor_waits_for_full_stride_and_checks_skipped_markers():
    rows = [[0.0, "", "1"]]
    for index in range(312):
        if index == 100:
            rows.append([float(index), "", "2"])
        rows.append(_sample(index))

    extracted = _run_extractor(rows, window_size=62, step_size=250)

    assert extracted == [
        [61.0, 0.0, 0.0, 1],
        [311.0, 250.0, 250.0, 2],
    ]


def test_trial_extractor_preserves_overlapping_window_stride():
    rows = [[0.0, "", "1"]]
    rows.extend(_sample(index) for index in range(8))

    extracted = _run_extractor(
        rows, window_size=4, step_size=2, padding_seconds=0.004
    )

    assert extracted[0][0] == 5.0
    assert extracted[0][1] == 2.0
    assert extracted[0][-1] == 1
