import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

module_path = Path(__file__).resolve().parents[1] / "modules" / "motor_imagery_experiment.py"
spec = importlib.util.spec_from_file_location("motor_imagery_experiment", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
wait_for_duration = module.wait_for_duration


def test_wait_for_duration_uses_short_poll_interval():
    start = 0.0
    fake_time_values = iter([0.0, 0.3])

    with patch.object(module, "perf_counter", side_effect=lambda: next(fake_time_values)):
        with patch.object(module.time, "sleep") as mock_sleep:
            wait_for_duration(start, duration=0.2, poll_interval=0.005)

    assert mock_sleep.call_count > 0
    assert max(call.args[0] for call in mock_sleep.call_args_list) <= 0.01
