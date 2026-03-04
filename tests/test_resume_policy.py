import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from train import _config_diff  # noqa: E402


def test_config_diff_empty_when_same():
    a = {"train": {"epochs": 1}, "system": {"seed": 0}}
    b = {"train": {"epochs": 1}, "system": {"seed": 0}}
    assert _config_diff(a, b) == []


def test_config_diff_detects_changes():
    a = {"train": {"epochs": 1}, "system": {"seed": 0}}
    b = {"train": {"epochs": 2}, "system": {"seed": 0}}
    diffs = _config_diff(a, b)
    assert any("train.epochs" in d for d in diffs)
