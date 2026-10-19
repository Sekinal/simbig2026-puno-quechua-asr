"""Filters of scripts/pseudo_label.py (hybrid expert-iteration labels)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "pseudo_label", Path(__file__).parents[1] / "scripts" / "pseudo_label.py"
)
pseudo_label = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pseudo_label)


def test_repeats_flags_looping_but_not_reduplication():
    loop = "kan una kan una kan una kan una kan una kan una"
    assert pseudo_label._repeats(loop)
    assert not pseudo_label._repeats("sapa sapa p'unchaw llamk'ayta qallarinchik")
    assert not pseudo_label._repeats("short")
