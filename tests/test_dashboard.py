import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from pathlib import Path

from streamlit.testing.v1 import AppTest

from imulab.experiment import ExperimentConfig, run_experiment

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_renders_without_exception(tmp_path, monkeypatch):
    for sc in ("healthy", "axis_flipped"):
        run_experiment(ExperimentConfig(name=sc, scenario=sc, duration=8.0, n_repeats=2), tmp_path / sc)
    monkeypatch.setenv("IMULAB_RUNS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("AXIS_FLIPPED" in str(m.value) for m in at.error)       # flag surfaced in the run-detail tab? (selected run may differ)
