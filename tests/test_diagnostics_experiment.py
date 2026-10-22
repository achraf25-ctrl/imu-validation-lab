import json

import numpy as np
import pytest

from imulab.__main__ import main
from imulab.experiment import ExperimentConfig, run_brush_study, run_experiment

EXPECT = {
    "healthy": (set(), "OK"),
    "delayed": ({"DELAY_TOO_LARGE"}, "FAIL"),
    "axis_flipped": ({"AXIS_FLIPPED"}, "FAIL"),
    "axis_remapped": ({"AXIS_REMAPPED"}, "WARN"),
    "biased": ({"GYRO_BIAS_HIGH", "ACCEL_BIAS_HIGH"}, "WARN"),
    "scaled": ({"SCALE_ERROR"}, "FAIL"),
    "misaligned": ({"MISALIGNMENT"}, "FAIL"),
    "lever_arm": ({"STRUCTURED_RESIDUAL"}, "WARN"),
}


@pytest.fixture(scope="module")
def reports():
    return {sc: run_experiment(ExperimentConfig(name=sc, scenario=sc, duration=10.0, n_repeats=2)) for sc in EXPECT}


@pytest.mark.parametrize("scenario", list(EXPECT))
def test_expected_flags_fire(reports, scenario):
    codes_exp, status = EXPECT[scenario]
    codes = {f["code"] for f in reports[scenario]["flags"]}
    assert codes_exp <= codes, f"{scenario}: missing {codes_exp - codes}, got {codes}"
    assert reports[scenario]["status"] == status, (scenario, codes)


def test_healthy_has_no_unexpected_flags(reports):
    assert reports["healthy"]["flags"] == []


def test_axis_remap_is_not_called_a_flip(reports):
    codes = {f["code"] for f in reports["axis_remapped"]["flags"]}
    assert "AXIS_FLIPPED" not in codes
    assert "AXIS_REMAPPED" not in {f["code"] for f in reports["axis_flipped"]["flags"]}


def test_calibration_vs_truth_is_tight(reports):
    for sc in ("healthy", "delayed", "biased", "scaled"):
        v = reports[sc]["calibration_vs_truth"]
        assert abs(v["delay_error_ms"]) < 1.0 and v["gyro_bias_error_dps"] < 0.15


def test_calibration_always_helps_fusion_when_sensor_is_broken(reports):
    for sc in ("axis_flipped", "axis_remapped", "delayed", "biased"):
        r = reports[sc]
        assert r["fusion_calibrated"]["ekf"]["total_rmse_deg"] < r["fusion_raw"]["ekf"]["total_rmse_deg"]


def test_run_writes_artifacts(tmp_path):
    rep = run_experiment(ExperimentConfig(name="t", scenario="delayed", duration=8.0, n_repeats=2), tmp_path / "t")
    assert rep["status"] == "FAIL"
    assert (tmp_path / "t" / "overview.png").stat().st_size > 10_000
    data = json.loads((tmp_path / "t" / "report.json").read_text())
    assert data["calibration"]["delay_ms"] == pytest.approx(60, abs=1.0)
    with np.load(tmp_path / "t" / "signals.npz") as z:
        assert {"gyro_raw", "gyro_cal", "gyro_gt", "err_cal_ekf_total_deg"} <= set(z.files)


def test_brush_study_orders_similarity(tmp_path):
    res = run_brush_study(duration=10.0, out_dir=tmp_path)
    assert np.allclose(np.diag(np.array(res["spectral_cosine"])), 1.0, atol=1e-6)
    assert (tmp_path / "brush_similarity.png").exists()


def test_cli_run_and_list(tmp_path, capsys):
    main(["list"])
    assert "scenarios" in capsys.readouterr().out
    main(["run", "--scenario", "axis_flipped", "--duration", "8", "--repeats", "2", "--out", str(tmp_path)])
    assert "AXIS_FLIPPED" in capsys.readouterr().out
