"""Automatic health diagnostics: turn calibration / metric numbers into flags.

Every flag has a code, a severity (info / warn / fail), a human message, the measured
value and the threshold that fired. Thresholds are configurable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Thresholds:
    delay_warn_ms: float = 20.0
    delay_fail_ms: float = 50.0
    misalign_warn_deg: float = 1.0
    misalign_fail_deg: float = 3.0
    scale_warn: float = 0.01
    scale_fail: float = 0.03
    gyro_bias_warn_dps: float = 0.5
    gyro_bias_fail_dps: float = 2.0
    accel_bias_warn: float = 0.1
    accel_bias_fail: float = 0.3
    gyro_noise_warn_dps: float = 0.5
    gyro_noise_fail_dps: float = 1.0
    accel_noise_warn: float = 0.2
    fit_r2_warn: float = 0.98
    fit_r2_fail: float = 0.90
    delay_conf_warn: float = 0.95
    structured_autocorr: float = 0.9   # residual lag-1 autocorrelation above this = systematic, not noise
    structured_min_rms: float = 0.03   # ...and only if the residual is above this accel level (m/s^2)
    replay_pos_warn_mm: float = 5.0
    replay_pos_fail_mm: float = 10.0
    heading_drift_warn_deg: float = 5.0
    heading_drift_fail_deg: float = 15.0
    tilt_warn_deg: float = 3.0


def _flag(code, severity, message, value=None, threshold=None) -> dict:
    return dict(code=code, severity=severity, message=message, value=value, threshold=threshold)


def _tier(val, warn, fail, code, msg_fn, flags, higher_is_bad=True):
    if (val >= fail) if higher_is_bad else (val <= fail):
        flags.append(_flag(code, "fail", msg_fn(val), val, fail))
    elif (val >= warn) if higher_is_bad else (val <= warn):
        flags.append(_flag(code, "warn", msg_fn(val), val, warn))


def diagnose(report: dict, expected_convention: str = "x,y,z", th: Thresholds | None = None) -> list[dict]:
    """`report` needs keys: calibration (Calibration.summary()), replay, plan, fusion_calibrated."""
    th = th or Thresholds()
    flags: list[dict] = []
    cal, g, a = report["calibration"], report["calibration"]["gyro"], report["calibration"]["accel"]

    # --- axes
    if g["convention"] == "ambiguous":
        flags.append(_flag("AXIS_AMBIGUOUS", "fail", "Could not identify an axis convention (motion too planar / low excitation)."))
    elif g["det"] < 0:
        flags.append(_flag("AXIS_FLIPPED", "fail",
                           f"Handedness mismatch: estimated convention '{g['convention']}' is a reflection (one axis sign inverted)."
                           " Fix the driver / axis convention; no pure rotation can correct this.", g["convention"], expected_convention))
    elif g["convention"].replace(" ", "") != expected_convention.replace(" ", ""):
        flags.append(_flag("AXIS_REMAPPED", "warn",
                           f"IMU axes are remapped: estimated '{g['convention']}' vs expected '{expected_convention}'. "
                           "Correctable with the calibration rotation.", g["convention"], expected_convention))
    if g["convention"] != a["convention"] and "ambiguous" not in (g["convention"], a["convention"]):
        flags.append(_flag("AXIS_INCONSISTENT", "fail",
                           f"Gyro ('{g['convention']}') and accelerometer ('{a['convention']}') disagree on the axis convention."))

    # --- timing
    d_ms = cal["delay_ms"]
    _tier(abs(d_ms), th.delay_warn_ms, th.delay_fail_ms, "DELAY_TOO_LARGE",
          lambda v: f"Effective IMU delay {d_ms:+.1f} ms (transport delay + filter group delay).", flags)
    if cal["delay_peak_corr"] < th.delay_conf_warn:
        flags.append(_flag("DELAY_LOW_CONFIDENCE", "warn",
                           "Delay estimate has low explained variance; motion may be too slow or too noisy.",
                           cal["delay_peak_corr"], th.delay_conf_warn))

    # --- gyro model
    _tier(g["misalign_deg"] if g["misalign_deg"] == g["misalign_deg"] else 0.0, th.misalign_warn_deg, th.misalign_fail_deg,
          "MISALIGNMENT", lambda v: f"Residual mounting misalignment of {v:.2f} deg after removing axis permutation.", flags)
    # accel gains are only trusted when the accel model fits without systematic residual
    # (a lever arm, for instance, distorts the fitted accel gains without any real scale error)
    accel_structured = a["resid_structured_rms"] >= th.structured_min_rms
    scales = g["scales"] + ([] if accel_structured else a["scales"])
    sc = max(abs(x - 1.0) for x in scales)
    _tier(sc, th.scale_warn, th.scale_fail, "SCALE_ERROR",
          lambda v: f"Per-axis scale error up to {100 * v:.1f} % (gyro scales {g['scales']}).", flags)
    gb = max(abs(x) for x in g["bias_dps"])
    _tier(gb, th.gyro_bias_warn_dps, th.gyro_bias_fail_dps, "GYRO_BIAS_HIGH",
          lambda v: f"Gyro bias up to {v:.2f} deg/s (heading drifts ~{v * 60:.0f} deg/min).", flags)
    ab = max(abs(x) for x in a["bias"])
    _tier(ab, th.accel_bias_warn, th.accel_bias_fail, "ACCEL_BIAS_HIGH",
          lambda v: f"Accelerometer bias up to {v:.2f} m/s^2.", flags)
    _tier(g["resid_rms_dps"], th.gyro_noise_warn_dps, th.gyro_noise_fail_dps, "NOISE_HIGH",
          lambda v: f"Gyro residual noise {v:.2f} deg/s rms after calibration.", flags)
    if a["resid_rms"] >= th.accel_noise_warn and not accel_structured:
        flags.append(_flag("NOISE_HIGH", "warn", f"Accelerometer residual noise {a['resid_rms']:.2f} m/s^2 rms.",
                           a["resid_rms"], th.accel_noise_warn))
    _tier(g["r2"], th.fit_r2_warn, th.fit_r2_fail, "FRAME_FIT_POOR",
          lambda v: f"Linear frame model explains only {100 * v:.1f} % of gyro variance.", flags, higher_is_bad=False)
    if accel_structured and g["r2"] >= th.fit_r2_warn:
        flags.append(_flag("STRUCTURED_RESIDUAL", "warn",
                           "Accelerometer residual is systematic (not white noise) while the gyro fits well: suspect a lever arm "
                           "(IMU not at the tracked point), a bandwidth limit or vibration. Accel gains are not trusted.",
                           a["resid_rms"], th.structured_min_rms))

    # --- replay / plan
    pe = report["replay"]["pos_rmse_mm"]
    _tier(pe, th.replay_pos_warn_mm, th.replay_pos_fail_mm, "REPLAY_ACCURACY_LOW",
          lambda v: f"Replay position error {v:.1f} mm rms vs commanded trajectory.", flags)
    pl = report["plan"]
    if pl["joint_limit_violations"] > 0:
        flags.append(_flag("JOINT_LIMIT_VIOLATION", "fail", f"{pl['joint_limit_violations']} joint-limit violations in the plan.",
                           pl["joint_limit_violations"], 0))
    if pl["retimed"]:
        flags.append(_flag("TRAJECTORY_SLOWED", "info",
                           f"Trajectory slowed x{pl['slowdown_factor']:.2f} to respect joint speed/acceleration limits "
                           f"(speed {pl['speed_ratio_before']:.2f}x, accel {pl['accel_ratio_before']:.2f}x of limit).",
                           pl["slowdown_factor"], 1.0))
    if pl["ik_converged_fraction"] < 1.0:
        flags.append(_flag("IK_FAILURE", "fail", f"IK converged for only {100 * pl['ik_converged_fraction']:.1f} % of samples.",
                           pl["ik_converged_fraction"], 1.0))

    # --- fusion (on calibrated data)
    ek = report["fusion_calibrated"]["ekf"]
    _tier(ek["heading_drift_deg"], th.heading_drift_warn_deg, th.heading_drift_fail_deg, "FUSION_HEADING_DRIFT",
          lambda v: f"EKF heading drifted {v:.1f} deg by the end of the run (no magnetometer).", flags)
    if ek["tilt_rmse_deg"] >= th.tilt_warn_deg:
        flags.append(_flag("FUSION_TILT_ERROR", "warn", f"EKF tilt error {ek['tilt_rmse_deg']:.1f} deg rms.",
                           ek["tilt_rmse_deg"], th.tilt_warn_deg))
    return flags


def overall_status(flags: list[dict]) -> str:
    sev = {f["severity"] for f in flags}
    return "FAIL" if "fail" in sev else "WARN" if "warn" in sev else "OK"


def flags_as_dicts(flags) -> list[dict]:
    return [asdict(f) if hasattr(f, "__dataclass_fields__") else f for f in flags]
