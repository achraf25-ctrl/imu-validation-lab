import numpy as np
import pytest

from imulab import quat as Q
from imulab.sensors import (G, IMUConfig, TrackerConfig, convention_string, ideal_imu_signals,
                            lowpass_group_delay_s, parse_axis_convention, simulate_imu, track)
from imulab.trajectory import Trajectory


def static_traj(n=240):
    t = np.arange(n) / 120
    return Trajectory(t, np.zeros((n, 3)), np.tile([1.0, 0, 0, 0], (n, 1)))


def test_static_imu_measures_gravity_up():
    s = ideal_imu_signals(static_traj())
    assert np.allclose(s["f"], [0, 0, G], atol=1e-9) and np.allclose(s["omega"], 0)


def test_tilted_static_imu_splits_gravity():
    tr = static_traj()
    q = Q.exp_rotvec(np.array([np.pi / 2, 0, 0]))        # roll 90 deg
    tr.quat[:] = q
    f = ideal_imu_signals(tr)["f"][0]
    assert np.allclose(f, [0, G, 0], atol=1e-9) or np.allclose(f, [0, -G, 0], atol=1e-9)


def test_axis_convention_parse_roundtrip():
    for s in ["x,y,z", "y,-x,z", "x,y,-z", "-z,x,-y"]:
        assert convention_string(parse_axis_convention(s)) == s
    with pytest.raises(ValueError):
        parse_axis_convention("x,x,z")
    with pytest.raises(ValueError):
        parse_axis_convention("x,y")


def test_reflection_has_negative_determinant():
    assert np.linalg.det(parse_axis_convention("x,y,-z")) < 0
    assert np.linalg.det(parse_axis_convention("y,-x,z")) > 0


def test_ideal_imu_equals_truth(truth):
    imu = simulate_imu(truth, IMUConfig())
    s = ideal_imu_signals(truth)
    assert np.allclose(imu.gyro, s["omega"]) and np.allclose(imu.accel, s["f"])


def test_delay_shifts_signal(truth):
    a = simulate_imu(truth, IMUConfig())
    b = simulate_imu(truth, IMUConfig(delay_s=0.05))
    k = 6  # 50 ms = 6 samples at 120 Hz
    assert np.allclose(b.gyro[k + 5:], a.gyro[5:-k], atol=1e-9)


def test_bias_scale_noise_applied(truth):
    base = simulate_imu(truth, IMUConfig())
    biased = simulate_imu(truth, IMUConfig(gyro_bias_dps=(1, 0, 0)))
    assert np.allclose((biased.gyro - base.gyro)[:, 0], np.deg2rad(1.0))
    scaled = simulate_imu(truth, IMUConfig(gyro_scale_err=(0.1, 0, 0)))
    assert np.allclose(scaled.gyro[:, 0], 1.1 * base.gyro[:, 0])
    noisy = simulate_imu(truth, IMUConfig(gyro_noise_dps=1.0, seed=3))
    assert 0.8 < np.rad2deg(np.std(noisy.gyro - base.gyro)) < 1.2


def test_lowpass_attenuates_and_lags(truth):
    cfg = IMUConfig(lowpass_hz=5.0)
    f = simulate_imu(truth, cfg)
    base = simulate_imu(truth, IMUConfig())
    assert np.std(np.diff(f.accel, axis=0)) < np.std(np.diff(base.accel, axis=0))
    assert 0.02 < lowpass_group_delay_s(cfg) < 0.05


def test_lever_arm_changes_accel_not_gyro(truth):
    a = simulate_imu(truth, IMUConfig())
    b = simulate_imu(truth, IMUConfig(lever_arm_m=(0, 0, 0.15)))
    assert np.allclose(a.gyro, b.gyro) and not np.allclose(a.accel, b.accel, atol=1e-3)


def test_tracker_noise_and_delay(truth):
    clean = track(truth)
    assert np.allclose(clean.pos, truth.pos)
    noisy = track(truth, TrackerConfig(pos_noise_m=1e-3, seed=1))
    assert 0.5e-3 < np.std(noisy.pos - truth.pos) < 1.5e-3
    late = track(truth, TrackerConfig(delay_s=0.05))
    assert np.allclose(late.pos[10:], truth.pos[4:-6], atol=1e-6)


def test_sample_rate_mismatch_rejected(truth):
    with pytest.raises(ValueError):
        simulate_imu(truth, IMUConfig(rate=100.0))
