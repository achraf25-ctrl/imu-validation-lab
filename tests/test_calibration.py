import numpy as np
import pytest

from imulab.calibration import calibrate, estimate_delay, estimate_delay_vec
from imulab.sensors import IMUConfig, ideal_imu_signals, simulate_imu


@pytest.mark.parametrize("delay_ms", [0, 10, 33.3, 80])
def test_delay_recovered_within_1ms(truth, delay_ms):
    imu = simulate_imu(truth, IMUConfig(delay_s=delay_ms / 1000, gyro_noise_dps=0.2, accel_noise=0.05, seed=2))
    cal = calibrate(imu, truth)
    assert abs(cal.delay.delay_s * 1000 - delay_ms) < 1.0


def test_delay_recovered_despite_axis_remap_scale_and_bias(truth):
    cfg = IMUConfig(delay_s=0.045, axis_convention="y,-x,z", gyro_scale_err=(0.03, -0.02, 0.04),
                    gyro_bias_dps=(1, -1, 0.5), gyro_noise_dps=0.3)
    assert abs(calibrate(simulate_imu(truth, cfg), truth).delay.delay_s * 1000 - 45) < 1.0


def test_scalar_xcorr_recovers_delay_on_rich_signal():
    rng = np.random.default_rng(0)
    x = np.convolve(rng.normal(size=3000), np.ones(8) / 8, mode="same")
    d = 7
    y = np.roll(x, d)
    est = estimate_delay(x, y, 1 / 120)
    assert abs(est.delay_s * 120 - d) < 0.1


@pytest.mark.parametrize("conv,flipped", [("x,y,z", False), ("y,-x,z", False), ("-x,-y,z", False),
                                          ("x,y,-z", True), ("-x,y,z", True), ("z,y,x", True)])
def test_axis_convention_and_flip_detection(truth, conv, flipped):
    cal = calibrate(simulate_imu(truth, IMUConfig(axis_convention=conv, gyro_noise_dps=0.1)), truth)
    assert cal.gyro.convention == conv and cal.accel.convention == conv
    assert (cal.gyro.det < 0) == flipped


def test_bias_scale_misalignment_recovery(truth):
    cfg = IMUConfig(gyro_bias_dps=(1.2, -0.7, 0.4), gyro_scale_err=(0.03, -0.02, 0.05),
                    mount_rotvec_deg=(2, -1, 3), accel_bias=(0.1, -0.2, 0.15), gyro_noise_dps=0.1, accel_noise=0.03)
    cal = calibrate(simulate_imu(truth, cfg), truth)
    assert np.allclose(np.rad2deg(cal.gyro.bias), cfg.gyro_bias_dps, atol=0.1)
    assert np.allclose(cal.gyro.scales, 1 + np.array(cfg.gyro_scale_err), atol=5e-3)
    assert abs(cal.gyro.misalign_deg - np.linalg.norm(cfg.mount_rotvec_deg)) < 0.2
    assert np.allclose(cal.accel.bias, cfg.accel_bias, atol=0.05)


def test_apply_calibration_restores_truth(truth):
    cfg = IMUConfig(delay_s=0.04, axis_convention="y,-x,z", gyro_bias_dps=(1, -1, 1), gyro_scale_err=(0.02, 0.03, -0.02),
                    mount_rotvec_deg=(1, 2, -2), gyro_noise_dps=0.05)
    imu = simulate_imu(truth, cfg)
    out = calibrate(imu, truth).apply(imu)
    ref = ideal_imu_signals(truth)
    k = 120
    err = np.rad2deg(np.sqrt(np.mean((out.gyro[k:-k] - ref["omega"][k:-k]) ** 2)))
    raw = np.rad2deg(np.sqrt(np.mean((imu.gyro[k:-k] - ref["omega"][k:-k]) ** 2)))
    assert err < 0.2 and raw > 20 * err


def test_lever_arm_leaves_gyro_good_but_accel_systematically_off(truth):
    cal = calibrate(simulate_imu(truth, IMUConfig(lever_arm_m=(0, 0, 0.15))), truth)
    assert cal.gyro.r2 > 0.9999
    assert cal.accel.resid_structured_rms > 0.03


def test_filter_lag_shows_up_as_effective_delay(truth):
    from imulab.sensors import lowpass_group_delay_s
    cfg = IMUConfig(lowpass_hz=8.0)
    est = calibrate(simulate_imu(truth, cfg), truth).delay.delay_s
    assert abs(est - lowpass_group_delay_s(cfg)) < 0.004
