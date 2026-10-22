import numpy as np

from imulab import fusion, quat as Q
from imulab.calibration import calibrate
from imulab.sensors import G, IMUConfig, simulate_imu


def test_gyro_integration_accurate_on_clean_data(truth):
    imu = simulate_imu(truth, IMUConfig())
    q = fusion.gyro_integration(truth.quat[0], imu.gyro, truth.dt)
    assert fusion.summarize_errors(fusion.orientation_errors(truth.quat, q))["total_rmse_deg"] < 1.0


def test_complementary_converges_from_wrong_tilt_at_rest():
    n, dt = 1200, 1 / 120
    gyro = np.zeros((n, 3)); acc = np.tile([0, 0, G], (n, 1))
    q0 = Q.exp_rotvec(np.array([np.deg2rad(20), np.deg2rad(-15), 0]))   # wrong initial tilt
    q = fusion.complementary(q0, gyro, acc, dt, kp=2.0)
    assert np.rad2deg(Q.tilt_error(np.array([1.0, 0, 0, 0]), q[-1])) < 0.5


def test_ekf_converges_from_wrong_tilt_at_rest():
    n, dt = 1200, 1 / 120
    gyro = np.zeros((n, 3)); acc = np.tile([0, 0, G], (n, 1))
    q0 = Q.exp_rotvec(np.array([np.deg2rad(20), np.deg2rad(-15), 0]))
    q, _ = fusion.ekf(q0, gyro, acc, dt, accel_dir_noise=0.05, p0_att=np.deg2rad(20))
    assert np.rad2deg(Q.tilt_error(np.array([1.0, 0, 0, 0]), q[-1])) < 1.0


def test_ekf_estimates_constant_gyro_bias_at_rest():
    n, dt = 6000, 1 / 120
    bias = np.deg2rad([0.5, -0.3, 0.2])
    gyro = np.tile(bias, (n, 1)); acc = np.tile([0, 0, G], (n, 1))
    q, b = fusion.ekf(np.array([1.0, 0, 0, 0]), gyro, acc, dt, p0_bias=np.deg2rad(1.0), bias_walk=1e-4, accel_dir_noise=0.02)
    assert np.allclose(np.rad2deg(b[-1][:2]), [0.5, -0.3], atol=0.1)   # tilt-observable axes


def test_calibration_improves_fusion_a_lot(truth):
    cfg = IMUConfig(axis_convention="y,-x,z", gyro_bias_dps=(1.5, -1, 0.8), delay_s=0.05,
                    gyro_noise_dps=0.2, accel_noise=0.05, gyro_scale_err=(0.03, -0.02, 0.03))
    imu = simulate_imu(truth, cfg)
    cal = calibrate(imu, truth).apply(imu)
    dt = truth.dt
    raw = fusion.summarize_errors(fusion.orientation_errors(truth.quat, fusion.ekf(truth.quat[0], imu.gyro, imu.accel, dt)[0]))
    fixed = fusion.summarize_errors(fusion.orientation_errors(truth.quat, fusion.ekf(truth.quat[0], cal.gyro, cal.accel, dt)[0]))
    assert fixed["total_rmse_deg"] < 5 and raw["total_rmse_deg"] > 5 * fixed["total_rmse_deg"]


def test_ekf_heading_does_not_run_away_on_healthy_data(truth):
    imu = simulate_imu(truth, IMUConfig(gyro_noise_dps=0.15, accel_noise=0.04))
    q, b = fusion.ekf(truth.quat[0], imu.gyro, imu.accel, truth.dt)
    s = fusion.summarize_errors(fusion.orientation_errors(truth.quat, q))
    assert s["heading_drift_deg"] < 5 and np.rad2deg(np.abs(b).max()) < 0.5
