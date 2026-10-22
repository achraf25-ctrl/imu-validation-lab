import numpy as np

from imulab import metrics
from imulab.sensors import IMUConfig, simulate_imu
from imulab.sim import KinematicSim, ReplayConfig
from imulab.experiment import get_plan


def test_replay_accuracy_zero_for_identical(truth):
    r = metrics.replay_accuracy(truth, truth)
    assert r["pos_rmse_mm"] == 0 and r["rot_rmse_deg"] < 1e-6


def test_replay_accuracy_reflects_servo_lag(arm):
    plan, _ = get_plan("circular", 10.0, 0, arm)
    slow = KinematicSim(arm, ReplayConfig(servo_tau_s=0.05, exec_noise_deg=0)).replay(plan).truth
    fast = KinematicSim(arm, ReplayConfig(servo_tau_s=0.005, exec_noise_deg=0)).replay(plan).truth
    assert metrics.replay_accuracy(plan.target, slow)["pos_rmse_mm"] > 3 * metrics.replay_accuracy(plan.target, fast)["pos_rmse_mm"]


def test_repeatability_zero_without_noise_and_positive_with(arm):
    plan, _ = get_plan("circular", 10.0, 0, arm)
    same = [KinematicSim(arm, ReplayConfig(exec_noise_deg=0, seed=s)).replay(plan).truth for s in range(4)]
    assert metrics.repeatability(same)["pos_rp_mm"] < 1e-9
    noisy = [KinematicSim(arm, ReplayConfig(exec_noise_deg=0.05, seed=s)).replay(plan).truth for s in range(4)]
    assert metrics.repeatability(noisy)["pos_rp_mm"] > 0.05


def test_dtw_properties():
    x = np.sin(np.linspace(0, 20, 300))
    assert metrics.dtw_distance(x, x) == 0
    assert metrics.dtw_distance(x, np.roll(x, 5)) < metrics.dtw_distance(x, np.random.default_rng(0).normal(size=300))


def test_brush_similarity_self_is_one_and_brushes_differ(arm):
    imus = {}
    for b in ("circular", "scrub"):
        plan, _ = get_plan(b, 10.0, 0, arm)
        imus[b] = simulate_imu(KinematicSim(arm, ReplayConfig(seed=0)).replay(plan).truth, IMUConfig())
    res = metrics.brush_similarity(imus)
    cos = np.array(res["spectral_cosine"])
    assert np.allclose(np.diag(cos), 1.0, atol=1e-6) and cos[0, 1] < 0.99
    assert res["features"]["scrub"]["accel_dyn_rms"] != res["features"]["circular"]["accel_dyn_rms"]
