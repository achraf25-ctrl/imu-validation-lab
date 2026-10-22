import numpy as np
import pytest

pb = pytest.importorskip("pybullet")
from imulab import quat as Q
from imulab.experiment import get_plan
from imulab.pybullet_backend import PyBulletSim, arm_urdf


def test_pybullet_fk_matches_our_fk(arm):
    qs = arm.q_home + np.random.default_rng(0).uniform(-0.6, 0.6, (8, 6))
    with PyBulletSim(arm) as sim:
        for q in qs:
            p_pb, q_pb = sim.fk(q)
            p, qq = arm.fk(q)
            assert np.linalg.norm(p_pb - p[0]) < 1e-5
            assert Q.angle_between(q_pb, qq[0]) < 1e-5


def test_pybullet_kinematic_replay_matches_plan(arm):
    plan, _ = get_plan("circular", 4.0, 0, arm)
    with PyBulletSim(arm) as sim:
        res = sim.replay_kinematic(plan)
    p, _ = arm.fk(plan.q)
    assert np.max(np.linalg.norm(res.truth.pos - p, axis=1)) < 1e-5


def test_pybullet_dynamic_replay_tracks_within_few_mm(arm):
    from imulab.metrics import replay_accuracy
    plan, _ = get_plan("circular", 6.0, 0, arm)
    with PyBulletSim(arm) as sim:
        res = sim.replay_dynamic(plan)
    assert replay_accuracy(plan.target, res.truth)["pos_rmse_mm"] < 5.0


def test_experiment_with_pybullet_backend():
    from imulab.experiment import ExperimentConfig, run_experiment
    r = run_experiment(ExperimentConfig(name="pb", scenario="delayed", duration=6.0, backend="pybullet"))
    assert r["config"]["backend"] == "pybullet" and "DELAY_TOO_LARGE" in {f["code"] for f in r["flags"]}
