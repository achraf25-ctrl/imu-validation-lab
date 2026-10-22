import numpy as np
from imulab import quat as Q
from imulab.experiment import get_plan


def test_ik_roundtrip(arm):
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = arm.q_home + rng.uniform(-0.3, 0.3, 6)
        p, qq = arm.fk(q)
        sol, ok = arm.ik(p[0], qq[0], arm.q_home)
        assert ok
        p2, q2 = arm.fk(sol)
        assert np.linalg.norm(p2 - p) < 1e-5 and Q.angle_between(q2[0], qq[0]) < 1e-5


def test_jacobian_matches_finite_difference(arm):
    q = arm.q_home + 0.1
    J = arm.jacobian(q)
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6); dq[i] = eps
        p1, q1 = arm.fk(q + dq)
        p0, q0 = arm.fk(q - dq)
        assert np.allclose((p1[0] - p0[0]) / (2 * eps), J[:3, i], atol=1e-5)
        w = Q.log_rotvec(Q.mul(q1[0], Q.conj(q0[0]))) / (2 * eps)  # world-frame
        assert np.allclose(w, J[3:, i], atol=1e-4)


def test_batch_fk_matches_single(arm):
    qs = arm.q_home + np.random.default_rng(2).uniform(-0.2, 0.2, (4, 6))
    pb, qb = arm.fk(qs)
    for i in range(4):
        ps, qs_ = arm.fk(qs[i])
        assert np.allclose(pb[i], ps[0]) and Q.angle_between(qb[i], qs_[0]) < 1e-9


def test_gentle_brush_is_not_retimed_and_within_limits(arm):
    plan, _ = get_plan("circular", 10.0, 0, arm)
    assert not plan.report["retimed"] and plan.report["within_limits"]
    assert plan.report["ik_converged_fraction"] == 1.0
    assert plan.report["plan_pos_err_mm"] < 0.1


def test_aggressive_brush_is_slowed_to_respect_limits(arm):
    plan, traj = get_plan("scrub", 10.0, 0, arm)
    assert plan.report["retimed"] and plan.report["slowdown_factor"] > 1.5
    assert plan.report["within_limits"]
    assert np.all(np.abs(plan.qd) <= arm.qd_max * 1.001)
    assert plan.t[-1] > traj.t[-1]           # took longer
    assert abs(plan.t[1] - plan.t[0] - 1 / 120) < 1e-12   # still 120 Hz
    assert np.all(plan.q >= arm.q_min - 1e-9) and np.all(plan.q <= arm.q_max + 1e-9)
