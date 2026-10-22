import numpy as np
from imulab import quat as Q

rng = np.random.default_rng(0)


def rand_q(n):
    return Q.normalize(rng.normal(size=(n, 4)))


def test_exp_log_roundtrip():
    rv = rng.normal(size=(50, 3))
    rv = rv / np.linalg.norm(rv, axis=1, keepdims=True) * rng.uniform(0, 3.0, (50, 1))
    assert np.allclose(Q.log_rotvec(Q.exp_rotvec(rv)), rv, atol=1e-9)


def test_small_angle_exp_is_stable():
    q = Q.exp_rotvec(np.array([1e-12, 0, 0]))
    assert np.allclose(q, [1, 5e-13, 0, 0], atol=1e-12)


def test_mul_conj_is_identity():
    q = rand_q(10)
    assert np.allclose(Q.mul(q, Q.conj(q)), [1, 0, 0, 0], atol=1e-12)


def test_matrix_roundtrip_and_orthogonality():
    q = rand_q(30)
    R = Q.to_matrix(q)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0)
    q2 = Q.from_matrix(R)
    assert np.allclose(Q.angle_between(q, q2), 0, atol=1e-6)


def test_rotate_matches_known_rotation():
    q = Q.exp_rotvec(np.array([0, 0, np.pi / 2]))
    assert np.allclose(Q.rotate(q, np.array([1.0, 0, 0])), [0, 1, 0], atol=1e-12)


def test_angular_velocity_roundtrip():
    dt = 1 / 120
    t = np.arange(0, 3, dt)
    w_true = np.stack([0.5 * np.sin(t), 0.3 * np.cos(2 * t), 0.2 + 0 * t], axis=1)
    q = np.empty((len(t), 4)); q[0] = [1, 0, 0, 0]
    for k in range(1, len(t)):
        q[k] = Q.integrate_body(q[k - 1], w_true[k - 1], dt)
    w = Q.angular_velocity_body(q, dt)
    assert np.max(np.abs(w[2:-2] - w_true[2:-2])) < 5e-3


def test_slerp_endpoints_and_midpoint():
    a, b = Q.exp_rotvec(np.array([0, 0, 0.0])), Q.exp_rotvec(np.array([0, 0, 1.0]))
    assert np.allclose(Q.slerp(a, b, 0.0), a) and np.allclose(Q.slerp(a, b, 1.0), b)
    assert np.allclose(Q.log_rotvec(Q.slerp(a, b, 0.5)), [0, 0, 0.5], atol=1e-9)


def test_yaw_and_tilt_error_separation():
    q = np.array([1.0, 0, 0, 0])
    yaw = Q.exp_rotvec(np.array([0, 0, np.deg2rad(10)]))
    tilt = Q.exp_rotvec(np.array([np.deg2rad(5), 0, 0]))
    assert np.isclose(np.rad2deg(Q.yaw_error(yaw, q)), 10, atol=1e-6)
    assert np.isclose(np.rad2deg(Q.tilt_error(yaw, q)), 0, atol=1e-6)      # pure yaw: no tilt
    assert np.isclose(np.rad2deg(Q.tilt_error(tilt, q)), 5, atol=1e-6)
    assert abs(np.rad2deg(Q.yaw_error(tilt, q))) < 1e-6                    # pure tilt: no heading error
