"""A 6-DOF serial arm (UR5-like DH parameters): FK, geometric Jacobian, damped IK."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quat as Q


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array(
        [[ct, -st * ca, st * sa, a * ct],
         [st, ct * ca, -ct * sa, a * st],
         [0, sa, ca, d],
         [0, 0, 0, 1.0]]
    )


@dataclass
class Arm:
    d: np.ndarray = field(default_factory=lambda: np.array([0.0892, 0.0, 0.0, 0.1093, 0.09475, 0.0825]))
    a: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.425, -0.392, 0.0, 0.0, 0.0]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]))
    tool_z: float = 0.10  # brush length along flange z (m)
    q_min: np.ndarray = field(default_factory=lambda: np.deg2rad([-170, -170, -150, -170, -170, -300.0]))
    q_max: np.ndarray = field(default_factory=lambda: np.deg2rad([170, 10, 150, 170, 170, 300.0]))
    qd_max: np.ndarray = field(default_factory=lambda: np.deg2rad([180, 180, 180, 220, 220, 220.0]))
    qdd_max: np.ndarray = field(default_factory=lambda: np.deg2rad([900, 900, 900, 1200, 1200, 1200.0]))
    q_home: np.ndarray = field(default_factory=lambda: np.deg2rad([0, -100, 110, -100, -90, 0.0]))

    # ---------------------------------------------------------------- kinematics
    def frames(self, q: np.ndarray) -> list[np.ndarray]:
        """Base-to-frame transforms for frames 0..6 (+ tool as the last entry)."""
        T = np.eye(4)
        out = [T]
        for i in range(6):
            T = T @ _dh(q[i], self.d[i], self.a[i], self.alpha[i])
            out.append(T)
        Tt = np.eye(4)
        Tt[2, 3] = self.tool_z
        out.append(T @ Tt)
        return out

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose for one configuration or a batch (N,6) -> pos (N,3), quat (N,4)."""
        q = np.atleast_2d(q)
        n = len(q)
        T = np.tile(np.eye(4), (n, 1, 1))
        for i in range(6):
            ct, st = np.cos(q[:, i]), np.sin(q[:, i])
            ca, sa = np.cos(self.alpha[i]), np.sin(self.alpha[i])
            A = np.zeros((n, 4, 4))
            A[:, 0, 0] = ct; A[:, 0, 1] = -st * ca; A[:, 0, 2] = st * sa; A[:, 0, 3] = self.a[i] * ct
            A[:, 1, 0] = st; A[:, 1, 1] = ct * ca; A[:, 1, 2] = -ct * sa; A[:, 1, 3] = self.a[i] * st
            A[:, 2, 1] = sa; A[:, 2, 2] = ca; A[:, 2, 3] = self.d[i]
            A[:, 3, 3] = 1.0
            T = T @ A
        pos = T[:, :3, 3] + self.tool_z * T[:, :3, 2]
        quat = Q.from_matrix(T[:, :3, :3])
        return pos, quat

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian (6x6) of the tool point; angular part in the world frame."""
        fr = self.frames(q)
        p = fr[-1][:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z = fr[i][:3, 2]
            o = fr[i][:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    def home_pose(self) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.fk(self.q_home)
        return p[0], q[0]

    # ------------------------------------------------------------------------ IK
    def ik(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        q0: np.ndarray,
        tol: float = 1e-7,
        max_iter: int = 60,
        damping: float = 0.02,
    ) -> tuple[np.ndarray, bool]:
        """Damped least-squares IK for a single pose, warm-started at q0."""
        q = np.array(q0, dtype=float)
        for _ in range(max_iter):
            p, qq = self.fk(q)
            e_p = pos - p[0]
            e_r = Q.log_rotvec(Q.mul(quat, Q.conj(qq[0])))  # world-frame rotation error
            e = np.concatenate([e_p, e_r])
            if np.linalg.norm(e) < tol:
                return q, True
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + dq, self.q_min, self.q_max)
        p, qq = self.fk(q)
        err = np.linalg.norm(np.concatenate([pos - p[0], Q.log_rotvec(Q.mul(quat, Q.conj(qq[0])))]))
        return q, bool(err < 1e-4)

    def ik_trajectory(self, pos: np.ndarray, quat: np.ndarray, q0: np.ndarray | None = None):
        q_prev = self.q_home if q0 is None else q0
        out = np.empty((len(pos), 6))
        ok = np.ones(len(pos), dtype=bool)
        for k in range(len(pos)):
            out[k], ok[k] = self.ik(pos[k], quat[k], q_prev)
            q_prev = out[k]
        return out, ok
