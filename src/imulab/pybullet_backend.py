"""Optional PyBullet backend: replay a joint plan in a physics engine.

* The URDF is generated from the same DH table as imulab.arm.Arm, so PyBullet's forward
  kinematics must agree with our own FK (checked in tests/test_pybullet.py).
* replay_kinematic: set joint states each sample, read the tool pose (pure FK cross-check).
* replay_dynamic: position-controlled joints stepped at 240 Hz with gravity and link
  masses -> a genuine physics-engine tracking error.

Install with:  pip install pybullet   (compiles from source on recent Python versions)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .arm import Arm
from .pipeline import JointPlan
from .sim import ReplayResult
from .trajectory import Trajectory

try:  # pragma: no cover - exercised only when pybullet is installed
    import pybullet as p
    HAVE_PYBULLET = True
except Exception:  # noqa: BLE001
    p = None
    HAVE_PYBULLET = False


def arm_urdf(arm: Arm) -> str:
    """URDF whose joint i origin is the fixed part F_{i-1} = Tz(d)Tx(a)Rx(alpha) of the DH chain."""
    def link(name: str) -> str:
        return (f'<link name="{name}"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>'
                '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>')

    parts = ['<?xml version="1.0"?>', '<robot name="imulab_arm">', link("base_link")]
    for i in range(6):
        parts.append(link(f"link_{i + 1}"))
    parts += [link("flange"), link("tool")]
    for i in range(6):
        parent = "base_link" if i == 0 else f"link_{i}"
        if i == 0:
            xyz, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        else:
            xyz, rpy = (arm.a[i - 1], 0.0, arm.d[i - 1]), (arm.alpha[i - 1], 0.0, 0.0)
        parts.append(
            f'<joint name="j{i + 1}" type="revolute"><parent link="{parent}"/><child link="link_{i + 1}"/>'
            f'<origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"/>'
            f'<axis xyz="0 0 1"/><limit lower="{arm.q_min[i]:.6f}" upper="{arm.q_max[i]:.6f}" effort="500" velocity="{arm.qd_max[i]:.6f}"/></joint>'
        )
    parts.append(
        f'<joint name="flange_fixed" type="fixed"><parent link="link_6"/><child link="flange"/>'
        f'<origin xyz="{arm.a[5]:.9f} 0 {arm.d[5]:.9f}" rpy="{arm.alpha[5]:.9f} 0 0"/></joint>'
    )
    parts.append(
        f'<joint name="tool_fixed" type="fixed"><parent link="flange"/><child link="tool"/>'
        f'<origin xyz="0 0 {arm.tool_z:.9f}" rpy="0 0 0"/></joint>'
    )
    parts.append("</robot>")
    return "\n".join(parts)


class PyBulletSim:
    TOOL_LINK = 7  # joints 0-5 revolute, 6 flange (fixed), 7 tool (fixed)

    def __init__(self, arm: Arm | None = None, gui: bool = False):
        if not HAVE_PYBULLET:
            raise ImportError("pybullet is not installed (pip install pybullet)")
        self.arm = arm or Arm()
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        self._tmp = tempfile.TemporaryDirectory()
        urdf = Path(self._tmp.name) / "arm.urdf"
        urdf.write_text(arm_urdf(self.arm))
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        self.body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=self.cid)

    def close(self) -> None:
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def set_q(self, q: np.ndarray) -> None:
        for i in range(6):
            p.resetJointState(self.body, i, float(q[i]), 0.0, physicsClientId=self.cid)

    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        st = p.getLinkState(self.body, self.TOOL_LINK, computeForwardKinematics=1, physicsClientId=self.cid)
        pos, (x, y, z, w) = np.array(st[4]), st[5]
        return pos, np.array([w, x, y, z])

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.set_q(q)
        return self.tool_pose()

    def replay_kinematic(self, plan: JointPlan) -> ReplayResult:
        pos, quat = zip(*(self.fk(q) for q in plan.q))
        return ReplayResult(plan.t.copy(), plan.q.copy(),
                            Trajectory(plan.t.copy(), np.array(pos), _continuous(np.array(quat))))

    def replay_dynamic(self, plan: JointPlan, sim_hz: float = 240.0, max_force: float = 500.0) -> ReplayResult:
        dt = float(plan.t[1] - plan.t[0])
        sub = int(round(sim_hz * dt))
        p.setTimeStep(1.0 / sim_hz, physicsClientId=self.cid)
        self.set_q(plan.q[0])
        qs, pos, quat = [], [], []
        for k in range(len(plan.t)):
            p.setJointMotorControlArray(
                self.body, list(range(6)), p.POSITION_CONTROL, targetPositions=plan.q[k].tolist(),
                targetVelocities=plan.qd[k].tolist(), forces=[max_force] * 6,
                positionGains=[0.5] * 6, velocityGains=[1.0] * 6, physicsClientId=self.cid)
            for _ in range(sub):
                p.stepSimulation(physicsClientId=self.cid)
            qs.append([p.getJointState(self.body, i, physicsClientId=self.cid)[0] for i in range(6)])
            ps, qq = self.tool_pose()
            pos.append(ps); quat.append(qq)
        return ReplayResult(plan.t.copy(), np.array(qs),
                            Trajectory(plan.t.copy(), np.array(pos), _continuous(np.array(quat))))


def _continuous(q: np.ndarray) -> np.ndarray:
    from . import quat as Q
    return Q.make_continuous(q)
