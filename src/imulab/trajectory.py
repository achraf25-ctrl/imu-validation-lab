"""Synthetic "brushing-like" motions: circles, sweeps, scrubs, rotations.

The motion is analytic (sum of sinusoids on position and on a body-frame rotation
vector), so it is smooth, bounded and reproducible from a seed. Sampled at 120 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quat as Q

RATE_HZ = 120.0


@dataclass
class Trajectory:
    t: np.ndarray       # (N,) seconds
    pos: np.ndarray     # (N, 3) metres, world frame
    quat: np.ndarray    # (N, 4) w,x,y,z, tool -> world

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


# amplitude (m), frequency (Hz), phase handled below
BRUSH_STYLES = {
    # circles in the horizontal plane + rocking of the tool
    "circular": dict(
        pos=[("x", 0.060, 0.8, 0.25), ("y", 0.060, 0.8, 0.0), ("z", 0.010, 1.6, 0.0)],
        rot=[(0.25, 0.8, 0.0), (0.25, 0.8, 0.25), (0.40, 0.2, 0.0)],
    ),
    # long linear strokes with a yaw twist
    "sweep": dict(
        pos=[("x", 0.120, 0.5, 0.0), ("y", 0.030, 0.1, 0.0), ("z", 0.008, 1.0, 0.25)],
        rot=[(0.15, 0.5, 0.0), (0.05, 0.5, 0.25), (0.50, 0.5, 0.0)],
    ),
    # fast small oscillations (high accelerations)
    "scrub": dict(
        pos=[("x", 0.025, 3.0, 0.0), ("y", 0.015, 2.0, 0.0), ("z", 0.004, 3.0, 0.25)],
        rot=[(0.10, 3.0, 0.0), (0.30, 3.0, 0.25), (0.20, 0.4, 0.0)],
    ),
    # slow big reorientation + circle + sweep
    "mixed": dict(
        pos=[("x", 0.080, 0.35, 0.0), ("y", 0.050, 0.55, 0.25), ("z", 0.015, 0.7, 0.0)],
        rot=[(0.35, 0.45, 0.0), (0.30, 0.30, 0.25), (0.80, 0.15, 0.0)],
    ),
}


def generate_brush_motion(
    style: str = "circular",
    duration: float = 20.0,
    rate: float = RATE_HZ,
    seed: int = 0,
    base_pos: np.ndarray | None = None,
    base_quat: np.ndarray | None = None,
    ramp: float = 1.0,
) -> Trajectory:
    """Generate a smooth trajectory around a base pose.

    The seed jitters frequencies (+-5 %) and phases so every "brush" run differs.
    A raised-cosine ramp-in avoids a velocity step at t = 0.
    """
    if style not in BRUSH_STYLES:
        raise ValueError(f"unknown brush style {style!r}; choose from {sorted(BRUSH_STYLES)}")
    spec = BRUSH_STYLES[style]
    rng = np.random.default_rng(seed)
    n = int(round(duration * rate)) + 1
    t = np.arange(n) / rate
    env = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / ramp, 0.0, 1.0)))

    base_pos = np.array([0.45, 0.0, 0.45]) if base_pos is None else np.asarray(base_pos, float)
    base_quat = np.array([0.0, 1.0, 0.0, 0.0]) if base_quat is None else np.asarray(base_quat, float)

    # frequency jitter is shared by all components with the same nominal frequency, so a
    # "circle" (x and y at the same frequency, 90 deg apart) stays a circle
    jitter: dict[float, float] = {}

    def fjit(f: float) -> float:
        if f not in jitter:
            jitter[f] = f * (1 + 0.05 * rng.uniform(-1, 1))
        return jitter[f]

    pos = np.tile(base_pos, (n, 1))
    for axis, amp, f, ph in spec["pos"]:
        k = "xyz".index(axis)
        fj = fjit(f)
        pj = 2 * np.pi * (ph + 0.03 * rng.uniform(-1, 1))
        pos[:, k] += env * amp * np.sin(2 * np.pi * fj * t + pj)

    rv = np.zeros((n, 3))
    for k, (amp, f, ph) in enumerate(spec["rot"]):
        fj = fjit(f)
        pj = 2 * np.pi * (ph + 0.03 * rng.uniform(-1, 1))
        rv[:, k] = env * amp * np.sin(2 * np.pi * fj * t + pj)
    quat = Q.make_continuous(Q.mul(base_quat, Q.exp_rotvec(rv)))
    return Trajectory(t=t, pos=pos, quat=quat)
