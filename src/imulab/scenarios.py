"""Named IMU fault / quality scenarios used by the demo, tests and dashboard."""
from __future__ import annotations

from .sensors import IMUConfig

_GOOD = dict(gyro_noise_dps=0.15, accel_noise=0.04)

SCENARIOS: dict[str, dict] = {
    "ideal": dict(),
    "healthy": dict(**_GOOD, gyro_bias_dps=(0.1, -0.08, 0.05), delay_s=0.004),
    "delayed": dict(**_GOOD, delay_s=0.060),
    "axis_flipped": dict(**_GOOD, axis_convention="x,y,-z"),
    "axis_remapped": dict(**_GOOD, axis_convention="y,-x,z"),
    "misaligned": dict(**_GOOD, mount_rotvec_deg=(3.0, -2.0, 4.0)),
    "biased": dict(**_GOOD, gyro_bias_dps=(1.5, -1.0, 0.8), accel_bias=(0.15, -0.1, 0.2)),
    "scaled": dict(**_GOOD, gyro_scale_err=(0.04, -0.03, 0.05), accel_scale_err=(0.03, -0.02, 0.04)),
    "noisy": dict(gyro_noise_dps=1.5, accel_noise=0.5, gyro_bias_walk_dps=0.05),
    "filtered": dict(**_GOOD, lowpass_hz=8.0),
    "lever_arm": dict(**_GOOD, lever_arm_m=(0.0, 0.0, 0.12)),
    "worst_case": dict(
        gyro_noise_dps=0.8, accel_noise=0.25, gyro_bias_dps=(1.0, -0.8, 0.6), accel_bias=(0.1, -0.1, 0.15),
        gyro_scale_err=(0.03, -0.02, 0.04), accel_scale_err=(0.02, -0.02, 0.03),
        axis_convention="y,-x,z", mount_rotvec_deg=(2.0, -1.5, 3.0), delay_s=0.045, lowpass_hz=15.0,
        gyro_bias_walk_dps=0.03,
    ),
}


def imu_config(name: str, **overrides) -> IMUConfig:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return IMUConfig(**{**SCENARIOS[name], **overrides})
