# Technical report — virtual IMU validation lab

All numbers below come from `python -m imulab demo` (circular brush, 20 s, 120 Hz, seed 0, 5 repeats) and are reproducible.

## 1. Setup
A UR5-like 6-DOF arm (joint limits 170°, 180–220 °/s, 900–1200 °/s²) replays an analytic brushing motion. IK is damped least squares, warm-started along the path. If speed or acceleration limits are exceeded the whole trajectory is time-scaled and resampled at 120 Hz, and the *retimed* target pose is kept as the reference. The joints follow commands through a first-order servo (τ = 12 ms) plus a smooth 0.02° run-to-run perturbation. The tool pose is the ground truth; the IMU is derived from it (gyro = ω in the tool frame, accelerometer = specific force Rᵀ(a − g)).

Injected IMU faults: delay, axis remap/flip, mounting misalignment, scale, bias, noise + bias random walk, 8 Hz low-pass, 12 cm lever arm.

## 2. Validation results
**Replay accuracy.** Position RMSE 2.5 mm (max 3.2 mm), orientation RMSE 0.62°, dominated by the servo lag. PyBullet dynamic replay of the same plan: 1.7 mm / 0.41°.
**Repeatability** (5 replays): ISO-9283-style position RP 0.60 mm, orientation 0.045°. Across those repeats the delay estimate has σ = 0.05 ms, misalignment σ = 0.006°, gyro bias σ = 0.006 °/s.
**Trajectory limits.** The gentle brushes fit inside the limits. The `scrub` brush needs 1.57× the speed limit and 5.3× the acceleration limit, so it is slowed ×2.35 automatically (flag `TRAJECTORY_SLOWED`).
**Calibration recovery** (vs the known simulated truth): delay error ≤ 0.07 ms for pure delays (1.2 ms in the stacked worst case), per-axis scale within 0.5 %, bias within 0.15 °/s, misalignment within 0.2°, and the signed-permutation axis convention is recovered exactly, including the reflection case.
**Brush similarity.** Spectral cosine between brushes: circular–mixed 0.87, sweep–mixed 0.94, circular–scrub 0.54, scrub–sweep 0.67. The fast, small `scrub` motion is clearly the odd one out (`docs/figures/brush_similarity.png`).

## 3. Failure analysis
| Scenario | What goes wrong | Gyro-only error raw → calibrated | Flag |
|---|---|---|---|
| axis_flipped (`x,y,-z`) | handedness reflection; no rotation can fix it | 105.6° → 0.32° | `AXIS_FLIPPED` |
| axis_remapped (`y,-x,z`) | wrong channel order | 20.0° → 0.33° | `AXIS_REMAPPED` |
| biased (1.5/−1/0.8 °/s) | heading drifts ≈ bias × time | 21.5° → 0.33° | `GYRO_BIAS_HIGH` |
| scaled (3–5 %) | proportional orientation error | 3.9° → 0.33° | `SCALE_ERROR` |
| delayed (60 ms) | IMU lags truth | 4.1° → 0.33° | `DELAY_TOO_LARGE` |
| worst_case (all stacked) | everything at once | 21.8° → 0.55° | 8 flags |

**Lessons**
1. *The norm of ω is a bad delay signal.* The first estimator cross-correlated |ω| (axis-invariant). For near-constant-speed circular motion it returned 5 ms for a true 60 ms. The fix was a multi-channel search: at each lag fit IMU = M·truth + b and keep the lag with the highest explained variance (exact to < 0.1 ms afterwards). Per-axis xcorr also failed on one axis (35 ms).
2. *Estimated delay is effective delay.* The 8 Hz low-pass has no transport delay yet is reported as 27.9 ms, matching its 28.1 ms group delay. Diagnostics say "transport + filter".
3. *A lever arm masquerades as other faults.* With the IMU 12 cm off the tracked point the gyro fits perfectly (R² > 0.9999) but the accelerometer fit is distorted (fitted gains 0.81/0.79, 8° "misalignment"). A naive diagnostic reports a scale error. The fix: estimate the white-noise floor from sample differences; if a structured residual remains, flag `STRUCTURED_RESIDUAL` and stop trusting accelerometer gains. `ACCEL_BIAS_HIGH` still fires as a side effect.
4. *Accelerometer gating by |f| fails for horizontal acceleration.* A 0.15 g sideways acceleration tilts measured "up" by ~9° but changes |f| by ~1 %. With optimistic noise settings the EKF absorbed this correlated error into its gyro-bias state: the bias estimate wandered to 3 °/s and heading error reached tens of degrees (tilt looked fine). Tightening the bias prior and inflating the direction noise (0.3 rad) fixed it.
5. *With a clean calibrated gyro, fusion does not help over 20 s.* Gyro-only dead reckoning gives 0.33°; the complementary filter and EKF give ≈ 2.0–2.2° total (tilt 0.5–0.8°) because the accelerometer correction is polluted by tool acceleration, and the EKF heading drifts ≈ 4.9° even on ideal data. Even in the noisy scenario gyro-only (0.86°) beats the filters (1.9–2.0°). I did not test long runs (minutes), where gyro drift should eventually favour fusion; that is an expectation, not a result.
6. *Calibration, not fusion, is what rescues a bad IMU:* EKF error on raw data 123.6° (flip), 20.2° (remap), 11.3° (bias) collapses to ≈ 2.1° after calibration.

## 4. Limitations
Gazebo/ROS 2 not implemented. Ground truth is exact by default. One arm model, simulated servo, no cable/vibration effects. Gyro and accel use one shared mounting map. Lever arm is detected but not estimated. Docker and CI files are written but were not run in the build environment; the 71 pytest tests and the demo were run.
