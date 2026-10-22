# imulab — simulated IMU validation lab

A 6-DOF arm replays a smooth, brushing-like motion (circles, sweeps, scrubs, rotations at 120 Hz). A **virtual IMU**
(configurable axes, bias, scale, noise, filtering, delay, lever arm) and an ideal **ground-truth tracker** measure it.
The software then checks how well the IMU matches the true motion, recovers what is wrong with it, and flags it.

```
trajectory (quaternions) → IK + limit check/retime → replay (kinematic or PyBullet) → truth pose
                                                              ├─ virtual IMU  (faults injected)
                                                              └─ virtual tracker
   delay sync (multi-channel xcorr) → frame/scale/bias calibration → fusion (gyro / complementary / EKF)
   → metrics (replay accuracy, repeatability, brush similarity) → diagnostics flags → Streamlit dashboard
```

## One command
```bash
pip install -e ".[dev]"      # add ,pybullet for the physics backend
make all                     # = python -m imulab demo  &&  streamlit run dashboard/app.py
# or: docker compose up --build   (dashboard on http://localhost:8501)
make test
```
Single run: `python -m imulab run --scenario delayed --brush scrub --override '{"delay_s": 0.08}'`
(`python -m imulab list` shows brushes and scenarios; `--backend pybullet` replays in the physics engine).

## What is where
| Step | Module |
|---|---|
| 1 Simulation: UR5-like 6-DOF arm, 120 Hz brush motions, PyBullet URDF generated from the same DH table | `arm.py`, `trajectory.py`, `sim.py`, `pybullet_backend.py` |
| 2 Virtual sensors: axis convention, mount misalignment, scale, bias (+random walk), noise, causal low-pass, delay, lever arm; tracker | `sensors.py`, `scenarios.py` |
| 3 Trajectory pipeline: quaternion poses → IK → joint position/speed/accel checks → automatic slow-down | `pipeline.py` |
| 4 Validation: delay estimation, frame calibration, replay accuracy, repeatability, brush similarity | `calibration.py`, `metrics.py` |
| 5 Fusion: gyro dead-reckoning, Mahony-style complementary filter, error-state EKF | `fusion.py` |
| 6 Dashboard + diagnostics (delay too large, axis flipped, bias, scale, lever arm…) | `diagnostics.py`, `dashboard/app.py` |
| 7 Polish: Docker, Makefile, 71 tests, GitHub Actions CI, report | `docs/REPORT.md` |

See **docs/REPORT.md** for results and the failure analysis.

## Honest limits
* Gazebo / ROS 2 is not implemented; PyBullet is the physics backend.
* "Ground truth" is the simulated tool pose; real mocap adds noise/latency (`TrackerConfig` models both, off by default).
* Orientation fusion has no magnetometer, so heading is unobservable and drifts; filters start from the true initial orientation.
* The Dockerfile and CI workflow were written but not executed in the environment this was built in.
