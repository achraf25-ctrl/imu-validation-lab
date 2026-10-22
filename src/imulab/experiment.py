"""End-to-end experiment: plan -> replay -> sensors -> calibrate -> metrics -> fusion -> diagnostics."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import fusion, metrics
from . import quat as Q
from .arm import Arm
from .calibration import calibrate
from .diagnostics import diagnose, overall_status
from .pipeline import JointPlan, plan_trajectory
from .scenarios import imu_config
from .sensors import IMUConfig, TrackerConfig, ideal_imu_signals, lowpass_group_delay_s, simulate_imu, track
from .sim import KinematicSim, ReplayConfig
from .trajectory import Trajectory, generate_brush_motion


@dataclass
class ExperimentConfig:
    name: str = "run"
    brush: str = "circular"
    scenario: str = "healthy"
    duration: float = 20.0
    seed: int = 0
    imu_overrides: dict = field(default_factory=dict)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    n_repeats: int = 5
    expected_convention: str = "x,y,z"
    backend: str = "kinematic"   # "kinematic" (default) or "pybullet" (dynamic physics replay)

    def imu(self) -> IMUConfig:
        return imu_config(self.scenario, seed=self.seed, **self.imu_overrides)


_ARM = Arm()
_PLAN_CACHE: dict = {}


def get_plan(brush: str, duration: float, seed: int, arm: Arm = _ARM) -> tuple[JointPlan, Trajectory]:
    key = (brush, duration, seed)
    if key not in _PLAN_CACHE:
        p, q = arm.home_pose()
        traj = generate_brush_motion(brush, duration=duration, seed=seed, base_pos=p, base_quat=q)
        _PLAN_CACHE[key] = (plan_trajectory(arm, traj), traj)
    return _PLAN_CACHE[key][0], _PLAN_CACHE[key][1]


def _fusion_block(q0, imu, dt, q_true):
    """Run the three estimators on `imu`; return summaries and error/state series."""
    runs = {
        "gyro_only": fusion.gyro_integration(q0, imu.gyro, dt),
    }
    runs["complementary"] = fusion.complementary(q0, imu.gyro, imu.accel, dt)
    runs["ekf"], bias = fusion.ekf(q0, imu.gyro, imu.accel, dt)
    errs = {k: fusion.orientation_errors(q_true, v) for k, v in runs.items()}
    return {k: fusion.summarize_errors(e) for k, e in errs.items()}, errs, bias


def run_experiment(cfg: ExperimentConfig, out_dir: str | Path | None = None, make_plots: bool = True) -> dict:
    t_start = time.time()
    arm = _ARM
    plan, _ = get_plan(cfg.brush, cfg.duration, cfg.seed, arm)
    imu_cfg = cfg.imu()

    # --- repeated replays (run-to-run variation through smooth execution noise)
    replays = []
    if cfg.backend == "pybullet":
        from .pybullet_backend import PyBulletSim
        with PyBulletSim(arm) as pb:
            replays.append(pb.replay_dynamic(plan))
    elif cfg.backend != "kinematic":
        raise ValueError(f"unknown backend {cfg.backend!r}")
    for i in range(0 if replays else max(cfg.n_repeats, 1)):
        rc = ReplayConfig(**{**asdict(cfg.replay), "seed": cfg.replay.seed + 1000 * cfg.seed + i})
        replays.append(KinematicSim(arm, rc).replay(plan))
    truth = replays[0].truth
    tracker_pose = track(truth, cfg.tracker)

    # --- sensors + validation
    imu = simulate_imu(truth, imu_cfg)
    cal = calibrate(imu, tracker_pose)
    imu_cal = cal.apply(imu)
    ref = ideal_imu_signals(tracker_pose)

    report: dict = dict(
        config=dict(name=cfg.name, brush=cfg.brush, scenario=cfg.scenario, duration=cfg.duration, seed=cfg.seed,
                    expected_convention=cfg.expected_convention, backend=cfg.backend, imu=imu_cfg.to_dict(), tracker=asdict(cfg.tracker),
                    replay=asdict(cfg.replay)),
        plan=plan.report,
        replay=metrics.replay_accuracy(plan.target, tracker_pose),
        repeatability=metrics.repeatability([r.truth for r in replays]) if len(replays) > 1 else {},
        calibration=cal.summary(),
        imu_error_raw=metrics.imu_error(imu, tracker_pose),
        imu_error_calibrated=metrics.imu_error(imu_cal, tracker_pose),
    )

    # --- calibration vs the (known) simulated truth
    Mg, Ma = imu_cfg.mapping()
    exp_delay = imu_cfg.delay_s + lowpass_group_delay_s(imu_cfg)
    report["calibration_vs_truth"] = dict(
        nominal_delay_ms=1000 * imu_cfg.delay_s,
        expected_effective_delay_ms=1000 * exp_delay,
        delay_error_ms=cal.delay.delay_s * 1000 - 1000 * exp_delay,
        gyro_M_error=float(np.linalg.norm(cal.gyro.M - Mg)),
        gyro_bias_error_dps=float(np.rad2deg(np.linalg.norm(cal.gyro.bias - np.deg2rad(imu_cfg.gyro_bias_dps)))),
        convention_true=imu_cfg.axis_convention,
        convention_estimated=cal.gyro.convention,
    )

    # --- calibration repeatability across repeats (new noise realisation + execution noise each time)
    if len(replays) > 1:
        delays, mis, gbias = [], [], []
        for i, rr in enumerate(replays):
            ic = IMUConfig(**{**imu_cfg.to_dict(), "seed": imu_cfg.seed + i})
            im = simulate_imu(rr.truth, ic)
            c = calibrate(im, track(rr.truth, cfg.tracker))
            delays.append(1000 * c.delay.delay_s); mis.append(c.gyro.misalign_deg)
            gbias.append(np.rad2deg(c.gyro.bias))
        gbias = np.array(gbias)
        report["repeatability"].update(
            delay_std_ms=float(np.std(delays, ddof=1)), misalign_std_deg=float(np.nanstd(mis, ddof=1)),
            gyro_bias_std_dps=float(np.linalg.norm(gbias.std(0, ddof=1))),
        )

    # --- fusion: calibrated vs raw
    dt = imu.dt
    q0 = tracker_pose.quat[0]
    fus_cal, errs_cal, bias_cal = _fusion_block(q0, imu_cal, dt, tracker_pose.quat)
    fus_raw, errs_raw, _ = _fusion_block(q0, imu, dt, tracker_pose.quat)
    report["fusion_calibrated"] = fus_cal
    report["fusion_raw"] = fus_raw

    report["flags"] = diagnose(report, cfg.expected_convention)
    report["status"] = overall_status(report["flags"])
    report["runtime_s"] = round(time.time() - t_start, 2)

    sig = dict(
        t=imu.t, gt_pos=tracker_pose.pos, gt_quat=tracker_pose.quat,
        target_pos=plan.target.pos, target_quat=plan.target.quat, joints=replays[0].q,
        gyro_raw=imu.gyro, accel_raw=imu.accel, gyro_cal=imu_cal.gyro, accel_cal=imu_cal.accel,
        gyro_gt=ref["omega"], accel_gt=ref["f"],
        lags_s=cal.delay.lags_s, delay_score=cal.delay.corr, ekf_bias=bias_cal,
    )
    for tag, errs in (("cal", errs_cal), ("raw", errs_raw)):
        for m, e in errs.items():
            for kind, v in e.items():
                sig[f"err_{tag}_{m}_{kind}"] = v

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
        np.savez_compressed(out / "signals.npz", **sig)
        if make_plots:
            from .plots import plot_run
            plot_run(report, sig, out / "overview.png")
    report["_signals"] = sig
    return report


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and (o != o):
        return None
    return o


def run_brush_study(brushes=("circular", "sweep", "scrub", "mixed"), scenario="healthy", duration=20.0, seed=0, out_dir=None) -> dict:
    """Similarity between virtual brushes (motion styles) as seen through the same IMU."""
    imus = {}
    for b in brushes:
        plan, _ = get_plan(b, duration, seed)
        truth = KinematicSim(_ARM, ReplayConfig(seed=seed)).replay(plan).truth
        imus[b] = simulate_imu(truth, imu_config(scenario, seed=seed))
    res = metrics.brush_similarity(imus)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "brush_similarity.json").write_text(json.dumps(_jsonable(res), indent=2))
        from .plots import plot_brush_similarity
        plot_brush_similarity(res, Path(out_dir) / "brush_similarity.png")
    return res
