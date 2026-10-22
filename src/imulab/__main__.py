"""CLI:  python -m imulab demo | run | list"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_brush_study, run_experiment
from .plots import plot_summary
from .scenarios import SCENARIOS
from .trajectory import BRUSH_STYLES

DEMO_SCENARIOS = ["ideal", "healthy", "delayed", "axis_flipped", "axis_remapped", "misaligned", "biased", "scaled",
                  "noisy", "filtered", "lever_arm", "worst_case"]


def _print_row(rep: dict) -> None:
    c = rep["calibration"]
    codes = ",".join(f["code"] for f in rep["flags"]) or "-"
    print(f"{rep['config']['name']:<28s} {rep['status']:<5s} delay={c['delay_ms']:+6.1f}ms conv={c['gyro']['convention']:<8s} "
          f"EKF raw/cal={rep['fusion_raw']['ekf']['total_rmse_deg']:6.2f}/{rep['fusion_calibrated']['ekf']['total_rmse_deg']:5.2f} deg  [{codes}]",
          flush=True)


def cmd_demo(a) -> None:
    out = Path(a.out)
    reports = []
    for sc in DEMO_SCENARIOS:
        cfg = ExperimentConfig(name=f"{a.brush}_{sc}", brush=a.brush, scenario=sc, duration=a.duration, n_repeats=a.repeats)
        rep = run_experiment(cfg, out / cfg.name, make_plots=not a.no_plots)
        _print_row(rep)
        reports.append(rep)
    if not a.no_plots:
        plot_summary(reports, out / "summary.png")
    res = run_brush_study(duration=a.duration, out_dir=out / "_brush_study" if not a.no_plots else None)
    print("brush similarity (spectral cosine):")
    for n, row in zip(res["names"], res["spectral_cosine"]):
        print(f"  {n:<9s}", " ".join(f"{v:5.2f}" for v in row))
    print(f"\nruns written to {out}/  ->  streamlit run dashboard/app.py")


def cmd_run(a) -> None:
    over = json.loads(a.override) if a.override else {}
    cfg = ExperimentConfig(name=a.name or f"{a.brush}_{a.scenario}", brush=a.brush, scenario=a.scenario,
                           duration=a.duration, seed=a.seed, imu_overrides=over, n_repeats=a.repeats, backend=a.backend)
    rep = run_experiment(cfg, Path(a.out) / cfg.name)
    _print_row(rep)
    for f in rep["flags"]:
        print(f"  [{f['severity']:<4s}] {f['code']}: {f['message']}")


def cmd_list(_a) -> None:
    print("brushes  :", ", ".join(BRUSH_STYLES))
    print("scenarios:", ", ".join(SCENARIOS))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="imulab", description="Virtual IMU validation lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run all fault scenarios and write runs/")
    d.add_argument("--out", default="runs"); d.add_argument("--brush", default="circular", choices=BRUSH_STYLES)
    d.add_argument("--duration", type=float, default=20.0); d.add_argument("--repeats", type=int, default=5)
    d.add_argument("--no-plots", action="store_true"); d.set_defaults(fn=cmd_demo)
    r = sub.add_parser("run", help="run one scenario")
    r.add_argument("--out", default="runs"); r.add_argument("--name"); r.add_argument("--brush", default="circular", choices=BRUSH_STYLES)
    r.add_argument("--scenario", default="healthy", choices=SCENARIOS); r.add_argument("--duration", type=float, default=20.0)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--backend", default="kinematic", choices=["kinematic", "pybullet"]); r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--override", help='JSON of IMUConfig overrides, e.g. \'{"delay_s": 0.08}\''); r.set_defaults(fn=cmd_run)
    l = sub.add_parser("list", help="list brushes and scenarios"); l.set_defaults(fn=cmd_list)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
