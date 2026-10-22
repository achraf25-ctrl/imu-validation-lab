"""Matplotlib figures (headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = dict(ok="#2e7d32", warn="#ef6c00", fail="#c62828", info="#1565c0")


def plot_run(report: dict, sig: dict, path: str | Path) -> None:
    t = sig["t"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    cfg = report["config"]
    fig.suptitle(f"{cfg['name']}  |  brush={cfg['brush']}  scenario={cfg['scenario']}  |  status: {report['status']}",
                 fontsize=13, color=COL[{"OK": "ok", "WARN": "warn", "FAIL": "fail"}[report["status"]]], weight="bold")

    a = ax[0, 0]
    a.plot(sig["target_pos"][:, 0], sig["target_pos"][:, 1], "k--", lw=1, label="commanded")
    a.plot(sig["gt_pos"][:, 0], sig["gt_pos"][:, 1], lw=1, label="replayed (tracker)")
    a.set_aspect("equal"); a.set_title(f"Tool path (top view), replay RMSE {report['replay']['pos_rmse_mm']:.2f} mm")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.legend(fontsize=8)

    a = ax[0, 1]
    m = (t >= 2) & (t <= 7)
    a.plot(t[m], np.rad2deg(sig["gyro_gt"][m, 0]), "k", lw=1.2, label="truth ω_x (tool frame)")
    a.plot(t[m], np.rad2deg(sig["gyro_raw"][m, 0]), lw=1, label="IMU ch.0 raw")
    a.plot(t[m], np.rad2deg(sig["gyro_cal"][m, 0]), lw=1, ls=":", label="IMU ch.0 calibrated")
    a.set_title("Gyro vs truth (5 s excerpt)"); a.set_xlabel("t [s]"); a.set_ylabel("deg/s"); a.legend(fontsize=7)

    a = ax[0, 2]
    a.plot(1000 * sig["lags_s"], sig["delay_score"], lw=1.2)
    d = report["calibration"]["delay_ms"]
    a.axvline(d, color="r", ls="--", label=f"estimated {d:+.1f} ms")
    a.set_title("Time sync: explained variance vs lag"); a.set_xlabel("lag [ms]"); a.set_ylabel("R²"); a.legend(fontsize=8)

    a = ax[1, 0]
    for m_, c in (("gyro_only", "C0"), ("complementary", "C1"), ("ekf", "C2")):
        a.plot(t, sig[f"err_cal_{m_}_total_deg"], c, lw=1, label=f"{m_} (calibrated)")
    a.plot(t, sig["err_raw_ekf_total_deg"], "C3", lw=1, ls="--", label="ekf on RAW IMU")
    a.set_yscale("symlog", linthresh=1.0); a.set_title("Orientation error vs ground truth")
    a.set_xlabel("t [s]"); a.set_ylabel("deg"); a.legend(fontsize=7)

    a = ax[1, 1]
    a.plot(t, sig["err_cal_ekf_tilt_deg"], label="EKF tilt")
    a.plot(t, sig["err_cal_ekf_yaw_deg"], label="EKF heading")
    a.plot(t, sig["err_cal_gyro_only_yaw_deg"], label="gyro-only heading", alpha=0.7)
    a.set_title("Tilt vs heading error (calibrated)"); a.set_xlabel("t [s]"); a.set_ylabel("deg"); a.legend(fontsize=8)

    a = ax[1, 2]
    a.axis("off")
    lines = [f"Flags ({len(report['flags'])}):"]
    for f in report["flags"][:9]:
        lines.append(f"[{f['severity'].upper():4s}] {f['code']}")
    if not report["flags"]:
        lines.append("none - sensor healthy")
    c = report["calibration"]
    lines += ["", f"delay {c['delay_ms']:+.1f} ms   conv {c['gyro']['convention']}",
              f"misalign {c['gyro']['misalign_deg']:.2f}°   gyro R² {c['gyro']['r2']:.4f}",
              f"slowdown ×{report['plan']['slowdown_factor']:.2f}"]
    a.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(reports: list[dict], path: str | Path) -> None:
    names = [r["config"]["scenario"] for r in reports]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    est = [r["calibration"]["delay_ms"] for r in reports]
    exp = [r["calibration_vs_truth"]["expected_effective_delay_ms"] for r in reports]
    ax[0].bar(x - 0.2, exp, 0.4, label="expected (delay + filter lag)")
    ax[0].bar(x + 0.2, est, 0.4, label="estimated by cross-correlation")
    ax[0].set_title("Delay recovery"); ax[0].set_ylabel("ms")
    cal = [r["fusion_calibrated"]["ekf"]["total_rmse_deg"] for r in reports]
    raw = [r["fusion_raw"]["ekf"]["total_rmse_deg"] for r in reports]
    ax[1].bar(x - 0.2, raw, 0.4, color="#c62828", label="EKF on raw IMU")
    ax[1].bar(x + 0.2, cal, 0.4, color="#2e7d32", label="EKF after calibration")
    ax[1].set_yscale("log"); ax[1].set_title("Orientation RMSE vs ground truth"); ax[1].set_ylabel("deg (log)")
    gr = [r["fusion_raw"]["gyro_only"]["total_rmse_deg"] for r in reports]
    gc = [r["fusion_calibrated"]["gyro_only"]["total_rmse_deg"] for r in reports]
    ax[2].bar(x - 0.2, gr, 0.4, color="#c62828", label="gyro-only, raw")
    ax[2].bar(x + 0.2, gc, 0.4, color="#2e7d32", label="gyro-only, calibrated")
    ax[2].set_yscale("log"); ax[2].set_title("Dead-reckoning RMSE"); ax[2].set_ylabel("deg (log)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=40, ha="right", fontsize=8); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_brush_similarity(res: dict, path: str | Path) -> None:
    n = res["names"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, key, title, cm in ((ax[0], "spectral_cosine", "Spectral similarity (cosine, 1 = same)", "viridis"),
                              (ax[1], "dtw_distance", "DTW distance of |ω| (0 = same)", "magma_r")):
        M = np.array(res[key])
        im = a.imshow(M, cmap=cm)
        a.set_xticks(range(len(n))); a.set_xticklabels(n); a.set_yticks(range(len(n))); a.set_yticklabels(n)
        for i in range(len(n)):
            for j in range(len(n)):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        a.set_title(title, fontsize=10); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
