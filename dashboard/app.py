"""Streamlit dashboard: browse runs, error plots, automatic health flags, live what-if runs.

    streamlit run dashboard/app.py          (reads ./runs, or $IMULAB_RUNS)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from imulab.experiment import ExperimentConfig, run_experiment
from imulab.scenarios import SCENARIOS
from imulab.trajectory import BRUSH_STYLES

st.set_page_config(page_title="IMU validation lab", layout="wide", page_icon="🦾")
RUNS = Path(os.environ.get("IMULAB_RUNS", "runs"))
BADGE = {"OK": "🟢", "WARN": "🟠", "FAIL": "🔴"}
SEV_ICON = {"fail": "🔴", "warn": "🟠", "info": "🔵"}


@st.cache_data(show_spinner=False)
def load_report(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def load_signals(path: str, mtime: float) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def list_runs() -> list[Path]:
    return sorted(p.parent for p in RUNS.glob("*/report.json")) if RUNS.exists() else []


def rep_of(run: Path) -> dict:
    f = run / "report.json"
    return load_report(str(f), f.stat().st_mtime)


def sig_of(run: Path) -> dict:
    f = run / "signals.npz"
    return load_signals(str(f), f.stat().st_mtime)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🦾 IMU validation lab")
st.sidebar.caption(f"runs directory: `{RUNS}`")

with st.sidebar.expander("▶ Run a live simulation", expanded=False):
    brush = st.selectbox("Brush motion", list(BRUSH_STYLES), key="lv_brush")
    base = st.selectbox("Start from scenario", list(SCENARIOS), index=list(SCENARIOS).index("healthy"), key="lv_base")
    delay_ms = st.slider("Extra delay (ms)", 0, 150, 0, key="lv_delay")
    axis = st.selectbox("Axis convention", ["x,y,z", "y,-x,z", "-x,-y,z", "x,y,-z", "-x,y,z", "z,x,y"], key="lv_axis")
    gbias = st.slider("Gyro bias, all axes (deg/s)", 0.0, 3.0, 0.0, 0.1, key="lv_gb")
    noise = st.slider("Gyro noise (deg/s)", 0.0, 3.0, 0.15, 0.05, key="lv_noise")
    lp = st.slider("Low-pass cutoff (Hz, 0 = off)", 0, 40, 0, key="lv_lp")
    dur = st.slider("Duration (s)", 8, 40, 15, key="lv_dur")
    if st.button("Run simulation", type="primary"):
        over = dict(axis_convention=axis, gyro_noise_dps=noise,
                    gyro_bias_dps=(gbias, -gbias, gbias), lowpass_hz=float(lp) if lp else None)
        if delay_ms:
            over["delay_s"] = delay_ms / 1000
        name = f"live_{brush}_{base}_{delay_ms}ms_{axis.replace(',', '').replace('-', 'm')}"
        with st.spinner("Planning, replaying, calibrating, fusing…"):
            run_experiment(ExperimentConfig(name=name, brush=brush, scenario=base, duration=float(dur),
                                            imu_overrides=over, n_repeats=3), RUNS / name)
        st.cache_data.clear()
        st.success(f"Saved {name}")

runs = list_runs()
if not runs:
    st.title("IMU validation lab")
    st.info("No runs found. Generate some with `python -m imulab demo` or use the live-run panel in the sidebar.")
    st.stop()

tab_over, tab_run, tab_cmp = st.tabs(["Overview", "Run detail", "Compare / brushes"])

# ----------------------------------------------------------------- overview
with tab_over:
    st.header("All runs")
    rows = []
    for r in runs:
        rp = rep_of(r)
        c = rp["calibration"]
        rows.append({
            "status": f"{BADGE[rp['status']]} {rp['status']}", "run": r.name, "brush": rp["config"]["brush"],
            "scenario": rp["config"]["scenario"], "delay [ms]": round(c["delay_ms"], 1),
            "axes": c["gyro"]["convention"], "misalign [°]": round(c["gyro"]["misalign_deg"], 2),
            "gyro bias max [°/s]": round(max(abs(x) for x in c["gyro"]["bias_dps"]), 2),
            "EKF raw [°]": round(rp["fusion_raw"]["ekf"]["total_rmse_deg"], 2),
            "EKF calibrated [°]": round(rp["fusion_calibrated"]["ekf"]["total_rmse_deg"], 2),
            "flags": ", ".join(f["code"] for f in rp["flags"]) or "–",
        })
    df = pd.DataFrame(rows)
    n_fail = sum("FAIL" in s for s in df["status"]); n_warn = sum("WARN" in s for s in df["status"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", len(df)); c2.metric("🟢 OK", len(df) - n_fail - n_warn)
    c3.metric("🟠 Warn", n_warn); c4.metric("🔴 Fail", n_fail)
    st.dataframe(df, use_container_width=True, hide_index=True)
    summ = RUNS / "summary.png"
    if summ.exists():
        st.image(str(summ), caption="Calibration recovers delay and removes most orientation error")

# --------------------------------------------------------------- run detail
with tab_run:
    sel = st.selectbox("Run", [r.name for r in runs], key="sel_run")
    run = RUNS / sel
    rp, sg = rep_of(run), sig_of(run)
    cal = rp["calibration"]
    banner = f"{BADGE[rp['status']]} **{rp['status']}** — {rp['config']['brush']} / {rp['config']['scenario']}"
    {"OK": st.success, "WARN": st.warning, "FAIL": st.error}[rp["status"]](banner)

    k = st.columns(5)
    k[0].metric("Effective delay", f"{cal['delay_ms']:+.1f} ms")
    k[1].metric("Axis convention", cal["gyro"]["convention"], "reflection!" if cal["gyro"]["det"] < 0 else None,
                delta_color="inverse")
    k[2].metric("Replay error", f"{rp['replay']['pos_rmse_mm']:.2f} mm")
    k[3].metric("Gyro fit R²", f"{cal['gyro']['r2']:.4f}")
    k[4].metric("EKF error raw → cal", f"{rp['fusion_raw']['ekf']['total_rmse_deg']:.1f}° → {rp['fusion_calibrated']['ekf']['total_rmse_deg']:.2f}°")

    st.subheader("Diagnostics")
    if not rp["flags"]:
        st.success("No flags: sensor behaves like the reference.")
    for f in rp["flags"]:
        txt = f"{SEV_ICON[f['severity']]} **{f['code']}** — {f['message']}"
        {"fail": st.error, "warn": st.warning, "info": st.info}[f["severity"]](txt)

    t = sg["t"]
    left, right = st.columns(2)
    with left:
        ax = st.radio("Gyro channel", ["x", "y", "z"], horizontal=True)
        i = "xyz".index(ax)
        t0, t1 = st.slider("Time window (s)", float(t[0]), float(t[-1]), (2.0, min(8.0, float(t[-1]))))
        m = (t >= t0) & (t <= t1)
        fig = go.Figure()
        fig.add_scatter(x=t[m], y=np.rad2deg(sg["gyro_gt"][m, i]), name="truth (tool frame)", line=dict(color="black"))
        fig.add_scatter(x=t[m], y=np.rad2deg(sg["gyro_raw"][m, i]), name="IMU raw (sensor axis)")
        fig.add_scatter(x=t[m], y=np.rad2deg(sg["gyro_cal"][m, i]), name="IMU calibrated", line=dict(dash="dot"))
        fig.update_layout(title="Gyro vs ground truth", xaxis_title="t [s]", yaxis_title="deg/s", height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=1000 * sg["lags_s"], y=sg["delay_score"], name="R²")
        fig.add_vline(x=cal["delay_ms"], line_dash="dash", line_color="red", annotation_text=f"{cal['delay_ms']:+.1f} ms")
        fig.update_layout(title="Time synchronisation (explained variance vs lag)", xaxis_title="lag [ms]",
                          yaxis_title="R²", height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    l2, r2 = st.columns(2)
    with l2:
        fig = go.Figure()
        for mth, col in (("gyro_only", "#1f77b4"), ("complementary", "#ff7f0e"), ("ekf", "#2ca02c")):
            fig.add_scatter(x=t, y=sg[f"err_cal_{mth}_total_deg"], name=f"{mth} (calibrated)", line=dict(color=col))
        fig.add_scatter(x=t, y=sg["err_raw_ekf_total_deg"], name="EKF on raw IMU", line=dict(color="#d62728", dash="dash"))
        fig.update_layout(title="Orientation error vs ground truth", xaxis_title="t [s]", yaxis_title="deg",
                          yaxis_type="log", height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_scatter(x=sg["target_pos"][:, 0], y=sg["target_pos"][:, 1], name="commanded", line=dict(color="black", dash="dash"))
        fig.add_scatter(x=sg["gt_pos"][:, 0], y=sg["gt_pos"][:, 1], name="replayed")
        fig.update_layout(title="Tool path (top view)", xaxis_title="x [m]", yaxis_title="y [m]", height=360, margin=dict(t=40))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Calibration details"):
        g, a = cal["gyro"], cal["accel"]
        st.write(pd.DataFrame({
            "": ["convention", "det (−1 = flipped)", "scales", "bias", "misalignment [°]", "fit R²", "residual rms"],
            "gyro": [g["convention"], g["det"], g["scales"], g["bias_dps"], round(g["misalign_deg"], 3), round(g["r2"], 5), round(g["resid_rms_dps"], 4)],
            "accel": [a["convention"], a["det"], a["scales"], a["bias"], round(a["misalign_deg"], 3), round(a["r2"], 5), round(a["resid_rms"], 4)],
        }).astype(str))
        st.json(rp["calibration_vs_truth"])
    with st.expander("Trajectory limits & repeatability"):
        st.json({"plan": rp["plan"], "replay": rp["replay"], "repeatability": rp["repeatability"]})
    with st.expander("Full report JSON"):
        st.json(rp)

# ------------------------------------------------------------------ compare
with tab_cmp:
    st.header("Compare runs")
    names = st.multiselect("Runs", [r.name for r in runs], default=[r.name for r in runs])
    if names:
        reps = {n: rep_of(RUNS / n) for n in names}
        fig = go.Figure()
        fig.add_bar(x=names, y=[reps[n]["fusion_raw"]["ekf"]["total_rmse_deg"] for n in names], name="EKF on raw IMU", marker_color="#d62728")
        fig.add_bar(x=names, y=[reps[n]["fusion_calibrated"]["ekf"]["total_rmse_deg"] for n in names], name="EKF after calibration", marker_color="#2ca02c")
        fig.update_layout(barmode="group", yaxis_type="log", yaxis_title="orientation RMSE [deg]", height=380, title="Effect of calibration")
        st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure()
        fig.add_bar(x=names, y=[reps[n]["calibration_vs_truth"]["expected_effective_delay_ms"] for n in names], name="expected")
        fig.add_bar(x=names, y=[reps[n]["calibration"]["delay_ms"] for n in names], name="estimated")
        fig.update_layout(barmode="group", yaxis_title="ms", height=340, title="Delay recovery (expected = transport delay + filter group delay)")
        st.plotly_chart(fig, use_container_width=True)
    bs = RUNS / "_brush_study" / "brush_similarity.json"
    if bs.exists():
        st.subheader("Similarity between virtual brushes")
        res = json.loads(bs.read_text())
        c1, c2 = st.columns(2)
        for col, key, title, scale in ((c1, "spectral_cosine", "Spectral similarity (1 = same)", "Viridis"),
                                       (c2, "dtw_distance", "DTW distance of |ω| (0 = same)", "Magma_r")):
            fig = go.Figure(go.Heatmap(z=res[key], x=res["names"], y=res["names"], colorscale=scale,
                                       text=np.round(res[key], 2), texttemplate="%{text}"))
            fig.update_layout(title=title, height=360)
            col.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(res["features"]).T.round(3))
