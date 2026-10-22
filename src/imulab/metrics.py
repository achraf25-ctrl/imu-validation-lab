"""Quantitative evaluation: replay accuracy, IMU error, repeatability, brush similarity."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from . import quat as Q
from .sensors import IMUData, ideal_imu_signals
from .trajectory import Trajectory


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def replay_accuracy(target: Trajectory, actual: Trajectory) -> dict:
    """How well did the arm replay the commanded tool trajectory? (target vs tracker)"""
    n = min(len(target), len(actual))
    e_p = np.linalg.norm(actual.pos[:n] - target.pos[:n], axis=1) * 1000.0
    e_r = np.rad2deg(Q.angle_between(actual.quat[:n], target.quat[:n]))
    return dict(
        pos_rmse_mm=_rms(e_p), pos_max_mm=float(e_p.max()), pos_p95_mm=float(np.percentile(e_p, 95)),
        rot_rmse_deg=_rms(e_r), rot_max_deg=float(e_r.max()),
    )


def imu_error(imu: IMUData, truth_pose: Trajectory, trim_s: float = 0.5) -> dict:
    """RMS error of IMU signals vs the tracker-derived truth (both must be in the tool frame)."""
    ref = ideal_imu_signals(truth_pose)
    k = int(trim_s / imu.dt)
    sl = slice(k, len(imu.t) - k)
    g = imu.gyro[sl] - ref["omega"][sl]
    a = imu.accel[sl] - ref["f"][sl]
    return dict(
        gyro_rmse_dps=float(np.rad2deg(_rms(g))),
        gyro_bias_dps=np.rad2deg(g.mean(0)).round(4).tolist(),
        accel_rmse=_rms(a),
        accel_bias=a.mean(0).round(4).tolist(),
        gyro_corr=float(np.mean([np.corrcoef(imu.gyro[sl][:, i], ref["omega"][sl][:, i])[0, 1] for i in range(3)])),
    )


def repeatability(runs: list[Trajectory]) -> dict:
    """ISO-9283-flavoured repeatability of repeated replays of the same trajectory.

    Per time sample: distance of each run to the centroid of the runs; RP = mean(l) + 3 S
    (position). Orientation repeatability: rms angle to the mean quaternion.
    """
    n = min(len(r) for r in runs)
    P = np.stack([r.pos[:n] for r in runs])                     # (R, N, 3)
    cen = P.mean(0)
    dist = np.linalg.norm(P - cen, axis=2) * np.sqrt(len(runs) / max(len(runs) - 1, 1))  # unbiased-ish
    l_bar = dist.mean(0)
    s = dist.std(0, ddof=1) if len(runs) > 2 else np.zeros(n)
    Qs = np.stack([Q.make_continuous(r.quat[:n]) for r in runs])
    # align signs to run 0 then average
    for i in range(1, len(runs)):
        flip = np.sum(Qs[i] * Qs[0], axis=-1) < 0
        Qs[i][flip] *= -1
    qm = Q.normalize(Qs.mean(0))
    ang = np.rad2deg(Q.angle_between(Qs, qm[None]))
    return dict(
        n_runs=len(runs),
        pos_rp_mm=float(1000 * (l_bar.mean() + 3 * s.mean())),
        pos_mean_dev_mm=float(1000 * l_bar.mean()),
        rot_rms_dev_deg=_rms(ang),
    )


# ------------------------------------------------------------ brush similarity
def _psd_signature(x: np.ndarray, fs: float, fmax: float = 20.0, nbins: int = 64) -> np.ndarray:
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(len(x), 512))
    grid = np.linspace(0.05, fmax, nbins)
    sig = np.interp(grid, f, p)
    sig = np.log10(sig + 1e-12)
    return sig - sig.mean()


def brush_features(imu: IMUData) -> dict:
    fs = 1.0 / imu.dt
    wn = np.linalg.norm(imu.gyro, axis=1)
    fn = np.linalg.norm(imu.accel - imu.accel.mean(0), axis=1)
    f, p = welch(wn - wn.mean(), fs=fs, nperseg=min(len(wn), 512))
    return dict(
        gyro_rms_dps=float(np.rad2deg(_rms(wn))),
        accel_dyn_rms=_rms(fn),
        dominant_hz=float(f[1:][np.argmax(p[1:])]),
        sig=np.concatenate([_psd_signature(wn, fs), _psd_signature(fn, fs)]),
    )


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 60) -> float:
    """Normalised DTW distance (Sakoe-Chiba band) between two 1-D series."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    w = max(band, abs(n - m))
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def brush_similarity(imus: dict[str, IMUData]) -> dict:
    """Pairwise similarity between brushes. Cosine of spectral signatures (1 = same
    spectral content) and normalised DTW distance of |omega| (0 = same waveform)."""
    names = list(imus)
    feats = {k: brush_features(v) for k, v in imus.items()}
    cos = np.zeros((len(names), len(names)))
    dtw = np.zeros_like(cos)
    wn = {k: np.linalg.norm(v.gyro, axis=1)[:: 6] for k, v in imus.items()}  # 20 Hz for DTW cost
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sa, sb = feats[a]["sig"], feats[b]["sig"]
            cos[i, j] = float(np.dot(sa, sb) / (np.linalg.norm(sa) * np.linalg.norm(sb) + 1e-12))
            dtw[i, j] = 0.0 if i == j else dtw_distance(wn[a][:400], wn[b][:400])
    return dict(
        names=names,
        spectral_cosine=cos.round(4).tolist(),
        dtw_distance=dtw.round(4).tolist(),
        features={k: {kk: vv for kk, vv in f.items() if kk != "sig"} for k, f in feats.items()},
    )
