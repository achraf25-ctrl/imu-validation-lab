"""Validation tools: time synchronisation and frame calibration of IMU vs ground truth.

1. Delay: scan candidate lags; at each lag fit the 3-axis linear map IMU = M x_true + b
   and keep the lag with the highest explained variance (a multi-channel cross-
   correlation that is invariant to axis convention, scale and bias), refined to
   sub-sample with a parabola. Positive delay = IMU lags the tracker. This is the
   *effective* delay (pure transport delay + group delay of any on-board low-pass
   filter). A 1-D FFT cross-correlation (estimate_delay) is kept for scalar signals,
   but the norm of omega is a poor choice for near-constant-speed motions.
2. Frame / gain / bias: after shifting the IMU back in time, solve the linear
   least-squares problem  y_imu = M x_true + b  (3x3 M, 3-vector b) for gyro and
   accel. M is then decomposed into per-axis scale x (signed permutation) x small
   rotation, which exposes axis flips, remapping, scale error and misalignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags

from . import quat as Q
from .sensors import IMUData, convention_string, ideal_imu_signals
from .trajectory import Trajectory


# ------------------------------------------------------------------- time sync
@dataclass
class DelayEstimate:
    delay_s: float
    peak_corr: float
    lags_s: np.ndarray
    corr: np.ndarray


def estimate_delay(ref: np.ndarray, sig: np.ndarray, dt: float, max_lag_s: float = 0.5) -> DelayEstimate:
    """Delay of `sig` relative to `ref` (both 1-D, same sampling)."""
    a = ref - ref.mean()
    b = sig - sig.mean()
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")
    m = np.abs(lags) <= int(round(max_lag_s / dt))
    c, lags = c[m], lags[m]
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            frac = 0.5 * (y0 - y2) / den
    return DelayEstimate(delay_s=float((lags[k] + frac) * dt), peak_corr=float(c[k]), lags_s=lags * dt, corr=c)


def shift_signal(t: np.ndarray, x: np.ndarray, shift: float) -> np.ndarray:
    """Return x(t + shift) by linear interpolation (shift>0 moves the signal earlier)."""
    return np.stack([np.interp(t + shift, t, x[:, i]) for i in range(x.shape[1])], axis=1)


def estimate_delay_vec(ref: np.ndarray, sig: np.ndarray, dt: float, max_lag_s: float = 0.5) -> DelayEstimate:
    """Delay of `sig` (N,3) vs `ref` (N,3) allowing an unknown linear map and bias."""
    L = int(round(max_lag_s / dt))
    n = len(ref)
    L = min(L, (n - 10) // 4)
    lags = np.arange(-L, L + 1)
    X = np.hstack([ref[L:n - L], np.ones((n - 2 * L, 1))])
    # projection onto the span of X (same for every lag)
    Qx, _ = np.linalg.qr(X)
    score = np.empty(len(lags))
    for i, k in enumerate(lags):
        Y = sig[L + k:n - L + k]
        Yc = Y - Y.mean(0)
        resid = Y - Qx @ (Qx.T @ Y)
        score[i] = 1.0 - np.sum(resid**2) / max(np.sum(Yc**2), 1e-12)
    k = int(np.argmax(score))
    frac = 0.0
    if 0 < k < len(score) - 1:
        y0, y1, y2 = score[k - 1], score[k], score[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            frac = 0.5 * (y0 - y2) / den
    return DelayEstimate(delay_s=float((lags[k] + frac) * dt), peak_corr=float(score[k]), lags_s=lags * dt, corr=score)


# ---------------------------------------------------------------- frame calibration
@dataclass
class AxisCalibration:
    M: np.ndarray             # 3x3 estimated true -> sensor map (incl. scale)
    bias: np.ndarray          # 3
    scales: np.ndarray        # per sensor axis
    convention: str           # nearest signed permutation, e.g. "y,-x,z"
    det: float                # det of the snapped permutation (-1 = reflection)
    misalign_deg: float       # residual rotation after removing the permutation
    r2: float
    resid_rms: float          # rms residual after applying the fit (signal units)
    ambiguous: bool           # snapped matrix was not a valid permutation
    resid_structured_rms: float = 0.0  # residual rms left after removing the white-noise floor (from sample-to-sample differences)
    resid_autocorr: float = 0.0  # lag-1 autocorrelation of fit residuals (0 = white, ->1 = systematic)

    def inverse_apply(self, x: np.ndarray) -> np.ndarray:
        """Map sensor readings back to the tool frame: x_true = M^-1 (y - b)."""
        return (x - self.bias) @ np.linalg.inv(self.M).T


def _fit_axis(true_sig: np.ndarray, imu_sig: np.ndarray) -> AxisCalibration:
    X = np.hstack([true_sig, np.ones((len(true_sig), 1))])
    B, *_ = np.linalg.lstsq(X, imu_sig, rcond=None)
    M, b = B[:3].T, B[3]
    resid = imu_sig - X @ B
    ss_res = np.sum(resid**2, axis=0)
    ss_tot = np.sum((imu_sig - imu_sig.mean(0)) ** 2, axis=0)
    r2 = float(1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12))

    scales = np.linalg.norm(M, axis=1)
    Rhat = M / np.maximum(scales[:, None], 1e-12)
    P = np.zeros((3, 3))
    for i in range(3):
        j = int(np.argmax(np.abs(Rhat[i])))
        P[i, j] = np.sign(Rhat[i, j]) or 1.0
    ambiguous = not np.allclose(np.abs(P).sum(0), 1)
    det = float(np.linalg.det(P)) if not ambiguous else float("nan")
    Rres = Rhat @ P.T
    ang = float(np.rad2deg(np.arccos(np.clip((np.trace(Rres) - 1) / 2, -1, 1)))) if not ambiguous else float("nan")
    ac = []
    for i in range(3):
        r_ = resid[:, i]
        ac.append(np.corrcoef(r_[:-1], r_[1:])[0, 1] if r_.std() > 1e-9 else 0.0)
    white = np.std(np.diff(resid, axis=0), axis=0) / np.sqrt(2.0)       # per-axis white-noise std
    total = np.sqrt(np.mean(resid**2, axis=0))
    structured = float(np.sqrt(np.mean(np.maximum(total**2 - white**2, 0.0))))
    return AxisCalibration(
        resid_structured_rms=structured,
        resid_autocorr=float(np.nanmean(ac)),
        M=M, bias=b, scales=scales,
        convention=convention_string(P) if not ambiguous else "ambiguous",
        det=det, misalign_deg=ang, r2=r2,
        resid_rms=float(np.sqrt(ss_res.sum() / resid.size)), ambiguous=ambiguous,
    )


@dataclass
class Calibration:
    delay: DelayEstimate
    gyro: AxisCalibration
    accel: AxisCalibration

    def apply(self, imu: IMUData) -> IMUData:
        """Time-align and map IMU data into the tool frame with calibrated gain/bias removed."""
        gyro = shift_signal(imu.t, imu.gyro, self.delay.delay_s)
        acc = shift_signal(imu.t, imu.accel, self.delay.delay_s)
        return IMUData(t=imu.t, gyro=self.gyro.inverse_apply(gyro), accel=self.accel.inverse_apply(acc))

    def summary(self) -> dict:
        return dict(
            delay_ms=1000 * self.delay.delay_s,
            delay_peak_corr=self.delay.peak_corr,
            gyro=dict(convention=self.gyro.convention, det=self.gyro.det, scales=self.gyro.scales.round(4).tolist(),
                      bias_dps=np.rad2deg(self.gyro.bias).round(4).tolist(), misalign_deg=self.gyro.misalign_deg,
                      r2=self.gyro.r2, resid_rms_dps=float(np.rad2deg(self.gyro.resid_rms)),
                      resid_autocorr=self.gyro.resid_autocorr, resid_structured_rms=self.gyro.resid_structured_rms),
            accel=dict(convention=self.accel.convention, det=self.accel.det, scales=self.accel.scales.round(4).tolist(),
                       bias=self.accel.bias.round(4).tolist(), misalign_deg=self.accel.misalign_deg,
                       r2=self.accel.r2, resid_rms=self.accel.resid_rms,
                       resid_autocorr=self.accel.resid_autocorr, resid_structured_rms=self.accel.resid_structured_rms),
        )


def calibrate(imu: IMUData, truth_pose: Trajectory, max_lag_s: float = 0.5, trim_s: float = 0.5) -> Calibration:
    """Estimate delay, then frame / scale / bias for gyro and accel against tracker truth."""
    ref = ideal_imu_signals(truth_pose)
    dt = imu.dt
    est = estimate_delay_vec(ref["omega"], imu.gyro, dt, max_lag_s)
    gyro = shift_signal(imu.t, imu.gyro, est.delay_s)
    acc = shift_signal(imu.t, imu.accel, est.delay_s)
    k = int(trim_s / dt)  # drop edge samples (finite-difference / interpolation edge effects)
    sl = slice(k, len(imu.t) - k)
    return Calibration(
        delay=est,
        gyro=_fit_axis(ref["omega"][sl], gyro[sl]),
        accel=_fit_axis(ref["f"][sl], acc[sl]),
    )
