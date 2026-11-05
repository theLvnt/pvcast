"""Build forecasting samples from the 1-minute SKIPP'D stream.

A sample at issue time t uses only information available at t:
  * sky images at t-15, t-10, t-5, t min        (image modality)
  * PV power at the same lags                     (history modality)
  * last completed hourly weather record at t     (weather modality)
  * deterministic solar geometry / clear-sky PV at t and at t+h
and the target is PV power at t+h (default h = 15 min).

Images are not copied into the samples: `frame_idx` points into the stream's
image array, so a 25k-sample dataset costs a few MB instead of >1 GB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.skippd import PV_CAPACITY_KW
from .data.weather import VARIABLES, align_to_minutes
from .solar import ClearSkyPV, clear_sky_index, solar_table

LAGS = (15, 10, 5, 0)          # minutes before issue time, oldest first


@dataclass
class Samples:
    frame_idx: np.ndarray       # (N, len(LAGS)) indices into stream images
    X: pd.DataFrame             # tabular features
    y: np.ndarray               # PV at t+h [kW]
    pv_cs_target: np.ndarray    # clear-sky PV at t+h [kW]
    k_now: np.ndarray           # clear-sky index at t (for smart persistence)
    pv_now: np.ndarray          # PV at t (for persistence)
    times: np.ndarray           # issue times (naive local)
    stream_idx: np.ndarray      # index of issue time in the stream

    def __len__(self):
        return len(self.y)

    def subset(self, m: np.ndarray) -> "Samples":
        return Samples(self.frame_idx[m], self.X[m].reset_index(drop=True), self.y[m], self.pv_cs_target[m],
                       self.k_now[m], self.pv_now[m], self.times[m], self.stream_idx[m])


def _minute_keys(times: np.ndarray) -> np.ndarray:
    return times.astype("datetime64[m]").astype(np.int64)


def build_samples(stream: dict, cs_model: ClearSkyPV, weather: pd.DataFrame | None = None,
                  horizon: int = 15, stride: int = 2, lags=LAGS) -> Samples:
    times, pv = stream["times"], stream["pv"].astype(np.float64)
    K = _minute_keys(times)
    order = np.argsort(K, kind="stable")
    if not np.all(order == np.arange(len(K))):
        raise ValueError("stream must be time-sorted")

    def locate(offset):
        q = K + offset
        j = np.clip(np.searchsorted(K, q), 0, len(K) - 1)
        return j, K[j] == q

    lag_idx, ok = [], np.ones(len(K), bool)
    for l in lags:
        j, m = locate(-l)
        lag_idx.append(j)
        ok &= m
    tgt, m = locate(horizon)
    ok &= m
    cand = np.nonzero(ok)[0]
    cand = cand[(K[cand] % stride) == 0]

    sol = solar_table(times)
    pv_cs = cs_model(sol["ghi_cs"].values)
    k = clear_sky_index(pv, pv_cs)
    F = np.stack(lag_idx, 1)[cand]                       # (N, L)
    t_i = tgt[cand]
    cap = PV_CAPACITY_KW

    X = pd.DataFrame({f"pv_lag{l}": pv[F[:, j]] / cap for j, l in enumerate(lags)})
    for j, l in enumerate(lags):
        X[f"k_lag{l}"] = k[F[:, j]]
    X["k_std"] = np.std([k[F[:, j]] for j in range(len(lags))], axis=0)
    X["k_trend"] = k[F[:, -1]] - k[F[:, 0]]
    X["pvcs_now"] = pv_cs[cand] / cap
    X["pvcs_target"] = pv_cs[t_i] / cap
    X["cos_zen_target"] = np.cos(np.radians(sol["zenith"].values[t_i]))
    az = np.radians(sol["azimuth"].values[t_i])
    X["sin_az_target"], X["cos_az_target"] = np.sin(az), np.cos(az)
    ts = pd.DatetimeIndex(times[cand])
    hod = ts.hour + ts.minute / 60
    X["sin_hod"], X["cos_hod"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    X["sin_doy"], X["cos_doy"] = np.sin(2 * np.pi * ts.dayofyear / 365.25), np.cos(2 * np.pi * ts.dayofyear / 365.25)
    if weather is not None:
        w = align_to_minutes(weather, times[cand])
        for v in VARIABLES:
            X[f"w_{v}"] = w[v].values
        X["w_sw_ratio"] = w["shortwave_radiation"].values / np.maximum(sol["ghi_cs"].values[cand], 50)

    return Samples(frame_idx=F, X=X.astype(np.float32), y=pv[t_i].astype(np.float32),
                   pv_cs_target=pv_cs[t_i].astype(np.float32), k_now=k[cand].astype(np.float32),
                   pv_now=pv[cand].astype(np.float32), times=times[cand], stream_idx=cand)


def split_by_day(s: Samples, val_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Day-level split (no minute of a validation day appears in training)."""
    days = s.times.astype("datetime64[D]")
    u = np.unique(days)
    rng = np.random.default_rng(seed)
    val_days = set(rng.choice(u, max(1, int(len(u) * val_frac)), replace=False).tolist())
    m_val = np.array([d in val_days for d in days.tolist()])
    return ~m_val, m_val
