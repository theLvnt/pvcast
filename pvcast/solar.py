"""Solar geometry, clear-sky irradiance and a clear-sky PV reference.

The clear-sky index k* = PV / PV_clearsky removes the deterministic daily and
seasonal cycle, so models learn cloud effects only. This is the standard trick
behind "smart persistence", the reference every solar forecast is scored against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from .data.skippd import SITE_ALT, SITE_LAT, SITE_LON, SITE_TZ


def localize(local_times: np.ndarray, tz: str = SITE_TZ) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(local_times).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")


def solar_table(local_times: np.ndarray, lat=SITE_LAT, lon=SITE_LON, alt=SITE_ALT) -> pd.DataFrame:
    """Zenith, azimuth and Ineichen clear-sky GHI/DNI/DHI for naive local timestamps."""
    idx = localize(local_times)
    ok = ~idx.isna()
    out = pd.DataFrame(np.nan, index=range(len(idx)),
                       columns=["zenith", "azimuth", "ghi_cs", "dni_cs", "dhi_cs"])
    if ok.any():
        loc = pvlib.location.Location(lat, lon, altitude=alt)
        sp = loc.get_solarposition(idx[ok])
        cs = loc.get_clearsky(idx[ok], model="ineichen", solar_position=sp)
        out.loc[ok, "zenith"] = sp["apparent_zenith"].values
        out.loc[ok, "azimuth"] = sp["azimuth"].values
        out.loc[ok, ["ghi_cs", "dni_cs", "dhi_cs"]] = cs[["ghi", "dni", "dhi"]].values
    return out


class ClearSkyPV:
    """PV_cs(t) = scale * GHI_cs(t)^gamma, fitted to the upper envelope of observed PV.

    The array's tilt/azimuth are not part of the dataset, so instead of a full
    plane-of-array model we fit a two-parameter envelope on high-sun samples
    (upper `quantile` of PV in GHI_cs bins). Good enough to normalise the
    diurnal/seasonal cycle; replace with pvlib ModelChain if orientation is known.
    """

    def __init__(self, quantile: float = 0.97):
        self.q = quantile
        self.scale, self.gamma = 0.03, 1.0

    def fit(self, pv: np.ndarray, ghi_cs: np.ndarray, n_bins: int = 25) -> "ClearSkyPV":
        m = (ghi_cs > 100) & np.isfinite(pv)
        g, p = ghi_cs[m], pv[m]
        edges = np.quantile(g, np.linspace(0, 1, n_bins + 1))
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (g >= a) & (g < b)
            if s.sum() > 50:
                xs.append(np.median(g[s]))
                ys.append(np.quantile(p[s], self.q))
        xs, ys = np.log(np.array(xs)), np.log(np.maximum(np.array(ys), 1e-3))
        self.gamma, b = np.polyfit(xs, ys, 1)
        self.scale = float(np.exp(b))
        return self

    def __call__(self, ghi_cs: np.ndarray) -> np.ndarray:
        g = np.clip(np.nan_to_num(ghi_cs), 0, None)
        return self.scale * g ** self.gamma


def clear_sky_index(pv: np.ndarray, pv_cs: np.ndarray, floor: float = 0.5, cap: float = 1.5) -> np.ndarray:
    """k* with a floor on the denominator to avoid blow-ups near sunrise/sunset."""
    return np.clip(pv / np.maximum(pv_cs, floor), 0.0, cap)
