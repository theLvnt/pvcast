"""Open-Meteo historical weather (ERA5-based archive API) for the SKIPP'D site.

Open-Meteo data is CC BY 4.0 (https://open-meteo.com). No API key needed.

Leakage control: the archive is hourly *reanalysis*, i.e. it describes what
actually happened. Linearly interpolating it to minutes would leak information
from the next hour into a 15-minute-ahead forecast. `align_to_minutes` therefore
uses the most recent *completed* hour only (value valid at HH:00 is used from
HH:00 onward, never earlier). In operations, replace this source with an NWP
forecast issued before the forecast time.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .skippd import SITE_LAT, SITE_LON, SITE_TZ

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
             "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "temperature_2m",
             "relative_humidity_2m", "wind_speed_10m"]


def _ranges(days: list[date], max_gap: int = 1) -> list[tuple[date, date]]:
    days = sorted(set(days))
    out, start, prev = [], days[0], days[0]
    for d in days[1:]:
        if (d - prev).days > max_gap:
            out.append((start, prev))
            start = d
        prev = d
    out.append((start, prev))
    return out


def fetch_open_meteo(days: list[date], out_csv: str | Path, lat: float = SITE_LAT, lon: float = SITE_LON,
                     pause: float = 0.3) -> Path:
    out = Path(out_csv)
    if out.exists():
        return out
    frames = []
    # group consecutive days into single requests; Open-Meteo accepts long ranges
    for a, b in _ranges(days, max_gap=10):
        q = {"latitude": lat, "longitude": lon, "start_date": str(a), "end_date": str(b),
             "hourly": ",".join(VARIABLES), "timezone": "UTC"}
        url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(q)}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    j = json.loads(r.read())
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        h = pd.DataFrame(j["hourly"])
        frames.append(h)
        time.sleep(pause)
    df = pd.concat(frames).drop_duplicates("time")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.sort_values("time").to_csv(out, index=False)
    return out


def load_weather(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def align_to_minutes(weather: pd.DataFrame, local_times: np.ndarray, tz: str = SITE_TZ) -> pd.DataFrame:
    """Return the last completed hourly record for each (naive local) timestamp."""
    idx = pd.DatetimeIndex(local_times).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    idx_utc = idx.tz_convert("UTC")
    pos = weather.index.searchsorted(idx_utc.fillna(weather.index[0]), side="right") - 1
    pos = np.clip(pos, 0, len(weather) - 1)
    out = weather.iloc[pos].reset_index(drop=True)
    out[idx.isna()] = np.nan
    return out
