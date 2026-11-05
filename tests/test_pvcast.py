import io
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pvcast.data.skippd import day_ranges, safe_load_object_npy  # noqa: E402
from pvcast.data.weather import align_to_minutes  # noqa: E402
from pvcast.evaluation import report, skill  # noqa: E402
from pvcast.features import LAGS, build_samples, split_by_day  # noqa: E402
from pvcast.models.baselines import persistence, smart_persistence  # noqa: E402
from pvcast.models.gbm import GBMForecaster  # noqa: E402
from pvcast.solar import ClearSkyPV, clear_sky_index, solar_table  # noqa: E402


def make_stream(days=3, start=datetime(2018, 6, 1, 7, 0), noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    times, pv = [], []
    for d in range(days):
        t0 = start + timedelta(days=d)
        for m in range(11 * 60):
            times.append(t0 + timedelta(minutes=m))
    times = np.array([np.datetime64(t, "s") for t in times])
    sol = solar_table(times)
    clear = 0.6 * np.clip(sol["ghi_cs"].values, 0, None) ** 0.56
    cloud = np.clip(1 - 0.5 * (np.sin(np.arange(len(times)) / 37.0) > 0.6) + rng.normal(0, noise, len(times)), 0, 1.2)
    pv = (clear * cloud).astype(np.float32)
    imgs = rng.integers(0, 255, (len(times), 64, 64, 3), dtype=np.uint8)
    return {"times": times, "pv": pv, "images": imgs}


def make_weather(times):
    idx = pd.date_range(pd.Timestamp(times[0]).tz_localize("America/Los_Angeles").tz_convert("UTC").floor("h") - pd.Timedelta("2h"),
                        periods=24 * 5, freq="h")
    cols = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover", "cloud_cover_low",
            "cloud_cover_mid", "cloud_cover_high", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    df = pd.DataFrame({c: np.arange(len(idx), dtype=float) for c in cols}, index=idx)
    df.index.name = "time"
    return df


# ---------------------------------------------------------------- safe loading
def test_safe_npy_loads_datetimes(tmp_path):
    arr = np.array([datetime(2017, 1, 1, 8, 0), datetime(2017, 1, 1, 8, 1)], dtype=object)
    p = tmp_path / "t.npy"
    np.save(p, arr, allow_pickle=True)
    out = safe_load_object_npy(p)
    assert list(out) == list(arr)


def test_safe_npy_blocks_arbitrary_code(tmp_path):
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned",))
    p = tmp_path / "evil.npy"
    np.save(p, np.array([Evil()], dtype=object), allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError):
        safe_load_object_npy(p)


def test_day_ranges():
    s = make_stream(days=2)
    r = day_ranges(s["times"].astype(object))
    assert len(r) == 2 and sum(b - a for a, b in r.values()) == len(s["times"])


# ---------------------------------------------------------------- solar / features
def test_clearsky_fit_recovers_envelope():
    s = make_stream(days=3, noise=0.0)
    sol = solar_table(s["times"])
    cs = ClearSkyPV(quantile=0.99).fit(s["pv"], sol["ghi_cs"].values)
    assert 0.4 < cs.gamma < 0.7
    k = clear_sky_index(s["pv"], cs(sol["ghi_cs"].values))
    assert np.nanmax(k) <= 1.5 and np.nanmin(k) >= 0


def test_samples_have_no_future_leakage():
    s = make_stream(days=2)
    sol = solar_table(s["times"])
    cs = ClearSkyPV().fit(s["pv"], sol["ghi_cs"].values)
    S = build_samples(s, cs, horizon=15, stride=1)
    K = s["times"].astype("datetime64[m]").astype(np.int64)
    issue = K[S.stream_idx]
    assert np.all(K[S.frame_idx[:, -1]] == issue)                       # newest frame is issue time
    assert np.all(K[S.frame_idx[:, 0]] == issue - max(LAGS))            # oldest frame t-15
    assert np.allclose(S.pv_now, s["pv"][S.stream_idx])


def test_weather_alignment_uses_completed_hour_only():
    s = make_stream(days=1)
    w = make_weather(s["times"])
    t = np.array([np.datetime64("2018-06-01T09:59:00"), np.datetime64("2018-06-01T10:00:00")])
    a = align_to_minutes(w, t)
    # 09:59 local must not see the 10:00 record; 10:00 sees it
    assert a["cloud_cover"].iloc[1] == a["cloud_cover"].iloc[0] + 1


def test_day_split_disjoint():
    s = make_stream(days=5)
    sol = solar_table(s["times"])
    S = build_samples(s, ClearSkyPV().fit(s["pv"], sol["ghi_cs"].values), stride=5)
    tr, va = split_by_day(S, 0.4)
    d_tr = set(S.times[tr].astype("datetime64[D]").tolist())
    d_va = set(S.times[va].astype("datetime64[D]").tolist())
    assert d_tr and d_va and not (d_tr & d_va)


# ---------------------------------------------------------------- models / metrics
def test_baselines_and_gbm_run():
    s = make_stream(days=4)
    w = make_weather(s["times"])
    sol = solar_table(s["times"])
    cs = ClearSkyPV().fit(s["pv"], sol["ghi_cs"].values)
    S = build_samples(s, cs, w, stride=3)
    tr, te = split_by_day(S, 0.25)
    g = GBMForecaster(max_iter=50).fit(S.subset(tr))
    p = g.predict(S.subset(te))
    assert p.shape == (te.sum(),) and np.all(p >= 0)
    r = report(S.y[te], {"persistence": persistence(S.subset(te)), "smart_persistence": smart_persistence(S.subset(te)),
                         "gbm": p})
    assert r["smart_persistence"]["skill_pct"] == pytest.approx(0.0)


def test_skill_definition():
    y = np.array([1.0, 2.0, 3.0])
    assert skill(y, y, y + 1) == pytest.approx(1.0)
    assert skill(y, y + 1, y + 1) == pytest.approx(0.0)


def test_net_forward_and_modality_dropout():
    torch = pytest.importorskip("torch")
    from pvcast.models.nets import NetForecaster
    s = make_stream(days=2)
    sol = solar_table(s["times"])
    S = build_samples(s, ClearSkyPV().fit(s["pv"], sol["ghi_cs"].values), stride=20)
    cols = list(S.X.columns)
    m = NetForecaster(cols, epochs=1, batch=32).fit(s["images"], S, verbose=False)
    p = m.predict(s["images"], S)
    p_off = m.predict(s["images"], S, drop_image=True)
    assert p.shape == (len(S),) and np.isfinite(p).all() and np.isfinite(p_off).all()
