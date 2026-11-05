"""Train all forecasters on the train/val days.

Models
  gbm_history          GBM, PV history + solar geometry only
  gbm_weather          GBM, + Open-Meteo weather
  cnn_image            sky images + PV history (SUNSET-style)
  fusion               sky images + PV history + weather + geometry (multimodal)

    python scripts/02_train.py --epochs 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pvcast.data.skippd import load_subset  # noqa: E402
from pvcast.data.weather import load_weather  # noqa: E402
from pvcast.features import build_samples, split_by_day  # noqa: E402
from pvcast.models.gbm import GBMForecaster  # noqa: E402
from pvcast.models.nets import NetForecaster  # noqa: E402
from pvcast.solar import ClearSkyPV, solar_table  # noqa: E402

HISTORY_COLS = None  # filled below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_raw/skippd"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)

    stream = load_subset(data / "trainval.npz")
    weather = load_weather(data / "weather.csv")
    sol = solar_table(stream["times"])
    cs = ClearSkyPV().fit(stream["pv"].astype(float), sol["ghi_cs"].values)
    print(f"clear-sky PV fit: scale={cs.scale:.4f}, gamma={cs.gamma:.3f}")
    s = build_samples(stream, cs, weather, horizon=a.horizon, stride=a.stride)
    tr, va = split_by_day(s, 0.2, seed=0)
    train, val = s.subset(tr), s.subset(va)
    print(f"samples: train={len(train)}, val={len(val)}, features={s.X.shape[1]}")

    cols_all = list(s.X.columns)
    cols_hist = [c for c in cols_all if not c.startswith("w_")]
    meta = {"horizon": a.horizon, "stride": a.stride, "cs_scale": cs.scale, "cs_gamma": cs.gamma,
            "cols_all": cols_all, "cols_hist": cols_hist, "train_days": int(np.unique(train.times.astype("datetime64[D]")).size),
            "val_days": int(np.unique(val.times.astype("datetime64[D]")).size), "n_train": len(train), "n_val": len(val)}
    pickle.dump(cs, open(out / "clearsky.pkl", "wb"))

    t0 = time.time()
    GBMForecaster(cols_hist).fit(train).save(out / "gbm_history.joblib")
    GBMForecaster(cols_all).fit(train).save(out / "gbm_weather.joblib")
    print(f"GBMs trained in {time.time() - t0:.0f}s")

    img = stream["images"]
    hist_tab = [c for c in cols_hist if c.startswith(("pv_lag", "k_lag", "pvcs"))]
    for name, cols, use_img in (("cnn_image", hist_tab, True), ("fusion", cols_all, True)):
        print(f"training {name} ...", flush=True)
        t0 = time.time()
        m = NetForecaster(cols, use_image=use_img, use_tab=True, epochs=a.epochs).fit(img, train, val)
        m.save(out / f"{name}.pt")
        meta[f"{name}_train_s"] = round(time.time() - t0)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
