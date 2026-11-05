"""Evaluate all forecasters on the held-out SKIPP'D test days.

Outputs
  reports/results.md / results.json   RMSE, MAE, nRMSE, skill — all test samples and cloudy test days
  reports/forecast_cloudy_day.png     observed vs forecasts on the most variable test day
  reports/skill.png                   forecast skill vs smart persistence
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pvcast.data.skippd import load_subset, safe_load_object_npy  # noqa: E402
from pvcast.data.weather import load_weather  # noqa: E402
from pvcast.evaluation import report, to_markdown  # noqa: E402
from pvcast.features import build_samples  # noqa: E402
from pvcast.models.baselines import persistence, smart_persistence  # noqa: E402
from pvcast.models.gbm import GBMForecaster  # noqa: E402
from pvcast.models.nets import NetForecaster  # noqa: E402

# Validated categorical palette (light surface #fcfcfb); observed series uses primary ink
INK, GRID, SURFACE = "#0b0b0b", "#e4e3df", "#fcfcfb"
SERIES = {"smart_persistence": ("#2a78d6", (0, (4, 2))), "gbm_weather": ("#eb6834", (0, (1, 1.5))),
          "fusion": ("#1baf7a", "solid")}


def cloudy_mask(data: Path, times: np.ndarray) -> np.ndarray:
    """Map the official cloudy-test flag (per minute) onto sample times via timestamps."""
    t = safe_load_object_npy(data / "times_test_nowcast.npy")
    flag = np.load(data / "cloudy_test_nowcast.npy", allow_pickle=False)
    lut = {np.datetime64(x, "s"): bool(f) for x, f in zip(t, flag)}
    return np.array([lut.get(x, False) for x in times])


def plots(out: Path, s, preds: dict, rep_all: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    days = s.times.astype("datetime64[D]")
    best_day, best_var = None, -1
    for d in np.unique(days):
        m = days == d
        if m.sum() > 150:
            v = np.std(np.diff(s.y[m]))
            if v > best_var:
                best_day, best_var = d, v
    m = days == best_day
    t = s.times[m].astype("datetime64[m]").astype(object)
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(t, s.y[m], color=INK, lw=2, label="Observed PV")
    for k, (c, ls) in SERIES.items():
        ax.plot(t, preds[k][m], color=c, lw=2, ls=ls, label=k.replace("_", " "))
    ax.set_title(f"15-min-ahead forecasts on a variable test day ({best_day})", loc="left", color=INK)
    ax.set_ylabel("PV power (kW)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(out / "forecast_cloudy_day.png", dpi=150)
    plt.close(fig)

    names = [k for k in rep_all if k != "smart_persistence"]
    vals = [rep_all[k]["skill_pct"] for k in names]
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(names, vals, color="#2a78d6", height=0.55)
    ax.axvline(0, color="#52514e", lw=1)
    for i, v in enumerate(vals):
        ax.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:.1f}%", va="center",
                ha="left" if v >= 0 else "right", color=INK)
    ax.set_xlabel("Forecast skill vs smart persistence (%)")
    ax.set_title("Skill on all test samples (higher is better)", loc="left", color=INK)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "skill.png", dpi=150)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    data, models, out = ROOT / "data_raw/skippd", ROOT / "models", ROOT / "reports"
    out.mkdir(exist_ok=True)
    meta = json.loads((models / "train_meta.json").read_text())
    cs = pickle.load(open(models / "clearsky.pkl", "rb"))
    stream = load_subset(data / "test.npz")
    s = build_samples(stream, cs, load_weather(data / "weather.csv"), horizon=meta["horizon"], stride=1)
    img = stream["images"]

    preds, timing = {"persistence": persistence(s), "smart_persistence": smart_persistence(s)}, {}
    for n in ("gbm_history", "gbm_weather"):
        t0 = time.time(); preds[n] = GBMForecaster.load(models / f"{n}.joblib").predict(s)
        timing[n] = (time.time() - t0) / len(s) * 1e3
    for n in ("cnn_image", "fusion"):
        m = NetForecaster.load(models / f"{n}.pt")
        t0 = time.time(); preds[n] = m.predict(img, s)
        timing[n] = (time.time() - t0) / len(s) * 1e3
        if n == "fusion":
            preds["fusion (camera offline)"] = m.predict(img, s, drop_image=True)

    cloudy = cloudy_mask(data, s.times)
    rep_all = report(s.y, preds)
    rep_cloudy = report(s.y[cloudy], {k: v[cloudy] for k, v in preds.items()})
    res = {"n_test_samples": len(s), "n_cloudy": int(cloudy.sum()),
           "test_days": int(np.unique(s.times.astype("datetime64[D]")).size),
           "all": rep_all, "cloudy": rep_cloudy, "inference_ms_per_sample_cpu": timing, "train": meta}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    md = ["# Results — 15-minute-ahead PV forecasting (SKIPP'D, Stanford 30 kW array)", "",
          f"Held-out test: {res['test_days']} days, {len(s)} samples (1-min issue times); "
          f"cloudy subset: {res['n_cloudy']} samples. Training: {meta['train_days']} days "
          f"(+{meta['val_days']} validation days).", "",
          to_markdown(rep_all, "All test samples"), "", to_markdown(rep_cloudy, "Cloudy test days"), "",
          "Inference (CPU, ms/sample): " + ", ".join(f"{k} {v:.2f}" for k, v in timing.items())]
    (out / "results.md").write_text("\n".join(md))
    plots(out, s, preds, rep_all)
    print("\n".join(md))


if __name__ == "__main__":
    main()
