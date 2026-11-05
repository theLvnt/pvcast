"""Download a SKIPP'D subset and the matching Open-Meteo weather.

    python scripts/01_download.py                    # 80 train/val days + full test set (~600 MB)
    python scripts/01_download.py --train-days 0     # all 2017-2019 train/val days (~4.5 GB)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pvcast.data import skippd, weather  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data_raw/skippd"))
    ap.add_argument("--train-days", type=int, default=80, help="0 = all days")
    ap.add_argument("--test-days", type=int, default=0, help="0 = all test days")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)

    paths = skippd.download_times(out)
    t_tr = skippd.safe_load_object_npy(paths["times_trainval_nowcast.npy"])
    t_te = skippd.safe_load_object_npy(paths["times_test_nowcast.npy"])
    d_tr = skippd.select_days(t_tr, a.train_days or None, a.seed)
    d_te = skippd.select_days(t_te, a.test_days or None, a.seed)
    print(f"train/val: {len(d_tr)} days, test: {len(d_te)} days")
    for split, t, days, name in (("test", t_te, d_te, "test.npz"), ("trainval", t_tr, d_tr, "trainval.npz")):
        if not (out / name).exists():
            skippd.download_subset(out / name, split, t, days)
    all_days = sorted(set(d_tr) | set(d_te))
    weather.fetch_open_meteo(all_days, out / "weather.csv")
    print("done ->", out)


if __name__ == "__main__":
    main()
