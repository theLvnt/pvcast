"""Download → train → evaluate.

    python scripts/run_all.py                  # 80 training days, CPU-friendly
    python scripts/run_all.py --train-days 0 --epochs 20   # full 2017-2019 training set (GPU recommended)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("\n▶", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", default="80")
    ap.add_argument("--epochs", default="8")
    a = ap.parse_args()
    run("scripts/01_download.py", "--train-days", a.train_days)
    run("scripts/02_train.py", "--epochs", a.epochs)
    run("scripts/03_evaluate.py")
