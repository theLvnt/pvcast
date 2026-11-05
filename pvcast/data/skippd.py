"""SKIPP'D dataset access (Stanford sky images + PV power, 2017-2019).

Nie Y. et al., "SKIPP'D: A SKy Images and Photovoltaic Power Generation Dataset
for short-term solar forecasting", Solar Energy 255 (2023) 171-179.
Hosted on Hugging Face: torchgeo/skippd — license CC BY 4.0.

The full HDF5 files are 4.5 GB (nowcast, 1-min images) and 26 GB (forecast).
This module reads the 1-min "nowcast" file *remotely* with HTTP range requests
and downloads only the days you ask for, so a development subset fits in a few
hundred MB. Forecasting samples (image history + PV history -> PV at t+h) are
then built locally from the 1-min stream.

Security note: the timestamp arrays are pickled object arrays. They are read with
a restricted unpickler that only allows `datetime.datetime` and the numpy
reconstruction helpers — never `np.load(..., allow_pickle=True)` on downloaded data.
"""
from __future__ import annotations

import io
import pickle
import random
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

HF_BASE = "https://huggingface.co/datasets/torchgeo/skippd/resolve/main"
NOWCAST_H5 = f"{HF_BASE}/2017_2019_images_pv_processed_nowcast.hdf5"
NOWCAST_ZIP = f"{HF_BASE}/2017_2019_images_pv_processed_nowcast.zip"

# Stanford, CA (the PV array and sky camera are on campus)
SITE_LAT, SITE_LON, SITE_ALT = 37.427, -122.174, 30.0
SITE_TZ = "America/Los_Angeles"
PV_CAPACITY_KW = 30.1

_ALLOWED = {
    ("datetime", "datetime"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def safe_load_object_npy(path_or_bytes) -> np.ndarray:
    """Load a pickled object .npy containing datetimes without executing arbitrary code."""
    fh = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb")
    with fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(fh)
        else:
            np.lib.format.read_array_header_2_0(fh)
        return _RestrictedUnpickler(fh).load()


def download_times(out_dir: str | Path) -> dict[str, Path]:
    """Fetch the small timestamp/cloudy-mask arrays from the zip (≈11 MB)."""
    from remotezip import RemoteZip

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ["times_trainval_nowcast.npy", "times_test_nowcast.npy", "cloudy_test_nowcast.npy"]
    with RemoteZip(NOWCAST_ZIP) as z:
        for n in names:
            if not (out / n).exists():
                z.extract(n, out)
    return {n: out / n for n in names}


def day_ranges(times: np.ndarray) -> dict[date, tuple[int, int]]:
    """Contiguous [start, stop) index range for every calendar day (data is time-sorted per split)."""
    rng: dict[date, list[int]] = defaultdict(lambda: [10**12, -1])
    for i, t in enumerate(times):
        r = rng[t.date()]
        r[0] = min(r[0], i)
        r[1] = max(r[1], i + 1)
    return {d: (a, b) for d, (a, b) in sorted(rng.items())}


def select_days(times: np.ndarray, n_days: int | None, seed: int = 0) -> list[date]:
    days = sorted(day_ranges(times))
    if n_days is None or n_days >= len(days):
        return days
    r = random.Random(seed)
    return sorted(r.sample(days, n_days))


def download_subset(out_file: str | Path, split: str, times: np.ndarray, days: list[date],
                    block_size: int = 2 ** 22, verbose: bool = True) -> Path:
    """Copy the chosen days of one split (images, PV, times) into a local compressed .npz."""
    import fsspec
    import h5py

    ranges = day_ranges(times)
    f = fsspec.open(NOWCAST_H5, block_size=block_size).open()
    h = h5py.File(f, "r")
    imgs, pv, ts = [], [], []
    for k, d in enumerate(days):
        a, b = ranges[d]
        imgs.append(h[f"{split}/images_log"][a:b])
        pv.append(h[f"{split}/pv_log"][a:b])
        ts.append(np.array([np.datetime64(t, "s") for t in times[a:b]]))
        if verbose and k % 5 == 0:
            print(f"  {split}: {k + 1}/{len(days)} days", flush=True)
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, images=np.concatenate(imgs), pv=np.concatenate(pv).astype(np.float32),
                        times=np.concatenate(ts))
    return out


def load_subset(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)          # local, non-object arrays only
    return {k: z[k] for k in z.files}
