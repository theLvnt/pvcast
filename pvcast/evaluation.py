"""Forecast verification metrics (following common solar-forecasting practice)."""
from __future__ import annotations

import numpy as np

from .data.skippd import PV_CAPACITY_KW


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def mbe(y, p):
    return float(np.mean(np.asarray(p) - np.asarray(y)))


def skill(y, p, ref):
    """Forecast skill vs a reference (smart persistence): 1 - RMSE/RMSE_ref. >0 means better."""
    return 1.0 - rmse(y, p) / rmse(y, ref)


def report(y, preds: dict[str, np.ndarray], ref_key: str = "smart_persistence",
           capacity: float = PV_CAPACITY_KW) -> dict[str, dict]:
    ref = preds[ref_key]
    out = {}
    for k, p in preds.items():
        out[k] = {"rmse_kw": rmse(y, p), "mae_kw": mae(y, p), "mbe_kw": mbe(y, p),
                  "nrmse_pct": 100 * rmse(y, p) / capacity, "skill_pct": 100 * skill(y, p, ref)}
    return out


def to_markdown(rep: dict[str, dict], title: str) -> str:
    lines = [f"**{title}**", "", "| Model | RMSE (kW) | MAE (kW) | nRMSE (%) | Skill vs smart persistence (%) |",
             "|---|---|---|---|---|"]
    for k, m in rep.items():
        lines.append(f"| {k} | {m['rmse_kw']:.3f} | {m['mae_kw']:.3f} | {m['nrmse_pct']:.2f} | {m['skill_pct']:.1f} |")
    return "\n".join(lines)
