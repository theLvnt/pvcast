"""Reference forecasts. Every model is scored by its skill against smart persistence."""
from __future__ import annotations

import numpy as np

from ..features import Samples


def persistence(s: Samples) -> np.ndarray:
    """PV(t+h) = PV(t)."""
    return s.pv_now.copy()


def smart_persistence(s: Samples) -> np.ndarray:
    """PV(t+h) = k*(t) · PV_clearsky(t+h): clouds persist, the sun keeps moving."""
    return s.k_now * s.pv_cs_target
