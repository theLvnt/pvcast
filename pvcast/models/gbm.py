"""Gradient-boosted trees on tabular features (PV history + weather + solar geometry).

Predicts the clear-sky index at t+h, then rescales by clear-sky PV — so the
trees model clouds, not the diurnal cycle.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..features import Samples
from ..solar import clear_sky_index


class GBMForecaster:
    def __init__(self, columns: list[str] | None = None, **kw):
        self.columns = columns
        params = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=30, random_state=0)
        params.update(kw)
        self.model = HistGradientBoostingRegressor(**params)

    def _X(self, s: Samples):
        return (s.X[self.columns] if self.columns else s.X).values

    def fit(self, s: Samples) -> "GBMForecaster":
        if self.columns is None:
            self.columns = list(s.X.columns)
        k_target = clear_sky_index(s.y, s.pv_cs_target)
        self.model.fit(self._X(s), k_target)
        return self

    def predict(self, s: Samples) -> np.ndarray:
        k = np.clip(self.model.predict(self._X(s)), 0, 1.5)
        return np.clip(k * s.pv_cs_target, 0, None)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "GBMForecaster":
        return joblib.load(path)
