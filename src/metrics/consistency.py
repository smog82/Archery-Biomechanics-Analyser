"""
src/metrics/consistency.py
Consistency Index (CI): quantifies how repeatable a movement pattern is
across multiple shots/sessions.

Method
------
For N time-series (each normalised to [0,1] time):
  1. Interpolate all to a common 200-point grid.
  2. Compute the inter-series standard deviation at each time point.
  3. CI = 100 * (1 - mean_std / range_of_all_values)
     → 100 = perfectly identical shots, 0 = maximally inconsistent.
"""

import numpy as np
from typing import List


class ConsistencyIndex:

    @staticmethod
    def compute(series_list: List[np.ndarray], n_points: int = 200) -> float:
        """
        Parameters
        ----------
        series_list : list of 1-D arrays (different lengths allowed)
        n_points    : resolution for interpolation

        Returns
        -------
        CI : float in [0, 100]
        """
        if len(series_list) < 2:
            return 100.0

        grid = np.linspace(0, 1, n_points)
        interp = []
        for s in series_list:
            arr = np.asarray(s, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) < 2:
                continue
            t = np.linspace(0, 1, len(arr))
            interp.append(np.interp(grid, t, arr))

        if len(interp) < 2:
            return 100.0

        mat       = np.array(interp)          # (N, n_points)
        std_curve = mat.std(axis=0)           # (n_points,)
        mean_std  = std_curve.mean()
        value_range = mat.max() - mat.min()

        if value_range == 0:
            return 100.0

        ci = float(np.clip(100 * (1 - mean_std / value_range), 0, 100))
        return round(ci, 2)

    @staticmethod
    def interpret(ci: float) -> str:
        if ci >= 80:  return "Excellent"
        if ci >= 60:  return "Good"
        if ci >= 40:  return "Moderate"
        return "Inconsistent"
