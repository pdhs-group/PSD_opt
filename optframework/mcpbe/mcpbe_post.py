# Post-processing utilities
from __future__ import annotations

from typing import Tuple

import numpy as np


class MCPBEPost:
    """Post-processing mixin: compute time-resolved moments µ(i,j,t)."""

    def calc_moments_over_time(
        self, max_i: int = 2, max_j: int = 2, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mu, t_vec) where mu[i,j,t] are mixed moments over components.

        For dim==1, the j-axis is used with j=0 only.
        If normalize=True, each time slice is divided by Vc(t).
        Only active particles saved at each time are used (capacity padding excluded).
        """
        T = min(len(self.V_save), len(self.Vc_save), len(self.t_vec))
        mu = np.zeros((max_i + 1, max_j + 1, T), dtype=float)

        for t in range(T):
            Vc = float(self.Vc_save[t]) if (normalize and self.Vc_save) else 1.0
            if self.dim == 1:
                V = np.asarray(self.V_save[t][0, :], dtype=float)
                for i in range(max_i + 1):
                    mu[i, 0, t] = np.sum(np.power(V, i)) / Vc
            else:
                V1 = np.asarray(self.V_save[t][0, :], dtype=float)
                V3 = np.asarray(self.V_save[t][1, :], dtype=float)
                for i in range(max_i + 1):
                    Vi = np.power(V1, i)
                    for j in range(max_j + 1):
                        mu[i, j, t] = float(np.dot(Vi, np.power(V3, j))) / Vc

        return mu, self.t_vec[:T]
