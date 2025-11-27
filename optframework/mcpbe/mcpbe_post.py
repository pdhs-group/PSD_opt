# Post-processing utilities
from __future__ import annotations

from typing import Tuple, List, Optional, Sequence, Any

import numpy as np
import math
import warnings


class MCPBEPost:
    """Post-processing mixin: compute time-resolved moments µ(i,j,t) and PSD."""

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

    # ------------------------------------------------------------------
    # PSD helpers (single realization)
    # ------------------------------------------------------------------
    def _compute_psd_cdf_from_snapshot(
        self,
        V_snap: np.ndarray,
        psd_basis: str = "volume",
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Build an empirical CDF from a particle snapshot.

        Parameters
        ----------
        V_snap : ndarray, shape (dim+1, a)
            Snapshot of particle volumes at a given time (last row is total volume).
        psd_basis : {"number", "volume"}
            Weighting basis for the CDF.

        Returns
        -------
        x_sorted : ndarray, shape (n_valid,)
            Sorted diameters.
        Q_sorted : ndarray, shape (n_valid,)
            Corresponding cumulative fractions in (0, 1], based on the chosen psd_basis.
        """
        if V_snap.size == 0 or V_snap.shape[1] == 0:
            return None

        V_tot = V_snap[-1, :]  # total volume per particle
        if psd_basis == "number":
            w = np.ones_like(V_tot, dtype=float)
        elif psd_basis == "volume":
            w = V_tot.astype(float)
        else:
            raise ValueError(f"psd_basis must be 'number' or 'volume', got {psd_basis!r}")

        # convert to diameter
        x = self._vol2diam(V_tot.astype(float))

        # filter out non-positive entries
        mask = (w > 0.0) & (x > 0.0)
        if not np.any(mask):
            return None
        x = x[mask]
        w = w[mask]

        idx = np.argsort(x)
        x_sorted = x[idx]
        w_sorted = w[idx]
        w_cum = np.cumsum(w_sorted)
        total = float(w_cum[-1])
        if total <= 0.0:
            return None
        Q_sorted = w_cum / total
        return x_sorted, Q_sorted

    @staticmethod
    def _eval_Q_of_x(
        x_sorted: np.ndarray,
        Q_sorted: np.ndarray,
        x_query: np.ndarray,
    ) -> np.ndarray:
        """Evaluate Q(x) on a given diameter grid using a step-wise empirical CDF."""
        xq = np.asarray(x_query, dtype=float)
        Qq = np.zeros_like(xq, dtype=float)
        idx = np.searchsorted(x_sorted, xq, side="right") - 1
        Qq[idx < 0] = 0.0
        valid = idx >= 0
        if np.any(valid):
            idx_clipped = np.clip(idx[valid], 0, len(Q_sorted) - 1)
            Qq[valid] = Q_sorted[idx_clipped]
        return Qq

    @staticmethod
    def _eval_x_of_Q(
        x_sorted: np.ndarray,
        Q_sorted: np.ndarray,
        Q_query: np.ndarray,
    ) -> np.ndarray:
        """Evaluate x(Q) (quantile function) on a given Q grid using the empirical CDF."""
        Qq = np.asarray(Q_query, dtype=float)
        xq = np.zeros_like(Qq, dtype=float)
        idx = np.searchsorted(Q_sorted, Qq, side="left")
        idx = np.clip(idx, 0, len(Q_sorted) - 1)
        xq = x_sorted[idx]
        return xq

    def compute_psd_cdf_over_time(
        self,
        psd_basis: str = "volume",
    ) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], np.ndarray]:
        """Compute empirical PSD CDF at all saved times for a single realization.

        Parameters
        ----------
        psd_basis : {"number", "volume"}
            Weighting basis for the CDF.

        Returns
        -------
        cdf_list : list of length T
            cdf_list[t] is either (x_sorted, Q_sorted) or None if no valid data at that time.
        t_vec : ndarray, shape (T,)
            Time vector aligned with cdf_list.
        """
        T = min(len(self.V_save), len(self.t_vec))
        t_vec = np.asarray(self.t_vec[:T], dtype=float)
        cdf_list: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []

        for t in range(T):
            V_snap = self.V_save[t]
            cdf = self._compute_psd_cdf_from_snapshot(V_snap, psd_basis=psd_basis)
            cdf_list.append(cdf)

        return cdf_list, t_vec

    # ------------------------------------------------------------------
    # PSD aggregation over repeats
    # ------------------------------------------------------------------
    def aggregate_psd_repeats(
        self,
        cdf_repeats: Sequence[Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]],
        t_vec: np.ndarray,
        psd_basis: str = "volume",
        psd_x_grid: Optional[np.ndarray] = None,
        psd_Q_grid: Optional[np.ndarray] = None,
    ) -> dict:
        """Aggregate PSD CDFs over multiple repeats into an averaged PSD on t_vec.

        Parameters
        ----------
        cdf_repeats : sequence of length N
            Each element is a list (length T) of (x_sorted, Q_sorted) or None.
        t_vec : ndarray, shape (T,)
            Time grid aligned with each cdf list.
        psd_basis : {"number", "volume"}
            Basis used when CDFs were computed.
        psd_x_grid : ndarray, optional
            If provided, the PSD is returned as Q(x) evaluated on this grid.
        psd_Q_grid : ndarray, optional
            If provided (and psd_x_grid is None), the PSD is returned as x(Q)
            evaluated on this Q grid.

        Returns
        -------
        psd_info : dict
            See docstring of MCPBEBase.solve_repeats for the exact structure.
        """
        # parse mode and grid choice
        psd_mode: Optional[str] = None   # "Q_of_x" or "x_of_Q"
        auto_x_grid = False

        x_grid_user: Optional[np.ndarray]
        Q_grid_user: Optional[np.ndarray]

        if psd_x_grid is not None and psd_Q_grid is not None:
            warnings.warn(
                "Both psd_x_grid and psd_Q_grid are provided; psd_x_grid will be used and Q(x) will be computed.",
                RuntimeWarning,
            )
            psd_mode = "Q_of_x"
            x_grid_user = np.asarray(psd_x_grid, dtype=float)
            Q_grid_user = None
        elif psd_x_grid is not None:
            psd_mode = "Q_of_x"
            x_grid_user = np.asarray(psd_x_grid, dtype=float)
            Q_grid_user = None
        elif psd_Q_grid is not None:
            psd_mode = "x_of_Q"
            Q_grid_user = np.asarray(psd_Q_grid, dtype=float)
            x_grid_user = None
        else:
            # auto-generate x_grid and output Q(x)
            psd_mode = "Q_of_x"
            auto_x_grid = True
            x_grid_user = None
            Q_grid_user = None
            warnings.warn(
                "psd_enable=True but neither psd_x_grid nor psd_Q_grid is provided; "
                "a log-spaced x_grid will be generated automatically and Q(x) will be returned.",
                RuntimeWarning,
            )

        T = int(len(t_vec))
        psd_info: dict[str, Any] = {
            "basis": psd_basis,
            "mode": psd_mode,
            "t_vec": np.asarray(t_vec, dtype=float),
            "note": (
                "PSD computed at all saved times (aligned with t_vec) "
                "and averaged over all repeats."
            ),
        }

        if T == 0 or not cdf_repeats:
            # no data
            if psd_mode == "Q_of_x":
                psd_info["x_grid"] = None
                psd_info["Q_mean"] = None
            else:
                psd_info["Q_grid"] = None
                psd_info["x_mean"] = None
            return psd_info

        # ensure all repeats have the same number of time steps
        for cdf_list in cdf_repeats:
            if len(cdf_list) != T:
                raise RuntimeError(
                    f"All CDF lists must have length T={T}, got {len(cdf_list)}."
                )

        # aggregation branches
        if psd_mode == "Q_of_x":
            # ------------------------------------------------------------------
            # Q(x) mode
            # ------------------------------------------------------------------
            if auto_x_grid:
                # determine global min/max diameter per time, then build a common x_grid
                global_min_x = np.full(T, np.inf, dtype=float)
                global_max_x = np.zeros(T, dtype=float)

                for cdf_list in cdf_repeats:
                    for it, cdf in enumerate(cdf_list):
                        if cdf is None:
                            continue
                        x_sorted, _ = cdf
                        xmin = float(x_sorted[0])
                        xmax = float(x_sorted[-1])
                        if xmin < global_min_x[it]:
                            global_min_x[it] = xmin
                        if xmax > global_max_x[it]:
                            global_max_x[it] = xmax

                # global range over all times (conservative choice)
                finite_min = global_min_x[np.isfinite(global_min_x)]
                if finite_min.size == 0:
                    warnings.warn(
                        "auto_x_grid mode: no valid CDFs collected; psd_info will be empty.",
                        RuntimeWarning,
                    )
                    psd_info["x_grid"] = None
                    psd_info["Q_mean"] = None
                    return psd_info

                xmin_global = float(np.min(finite_min))
                xmax_global = float(np.max(global_max_x))
                xmin_global = max(xmin_global, 1e-20)
                if xmax_global <= xmin_global:
                    xmax_global = xmin_global * 1.01

                x_grid = np.logspace(
                    math.log10(xmin_global),
                    math.log10(xmax_global),
                    num=200,
                    base=10.0,
                )
            else:
                if x_grid_user is None:
                    raise RuntimeError("x_grid_user is None in Q_of_x mode.")
                x_grid = x_grid_user

            # now accumulate Q(x) over repeats for each time
            M = x_grid.shape[0]
            Q_sum = np.zeros((T, M), dtype=float)
            Q_count = np.zeros(T, dtype=int)

            for cdf_list in cdf_repeats:
                for it, cdf in enumerate(cdf_list):
                    if cdf is None:
                        continue
                    x_sorted, Q_sorted = cdf
                    Q_r = self._eval_Q_of_x(x_sorted, Q_sorted, x_grid)
                    Q_sum[it] += Q_r
                    Q_count[it] += 1

            Q_mean = np.empty_like(Q_sum)
            for it in range(T):
                if Q_count[it] > 0:
                    Q_mean[it] = Q_sum[it] / float(Q_count[it])
                else:
                    Q_mean[it] = np.nan

            psd_info["x_grid"] = x_grid
            psd_info["Q_mean"] = Q_mean

        elif psd_mode == "x_of_Q":
            # ------------------------------------------------------------------
            # x(Q) mode
            # ------------------------------------------------------------------
            if Q_grid_user is None:
                raise RuntimeError("Q_grid_user is None in x_of_Q mode.")
            Q_grid = Q_grid_user
            M = Q_grid.shape[0]
            x_sum = np.zeros((T, M), dtype=float)
            x_count = np.zeros(T, dtype=int)

            for cdf_list in cdf_repeats:
                for it, cdf in enumerate(cdf_list):
                    if cdf is None:
                        continue
                    x_sorted, Q_sorted = cdf
                    x_r = self._eval_x_of_Q(x_sorted, Q_sorted, Q_grid)
                    x_sum[it] += x_r
                    x_count[it] += 1

            x_mean = np.empty_like(x_sum)
            for it in range(T):
                if x_count[it] > 0:
                    x_mean[it] = x_sum[it] / float(x_count[it])
                else:
                    x_mean[it] = np.nan

            psd_info["Q_grid"] = Q_grid
            psd_info["x_mean"] = x_mean

        else:
            raise RuntimeError(f"Unknown psd_mode={psd_mode!r}.")

        return psd_info
