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
    
        Weighted version:
            mu = sum_k W_k * V1_k^i * V3_k^j / Vc(t)   (dim>1)
            mu = sum_k W_k * V_k^i / Vc(t)            (dim==1)
        If no W_save exists, fall back to W=1 (backward compatible).
        """
        T = min(len(self.V_save), len(self.Vc_save), len(self.t_vec))
        mu = np.zeros((max_i + 1, max_j + 1, T), dtype=float)
    
        has_W = hasattr(self, "W_save") and self.W_save is not None and len(self.W_save) >= T
    
        for t in range(T):
            Vc = float(self.Vc_save[t]) if (normalize and self.Vc_save) else 1.0
            V_snap = np.asarray(self.V_save[t], dtype=float)
    
            if has_W:
                W = np.asarray(self.W_save[t], dtype=float)
            else:
                W = np.ones(V_snap.shape[1], dtype=float)
    
            if self.dim == 1:
                V = np.asarray(V_snap[0, :], dtype=float)
                for i in range(max_i + 1):
                    mu[i, 0, t] = float(np.sum(W * np.power(V, i))) / Vc
            else:
                V1 = np.asarray(V_snap[0, :], dtype=float)
                V3 = np.asarray(V_snap[1, :], dtype=float)
                for i in range(max_i + 1):
                    Vi = np.power(V1, i)
                    for j in range(max_j + 1):
                        mu[i, j, t] = float(np.sum(W * Vi * np.power(V3, j))) / Vc
    
        return mu, self.t_vec[:T]


    # ------------------------------------------------------------------
    # PSD helpers (single realization)
    # ------------------------------------------------------------------
    def _compute_psd_cdf_from_snapshot(
        self,
        V_snap: np.ndarray,
        psd_basis: str = "volume",
        W_snap: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Build an empirical CDF from a particle snapshot (weighted).
    
        Parameters
        ----------
        V_snap : ndarray, shape (dim+1, a)
            Snapshot of particle volumes at a given time (last row is total volume).
        psd_basis : {"number", "volume"}
            Weighting basis for the CDF.
        W_snap : ndarray, shape (a,), optional
            Statistical/number weight for each compute particle (how many real particles it represents).
            If None, assumes W=1 (backward compatible).
    
        Returns
        -------
        (x_sorted, Q_sorted) or None
        """
        if V_snap.size == 0 or V_snap.shape[1] == 0:
            return None
    
        V_tot = np.asarray(V_snap[-1, :], dtype=float)  # total volume per particle
    
        if W_snap is None:
            W = np.ones_like(V_tot, dtype=float)
        else:
            W = np.asarray(W_snap, dtype=float)
            if W.shape[0] != V_tot.shape[0]:
                raise ValueError("W_snap length must match number of particles in V_snap")
    
        # CDF weight
        if psd_basis == "number":
            w = W
        elif psd_basis == "volume":
            w = W * V_tot
        else:
            raise ValueError(f"psd_basis must be 'number' or 'volume', got {psd_basis!r}")
    
        # convert to diameter
        x = self._vol2diam(V_tot)
    
        mask = (w > 0.0) & (x > 0.0) & np.isfinite(w) & np.isfinite(x)
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
        time_scheme: str = "right",
    ) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], np.ndarray]:
        """Compute empirical PSD CDF at all saved times for a single realization (weighted)."""
        time_scheme = str(time_scheme).lower()
        if time_scheme not in ("right", "left", "interp", "nearest"):
            raise ValueError(
                f"time_scheme must be one of 'right', 'left', 'interp', 'nearest', "
                f"got {time_scheme!r}."
            )
    
        T_base = min(len(self.V_save), len(self.t_vec))
        if T_base == 0:
            return [], np.asarray([], dtype=float)
    
        use_left_side = time_scheme in ("left", "interp", "nearest")
        if use_left_side:
            if not hasattr(self, "V_save_left") or not hasattr(self, "t_left") or not hasattr(self, "t_right"):
                raise RuntimeError(
                    "time_scheme uses left/right information but V_save_left / t_left / t_right "
                    "are not available."
                )
            T = min(T_base, len(self.V_save_left), len(self.t_left), len(self.t_right))
        else:
            T = T_base
    
        if T == 0:
            return [], np.asarray([], dtype=float)
    
        t_vec = np.asarray(self.t_vec[:T], dtype=float)
    
        has_W_right = hasattr(self, "W_save") and self.W_save is not None and len(self.W_save) >= T
        has_W_left = hasattr(self, "W_save_left") and self.W_save_left is not None and len(self.W_save_left) >= T
    
        def _cdf_from_side(idx: int, side: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            if side == "right":
                V_snap = self.V_save[idx]
                W_snap = self.W_save[idx] if has_W_right else None
            elif side == "left":
                V_snap = self.V_save_left[idx]
                W_snap = self.W_save_left[idx] if has_W_left else None
            else:
                raise ValueError(f"Unknown side {side!r} in _cdf_from_side.")
            return self._compute_psd_cdf_from_snapshot(V_snap, psd_basis=psd_basis, W_snap=W_snap)
    
        cdf_list: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    
        for t in range(T):
            if time_scheme == "right":
                cdf_list.append(_cdf_from_side(t, "right"))
                continue
    
            if time_scheme == "left":
                cdf_list.append(_cdf_from_side(t, "left"))
                continue
    
            if time_scheme == "nearest":
                tl = float(self.t_left[t])
                tr = float(self.t_right[t])
                target = float(t_vec[t])
                side = "left" if abs(target - tl) <= abs(tr - target) else "right"
                cdf_list.append(_cdf_from_side(t, side))
                continue
    
            # interp
            # compute left/right CDF and interpolate in time
            cL = _cdf_from_side(t, "left")
            cR = _cdf_from_side(t, "right")
            if cL is None and cR is None:
                cdf_list.append(None)
                continue
            if cL is None:
                cdf_list.append(cR)
                continue
            if cR is None:
                cdf_list.append(cL)
                continue
    
            xL, QL = cL
            xR, QR = cR
    
            x_union = np.unique(np.concatenate([xL, xR]))
            QL_u = self._eval_Q_of_x(xL, QL, x_union)
            QR_u = self._eval_Q_of_x(xR, QR, x_union)
    
            tl = float(self.t_left[t])
            tr = float(self.t_right[t])
            target = float(t_vec[t])
            if tr <= tl + 1e-15:
                alpha = 1.0
            else:
                alpha = float((target - tl) / (tr - tl))
                alpha = float(np.clip(alpha, 0.0, 1.0))
    
            Q_interp = (1.0 - alpha) * QL_u + alpha * QR_u
            cdf_list.append((x_union, Q_interp))
    
        return cdf_list, t_vec

    def _invert_cdf_monotone(self, x_axis: np.ndarray, Q_vals: np.ndarray, q: float = 0.5) -> float:
        """Invert a (nearly) monotone CDF to find x at probability q.
    
        Enforces monotonicity (cummax) to reduce numerical noise, then uses
        linear interpolation in (Q, x). Returns NaN if q is outside range.
        """
        x_axis = np.asarray(x_axis, dtype=float)
        Q_vals = np.asarray(Q_vals, dtype=float)
        if x_axis.size == 0 or Q_vals.size == 0 or x_axis.size != Q_vals.size:
            return float("nan")
    
        mask = np.isfinite(x_axis) & np.isfinite(Q_vals)
        if not np.any(mask):
            return float("nan")
    
        x = x_axis[mask]
        Q = Q_vals[mask]
    
        order = np.argsort(x)
        x = x[order]
        Q = Q[order]
    
        # enforce monotone non-decreasing
        Q = np.maximum.accumulate(Q)
    
        if Q[0] > q or Q[-1] < q:
            return float("nan")
    
        k = int(np.searchsorted(Q, q, side="left"))
        if k <= 0:
            return float(x[0])
        if k >= Q.size:
            return float(x[-1])
    
        q0, q1 = float(Q[k - 1]), float(Q[k])
        x0, x1 = float(x[k - 1]), float(x[k])
        if q1 <= q0 + 1e-15:
            return float(x1)
        return float(x0 + (q - q0) * (x1 - x0) / (q1 - q0))

    def _filter_and_renormalize_cdf_by_xmin(
        self,
        x_sorted: np.ndarray,
        Q_sorted: np.ndarray,
        x_min: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Filter an empirical CDF (x_sorted, Q_sorted) by removing x < x_min,
        then re-normalize so that:
          - Q(x_min) = 0
          - Q(max)   = 1

        Returns None if nothing remains.
        """
        x_sorted = np.asarray(x_sorted, dtype=float).ravel()
        Q_sorted = np.asarray(Q_sorted, dtype=float).ravel()
        x_min = float(x_min)

        if x_sorted.size == 0 or Q_sorted.size == 0 or x_sorted.size != Q_sorted.size:
            return None

        # enforce monotone & bounds (numerical safety)
        Q_sorted = np.clip(Q_sorted, 0.0, 1.0)
        Q_sorted = np.maximum.accumulate(Q_sorted)

        # if x_min is below support -> nothing to do
        if x_min <= float(x_sorted[0]):
            return x_sorted, Q_sorted

        # if x_min is above support -> everything removed
        if x_min >= float(x_sorted[-1]):
            return None

        # Q at cutoff (stepwise: right-continuous convention consistent with _eval_Q_of_x)
        # For x<x_sorted[0], Q=0. Here x_min within support.
        j = int(np.searchsorted(x_sorted, x_min, side="right") - 1)
        j = max(j, 0)
        Q_cut = float(Q_sorted[j])

        denom = 1.0 - Q_cut
        if denom <= 0.0:
            return None

        # keep points with x >= x_min
        k0 = int(np.searchsorted(x_sorted, x_min, side="left"))
        x_tail = x_sorted[k0:]
        Q_tail = Q_sorted[k0:]

        # ensure x_min included as first point (use Q_cut at x_min)
        x_new = np.concatenate(([x_min], x_tail))
        Q_new_raw = np.concatenate(([Q_cut], Q_tail))

        # shift & renormalize: Q' = (Q - Q_cut)/(1 - Q_cut)
        Q_new = (Q_new_raw - Q_cut) / denom
        Q_new = np.clip(Q_new, 0.0, 1.0)
        Q_new = np.maximum.accumulate(Q_new)
        Q_new[0] = 0.0
        Q_new[-1] = 1.0

        return x_new, Q_new

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
        x_50 = np.full(T, np.nan, dtype=float)
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
                    psd_info["x_50"] = np.full(T, np.nan, dtype=float)
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

            use_qx_filter = bool(getattr(self, "Qx_filter", False)) and (not auto_x_grid)
            x_min_user = None
            if use_qx_filter:
                x_min_user = float(np.min(x_grid))

            for cdf_list in cdf_repeats:
                for it, cdf in enumerate(cdf_list):
                    if cdf is None:
                        continue
                    x_sorted, Q_sorted = cdf

                    if use_qx_filter and x_min_user is not None:
                        cdf2 = self._filter_and_renormalize_cdf_by_xmin(
                            x_sorted,
                            Q_sorted,
                            x_min_user,
                        )
                        if cdf2 is None:
                            continue
                        x_sorted, Q_sorted = cdf2

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
            psd_info["Q_mean"] = Q_mean.T
            
            for it in range(T):
                x_50[it] = self._invert_cdf_monotone(x_grid, Q_mean[it, :], q=0.5)
            psd_info["x_50"] = x_50

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
            
            q = 0.5
            hit = np.where(np.isclose(Q_grid, q, rtol=0.0, atol=1e-12))[0]
            if hit.size > 0:
                j = int(hit[0])
                x_50 = x_mean[:, j].astype(float, copy=False)
            else:
                for it in range(T):
                    xq = np.asarray(x_mean[it], dtype=float)
                    mask = np.isfinite(Q_grid) & np.isfinite(xq)
                    if not np.any(mask):
                        x_50[it] = float("nan")
                        continue
                    Qm = Q_grid[mask]
                    xm = xq[mask]
                    order = np.argsort(Qm)
                    Qm = Qm[order]
                    xm = xm[order]
                    xm = np.maximum.accumulate(xm)
                    if Qm[0] > q or Qm[-1] < q:
                        x_50[it] = float("nan")
                    else:
                        x_50[it] = float(np.interp(q, Qm, xm))
            psd_info["x_50"] = x_50
        else:
            raise RuntimeError(f"Unknown psd_mode={psd_mode!r}.")

        return psd_info
