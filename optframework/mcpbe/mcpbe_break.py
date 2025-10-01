# Breakage mixin: breakage rates (full & single), two-level CDF builder, fragment production, single break event.
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# External JIT kernels
from optframework.utils.func.jit_kernel_agg import calc_beta as _kb_beta
from optframework.utils.func.jit_mcpbe import _build_table_1d_jit, _build_tables_2d_jit
import optframework.utils.func.jit_kernel_break as JKB
from optframework.utils.func.jit_kernel_break import (
    calc_B_R_1d as _kb_BR1_all,
    calc_B_R_2d_flat as _kb_BR2_all,
    calc_break_rate_1d as _kb_br1_single,
    calc_break_rate_2d_flat as _kb_br2_single,
)


class MCPBEBreak:
    """Breakage logic:
    - breakage rate table (full) from external JIT kernels
    - single-point breakage rate for incremental updates
    - two-level CDF builder (1D/2D) and discrete samplers
    - multi-fragment event by stochastic rounding of expected fragment count
    """

    # ------------------------------------------------------------------
    # Breakage rate (full table and single-point)
    # ------------------------------------------------------------------
    def _calc_break_rates_full(self):
        """Compute breakage rates B_R for active slice using external JIT kernels."""
        a = self.a_tot
        if not hasattr(self, "_break_rate") or self._break_rate is None or self._break_rate.shape[0] < getattr(self, "_cap", a):
            self._break_rate = np.zeros(getattr(self, "_cap", max(8, a)), dtype=float)

        self.V = self.V_flat[-1, :a]  # match external wrappers' expectation
        self.B_R = np.zeros(a, dtype=float)

        if self.dim == 1 and hasattr(JKB, "calc_B_R_1d"):
            _kb_BR1_all(self)  # fills self.B_R
        elif self.dim == 2 and hasattr(JKB, "calc_B_R_2d_flat"):
            _kb_BR2_all(self)  # fills self.B_R
        else:
            raise RuntimeError("Required breakage kernels not found in jit_kernel_break.")

        self._break_rate[:a] = np.asarray(self.B_R, dtype=float)
        if self._break_rate.shape[0] > a:
            self._break_rate[a:] = 0.0

    def _break_rate_single(self, i: int) -> float:
        """Single-particle breakage rate using external JIT kernels (no fallbacks)."""
        a = self.a_tot
        if i < 0 or i >= a:
            return 0.0
        if self.dim == 1:
            return float(
                _kb_br1_single(
                    self.V_flat[-1, :a],
                    float(getattr(self, "pl_P1", 1.0)),
                    float(getattr(self, "pl_P2", 1.0)),
                    float(getattr(self, "G", 1.0)),
                    int(getattr(self, "BREAKRVAL", 1)),
                    i,
                )
            )
        else:
            return float(
                _kb_br2_single(
                    self.V_flat[-1, :a],
                    self.V_flat[0, :a],
                    self.V_flat[1, :a],
                    float(getattr(self, "G", 1.0)),
                    float(getattr(self, "pl_P1", 1.0)),
                    float(getattr(self, "pl_P2", 1.0)),
                    float(getattr(self, "pl_P3", 1.0)),
                    float(getattr(self, "pl_P4", 1.0)),
                    int(getattr(self, "BREAKRVAL", 1)),
                    int(getattr(self, "BREAKFVAL", 1)),
                    i,
                )
            )

    # ------------------------------------------------------------------
    # Two-level CDF builder (cached)
    # ------------------------------------------------------------------
    def _build_break_function(self, num_points: int = 1000):
        """Build two-level CDF tables for breakage fragment distributions (1D/2D)."""
        key = (int(self.dim), int(num_points), int(getattr(self, "BREAKFVAL", 1)), float(getattr(self, "pl_v", 1.0)), float(getattr(self, "pl_q", 1.0)))
        cached = self._bf_cache.get(key, None)
        if cached is not None:
            # restore cached tables
            if self.dim == 1:
                self._bf1_rel, self._bf1_cdf = cached
            else:
                self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf = cached
            self._bf_ready = True
            return

        # grids
        if self.dim == 1:
            rel = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            v = float(getattr(self, "pl_v", 1.0)); q = float(getattr(self, "pl_q", 1.0))
            bf = int(getattr(self, "BREAKFVAL", 1))
            cdf = _build_table_1d_jit(rel, v, q, bf)
            self._bf1_rel = rel
            self._bf1_cdf = cdf
            self._bf_cache[key] = (self._bf1_rel, self._bf1_cdf)
        else:
            rel1 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            rel3 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            v = float(getattr(self, "pl_v", 1.0)); q = float(getattr(self, "pl_q", 1.0))
            bf = int(getattr(self, "BREAKFVAL", 1))
            rowsum_cdf, row_cdf = _build_tables_2d_jit(rel1, rel3, v, q, bf)
            self._bf2_rel1 = rel1
            self._bf2_rel3 = rel3
            self._bf2_rowsum_cdf = rowsum_cdf
            self._bf2_row_cdf = row_cdf
            self._bf_cache[key] = (self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf)

        self._bf_ready = True

    # ------------------------------------------------------------------
    # Discrete samplers (from built tables)
    # ------------------------------------------------------------------
    def _discrete_sample(self, rel: np.ndarray, cdf: np.ndarray, u: float) -> float:
        """Sample 1D relative ratio by inverse CDF on discrete grid."""
        k = int(np.searchsorted(cdf, u, side="right"))
        if k >= rel.size:
            k = rel.size - 1
        return float(rel[k])

    def _discrete_sample_2d(self, u1: float, u2: float) -> Tuple[float, float]:
        """Sample (r1, r3) by two-level (row then within-row) discrete inverse CDF."""
        # pick row by rowsum CDF
        i = int(np.searchsorted(self._bf2_rowsum_cdf, u1, side="right"))
        if i >= self._bf2_rel1.size:
            i = self._bf2_rel1.size - 1
        # pick col by row-wise CDF
        row_cdf = self._bf2_row_cdf[i]
        j = int(np.searchsorted(row_cdf, u2, side="right"))
        if j >= self._bf2_rel3.size:
            j = self._bf2_rel3.size - 1
        return float(self._bf2_rel1[i]), float(self._bf2_rel3[j])

    # ------------------------------------------------------------------
    # Produce one fragment from remaining volume vector
    # ------------------------------------------------------------------
    def _produce_one_frag_from_remaining(self, Vrem: np.ndarray) -> np.ndarray:
        """Draw a single fragment that conserves per-component volume from Vrem (dim=1 or 2)."""
        if not getattr(self, "_bf_ready", False):
            self._build_break_function()

        if self.dim == 1:
            u = float(self._rng.random())
            r = self._discrete_sample(self._bf1_rel, self._bf1_cdf, u)
            r = max(0.0, min(1.0, r))
            frag = np.array([r * Vrem[0]], dtype=float)
        else:
            u1 = float(self._rng.random()); u2 = float(self._rng.random())
            r1, r3 = self._discrete_sample_2d(u1, u2)
            r1 = max(0.0, min(1.0, r1))
            r3 = max(0.0, min(1.0, r3))
            frag = np.array([r1 * Vrem[0], r3 * Vrem[1]], dtype=float)

        # Robustness: clamp fragment not to exceed remaining
        frag = np.minimum(frag, Vrem)
        frag = np.maximum(frag, 0.0)
        return frag

    # ------------------------------------------------------------------
    # Single breakage event (multi-fragment)
    # ------------------------------------------------------------------
    def _do_one_break(self):
        a = self.a_tot
        if a < 1:
            return

        # Ensure sampler exists & pick particle
        self._ensure_break_sampler()
        k = self._break_sampler.sample(self._rng)
        if k < 0 or k >= self.a_tot:
            return

        # stochastic rounding of expected fragment count
        p = float(getattr(self, "frag_num", 2.0))
        fl = int(math.floor(p)); ce = int(math.ceil(p))
        if fl <= 1:
            raise ValueError("Expected number of fragments <= 1; check BREAKFVAL/pl_v.")
        n = ce if (self._rng.random() < (p - fl)) else fl

        # remaining per-component volumes for selected particle
        Vrem = self.V_flat[: self.dim, k].copy()
        new_indices = []

        # produce n-1 fragments
        for _ in range(n - 1):
            frag = self._produce_one_frag_from_remaining(Vrem)
            # append fragment (capacity-aware, samplers rebuilt by base)
            self._append_particle_column(frag)
            new_idx = self.a_tot - 1
            new_indices.append(new_idx)
            # update breakage rate for new index
            br_new = self._break_rate_single(new_idx)
            self._break_rate[new_idx] = br_new
            if self._break_sampler is not None:
                self._break_sampler.update(new_idx, br_new)
            # subtract from remaining
            Vrem -= frag

        # last fragment stays in-place at k
        self.V_flat[: self.dim, k] = Vrem
        self.V_flat[-1, k] = float(np.sum(Vrem))
        self.X[k] = float(self._vol2diam(self.V_flat[-1, k]))

        # update breakage rate for k
        br_k = self._break_rate_single(k)
        self._break_rate[k] = br_k
        if self._break_sampler is not None:
            self._break_sampler.update(k, br_k)

        # agglomeration sampler maintenance (approximate incremental updates)
        pt = getattr(self, "process_type", "agglomeration")
        if pt in ("agglomeration", "mix") and self._agg_sampler is not None:
            # recompute r for k and each newly added index
            a_now = self.a_tot
            R_now = (self.X[:a_now] * 0.5).astype(np.float64)

            # (1) recalc r_k exactly
            rk = 0.0
            for m in range(a_now):
                if m == k:
                    continue
                rk += _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_now, k, m)
            self._r_agg[k] = rk
            self._agg_sampler.update(k, rk)

            # (2) for each new index, compute r_new exactly, and add its impact to others approximately
            for new_idx in new_indices:
                rnew = 0.0
                for m in range(a_now):
                    if m == new_idx:
                        continue
                    rnew += _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_now, new_idx, m)
                # write r_new
                if hasattr(self, "_r_agg") and self._r_agg.shape[0] >= a_now:
                    self._r_agg[new_idx] = rnew
                # sampler update
                self._agg_sampler.update(new_idx, rnew)

                # approximate: add β(m, new_idx) to r_m (without subtracting old β(m,k_old))
                for m in range(a_now):
                    if m == new_idx:
                        continue
                    bmnew = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_now, m, new_idx)
                    self._r_agg[m] += bmnew
                    self._agg_sampler.update(m, self._r_agg[m])
