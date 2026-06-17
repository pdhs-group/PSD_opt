# Agglomeration mixin: beta/alpha hooks, r_agg rebuild (JIT), single agglomeration event.
from __future__ import annotations

import numpy as np

from .fenwick import FenwickSampler
from pbe_core.func.jit_mcpbe import (
    nb_pick_partner_weighted_pair_delta,
    nb_rebuild_ragg_weighted_pair_delta,
)

# External JIT kernel for Î²(i,j)
from pbe_core.func.jit_kernel_agg import calc_beta as _kb_beta


class MCPBEAgg:
    """Agglomeration logic:
    - _beta: wrapper to JIT kernel
    - _alpha_ccm: 2D alpha based on component fractions (kept for parity)
    - _rebuild_all_propensities: parallel JIT rebuild of pair-delta corrected r_i
    - _do_one_agg: single event with incremental r updates + swap-pop removal
    """
    agg_dW_max: float = 1.0
    agg_dW_min: float = 1.0
    agg_dW_mode: str = "const"
    agg_dW_alpha: float = 100.0
    SIZEEVAL: int = 1
    X_SEL: float = 0.31
    Y_SEL: float = 1.06

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _alpha_ccm(self, idx1: int, idx2: int) -> float:
        """2D collision efficiency from component fractions and alpha_prim (length 4)."""
        if self.dim == 1:
            return float(self.alpha_prim if np.ndim(self.alpha_prim) == 0 else np.mean(self.alpha_prim))

        V = self.V_flat
        Vi0 = V[0, idx1]; Vi1 = V[1, idx1]; Vti = Vi0 + Vi1
        Vj0 = V[0, idx2]; Vj1 = V[1, idx2]; Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            return 0.0
        p0 = (Vi0 / Vti) * (Vj0 / Vtj)
        p1 = (Vi0 / Vti) * (Vj1 / Vtj)
        p2 = (Vi1 / Vti) * (Vj0 / Vtj)
        p3 = (Vi1 / Vti) * (Vj1 / Vtj)
        ap = np.asarray(self.alpha_prim, dtype=float)
        if ap.size != 4:
            raise ValueError("`alpha_prim` must contain 4 entries for dim=2 agglomeration.")
        return float(p0 * ap[0] + p1 * ap[1] + p2 * ap[2] + p3 * ap[3])

    def _beta(self, i: int, j: int) -> float:
        """Pair kernel Î²(i,j) via external JIT calc_beta (radii array = X/2)."""
        return float(_kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(self.G), self.X / 2.0, i, j))

    # ------------------------------------------------------------------
    # r_agg maintenance (full rebuild)
    # ------------------------------------------------------------------
    def _rebuild_all_propensities(self):
        """Rebuild weighted agglomeration propensities with shared packet correction."""
        a = self.a_tot
        if a <= 0:
            self._r_agg[:] = 0.0
            self._delta_agg[:] = 0.0
            return

        R = (self.X[:a] * 0.5).astype(np.float64)
        W = self.W[:a].astype(np.float64)
        dW_const = float(self._agg_dW_const)
        delta = self._delta_from_weights(W, dW_const=dW_const)
        if self._delta_agg.shape[0] < self._cap:
            raise ValueError("`_delta_agg` capacity is smaller than solver._cap.")
        self._delta_agg[:a] = delta
        r = nb_rebuild_ragg_weighted_pair_delta(
            int(self.COLEVAL),
            float(self.CORR_BETA),
            float(self.G),
            R,
            W,
            delta,
        )
        np.maximum(r, 0.0, out=r)

        if self._r_agg.shape[0] < self._cap:
            raise ValueError("`_r_agg` capacity is smaller than solver._cap.")
        self._r_agg[:a] = r
        if self._r_agg.shape[0] > a:
            self._r_agg[a:] = 0.0
        if self._delta_agg.shape[0] > a:
            self._delta_agg[a:] = 0.0

    def _compute_agg_dW(self, i: int, j: int, pair_prop: float, sum_prop_before: float) -> float:
        """Compute packet size Î”W for one agglomeration event on pair (i,j)."""
        Wi = float(self.W[i])
        Wj = float(self.W[j])
        if Wi <= 0.0 or Wj <= 0.0:
            return 0.0

        dW_max = float(self.agg_dW_max)
        dW_min = float(self.agg_dW_min)
        if dW_max <= 0.0:
            raise ValueError("`agg_dW_max` must be positive.")
        if dW_min < 0.0:
            raise ValueError("`agg_dW_min` must be non-negative.")

        mode = str(self.agg_dW_mode).lower()
        if mode == "const":
            dW = dW_max
        elif mode in ("linear", "sqrt"):
            f = 0.0 if sum_prop_before <= 0.0 else float(pair_prop) / float(sum_prop_before)
            f = float(np.clip(f, 0.0, 1.0))
            alpha = float(self.agg_dW_alpha)
            if alpha <= 0.0:
                raise ValueError("`agg_dW_alpha` must be positive.")
            if mode == "sqrt":
                dW = dW_min + alpha * (f ** 0.5) * (dW_max - dW_min)
            else:
                dW = dW_min + alpha * f * (dW_max - dW_min)
        else:
            raise ValueError("`agg_dW_mode` must be 'const', 'linear', or 'sqrt'.")

        if dW < dW_min:
            dW = dW_min
        if dW > dW_max:
            dW = dW_max
        delta_i = self._update_delta_single(i, attr_name="_delta_agg", dW_const=float(self._agg_dW_const))
        delta_j = self._update_delta_single(j, attr_name="_delta_agg", dW_const=float(self._agg_dW_const))
        if i == j:
            if delta_i <= 0.0 or Wi <= 1.0:
                return 0.0
            delta_ii = min(float(delta_i), 0.5 * Wi)
            if delta_ii <= 0.0:
                return 0.0
            if dW > delta_ii:
                dW = delta_ii
        else:
            if dW > Wi:
                dW = Wi
            if dW > Wj:
                dW = Wj
            if delta_i > 0.0 and dW > delta_i:
                dW = delta_i
            if delta_j > 0.0 and dW > delta_j:
                dW = delta_j
        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)

    # ------------------------------------------------------------------
    # Single agglomeration event
    # ------------------------------------------------------------------
    def _do_one_agg(self):
        a = self.a_tot
        if a < 2:
            self._last_agg_dW = 0.0
            return

        # default packet for rejected/empty attempts
        self._last_agg_dW = 0.0

        # 1) pick first partner by r_i
        i = self._agg_sampler.sample(self._rng)

        # 2) weighted partner sampling + acceptance (numba)
        R = (self.X[:a] * 0.5).astype(np.float64)
        W = self.W[:a].astype(np.float64)
        if self.dim == 1:
            alpha1d = float(self.alpha_prim if np.ndim(self.alpha_prim) == 0 else np.mean(self.alpha_prim))
            alpha4 = np.zeros(4, dtype=np.float64)
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = np.zeros_like(V0)
        else:
            alpha1d = 1.0
            ap = np.asarray(self.alpha_prim, dtype=np.float64)
            if ap.size != 4:
                raise ValueError("`alpha_prim` must contain 4 entries for dim=2 agglomeration.")
            alpha4 = ap
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = self.V_flat[1, :a].astype(np.float64)

        SIZEEVAL = int(self.SIZEEVAL)
        X_SEL = float(self.X_SEL)
        Y_SEL = float(self.Y_SEL)
        Vmean2 = float(np.mean(self.V0[-1, :]) ** 2)

        u_sel = float(self._rng.random())
        u_acc = float(self._rng.random())
        Wi_selected = float(W[i])
        partner_total = (
            float(self._r_agg[i]) / Wi_selected
            if Wi_selected > 0.0 and hasattr(self, "_r_agg")
            else 0.0
        )
        j, pick_w = nb_pick_partner_weighted_pair_delta(
            i,
            int(self.COLEVAL), float(self.CORR_BETA), float(self.G),
            R,
            W,
            self._delta_agg[:a].astype(np.float64),
            partner_total,
            V0, V1, int(self.dim),
            float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2,
            u_sel, u_acc,
        )
        if j < 0 or pick_w <= 0.0:
            return

        # 3) packet size Î”W for this accepted event
        sum_prop_before = float(self._agg_sampler.total())
        Wi = float(self.W[i])
        # pick_w already includes the pair-delta correction used in r_i.
        pair_prop = Wi * pick_w
        dW = self._compute_agg_dW(i, j, pair_prop, sum_prop_before)
        if dW <= 0.0:
            return
        self._last_agg_dW = dW

        # 4) create one new compute particle for merged products with weight dW
        Vi = self.V_flat[: self.dim, i].copy()
        Vj = self.V_flat[: self.dim, j].copy()
        Vnew = Vi + Vj

        self._append_particle_column(Vnew)
        new_idx = self.a_tot - 1
        self.W[new_idx] = dW

        # 5) consume dW from parents (self-agglomeration consumes 2*dW from one packet)
        if i == j:
            w_now = float(self.W[i])
            w_rem = w_now - 2.0 * dW
            if w_rem > 0.0:
                self.W[i] = w_rem
            else:
                self._remove_particle_column(i)
        else:
            for idx in sorted({int(i), int(j)}, reverse=True):
                w_now = float(self.W[idx])
                w_rem = w_now - dW
                if w_rem > 0.0:
                    self.W[idx] = w_rem
                else:
                    self._remove_particle_column(idx)

        # 6) full weighted agglomeration propensity refresh (simple and consistent)
        self._rebuild_all_propensities()
        self._agg_sampler = FenwickSampler(self._r_agg[: self.a_tot])

        # 7) breakage sampler refresh (if active)
        if self.process_type in ("breakage", "mix"):
            self._calc_break_rates_full()  # provided by BreakageMixin
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])

