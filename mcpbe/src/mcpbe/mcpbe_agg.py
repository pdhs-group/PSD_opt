# Agglomeration mixin: beta/alpha hooks, r_agg rebuild (JIT), single agglomeration event.
from __future__ import annotations

import numpy as np

from .fenwick import FenwickSampler
from pbe_core.func.jit_mcpbe import nb_rebuild_ragg, nb_pick_partner

# External JIT kernel for Î²(i,j)
from pbe_core.func.jit_kernel_agg import calc_beta as _kb_beta


class MCPBEAgg:
    """Agglomeration logic:
    - _beta: wrapper to JIT kernel
    - _alpha_ccm: 2D alpha based on component fractions (kept for parity)
    - _rebuild_all_propensities: parallel JIT rebuild r_i = sum_j beta(i,j)
    - _do_one_agg: single event with incremental r updates + swap-pop removal
    """

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
            ap = np.ones(4, dtype=float)
        return float(p0 * ap[0] + p1 * ap[1] + p2 * ap[2] + p3 * ap[3])

    def _beta(self, i: int, j: int) -> float:
        """Pair kernel Î²(i,j) via external JIT calc_beta (radii array = X/2)."""
        if i == j:
            return 0.0
        return float(_kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), self.X / 2.0, i, j))

    # ------------------------------------------------------------------
    # r_agg maintenance (full rebuild)
    # ------------------------------------------------------------------
    def _rebuild_all_propensities(self):
        """Parallel rebuild of r_agg with numba kernel."""
        a = self.a_tot
        if a <= 0:
            if not hasattr(self, "_r_agg") or self._r_agg is None or self._r_agg.shape[0] < getattr(self, "_cap", a):
                self._r_agg = np.zeros(getattr(self, "_cap", max(8, a)), dtype=float)
            else:
                self._r_agg[:] = 0.0
            return

        R = (self.X[:a] * 0.5).astype(np.float64)
        r = nb_rebuild_ragg(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R)

        if not hasattr(self, "_r_agg") or self._r_agg is None or self._r_agg.shape[0] < getattr(self, "_cap", a):
            self._r_agg = np.zeros(getattr(self, "_cap", a), dtype=float)
        self._r_agg[:a] = r
        if self._r_agg.shape[0] > a:
            self._r_agg[a:] = 0.0

    # ------------------------------------------------------------------
    # Single agglomeration event
    # ------------------------------------------------------------------
    def _do_one_agg(self):
        a = self.a_tot
        if a < 2:
            return

        # 1) pick first partner by r_i
        i = self._agg_sampler.sample(self._rng)

        # 2) numba-assisted partner sampling & acceptance test
        R = (self.X[:a] * 0.5).astype(np.float64)
        if self.dim == 1:
            alpha1d = float(self.alpha_prim if np.ndim(self.alpha_prim) == 0 else np.mean(self.alpha_prim))
            alpha4 = np.zeros(4, dtype=np.float64)
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = np.zeros_like(V0)
        else:
            alpha1d = 1.0
            ap = np.asarray(self.alpha_prim, dtype=np.float64)
            alpha4 = ap if ap.size == 4 else np.ones(4, dtype=np.float64)
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = self.V_flat[1, :a].astype(np.float64)

        SIZEEVAL = int(getattr(self, "SIZEEVAL", 1))
        X_SEL = float(getattr(self, "X_SEL", 0.31))
        Y_SEL = float(getattr(self, "Y_SEL", 1.06))
        Vmean2 = float(np.mean(self.V0[-1, :]) ** 2) if hasattr(self, "V0") else float(np.mean(self.V_flat[-1, :a]) ** 2)

        u_sel = float(self._rng.random())
        u_acc = float(self._rng.random())
        j = nb_pick_partner(
            i,
            int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)),
            R, V0, V1, int(self.dim),
            float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2,
            u_sel, u_acc,
        )
        if j < 0 or j == i:
            return

        # 3) perform merge i <- i âˆª j
        Vi = self.V_flat[: self.dim, i].copy()
        Vj = self.V_flat[: self.dim, j].copy()
        Vnew = Vi + Vj
        Xnew = float(self._vol2diam(np.sum(Vnew)))

        # 4) incremental propensity updates for all m != i,j
        #    use a single R_new for Î²(m,i_new)
        R_new = R.copy()
        R_new[i] = 0.5 * Xnew
        for m in range(a):
            if m == i or m == j:
                continue
            beta_m_i = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R, m, i)
            beta_m_j = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R, m, j)
            beta_m_new = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_new, m, i)
            rm_new = self._r_agg[m] - beta_m_i - beta_m_j + beta_m_new
            self._r_agg[m] = rm_new
            self._agg_sampler.update(m, rm_new)

        # 5) commit: write new column at i; remove j (swap-pop)
        self.V_flat[: self.dim, i] = Vnew
        self.V_flat[-1, i] = float(np.sum(Vnew))
        self.X[i] = Xnew

        last = a - 1
        i_after = j if (j != last and i == last) else i
        self._remove_particle_column(j)  # updates a_tot and rebuilds samplers
        i = i_after

        # 6) recompute r_i exactly (others unchanged from step 4)
        rk = 0.0
        a_now = self.a_tot
        R_now = (self.X[:a_now] * 0.5).astype(np.float64)
        for m in range(a_now):
            if m == i:
                continue
            rk += _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_now, i, m)
        self._r_agg[i] = rk
        self._agg_sampler.update(i, rk)

        # 7) breakage sampler refresh (if active)
        if getattr(self, "process_break", False) or getattr(self, "process_type", "agglomeration") in ("breakage", "mix"):
            self._calc_break_rates_full()  # provided by BreakageMixin
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])

