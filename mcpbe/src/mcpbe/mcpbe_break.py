# Breakage mixin: breakage rates (full & single), two-level CDF builder, fragment production, single break event.
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# External JIT kernels
from pbe_core.func.jit_kernel_agg import calc_beta as _kb_beta
from pbe_core.func.jit_mcpbe import _build_table_1d_jit, _build_tables_2d_jit
import pbe_core.func.jit_kernel_break as JKB
from pbe_core.func.jit_kernel_break import (
    calc_B_R_1d as _kb_BR1_all,
    calc_B_R_2d_flat as _kb_BR2_all,
    calc_break_rate_1d as _kb_br1_single,
    calc_break_rate_2d_flat as _kb_br2_single,
)

_ONE_SHOT_ADAPTER_NAMES = {"LMCRankAdapter", "LMCCopulaAdapter", "LMCFlowAdapter"}
_TABLE_ADAPTER_NAME = "LMCTableAdapter"
_LIVE_FALLBACK_ERROR = "LMCLiveFallback"
_LIVE_DISABLE_ERROR = "LMCLiveDisable"

class MCPBEBreak:
    """Breakage logic:
    - breakage rate table (full) from external JIT kernels
    - single-point breakage rate for incremental updates
    - two-level CDF builder (1D/2D) and discrete samplers
    - multi-fragment event by stochastic rounding of expected fragment count
    """

    @staticmethod
    def _adapter_type_name(adapter) -> str:
        return type(adapter).__name__ if adapter is not None else ""

    # ------------------------------------------------------------------
    # Breakage rate (full table and single-point)
    # ------------------------------------------------------------------
    def _calc_break_rates_full(self):
        """Compute breakage rates B_R for active slice.

          Priority:
             1) If the LMC-MLP breakage model is enabled
                 (lmc_use_breakage_model=True and adapter is available),
                 call self.lmc_breakage_adapter.compute_rates_full(self).
             2) Otherwise, use the external JIT kernels (original behavior).
        """
        a = self.a_tot
        cap = getattr(self, "_cap", a)
        if (not hasattr(self, "_break_rate")
                or self._break_rate is None
                or self._break_rate.shape[0] < cap):
            self._break_rate = np.zeros(max(8, cap), dtype=float)

        # --------- Branch 1: use MLP breakage-rate model ---------
        use_mlp = bool(getattr(self, "lmc_use_breakage_model", False)) and (
            getattr(self, "lmc_breakage_adapter", None) is not None
        )
        if use_mlp:
            # Let the adapter compute rates from current PBE state
            # (V_flat, dim, lmc_* parameters, etc.).
            rates = self.lmc_breakage_adapter.compute_rates_full(self)
            rates = np.asarray(rates, dtype=float)

            if rates.shape[0] < a:
                # If returned length is shorter than active size, pad with zeros.
                tmp = np.zeros(a, dtype=float)
                tmp[: rates.shape[0]] = rates
                rates = tmp
            elif rates.shape[0] > a:
                # If returned length is larger than active size, truncate.
                rates = rates[:a]

            self._break_rate[:a] = rates
            if self._break_rate.shape[0] > a:
                self._break_rate[a:] = 0.0
            return

        # --------- Branch 2: original JIT-kernel path ---------
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
        """Single-particle breakage rate.

                Priority:
                    1) If LMC-MLP breakage model is enabled, call
                         adapter.compute_rate_single(self, i).
                    2) Otherwise, use the original JIT single-point formula.
        """
        a = self.a_tot
        if i < 0 or i >= a:
            return 0.0

        # --------- Branch 1: use MLP breakage-rate model ---------
        use_mlp = bool(getattr(self, "lmc_use_breakage_model", False)) and (
            getattr(self, "lmc_breakage_adapter", None) is not None
        )
        if use_mlp:
            return float(self.lmc_breakage_adapter.compute_rate_single(self, i))

        # --------- Branch 2: original JIT-kernel path ---------
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
            v = float(getattr(self, "pl_v", 1.0))
            q = float(getattr(self, "pl_q", 1.0))
            bf = int(getattr(self, "BREAKFVAL", 1))
            cdf = _build_table_1d_jit(rel, v, q, bf)
            self._bf1_rel = rel
            self._bf1_cdf = cdf
            self._bf_cache[key] = (self._bf1_rel, self._bf1_cdf)
        else:
            rel1 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            rel3 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            v = float(getattr(self, "pl_v", 1.0))
            q = float(getattr(self, "pl_q", 1.0))
            bf = int(getattr(self, "BREAKFVAL", 1))
            rowsum_cdf, row_cdf = _build_tables_2d_jit(rel1, rel3, v, q, bf)
            self._bf2_rel1 = rel1
            self._bf2_rel3 = rel3
            self._bf2_rowsum_cdf = rowsum_cdf
            self._bf2_row_cdf = row_cdf
            self._bf_cache[key] = (self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf)

        self._bf_ready = True

    # [LMC-ADAPT] helpers
    def _state_AX1_from_Vrem(self, Vrem: np.ndarray) -> Tuple[float, float]:
        """Infer parent-particle state (A, X1) from remaining volume Vrem.

        A is total amount; X1 is phase-1 fraction.
        Uses a safe fallback when denominator is zero.
        """
        if self.dim == 1:
            A = float(Vrem[0])
            X1 = 1.0  # Single-component case; X1 is not used in practice.
        else:
            v1 = float(Vrem[0])
            v3 = float(Vrem[1])
            A = v1 + v3
            X1 = (v1 / A) if A > 0.0 else 0.5
        return A, X1

    def _get_break_tables_for_state(self, Vrem: np.ndarray):
        """
        Return CDF tables for the current parent-particle state.

        - If precomputed LMC adapter is enabled and available:
          fetch 1D/2D tables from adapter.
        - Otherwise:
          use JIT-generated tables via _build_break_function() and self._bf*.
        """
        use_lmc = bool(getattr(self, "use_lmc_pre_model", False) and getattr(self, "lmc_adapter", None) is not None)
        if not use_lmc:
            if not getattr(self, "_bf_ready", False):
                self._build_break_function()
            if self.dim == 1:
                return ("1d", self._bf1_rel, self._bf1_cdf, None, None, None, None, float(getattr(self, "frag_num", 2.0)))
            else:
                return ("2d", self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf, None, None, float(getattr(self, "frag_num", 2.0)))

        # LMC table-driven path
        A, X1 = self._state_AX1_from_Vrem(Vrem)

        # Use 1D table for pure-phase/degenerate states; otherwise 2D.
        if self.dim == 2 and (Vrem[0] <= 0.0 or Vrem[1] <= 0.0):
            rel1d, cdf1d, zmin1d, pexp = self.lmc_adapter.get_1d(A, X1)
            return ("1d", rel1d, cdf1d, None, None, zmin1d, None, float(pexp))
        if self.dim == 1:
            rel1d, cdf1d, zmin1d, pexp = self.lmc_adapter.get_1d(A, X1)
            return ("1d", rel1d, cdf1d, None, None, zmin1d, None, float(pexp))
        else:
            rel1, rel3, rowsum_cdf, row_cdf, zmin1, zmin3, pexp = self.lmc_adapter.get_2d(A, X1)
            return ("2d", rel1, rel3, rowsum_cdf, row_cdf, zmin1, zmin3, float(pexp))
        
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
        mode, rA, rB, rowsum_cdf, row_cdf, zminA, zminB, _pexp = self._get_break_tables_for_state(Vrem)

        if mode == "1d":
            # 1D: r in [0, 1], interpreted as a fragment ratio.
            u = float(self._rng.random())
            # _get_break_tables_for_state("1d") returns (rel1d, cdf1d, ...),
            # so here we map rel=rA and cdf=rB explicitly for clarity.
            rel = rA
            cdf = rB if rB is not None else None
            if cdf is None:
                raise RuntimeError("1D breakage table missing CDF.")
            j = int(np.searchsorted(cdf, u, side="right"))
            if j >= rel.size: j = rel.size - 1
            r = float(rel[j])
            r = max(0.0, min(1.0, r))
            if self.dim == 1:
                return np.array([r * Vrem[0]], dtype=float)
            else:
                # 2D degraded to 1D: place ratio on the non-zero phase.
                if Vrem[0] > 0.0 and Vrem[1] <= 0.0:
                    return np.array([r * Vrem[0], 0.0], dtype=float)
                elif Vrem[1] > 0.0 and Vrem[0] <= 0.0:
                    return np.array([0.0, r * Vrem[1]], dtype=float)
                else:
                    # Both phases are zero: return zero fragment.
                    return np.array([0.0, 0.0], dtype=float)

        else:
            # 2D two-level CDF
            u1 = float(self._rng.random())
            u2 = float(self._rng.random())
            i = int(np.searchsorted(rowsum_cdf, u1, side="right"))
            if i >= rA.size: i = rA.size - 1
            row = row_cdf[i]
            j = int(np.searchsorted(row, u2, side="right"))
            if j >= rB.size: j = rB.size - 1
            r1 = float(rA[i])
            r3 = float(rB[j])
            r1 = max(0.0, min(1.0, r1))
            r3 = max(0.0, min(1.0, r3))
            frag = np.array([r1 * Vrem[0], r3 * Vrem[1]], dtype=float)
            frag = np.minimum(frag, Vrem)
            frag = np.maximum(frag, 0.0)
            return frag

    # ------------------------------------------------------------------
    # Single breakage event (multi-fragment)
    # ------------------------------------------------------------------
    
    # Unified post-processing: apply fragments and maintain break/agg states.
    def _break_apply_and_maintain(self, k: int, frags: list[np.ndarray]) -> None:
        if not frags:
            return
        # n = len(frags)
        new_indices = []
        # First append the first n-1 fragments.
        for f in frags[:-1]:
            self._append_particle_column(f)
            new_idx = self.a_tot - 1
            new_indices.append(new_idx)
            br_new = self._break_rate_single(new_idx)
            self._break_rate[new_idx] = br_new
            if self._break_sampler is not None:
                self._break_sampler.update(new_idx, br_new)
    
        # Write the last fragment back to original index k.
        last = frags[-1]
        self.V_flat[: self.dim, k] = last
        self.V_flat[-1, k] = float(np.sum(last))
        self.X[k] = float(self._vol2diam(self.V_flat[-1, k]))
    
        br_k = self._break_rate_single(k)
        self._break_rate[k] = br_k
        if self._break_sampler is not None:
            self._break_sampler.update(k, br_k)
    
        # Unified agglomeration maintenance.
        pt = getattr(self, "process_type", "agglomeration")
        if pt in ("agglomeration", "mix") and self._agg_sampler is not None:
            a_now = self.a_tot
            R_now = (self.X[:a_now] * 0.5).astype(np.float64)
    
            # recalc r_k
            rk = 0.0
            for m in range(a_now):
                if m == k:
                    continue
                rk += _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_now, k, m)
            self._r_agg[k] = rk
            self._agg_sampler.update(k, rk)
    
            # new indices
            for new_idx in new_indices:
                rnew = 0.0
                for m in range(a_now):
                    if m == new_idx:
                        continue
                    rnew += _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_now, new_idx, m)
                self._r_agg[new_idx] = rnew
                self._agg_sampler.update(new_idx, rnew)
                for m in range(a_now):
                    if m == new_idx:
                        continue
                    bmnew = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)), R_now, m, new_idx)
                    self._r_agg[m] += bmnew
                    self._agg_sampler.update(m, self._r_agg[m])
    
    # Mark particle as unbreakable: zero out breakage rate and update sampler.
    def _mark_unbreakable(self, k: int) -> None:
        self._break_rate[k] = 0.0
        if self._break_sampler is not None:
            self._break_sampler.update(k, 0.0)
    
    # Wrap legacy stepwise splitting into a helper that samples fragments
    # from table/builtin CDFs one by one.
    def _build_fragments_stepwise(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
        mode, rA, rB, rowsum_cdf, row_cdf, zminA, zminB, pexp = self._get_break_tables_for_state(Vrem_k)
        p = float(pexp) if (pexp is not None and getattr(self, "use_lmc_tables", False)) else float(getattr(self, "frag_num", 2.0))
        fl = int(math.floor(p))
        ce = int(math.ceil(p))
        if fl <= 1:
            fl = 2
            ce = 2
        n = ce if (self._rng.random() < (p - fl)) else fl
    
        Vrem = Vrem_k.copy()
        frags = []
        for _ in range(n - 1):
            frag = self._produce_one_frag_from_remaining(Vrem)
            frags.append(frag)
            Vrem -= frag
        frags.append(Vrem)
        return frags
    
    # Live LMC small-particle fallback: generate NO_FRAG uniform fragments.
    def _build_uniform_live_fragments(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
        """
        When live LMC raises LMCLiveFallback (particle too small to host
        the desired number of lattice cells), fall back to a simple,
        deterministic uniform split into NO_FRAG fragments.

        - For dim=1: split total volume V into NO_FRAG equal parts.
        - For dim=2: split each phase volume (VA, VB) evenly into NO_FRAG
          fragments, keeping the overall composition unchanged.
        """
        # Prefer NO_FRAG from the live LMC adapter; fall back to 2 if missing.
        n = int(getattr(getattr(self, "lmc_live", None), "NO_FRAG", 0))
        if n < 2:
            n = 2

        frags: list[np.ndarray] = []

        if self.dim == 1:
            V = float(Vrem_k[0])
            v = V / float(n) if n > 0 else 0.0
            for _ in range(n):
                frags.append(np.array([v], dtype=float))
        else:
            VA = float(Vrem_k[0])
            VB = float(Vrem_k[1])
            vA = VA / float(n) if n > 0 else 0.0
            vB = VB / float(n) if n > 0 else 0.0
            for _ in range(n):
                frags.append(np.array([vA, vB], dtype=float))

        return frags
    
        # Unified fragment-source dispatcher: Rank one-shot / Live LMC / stepwise split.
    def _break_build_fragments(self, k: int, Vrem_k: np.ndarray) -> tuple[str, list[np.ndarray]]:
        """
                Dispatch order:
                    1) Live LMC (if enabled): on Fallback/Disable, fall back or disable.
                    2) Rank/Copula/Flow tables (if available): one-shot sampling.
                    3) Marginal table or analytic function: stepwise splitting.

                Returns:
                    ("ok", frags) or ("disable", [])
        """
                # 1) Live LMC first
        if getattr(self, "use_lmc_live", False) and (getattr(self, "lmc_live", None) is not None):
            try:
                frags, _E = self.lmc_live.sample_one_shot(Vrem_k, self._rng)
                return "ok", frags

            except Exception as exc:
                exc_name = type(exc).__name__
                if exc_name == _LIVE_FALLBACK_ERROR:
                    # Conditional fallback:
                    #   - if a table/rank model (or adapter) is available, keep the
                    #     original behavior and fall through to those models;
                    #   - otherwise, fall back to a simple uniform NO_FRAG split.
                    has_tables = bool(getattr(self, "use_lmc_tables", False))
                    has_adapter = getattr(self, "lmc_adapter", None) is not None

                    if has_tables or has_adapter:
                        # Old behavior: do nothing here and let the code fall through
                        # to the table / rank-based breakage models below.
                        pass
                    else:
                        # New behavior: no table/rank model available, so we use a
                        # simple deterministic uniform split into NO_FRAG fragments.
                        frags = self._build_uniform_live_fragments(Vrem_k)
                        return "ok", frags
                elif exc_name == _LIVE_DISABLE_ERROR:
                    # Caller can mark this particle as unbreakable after receiving "disable".
                    return "disable", []
                else:
                    raise
    
        # 2) One-shot distribution adapters: rank / copula / flow
        lmc_ad = getattr(self, "lmc_adapter", None)
        ad_name = self._adapter_type_name(lmc_ad)
        if ad_name in _ONE_SHOT_ADAPTER_NAMES:
        # if ad_name in {"LMCRankAdapter", "LMCCopulaAdapter"}:
            # --- Small-particle policy check (only required for "disable") ---
            if getattr(lmc_ad, "small_particle_policy", "fallback") == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
            # Build A and X1
            if self.dim == 1:
                A = float(Vrem_k[0])
                X1 = 1.0
            else:
                A = float(Vrem_k[0] + Vrem_k[1])
                X1 = float(Vrem_k[0] / A) if A > 0.0 else 0.5
    
            # One-shot method signatures differ by adapter type.
            if ad_name == "LMCFlowAdapter":
                # flow: sample_one_shot(A, X1, rng, N=None)
                rA_list, rB_list = lmc_ad.sample_one_shot(A, X1, self._rng, N=None)
            else:
            # rank / copula: sample_one_shot(A, X1, rng, N=None, K_use=None, tail_strategy="equal")
                rA_list, rB_list = lmc_ad.sample_one_shot(
                    A, X1, self._rng, N=None, K_use=None, tail_strategy="equal"
                )
    
            # Reconstruct volume fragments by dimensionality.
            if self.dim == 1:
                frags = [np.array([r * Vrem_k[0]], dtype=float) for r in rA_list]
            else:
                frags = [
                    np.array([rA * Vrem_k[0], rB * Vrem_k[1]], dtype=float)
                    for (rA, rB) in zip(rA_list, rB_list)
                ]
            return "ok", frags
    
        # 3) Marginal-table / analytic-function path: stepwise split
        #    (LMCTableAdapter or pure JIT analytic model)
        if ad_name == _TABLE_ADAPTER_NAME:
            if getattr(lmc_ad, "small_particle_policy", "fallback") == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
        # Use original stepwise splitting logic.
        return "ok", self._build_fragments_stepwise(Vrem_k)
    
    # Main entry: preprocess -> generate fragments -> unified maintenance.
    def _do_one_break(self):
        a = self.a_tot
        if a < 1:
            return
        self._ensure_break_sampler()
        # If no breakable weight exists, return (no breakage event this round).
        if self._break_sampler.total() <= 0.0:
            return
        # -------- Resample within the same event until a breakable particle is found --------
        attempts = 0
        max_attempts = max(1, self.a_tot)  # Avoid infinite loops.
        while attempts < max_attempts:
            # Exit if all breakage weights become zero during the loop.
            if self._break_sampler.total() <= 0.0:
                return
            k = self._break_sampler.sample(self._rng)
            # assert 0 <= k < self._break_sampler._n, (k, self._break_sampler._n, self._break_sampler.total())
            # Remaining-volume vector (passed into branch-specific handlers).
            if self.dim == 1:
                Vrem_k = np.array([self.V_flat[0, k]], dtype=float)
            else:
                Vrem_k = np.array([self.V_flat[0, k], self.V_flat[1, k]], dtype=float)
            status, frags = self._break_build_fragments(k, Vrem_k)
            if status == "disable":
                # Mark this particle as unbreakable and continue resampling.
                self._mark_unbreakable(k)
                attempts += 1
                continue
            if status == "ok":
                # Fragments generated successfully; proceed to update stage.
                break
            # Any other status: treat as failure for this attempt and retry.
            attempts += 1
        # If no breakable particle is found within attempts, do nothing this event.
        if attempts >= max_attempts or self._break_sampler.total() <= 0.0:
            return
        self._break_apply_and_maintain(k, frags)

