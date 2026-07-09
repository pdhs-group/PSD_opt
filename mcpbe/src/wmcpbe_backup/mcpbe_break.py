# Breakage mixin: breakage rates (full & single), two-level CDF builder, fragment production, single break event.
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# External JIT kernels
from pbe_core.func.jit_mcpbe import _build_table_1d_jit, _build_tables_2d_jit
from pbe_core.func.jit_kernel_break import (
    calc_B_R_1d as _kb_BR1_all,
    calc_B_R_2d_flat as _kb_BR2_all,
    calc_break_rate_1d as _kb_br1_single,
    calc_break_rate_2d_flat as _kb_br2_single,
)
from .fenwick_new import FenwickSampler

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

    def _prepare_break_config(self) -> None:
        """Cache breakage configuration that is effectively constant during one solve run."""
        self._bf_ready = False
        self._break_G = float(getattr(self, "G", 1.0))
        self._break_pl_P1 = float(getattr(self, "pl_P1", 1.0))
        self._break_pl_P2 = float(getattr(self, "pl_P2", 1.0))
        self._break_pl_P3 = float(getattr(self, "pl_P3", 1.0))
        self._break_pl_P4 = float(getattr(self, "pl_P4", 1.0))
        self._break_BREAKRVAL = int(getattr(self, "BREAKRVAL", 1))
        self._break_BREAKFVAL = int(getattr(self, "BREAKFVAL", 1))
        self._break_pl_v = float(getattr(self, "pl_v", 1.0))
        self._break_pl_q = float(getattr(self, "pl_q", 1.0))

        self._prepare_break_delta_config()
        self._break_dW_const = float(getattr(self, "_break_dW_const", 50.0))

    # ------------------------------------------------------------------
    # Breakage rate (full table and single-point)
    # ------------------------------------------------------------------
    def _calc_break_rates_full(self):
        """Compute BREAKAGE PROPENSITIES for active slice.
    
        Stored in self._break_rate[:a] as:
            propensity_i = W[i] * S_i / delta_i
        with delta_i = min(break_dW_const, W[i]).
        where S_i is the single-particle breakage rate from MLP/JIT.
        """
        a = self.a_tot
        if a <= 0:
            return
        cap = self._cap
        if (not hasattr(self, "_break_rate")
                or self._break_rate is None
                or self._break_rate.shape[0] < cap):
            self._break_rate = np.zeros(max(8, cap), dtype=float)
        if (not hasattr(self, "_delta_break")
                or self._delta_break is None
                or self._delta_break.shape[0] < cap):
            self._delta_break = np.zeros(max(8, cap), dtype=float)
        W = self.W[:a]
        delta = self._delta_from_weights(W, dW_const=float(self._break_dW_const))
        self._delta_break[:a] = delta
    
        # --------- Branch 1: MLP model ---------
        use_mlp = bool(self.lmc_use_breakage_model) and (self.lmc_breakage_adapter is not None)
        if use_mlp:
            rates = self.lmc_breakage_adapter.compute_rates_full(self)
            rates = np.asarray(rates, dtype=float)

            prop = np.divide(
                W * rates,
                delta,
                out=np.zeros_like(W, dtype=float),
                where=delta > 0.0,
            )
            np.maximum(prop, 0.0, out=prop)
            self._break_rate[:a] = prop
            if self._break_rate.shape[0] > a:
                self._break_rate[a:] = 0.0
                self._delta_break[a:] = 0.0
            return
    
        # --------- Branch 2: original JIT kernels (single-particle rates) ---------
        self.V = self.V_flat[-1, :a]
        self.B_R = np.zeros(a, dtype=float)

        if self.dim == 1:
            _kb_BR1_all(self)
        elif self.dim == 2:
            _kb_BR2_all(self)
        else:
            raise RuntimeError(f"Unsupported dim={self.dim} for breakage kernels.")
    
        rates = np.asarray(self.B_R, dtype=float)
        prop = np.divide(
            W * rates,
            delta,
            out=np.zeros_like(W, dtype=float),
            where=delta > 0.0,
        )
        prop[prop < 0.0] = 0.0
        self._break_rate[:a] = prop
        if self._break_rate.shape[0] > a:
            self._break_rate[a:] = 0.0
            self._delta_break[a:] = 0.0

    def _break_rate_single(self, i: int) -> float:
        """Single-particle BREAKAGE PROPENSITY.
    
        Returns:
            propensity_i = W[i] * S_i / delta_i
        with delta_i = min(break_dW_const, W[i]).
        where S_i is the single-particle breakage rate from MLP/JIT.
        """
        a = self.a_tot
        if i < 0 or i >= a:
            return 0.0

        Wi = float(self.W[i])
        if Wi <= 0.0:
            if hasattr(self, "_delta_break") and self._delta_break is not None:
                self._delta_break[i] = 0.0
            return 0.0
        delta_i = self._update_delta_single(i, attr_name="_delta_break", dW_const=float(self._break_dW_const))
        if delta_i <= 0.0:
            return 0.0
    
        # --------- Branch 1: MLP model ---------
        use_mlp = bool(self.lmc_use_breakage_model) and (self.lmc_breakage_adapter is not None)
        if use_mlp:
            Si = float(self.lmc_breakage_adapter.compute_rate_single(self, i))
            val = Wi * Si / delta_i
            return float(val) if val > 0.0 else 0.0
    
        # --------- Branch 2: original JIT single-particle rate ---------
        if self.dim == 1:
            Si = float(
                _kb_br1_single(
                    self.V_flat[-1, :a],
                    self._break_pl_P1,
                    self._break_pl_P2,
                    self._break_G,
                    self._break_BREAKRVAL,
                    i,
                )
            )
        else:
            Si = float(
                _kb_br2_single(
                    self.V_flat[-1, :a],
                    self.V_flat[0, :a],
                    self.V_flat[1, :a],
                    self._break_G,
                    self._break_pl_P1,
                    self._break_pl_P2,
                    self._break_pl_P3,
                    self._break_pl_P4,
                    self._break_BREAKRVAL,
                    self._break_BREAKFVAL,
                    i,
                )
            )
    
        val = Wi * Si / delta_i
        return float(val) if val > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Two-level CDF builder (cached)
    # ------------------------------------------------------------------
    def _build_break_function(self, num_points: int = 1000):
        """Build two-level CDF tables for breakage fragment distributions (1D/2D)."""
        key = (
            int(self.dim),
            int(num_points),
            self._break_BREAKFVAL,
            self._break_pl_v,
            self._break_pl_q,
        )
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
            v = self._break_pl_v
            q = self._break_pl_q
            bf = self._break_BREAKFVAL
            cdf = _build_table_1d_jit(rel, v, q, bf)
            self._bf1_rel = rel
            self._bf1_cdf = cdf
            self._bf_cache[key] = (self._bf1_rel, self._bf1_cdf)
        else:
            rel1 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            rel3 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            v = self._break_pl_v
            q = self._break_pl_q
            bf = self._break_BREAKFVAL
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
        use_lmc = bool(self.use_lmc_pre_model and self.lmc_adapter is not None)
        if not use_lmc:
            if not self._bf_ready:
                self._build_break_function()
            if self.dim == 1:
                return ("1d", self._bf1_rel, self._bf1_cdf, None, None, None, None, float(self.frag_num))
            else:
                return ("2d", self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf, None, None, float(self.frag_num))

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
    # Produce one fragment from remaining volume vector
    # ------------------------------------------------------------------
    def _produce_one_frag_from_remaining(self, Vrem: np.ndarray) -> np.ndarray:
        mode, rA, rB, rowsum_cdf, row_cdf, *_ = self._get_break_tables_for_state(Vrem)

        if mode == "1d":
            u = float(self._rng.random())
            rel, cdf = rA, rB
            j = int(np.searchsorted(cdf, u, side="right"))
            if j >= rel.size: j = rel.size - 1
            r = float(rel[j])
            r = max(0.0, min(1.0, r))
            if self.dim == 1:
                return np.array([r * Vrem[0]], dtype=float)
            else:
                if Vrem[0] > 0.0 and Vrem[1] <= 0.0:
                    return np.array([r * Vrem[0], 0.0], dtype=float)
                elif Vrem[1] > 0.0 and Vrem[0] <= 0.0:
                    return np.array([0.0, r * Vrem[1]], dtype=float)
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

    @staticmethod
    def _frag_total_volume(frag: np.ndarray) -> float:
        return float(np.sum(np.asarray(frag, dtype=float)))

    def _fragments_have_zero_volume(self, frags: list[np.ndarray]) -> bool:
        if not frags:
            return False
        for frag in frags:
            if self._frag_total_volume(frag) <= 0.0:
                return True
        return False

    def _filter_positive_volume_fragments(self, frags: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for frag in frags:
            if self._frag_total_volume(frag) > 0.0:
                out.append(np.asarray(frag, dtype=float))
        return out

    # Unified post-processing: apply fragments and maintain break/agg states.
    def _break_apply_and_maintain(self, k: int, frags: list[np.ndarray], dW: float, Vrem_k: np.ndarray) -> None:
        """Apply one *packet* breakage event.
    
        Interpretation:
          - Parent compute particle k represents W[k] real particles of volume Vk.
          - This call breaks dW of those real particles (dW can be non-integer).
          - The remaining (W[k]-dW) real particles stay at the same Vk (same compute particle k).
          - Broken products are represented by appending fragment compute particles,
            each with weight = dW and volume equal to the fragment volume of ONE real parent.
        """
        if (not frags) or (dW <= 0.0):
            return
    
        w_parent_old = float(self.W[k])
        if w_parent_old <= 0.0:
            self._mark_unbreakable(k)
            return
    
        dW = float(min(dW, w_parent_old))
        if dW <= 0.0:
            self._mark_unbreakable(k)
            return

        Vrem_ref = np.asarray(Vrem_k, dtype=float).copy()
        resample_attempts = 0
        max_resample = 1000
        while self._fragments_have_zero_volume(frags) and resample_attempts < max_resample:
            status_retry, frags_retry = self._break_build_fragments(Vrem_ref.copy())
            if status_retry == "disable":
                self._mark_unbreakable(k)
                return
            if status_retry == "ok" and frags_retry:
                frags = frags_retry
            resample_attempts += 1

        frags = self._filter_positive_volume_fragments(frags)
        if not frags:
            return
    
        new_indices: list[int] = []
    
        # 1) Append ALL fragments as new particles, each carrying weight dW
        for f in frags:
            self._append_particle_column(f)
            new_idx = self.a_tot - 1
            new_indices.append(new_idx)
    
            self.W[new_idx] = dW
            self._update_delta_single(new_idx, attr_name="_delta_break", dW_const=float(self._break_dW_const))
    
            br_new = self._break_rate_single(new_idx)  # already returns W*Si
            self._break_rate[new_idx] = br_new
            if self._break_sampler is not None:
                self._break_sampler.update(new_idx, br_new)
    
        # 2) Reduce parent weight but keep its volume unchanged
        w_rem = w_parent_old - dW
        self.W[k] = w_rem
        if w_rem > 0.0:
            self._update_delta_single(k, attr_name="_delta_break", dW_const=float(self._break_dW_const))
            br_k = self._break_rate_single(k)  # uses new W[k]
            self._break_rate[k] = br_k
            if self._break_sampler is not None:
                self._break_sampler.update(k, br_k)
        else:
            # Parent population fully consumed -> remove compute particle k
            self._remove_particle_column(k)

        # 3) Agglomeration maintenance (mix mode): full weighted rebuild for consistency
        pt = self.process_type
        if pt in ("agglomeration", "mix") and self._agg_sampler is not None:
            self._rebuild_all_propensities()
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
    
    # Mark particle as unbreakable: zero out breakage rate and update sampler.
    def _mark_unbreakable(self, k: int) -> None:
        self._break_rate[k] = 0.0
        if hasattr(self, "_delta_break") and self._delta_break is not None:
            self._delta_break[k] = 0.0
        if self._break_sampler is not None:
            self._break_sampler.update(k, 0.0)
    
    # Wrap legacy stepwise splitting into a helper that samples fragments
    # from table/builtin CDFs one by one.
    def _build_fragments_stepwise(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
        *_, pexp = self._get_break_tables_for_state(Vrem_k)
        p = float(pexp) if (pexp is not None and self.use_lmc_pre_model) else float(self.frag_num)
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
        n = int(self.lmc_NO_FRAG)
        if n < 2:
            n = 2

        if self.dim == 1:
            V = float(Vrem_k[0])
            v = V / float(n)
            return [np.array([v], dtype=float) for _ in range(n)]

        VA = float(Vrem_k[0])
        VB = float(Vrem_k[1])
        vA = VA / float(n)
        vB = VB / float(n)
        return [np.array([vA, vB], dtype=float) for _ in range(n)]
    
    # Unified fragment-source dispatcher: Rank one-shot / Live LMC / stepwise split.
    def _break_build_fragments(self, Vrem_k: np.ndarray) -> tuple[str, list[np.ndarray]]:
        """
        Dispatch order:
          1) Live LMC (if enabled): on Fallback/Disable, fall back or disable.
          2) Rank/Copula/Flow tables (if available): one-shot sampling.
          3) Marginal table or analytic function: stepwise splitting.

        Returns:
          ("ok", frags) or ("disable", [])
        """
        # 1) Live LMC first
        if self.use_lmc_live and (self.lmc_live is not None):
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
                    has_tables = bool(self.use_lmc_pre_model)
                    has_adapter = self.lmc_adapter is not None

                    if has_tables or has_adapter:
                        # Old behavior: do nothing here and let the code fall through
                        # to the table / rank-based breakage models below.
                        pass
                    else:
                        # New behavior: no table/rank model available, so use a
                        # simple deterministic uniform split into NO_FRAG fragments.
                        frags = self._build_uniform_live_fragments(Vrem_k)
                        return "ok", frags
                elif exc_name == _LIVE_DISABLE_ERROR:
                    # Caller can mark this particle as unbreakable after receiving "disable".
                    return "disable", []
                else:
                    raise
    
        # 2) One-shot distribution adapters: rank / copula / flow
        lmc_ad = self.lmc_adapter
        ad_name = self._adapter_type_name(lmc_ad)
        if ad_name in _ONE_SHOT_ADAPTER_NAMES:
        # if ad_name in {"LMCRankAdapter", "LMCCopulaAdapter"}:
            # --- Small-particle policy check (only required for "disable") ---
            if lmc_ad.small_particle_policy == "disable":
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
            if lmc_ad.small_particle_policy == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
        # Use original stepwise splitting logic.
        return "ok", self._build_fragments_stepwise(Vrem_k)

    def _compute_dW(self, k: int) -> float:
        """Compute packet size Î”W for breakage using a single constant event size."""
        Wk = float(self.W[k])
        if Wk <= 0.0 or not np.isfinite(Wk):
            return 0.0
        dW = min(float(self._break_dW_const), Wk)
    
        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)
    
    def _compute_dW_packet(self, k: int) -> float:
        """Compute packet size delta_i for breakage."""
        if k < 0 or k >= self.a_tot:
            return 0.0
        if hasattr(self, "_delta_break") and self._delta_break is not None:
            dW = float(self._delta_break[k])
        else:
            dW = self._update_delta_single(k, attr_name="_delta_break", dW_const=float(self._break_dW_const))
        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)

    def _do_one_break(self): 
        # Main entry: preprocess -> generate fragments -> unified maintenance.
        self._last_break_dW = 0.0
        if self.a_tot < 1:
            return
    
        self._ensure_break_sampler()
    
        # If no breakable weight exists, return directly.
        if self._break_sampler.total() <= 0.0:
            return
    
        for _ in range(max(1, self.a_tot)):
            if self._break_sampler.total() <= 0.0:
                return
    
            k = self._break_sampler.sample(self._rng)
    
            # Available weight represented by this compute particle
            Wk0 = float(self.W[k])
            if Wk0 <= 0.0:
                self._mark_unbreakable(k)
                continue
    
            dW_total = self._compute_dW_packet(k)
            if dW_total <= 0.0:
                self._mark_unbreakable(k)
                continue

            # Store real-event count consumed by this packet event for dt update in solve().
            self._last_break_dW = float(dW_total)

            # Single packet event (no additional chunking, fixed N=1).
            if k >= self.a_tot:
                return

            Wk_now = float(self.W[k])
            if Wk_now <= 0.0:
                return

            dW = min(float(dW_total), Wk_now)
            if dW <= 0.0:
                return

            if self.dim == 1:
                Vrem_k = np.array([self.V_flat[0, k]], dtype=float)
            else:
                Vrem_k = np.array([self.V_flat[0, k], self.V_flat[1, k]], dtype=float)

            status, frags = self._break_build_fragments(Vrem_k)

            if status == "disable":
                self._mark_unbreakable(k)
                return

            if status == "ok":
                self._break_apply_and_maintain(k, frags, dW, Vrem_k)
            else:
                return
    
            return  # Current breakage event completed.

        return


