# Breakage mixin: breakage rates (full & single), two-level CDF builder, fragment production, single break event.
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# External JIT kernels
from optframework.utils.func.jit_mcpbe import _build_table_1d_jit, _build_tables_2d_jit
from optframework.utils.func.jit_kernel_break import (
    calc_B_R_1d as _kb_BR1_all,
    calc_B_R_2d_flat as _kb_BR2_all,
    calc_break_rate_1d as _kb_br1_single,
    calc_break_rate_2d_flat as _kb_br2_single,
)
from .lmc_adapter import (
    LMCTableAdapter,
    LMCRankAdapter,
    LMCCopulaAdapter,
    LMCFlowAdapter,
    LMCLiveFallback,
    LMCLiveDisable,
)
from .fenwick_new import FenwickSampler

class MCPBEBreak:
    """Breakage logic:
    - breakage rate table (full) from external JIT kernels
    - single-point breakage rate for incremental updates
    - two-level CDF builder (1D/2D) and discrete samplers
    - multi-fragment event by stochastic rounding of expected fragment count
    """

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

        dW_mode = str(getattr(self, "break_dW_mode", "linear")).lower()
        if dW_mode not in ("linear", "sqrt", "const", "ratio"):
            dW_mode = "linear"
        self._break_dW_mode = dW_mode

        dW_max = float(getattr(self, "break_dW_max", 50.0))
        if (not np.isfinite(dW_max)) or dW_max <= 0.0:
            dW_max = 50.0
        self._break_dW_max = dW_max

        dW_min = float(getattr(self, "break_dW_min", 1.0))
        if (not np.isfinite(dW_min)) or dW_min < 0.0:
            dW_min = 0.0
        self._break_dW_min = dW_min

        alpha = float(getattr(self, "break_dW_alpha", 100.0))
        if (not np.isfinite(alpha)) or alpha <= 0.0:
            alpha = 1.0
        self._break_dW_alpha = alpha

        rho_min = float(getattr(self, "break_dW_ratio_min", 0.10))
        rho_max = float(getattr(self, "break_dW_ratio_max", 0.50))
        if (not np.isfinite(rho_min)) or rho_min < 0.0:
            rho_min = 0.0
        if (not np.isfinite(rho_max)) or rho_max <= 0.0:
            rho_max = 0.5
        if rho_max < rho_min:
            rho_min, rho_max = rho_max, rho_min
        self._break_dW_ratio_min = rho_min
        self._break_dW_ratio_max = rho_max

        gamma = float(getattr(self, "break_dW_ratio_gamma", 1.0))
        if (not np.isfinite(gamma)) or gamma <= 0.0:
            gamma = 1.0
        self._break_dW_ratio_gamma = gamma

        self._break_dW_ratio_use_log = bool(getattr(self, "break_dW_ratio_use_log", True))
        V0_cfg = getattr(self, "break_dW_ratio_V0", None)
        if V0_cfg is None:
            self._break_dW_ratio_V0 = None
        else:
            V0_val = float(V0_cfg)
            self._break_dW_ratio_V0 = V0_val if (np.isfinite(V0_val) and V0_val > 0.0) else None

        self._break_N_adaptive = bool(getattr(self, "break_N_adaptive", True))

        N_fixed = int(getattr(self, "break_N", 1))
        if N_fixed < 1:
            N_fixed = 1
        self._break_N = N_fixed

        chunk_ratio = float(getattr(self, "break_dW_chunk_max_ratio", 0.10))
        if (not np.isfinite(chunk_ratio)) or chunk_ratio <= 0.0:
            chunk_ratio = 0.10
        self._break_dW_chunk_max_ratio = chunk_ratio

        chunk_abs = float(getattr(self, "break_dW_chunk_max_abs", 0.0))
        if not np.isfinite(chunk_abs):
            chunk_abs = 0.0
        self._break_dW_chunk_max_abs = chunk_abs

        N_max = int(getattr(self, "break_N_max", 64))
        if N_max < 1:
            N_max = 1
        self._break_N_max = N_max

    # ------------------------------------------------------------------
    # Breakage rate (full table and single-point)
    # ------------------------------------------------------------------
    def _calc_break_rates_full(self):
        """Compute BREAKAGE PROPENSITIES for active slice.
    
        Stored in self._break_rate[:a] as:
            propensity_i = W[i] * S_i
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
        W = self.W[:a]
    
        # --------- Branch 1: MLP model ---------
        use_mlp = bool(self.lmc_use_breakage_model) and (self.lmc_breakage_adapter is not None)
        if use_mlp:
            rates = self.lmc_breakage_adapter.compute_rates_full(self)
            rates = np.asarray(rates, dtype=float)

            prop = W * rates
            np.maximum(prop, 0.0, out=prop)
            self._break_rate[:a] = prop
            if self._break_rate.shape[0] > a:
                self._break_rate[a:] = 0.0
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
        prop = W * rates
        prop[prop < 0.0] = 0.0
        self._break_rate[:a] = prop
        if self._break_rate.shape[0] > a:
            self._break_rate[a:] = 0.0

    def _break_rate_single(self, i: int) -> float:
        """Single-particle BREAKAGE PROPENSITY.
    
        Returns:
            propensity_i = W[i] * S_i
        where S_i is the single-particle breakage rate from MLP/JIT.
        """
        a = self.a_tot
        if i < 0 or i >= a:
            return 0.0

        Wi = float(self.W[i])
        if Wi <= 0.0:
            return 0.0
    
        # --------- Branch 1: MLP model ---------
        use_mlp = bool(self.lmc_use_breakage_model) and (self.lmc_breakage_adapter is not None)
        if use_mlp:
            Si = float(self.lmc_breakage_adapter.compute_rate_single(self, i))
            val = Wi * Si
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
    
        val = Wi * Si
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
        """从当前剩余体积向量 Vrem 推出母颗粒的 (A, X1)。A 是总量，X1 是材料1比例；守护 0 分母。"""
        if self.dim == 1:
            A = float(Vrem[0])
            X1 = 1.0  # 单组分，X1 不会被实际使用
        else:
            v1 = float(Vrem[0])
            v3 = float(Vrem[1])
            A = v1 + v3
            X1 = (v1 / A) if A > 0.0 else 0.5
        return A, X1
            
    def _get_break_tables_for_state(self, Vrem: np.ndarray):
        """
        返回当前母颗粒状态对应的 CDF 表：
        - 若 use_lmc_tables=True 且 lmc_adapter 可用：从 adapter 取 1D/2D 表
        - 否则：走 JIT 路径（调用 _build_break_function()，使用 self._bf* 属性）
        """
        use_lmc = bool(self.use_lmc_pre_model and self.lmc_adapter is not None)
        if not use_lmc:
            if not self._bf_ready:
                self._build_break_function()
            if self.dim == 1:
                return ("1d", self._bf1_rel, self._bf1_cdf, None, None, None, None, float(self.frag_num))
            else:
                return ("2d", self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf, None, None, float(self.frag_num))

        # LMC 表驱动
        A, X1 = self._state_AX1_from_Vrem(Vrem)

        # 纯相/退化时用 1D 表；正常 2D
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
            # 2D 两级 CDF
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

    def _current_break_sum_prop(self) -> float:
        if self._break_sampler is not None:
            return float(self._break_sampler.total())
        return float(np.sum(self._break_rate[:self.a_tot]))
    
    # 统一后处理：应用碎片并维护 break/agg
    def _break_apply_and_maintain(self, k: int, frags: list[np.ndarray], dW: float) -> None:
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
    
        new_indices: list[int] = []
    
        # 1) Append ALL fragments as new particles, each carrying weight dW
        for f in frags:
            self._append_particle_column(f)
            new_idx = self.a_tot - 1
            new_indices.append(new_idx)
    
            self.W[new_idx] = dW
    
            br_new = self._break_rate_single(new_idx)  # already returns W*Si
            self._break_rate[new_idx] = br_new
            if self._break_sampler is not None:
                self._break_sampler.update(new_idx, br_new)
    
        # 2) Reduce parent weight but keep its volume unchanged
        w_rem = w_parent_old - dW
        self.W[k] = w_rem
    
        if w_rem > 0.0:
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
    
    # 小颗粒禁用：破碎率清零并更新采样器
    def _mark_unbreakable(self, k: int) -> None:
        self._break_rate[k] = 0.0
        if self._break_sampler is not None:
            self._break_sampler.update(k, 0.0)
    
    # 旧分步切逻辑封装成一个“从表/内置 CDF 逐片生成”的助手
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
    
    # Live LMC 小颗粒 fallback：均匀产生 NO_FRAG 个碎片
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
    
    # 统一的“构建碎片来源分派”：Rank one-shot / Live LMC / 分步切
    def _break_build_fragments(self, Vrem_k: np.ndarray) -> tuple[str, list[np.ndarray]]:
        """
        分派顺序：
          1) Live LMC（若启用）——若触发 Fallback/Disable，分别回退或禁用
          2) Rank tables（若存在）——one-shot 采样（无需依赖 break_one_shot 标志）
          3) 边际表 / 数学函数 ——分步切
        返回:
          ("ok", frags) 或 ("disable", [])
        """
        # 1) Live LMC 优先
        if self.use_lmc_live and (self.lmc_live is not None):
            try:
                frags, _E = self.lmc_live.sample_one_shot(Vrem_k, self._rng)
                return "ok", frags

            except LMCLiveFallback:
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
                    # New behavior: no table/rank model available, so we use a
                    # simple deterministic uniform split into NO_FRAG fragments.
                    frags = self._build_uniform_live_fragments(Vrem_k)
                    return "ok", frags

            except LMCLiveDisable:
                # 上层看到 "disable" 后可以把该粒子的破碎率置零
                return "disable", []
    
        # 2) 一次性分布类适配器：rank / copula / flow
        lmc_ad = self.lmc_adapter
        if isinstance(lmc_ad, (LMCRankAdapter, LMCCopulaAdapter, LMCFlowAdapter)):
        # if isinstance(lmc_ad, (LMCRankAdapter, LMCCopulaAdapter)):
            # --- 小颗粒策略（仅当策略为 disable 时才判定；fallback 直接用） ---
            if lmc_ad.small_particle_policy == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
            # 构造 A, X1
            if self.dim == 1:
                A = float(Vrem_k[0])
                X1 = 1.0
            else:
                A = float(Vrem_k[0] + Vrem_k[1])
                X1 = float(Vrem_k[0] / A) if A > 0.0 else 0.5
    
            # 不同适配器的 one-shot 调用签名略有区别，这里分开调
            if isinstance(lmc_ad, LMCFlowAdapter):
                # flow: sample_one_shot(A, X1, rng, N=None)
                rA_list, rB_list = lmc_ad.sample_one_shot(A, X1, self._rng, N=None)
            else:
            # rank / copula: sample_one_shot(A, X1, rng, N=None, K_use=None, tail_strategy="equal")
                rA_list, rB_list = lmc_ad.sample_one_shot(
                    A, X1, self._rng, N=None, K_use=None, tail_strategy="equal"
                )
    
            # 按维数还原成体积碎片
            if self.dim == 1:
                frags = [np.array([r * Vrem_k[0]], dtype=float) for r in rA_list]
            else:
                frags = [
                    np.array([rA * Vrem_k[0], rB * Vrem_k[1]], dtype=float)
                    for (rA, rB) in zip(rA_list, rB_list)
                ]
            return "ok", frags
    
        # 3) 边际表 / 数学函数：分步切（LMCTableAdapter 或 纯 JIT 数学函数）
        if isinstance(lmc_ad, LMCTableAdapter):
            if lmc_ad.small_particle_policy == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
        # 走原来的逐步切分逻辑
        return "ok", self._build_fragments_stepwise(Vrem_k)

    def _compute_dW(self, k: int, sum_prop_before: float) -> float:
        """Compute packet size ΔW for a breakage event on particle k.
    
        Existing modes:
          - "linear": ΔW ~ f
          - "sqrt"  : ΔW ~ sqrt(f)
          - "const" : ΔW = dW_max
    
        New mode:
          - "ratio" : ΔW = rho(V) * Wk, with rho in [rho_min, rho_max]
                     where rho decreases with particle volume (larger V -> smaller rho).
    
        Notes
        -----
        - In "ratio" mode, rho depends ONLY on particle volume (Vtot), not on propensity fraction f.
        - Always clamps by [break_dW_min, break_dW_max] and available Wk.
    
        Required attributes
        -------------------
        break_dW_mode : str, default "linear"
        break_dW_max  : float, default 50.0
        break_dW_min  : float, default 1.0
    
        ratio-mode knobs
        ---------------
        break_dW_ratio_min    : float, default 0.10     # minimum fraction of Wk per event (for large V)
        break_dW_ratio_max    : float, default 0.50     # maximum fraction of Wk per event (for small V)
        break_dW_ratio_V0     : float, default None     # knee volume in normalized V space; if None -> median(Vtot)
        break_dW_ratio_gamma  : float, default 1.0      # shape exponent; larger -> sharper transition
        break_dW_ratio_use_log: bool,  default True     # use log(V) mapping to handle wide size ranges
        """
        Wk = float(self.W[k])
        if Wk <= 0.0 or not np.isfinite(Wk):
            return 0.0
    
        dW_max = float(self._break_dW_max)
        dW_min = float(self._break_dW_min)
    
        # If total propensity is invalid, we can still compute ratio-mode purely from volume.
        # For other modes, we need sum_prop_before and pk.
        mode = self._break_dW_mode
    
        if mode == "ratio":
            # --- volume-only relative packet size ---
            rho_min = float(self._break_dW_ratio_min)
            rho_max = float(self._break_dW_ratio_max)
    
            gamma = float(self._break_dW_ratio_gamma)
    
            use_log = bool(self._break_dW_ratio_use_log)
    
            # Particle volume (assumed normalized Vtot)
            a = int(self.a_tot)
            V = float(self.V_flat[-1, k])  # Vtot[k]
            if not np.isfinite(V) or V <= 0.0:
                V = 0.0
    
            # Choose knee volume V0:
            #   - If provided: use it
            #   - Else: use current median of active Vtot (robust and scale-adaptive)
            V0 = self._break_dW_ratio_V0
            if V0 is None:
                if a > 0:
                    Vactive = np.asarray(self.V_flat[-1, :a], dtype=float)
                    m = np.isfinite(Vactive) & (Vactive > 0.0)
                    if np.any(m):
                        V0 = float(np.median(Vactive[m]))
                    else:
                        V0 = 1.0
                else:
                    V0 = 1.0
            V0 = float(V0)
    
            # Map volume to t in [0,1], where:
            #   - small V -> t ~ 0 -> rho ~ rho_max
            #   - large V -> t ~ 1 -> rho ~ rho_min
            # Use smooth saturating function:
            #   s = V/(V+V0) in (0,1)
            #   t = s^gamma
            #   rho = rho_max - (rho_max-rho_min)*t
            if use_log:
                # log-soften: use V' = log(1+V) and V0' = log(1+V0)
                Vp = math.log1p(max(V, 0.0))
                V0p = math.log1p(V0)
                s = Vp / (Vp + V0p) if (Vp + V0p) > 0.0 else 0.0
            else:
                s = V / (V + V0) if (V + V0) > 0.0 else 0.0
    
            s = float(np.clip(s, 0.0, 1.0))
            t = float(s ** gamma)
    
            rho = rho_max - (rho_max - rho_min) * t
            dW = rho * Wk
    
        elif mode == "const":
            dW = dW_max
    
        else:
            # For linear/sqrt we need pk and sum_prop_before
            if not np.isfinite(sum_prop_before) or sum_prop_before <= 0.0:
                return float(min(Wk, dW_max))
    
            pk = float(self._break_rate[k])  # W*S
            if not np.isfinite(pk) or pk <= 0.0:
                return 0.0
    
            f = pk / float(sum_prop_before)
            f = float(np.clip(f, 0.0, 1.0))
    
            alpha = float(self._break_dW_alpha)
    
            span = (dW_max - dW_min)
            if span < 0.0:
                span = 0.0
    
            if mode == "sqrt":
                dW = alpha * (f ** 0.5) * span
            else:
                # default: linear
                dW = alpha * f * span
    
        # clamp by [min, max] and available weight
        if dW < dW_min:
            dW = dW_min
        if dW > dW_max:
            dW = dW_max
        if dW > Wk:
            dW = Wk
    
        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)
    
    def _do_one_break(self): 
        # 主入口：预处理 -> 生成碎片 -> 统一维护
        if self.a_tot < 1:
            return

        # Optional per-event quadrature data for dt integration in base solver.
        # For N>1 chunked events, we fill these; otherwise keep None and use log-mean fallback.
        self._last_break_prop_nodes = None
        self._last_break_dW_chunks = None
    
        self._ensure_break_sampler()
    
        # 若当前没有可破碎权重，直接退出
        if self._break_sampler.total() <= 0.0:
            return
    
        for _ in range(max(1, self.a_tot)):
            if self._break_sampler.total() <= 0.0:
                return
    
            k = self._break_sampler.sample(self._rng)
    
            # 该计算颗粒的可用权重
            Wk0 = float(self.W[k])
            if Wk0 <= 0.0:
                self._mark_unbreakable(k)
                continue
    
            # 本次事件总消耗的权重包
            sum_prop_before = float(self._break_sampler.total())
            dW_total = self._compute_dW(k, sum_prop_before)
            if dW_total <= 0.0:
                self._mark_unbreakable(k)
                continue
    
            # --------------------------
            # Adaptive chunking: choose N based on dW_total
            # --------------------------
            # Backward-compatible switch:
            #   break_N_adaptive=True  -> adaptive N (default)
            #   break_N_adaptive=False -> fixed N = break_N
            use_adaptive = bool(self._break_N_adaptive)
    
            if not use_adaptive:
                N = int(self._break_N)
            else:
                # Maximum chunk size (controls batch-wise correlation):
                # chunk_max = min( break_dW_chunk_max_abs,
                #                  break_dW_chunk_max_ratio * Wk0 )
                ratio = float(self._break_dW_chunk_max_ratio)  # 10% of Wk0 by default
                chunk_max = ratio * float(max(Wk0, 1e-12))
    
                abs_cap = float(self._break_dW_chunk_max_abs)
                if abs_cap > 0.0:
                    chunk_max = min(chunk_max, abs_cap)
    
                # ensure sensible lower bound
                if not np.isfinite(chunk_max) or chunk_max <= 0.0:
                    chunk_max = max(1.0, 0.10 * dW_total)
    
                N = int(math.ceil(float(dW_total) / float(chunk_max)))
                if N < 1:
                    N = 1
    
                N_max = int(self._break_N_max)
                if N > N_max:
                    N = N_max
    
            # Keep legacy cap: don't create more chunks than total weight (helps avoid tiny chunks)
            if N > dW_total:
                N = int(dW_total)
            if N < 1:
                N = 1
    
            # 记录本次事件对应的真实事件数（总消耗），用于 solve() 里的 dt 计算
            self._last_break_dW = float(dW_total)

            # For chunked breakage (N>1), record propensity nodes and chunk masses
            # to enable trapezoidal integration in dt computation.
            use_trap = N > 1
            if use_trap:
                prop_nodes = [float(sum_prop_before)]
                dW_chunks = []
    
            # 均分权重（允许非整数）；最后一份用“剩余量”兜底，避免累计误差
            base_chunk = dW_total / float(N)
            remaining = float(dW_total)
    
            for n in range(N):
                # 若 parent 已被移除或索引已越界，停止（避免 swap/remove 后继续用旧 k）
                if k >= self.a_tot:
                    break
    
                Wk_now = float(self.W[k])
                if Wk_now <= 0.0:
                    break
    
                # 本份权重
                dW = base_chunk if (n < N - 1) else remaining
                if dW > remaining:
                    dW = remaining
                if dW > Wk_now:
                    dW = Wk_now
                if dW <= 0.0:
                    break
    
                # 母颗粒体积（用于生成“单个真实颗粒”的碎片体积分配）
                if self.dim == 1:
                    Vrem_k = np.array([self.V_flat[0, k]], dtype=float)
                else:
                    Vrem_k = np.array([self.V_flat[0, k], self.V_flat[1, k]], dtype=float)
    
                status, frags = self._break_build_fragments(Vrem_k)
    
                if status == "disable":
                    # 该粒子不可破碎：清零 propensity，整次事件作废（返回）
                    self._mark_unbreakable(k)
                    return
    
                if status == "ok":
                    # 应用该份碎片（以 dW_chunk 的权重 append），并减少父颗粒权重
                    self._break_apply_and_maintain(k, frags, dW)
                    remaining -= dW

                    if use_trap:
                        dW_chunks.append(float(dW))
                        prop_nodes.append(self._current_break_sum_prop())

                    if remaining <= 0.0:
                        break
                else:
                    # 未知状态：直接退出
                    return

            if use_trap and len(dW_chunks) >= 1 and len(prop_nodes) == len(dW_chunks) + 1:
                self._last_break_dW_chunks = dW_chunks
                self._last_break_prop_nodes = prop_nodes
    
            return  # 本次 break 事件完成（无论是否完全用尽 remaining）

        return

