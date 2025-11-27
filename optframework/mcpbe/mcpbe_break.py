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
from .lmc_adapter import (
    LMCTableAdapter,
    LMCRankAdapter,
    LMCCopulaAdapter,
    # LMCFlowAdapter,
    LMCLiveFallback,
    LMCLiveDisable,
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
        use_lmc = bool(getattr(self, "use_lmc_pre_model", False) and getattr(self, "lmc_adapter", None) is not None)
        if not use_lmc:
            if not getattr(self, "_bf_ready", False):
                self._build_break_function()
            if self.dim == 1:
                return ("1d", self._bf1_rel, self._bf1_cdf, None, None, None, None, float(getattr(self, "frag_num", 2.0)))
            else:
                return ("2d", self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf, None, None, float(getattr(self, "frag_num", 2.0)))

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
            # 1D: r ∈ [0,1] -> 单组分或总碎片比分布
            u = float(self._rng.random())
            # k = int(np.searchsorted(rB if rB is not None else rA, u, side="right"))  # 这里 rA 是 rel，rB 是 cdf？
            # 注意：我们在 _get_break_tables_for_state("1d") 里返回 (rel1d, cdf1d, ...)，
            # 这里应取 cdf = rB? 为避免混淆重写更清晰：
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
                # 2D 退化为 1D：按总量比分配到“非零”的那一相
                if Vrem[0] > 0.0 and Vrem[1] <= 0.0:
                    return np.array([r * Vrem[0], 0.0], dtype=float)
                elif Vrem[1] > 0.0 and Vrem[0] <= 0.0:
                    return np.array([0.0, r * Vrem[1]], dtype=float)
                else:
                    # 都为 0 直接 0 碎片
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
    
    # 统一后处理：应用碎片并维护 break/agg
    def _break_apply_and_maintain(self, k: int, frags: list[np.ndarray]) -> None:
        if not frags:
            return
        # n = len(frags)
        new_indices = []
        # 先 append 前 n-1 个
        for f in frags[:-1]:
            self._append_particle_column(f)
            new_idx = self.a_tot - 1
            new_indices.append(new_idx)
            br_new = self._break_rate_single(new_idx)
            self._break_rate[new_idx] = br_new
            if self._break_sampler is not None:
                self._break_sampler.update(new_idx, br_new)
    
        # 最后一块回写到 k
        last = frags[-1]
        self.V_flat[: self.dim, k] = last
        self.V_flat[-1, k] = float(np.sum(last))
        self.X[k] = float(self._vol2diam(self.V_flat[-1, k]))
    
        br_k = self._break_rate_single(k)
        self._break_rate[k] = br_k
        if self._break_sampler is not None:
            self._break_sampler.update(k, br_k)
    
        # 统一的 agg 维护（和你之前一致）
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
    
    # 小颗粒禁用：破碎率清零并更新采样器
    def _mark_unbreakable(self, k: int) -> None:
        self._break_rate[k] = 0.0
        if self._break_sampler is not None:
            self._break_sampler.update(k, 0.0)
    
    # 旧分步切逻辑封装成一个“从表/内置 CDF 逐片生成”的助手
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
    
    # 统一的“构建碎片来源分派”：Rank one-shot / Live LMC / 分步切
    def _break_build_fragments(self, k: int, Vrem_k: np.ndarray) -> tuple[str, list[np.ndarray]]:
        """
        分派顺序：
          1) Live LMC（若启用）——若触发 Fallback/Disable，分别回退或禁用
          2) Rank tables（若存在）——one-shot 采样（无需依赖 break_one_shot 标志）
          3) 边际表 / 数学函数 ——分步切
        返回:
          ("ok", frags) 或 ("disable", [])
        """
        # 1) Live LMC 优先
        if getattr(self, "use_lmc_live", False) and (getattr(self, "lmc_live", None) is not None):
            try:
                frags, _E = self.lmc_live.sample_one_shot(Vrem_k, self._rng)
                return "ok", frags
            except LMCLiveFallback:
                # 继续往下走，用离线模型兜底
                pass
            except LMCLiveDisable:
                # 上层看到 "disable" 后可以把该粒子的破碎率置零
                return "disable", []
    
        # 2) 一次性分布类适配器：rank / copula / flow
        lmc_ad = getattr(self, "lmc_adapter", None)
        # if isinstance(lmc_ad, (LMCRankAdapter, LMCCopulaAdapter, LMCFlowAdapter)):
        if isinstance(lmc_ad, (LMCRankAdapter, LMCCopulaAdapter)):
            # --- 小颗粒策略（仅当策略为 disable 时才判定；fallback 直接用） ---
            if getattr(lmc_ad, "small_particle_policy", "fallback") == "disable":
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
            # if isinstance(lmc_ad, LMCFlowAdapter):
            #     # flow: sample_one_shot(A, X1, rng, N=None)
            #     rA_list, rB_list = lmc_ad.sample_one_shot(A, X1, self._rng, N=None)
            # else:
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
            if getattr(lmc_ad, "small_particle_policy", "fallback") == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
        # 走原来的逐步切分逻辑
        return "ok", self._build_fragments_stepwise(Vrem_k)
    
    # 主入口：预处理 -> 生成碎片 -> 统一维护
    def _do_one_break(self):
        a = self.a_tot
        if a < 1:
            return
        self._ensure_break_sampler()
        # 若当前没有可破碎权重，直接退出（本次事件不发生破碎）
        if self._break_sampler.total() <= 0.0:
            return
        # -------- 在同一事件内循环采样，直到选到可破碎的颗粒 --------
        attempts = 0
        max_attempts = max(1, self.a_tot)  # 最多尝试当前颗粒数次，避免死循环
        while attempts < max_attempts:
            # 若在循环过程中所有破碎权重被清零，则退出
            if self._break_sampler.total() <= 0.0:
                return
            k = self._break_sampler.sample(self._rng)
            # 剩余体积向量（便于传给各分支）
            if self.dim == 1:
                Vrem_k = np.array([self.V_flat[0, k]], dtype=float)
            else:
                Vrem_k = np.array([self.V_flat[0, k], self.V_flat[1, k]], dtype=float)
            status, frags = self._break_build_fragments(k, Vrem_k)
            if status == "disable":
                # 标记该颗粒不可破碎，并继续在同一事件内重采样其它颗粒
                self._mark_unbreakable(k)
                attempts += 1
                continue
            if status == "ok":
                # 正常生成碎片，退出循环，进入后续的更新逻辑
                break
            # 其它状态（稳妥）也视作本次失败，继续尝试
            attempts += 1
        # 若尝试用尽仍未获得可破碎颗粒，则本次事件无操作返回
        if attempts >= max_attempts or self._break_sampler.total() <= 0.0:
            return
        self._break_apply_and_maintain(k, frags)
