from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import math
from numba import njit
from lmcann.core.lmc.lmc import LMCSimulator


# ==============================
# Base adapter for LMC tables
# ==============================
class LMCBaseAdapter:
    """
    公共基类：封装网格/插值/尺度换算/PMF-CDF 工具与缓存。
    子类只需：
      - __init__ 中加载各自的表格，并调用 self._init_common(...)
      - 实现 get_* 查询接口 / 采样接口
    """

    def _init_common(
        self,
        A_grid: np.ndarray,
        X1_grid: np.ndarray,
        meta: Dict[str, Any],
        *,
        interp: str = "bilinear",
        A0_run: Optional[float] = None,
        cache_enabled: bool = True,
    ):
        self.A_grid = np.asarray(A_grid, dtype=float)
        self.X1_grid = np.asarray(X1_grid, dtype=float)
        self.meta: Dict[str, Any] = meta if isinstance(meta, dict) else dict(meta)
        self.eps = 1e-16
        
        # --- 小颗粒策略（表格路径专用） ---
        self.small_particle_policy = "fallback"  # 或 "disable"
        self.delta_cells = 0.1                   # 与 live 相同的安全边际
        # 表里保存的目标 NO_FRAG（用于判定能否“生成设定数量的碎片”）
        self.NO_FRAG = int(self.meta.get("NO_FRAG", 4))

        if np.any(self.A_grid <= 0):
            raise ValueError("A_grid must be positive.")
        self.logA_grid = np.log(self.A_grid)

        if interp not in ("nearest", "bilinear"):
            raise ValueError("interp must be 'nearest' or 'bilinear'")
        self.interp = interp

        # LMC 表的单元尺度（表内的 A0）与运行期的 A0（可不同）
        self.A0_tab = float(self.meta.get("A0", 1.0))
        self.A0_run = self.A0_tab if (A0_run is None) else float(A0_run)

        # 缓存（key: round(logA,6), round(X1,6), interp, tag）
        self.cache_enabled: bool = bool(cache_enabled)
        self._cache: Dict[Tuple[float, float, str, int], Any] = {}

    # ------- 缓存工具（让子类统一调用这两个） -------
    def _cache_get(self, key):
        if not self.cache_enabled:
            return None
        return self._cache.get(key)

    def _cache_set(self, key, value):
        if not self.cache_enabled:
            return
        self._cache[key] = value

    def clear_cache(self):
        """主动清空缓存，释放内存。"""
        self._cache.clear()
        
    def set_small_particle_policy(self, policy: str = "fallback", delta_cells: float = 0.1):
        if policy not in ("fallback", "disable"):
            raise ValueError("small_particle_policy must be 'fallback' or 'disable'")
        self.small_particle_policy = policy
        self.delta_cells = float(delta_cells)
        
    def eligible_for_tables(self, A_run: float) -> bool:
        """
        表格驱动路径的“小颗粒可破碎性”判断：
        按 live 的逻辑： floor(A/A0_run - delta) >= NO_FRAG 才认为可一次性产生设定的碎片数。
        当策略为 'fallback' 时，solver 不会调用此判断；当策略为 'disable' 时会启用。
        """
        n_cells = float(A_run) / max(self.A0_run, self.eps)
        NO_FRAG_raw = int(math.floor(max(n_cells - self.delta_cells, 0.0)))
        return (NO_FRAG_raw >= int(self.NO_FRAG))
        
    # ----- 邻域与权重（log A 上双线性, X1 线性）-----
    def _neighbors_weights(self, A: float, X1: float):
        logA = np.log(max(A, self.eps))
        i1 = np.searchsorted(self.logA_grid, logA, side="right")
        i0 = max(0, i1 - 1)
        i1 = min(i1, self.logA_grid.size - 1)

        j1 = np.searchsorted(self.X1_grid, X1, side="right")
        j0 = max(0, j1 - 1)
        j1 = min(j1, self.X1_grid.size - 1)

        if self.interp == "nearest" or (i0 == i1 and j0 == j1):
            return (i0, i0, j0, j0, 1.0, 0.0, 0.0, 0.0)

        xA0, xA1 = self.logA_grid[i0], self.logA_grid[i1]
        tA = 0.0 if (xA1 <= xA0 + 1e-15) else (logA - xA0) / (xA1 - xA0)

        xX0, xX1 = self.X1_grid[j0], self.X1_grid[j1]
        tX = 0.0 if (xX1 <= xX0 + 1e-15) else (X1 - xX0) / (xX1 - xX0)

        w00 = (1.0 - tA) * (1.0 - tX)
        w01 = (1.0 - tA) * tX
        w10 = tA * (1.0 - tX)
        w11 = tA * tX
        return (i0, i1, j0, j1, w00, w01, w10, w11)

    # ----- A 尺度换算：保持“单元数 n=A/A0”一致 -----
    def _A_lookup(self, A_run: float) -> float:
        s = self.A0_tab / max(self.A0_run, self.eps)
        return float(A_run) * s

# ----- CDF/PMF helpers -----
@njit(fastmath=True)
def _pmf_from_cdf_1d(cdf: np.ndarray) -> np.ndarray:
    N = cdf.shape[0]
    p = np.empty_like(cdf)
    p[0] = cdf[0]
    for k in range(1, N):
        p[k] = cdf[k] - cdf[k - 1]
    return p

@njit(fastmath=True)
def _cdf_from_pmf_1d(p: np.ndarray) -> np.ndarray:
    c = np.cumsum(p)
    if c[-1] != 1.0:
        c[-1] = 1.0
    return c

@njit(fastmath=True)
def _pmf2_from_twolevel(rowsum_cdf: np.ndarray, row_cdf: np.ndarray) -> np.ndarray:
    N = rowsum_cdf.shape[0]
    P = np.zeros_like(row_cdf)
    # 行质量
    r = np.empty(N, dtype=np.float64)
    r[0] = rowsum_cdf[0]
    for i in range(1, N):
        r[i] = rowsum_cdf[i] - rowsum_cdf[i - 1]
    r = np.maximum(r, 0.0)
    # 每行列 pmf
    for i in range(N):
        p_row = np.empty(N, dtype=np.float64)
        p_row[0] = row_cdf[i, 0]
        for j in range(1, N):
            p_row[j] = row_cdf[i, j] - row_cdf[i, j - 1]
        p_row = np.maximum(p_row, 0.0)
        s = p_row.sum()
        if s <= 0.0:
            p_row[:] = 1.0 / N
        else:
            p_row /= s
        P[i, :] = r[i] * p_row
    # 归一
    S = P.sum()
    if S > 0.0:
        P /= S
    return P

@njit(fastmath=True)
def _twolevel_from_pmf2(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = P.shape[0]
    rowsum = P.sum(axis=1)
    tot = rowsum.sum()
    if tot <= 0.0:
        rowsum[:] = 1.0 / N
        tot = 1.0
    rowsum /= tot
    rowsum_cdf = np.cumsum(rowsum)
    row_cdf = np.zeros_like(P)
    for i in range(N):
        s = P[i, :].sum()
        if s <= 0.0:
            row_cdf[i, :] = np.cumsum(np.full(N, 1.0 / N))
        else:
            row_cdf[i, :] = np.cumsum(P[i, :] / s)
    rowsum_cdf[-1] = 1.0
    row_cdf[:, -1] = 1.0
    return rowsum_cdf, row_cdf


# ==============================
# Marginal (旧) 表的适配器
# ==============================
class LMCTableAdapter(LMCBaseAdapter):
    """
    读取 preprocess_lmc_to_tables.py 生成的边际表（.npz），
    提供按 (A, X1) 查询的 1D/2D 两级 CDF（边际）。
    """

    def __init__(self, npz_path: str, interp: str = "bilinear", A0_run: float = None,
                 cache_enabled: bool = True):
        d = np.load(npz_path, allow_pickle=True)
        A_grid = np.asarray(d["A_grid"], dtype=float)
        X1_grid = np.asarray(d["X1_grid"], dtype=float)
        meta = d["meta"].item() if isinstance(d["meta"], np.ndarray) else dict(d["meta"])
        super()._init_common(A_grid, X1_grid, meta, interp=interp, A0_run=A0_run, cache_enabled=cache_enabled)

        self.rel1d = np.asarray(d["rel1d"], dtype=float)          # (N,)
        self.cdf1d_grid = d["cdf1d_grid"]                          # object[nA,nX] of float64[N]
        self.zmin1d_grid = np.asarray(d["zmin1d_grid"], dtype=float)

        self.rel1 = np.asarray(d["rel1"], dtype=float)            # (N,)
        self.rel3 = np.asarray(d["rel3"], dtype=float)            # (N,)
        self.rowsum_cdf_grid = d["rowsum_cdf_grid"]               # object[nA,nX]
        self.row_cdf_grid = d["row_cdf_grid"]                     # object[nA,nX]
        self.zmin1_grid = np.asarray(d["zmin1_grid"], dtype=float)
        self.zmin3_grid = np.asarray(d["zmin3_grid"], dtype=float)

        self.p_expected_grid = np.asarray(d["p_expected_grid"], dtype=float)
        self.n_obs_grid = np.asarray(d["n_obs_grid"], dtype=np.int64)

        self.N = int(self.rel1.shape[0])

    # ------ Public queries ------
    def get_1d(self, A: float, X1: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        A_lookup = self._A_lookup(A)
        key = (round(float(np.log(max(A_lookup, self.eps))), 6), round(float(X1), 6), self.interp, 1)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)

        def pmf_at(i, j):
            c = np.asarray(self.cdf1d_grid[i, j], dtype=np.float64)
            return _pmf_from_cdf_1d(c)

        p = w00 * pmf_at(i0, j0) + w01 * pmf_at(i0, j1) + w10 * pmf_at(i1, j0) + w11 * pmf_at(i1, j1)
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s > 0.0:
            p /= s
        cdf = _cdf_from_pmf_1d(p)

        pe = w00 * self.p_expected_grid[i0, j0] + w01 * self.p_expected_grid[i0, j1] + w10 * self.p_expected_grid[i1, j0] + w11 * self.p_expected_grid[i1, j1]
        zmin1d = float(np.clip(self.A0_run / max(A, 1e-20), 0.0, 1.0))

        out = (self.rel1d, cdf, zmin1d, float(pe))
        self._cache_set(key, out)
        return out

    def get_2d(self, A: float, X1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        A_lookup = self._A_lookup(A)
        key = (round(float(np.log(max(A_lookup, self.eps))), 6), round(float(X1), 6), self.interp, 2)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)

        def P_at(i, j):
            rowsum_cdf = np.asarray(self.rowsum_cdf_grid[i, j], dtype=np.float64)
            row_cdf = np.asarray(self.row_cdf_grid[i, j], dtype=np.float64)
            return _pmf2_from_twolevel(rowsum_cdf, row_cdf)

        P = w00 * P_at(i0, j0) + w01 * P_at(i0, j1) + w10 * P_at(i1, j0) + w11 * P_at(i1, j1)
        P = np.maximum(P, 0.0)
        S = P.sum()
        if S > 0.0:
            P /= S
        rowsum_cdf, row_cdf = _twolevel_from_pmf2(P)

        pe = w00 * self.p_expected_grid[i0, j0] + w01 * self.p_expected_grid[i0, j1] + w10 * self.p_expected_grid[i1, j0] + w11 * self.p_expected_grid[i1, j1]
        denom1 = A * X1
        denom3 = A * (1.0 - X1)
        zmin1 = float(np.clip(self.A0_run / max(denom1, 1e-20), 0.0, 1.0)) if denom1 > self.eps else 0.0
        zmin3 = float(np.clip(self.A0_run / max(denom3, 1e-20), 0.0, 1.0)) if denom3 > self.eps else 0.0

        out = (self.rel1, self.rel3, rowsum_cdf, row_cdf, zmin1, zmin3, float(pe))
        self._cache_set(key, out)
        return out


# ==============================
# Rank-wise (Top-K + Tail) 表的适配器
# ==============================
class LMCRankAdapter(LMCBaseAdapter):
    """
    读取 preprocess_lmc_to_mcpbe_joint.py 生成的 rank-wise 表（.npz）：
      - 每个 (A,X1) 保存 Top-K 的二维两级 CDF（前 K 个秩次）
      - 保存 P(N) 直方图（pn_hist），尾部总量 CDF（tail_T_cdf），以及（可选）尾部两相配比 Beta(α,β)

    提供：
      - get_rank_tables(A,X1) -> (rel1, rel3, rowsum_cdf_list[K], row_cdf_list[K], zmin1, zmin3, EN, tail_T_cdf, tail_mode, tail_alpha, tail_beta)
      - sample_one_shot(A,X1,rng,N=None,K_use=None) -> (rA_list, rB_list)
        （一次性采样 N 个碎片的相对体积分布；最后一块为余量确保守恒；K_use 缺省为 min(K, N-1)）
    """

    def __init__(self, npz_path: str, interp: str = "bilinear", A0_run: float = None, 
                 cache_enabled: bool = True):
        d = np.load(npz_path, allow_pickle=True)
        A_grid = np.asarray(d["A_grid"], dtype=float)
        X1_grid = np.asarray(d["X1_grid"], dtype=float)
        meta = d["meta"].item() if isinstance(d["meta"], np.ndarray) else dict(d["meta"])
        super()._init_common(A_grid, X1_grid, meta, interp=interp, A0_run=A0_run, cache_enabled=cache_enabled,)

        self.rank_K = int(d["rank_K"])
        self.N = int(d["N_bins"])
        self.rel1 = np.asarray(d["rel1"], dtype=float)
        self.rel3 = np.asarray(d["rel3"], dtype=float)
        self.rel1d = np.asarray(d["rel1d"], dtype=float)
        self.zmin1d_grid = np.asarray(d["zmin1d_grid"], dtype=float)

        # 每个 rank 的对象网格
        self.rowsum_cdf_rank = []
        self.row_cdf_rank = []
        self.cdf1d_rank = []
        for r in range(self.rank_K):
            self.rowsum_cdf_rank.append(d[f"rowsum_cdf_rank{r+1}_grid"])
            self.row_cdf_rank.append(d[f"row_cdf_rank{r+1}_grid"])
            self.cdf1d_rank.append(d[f"cdf1d_rank{r+1}_grid"])

        self.pn_hist_grid = d["pn_hist_grid"]  # object[nA,nX] of int64[NO_FRAG+1]
        self.n_obs_ranks_grid = d["n_obs_ranks_grid"]  # object[nA,nX] of int64[K]

        self.tail_T_cdf_grid = d["tail_T_cdf_grid"]  # object[nA,nX] of float64[N]
        self.tail_mode = str(d["tail_mode"]) if "tail_mode" in d else "equal"
        self.tail_alpha_grid = d["tail_alpha_grid"] if "tail_alpha_grid" in d else None
        self.tail_beta_grid = d["tail_beta_grid"] if "tail_beta_grid" in d else None
        
        self.freq_adj_grid = d["freq_adj_grid"]
        self.M2_true_grid  = d["M2_true_grid"]
    # ------- 内部：插值 rowsum/rowcdf（对每个 rank） -------
    def _interp_rank_tables(self, A: float, X1: float):
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A, X1)

        rowsum_list: List[np.ndarray] = []
        rowcdf_list: List[np.ndarray] = []

        for r in range(self.rank_K):
            def tab(i, j):
                rs = np.asarray(self.rowsum_cdf_rank[r][i, j], dtype=np.float64)
                rc = np.asarray(self.row_cdf_rank[r][i, j], dtype=np.float64)
                return rs, rc

            rs00, rc00 = tab(i0, j0)
            rs01, rc01 = tab(i0, j1)
            rs10, rc10 = tab(i1, j0)
            rs11, rc11 = tab(i1, j1)

            # 直接在 CDF 上凸组合（工程上稳定）
            rs = w00 * rs00 + w01 * rs01 + w10 * rs10 + w11 * rs11
            rc = w00 * rc00 + w01 * rc01 + w10 * rc10 + w11 * rc11

            rs = np.clip(rs, 0.0, 1.0)
            rs[-1] = 1.0
            rc = np.clip(rc, 0.0, 1.0)
            rc[:, -1] = 1.0

            rowsum_list.append(rs)
            rowcdf_list.append(rc)

        return rowsum_list, rowcdf_list

    def _interp_EN(self, A: float, X1: float) -> float:
        # 用 pn_hist 插值 EN
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A, X1)

        def EN_at(i, j):
            hist = np.asarray(self.pn_hist_grid[i, j], dtype=np.int64)
            tot = float(hist.sum()) if hist.size > 0 else 0.0
            if tot <= 0.0:
                return 2.0
            idx = np.arange(hist.size, dtype=float)
            return float((idx * hist).sum() / tot)

        return (
            w00 * EN_at(i0, j0) + w01 * EN_at(i0, j1)
            + w10 * EN_at(i1, j0) + w11 * EN_at(i1, j1)
        )

    def _interp_tail(self, A: float, X1: float):
        # 尾部总量 CDF + （可选）Beta 参数
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A, X1)

        def cdf_at(i, j):
            return np.asarray(self.tail_T_cdf_grid[i, j], dtype=np.float64)

        c00 = cdf_at(i0, j0)
        c01 = cdf_at(i0, j1)
        c10 = cdf_at(i1, j0)
        c11 = cdf_at(i1, j1)
        cdf = w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf[-1] = 1.0

        alpha = beta = None
        if self.tail_mode == "beta" and (self.tail_alpha_grid is not None) and (self.tail_beta_grid is not None):
            a00 = float(self.tail_alpha_grid[i0, j0])
            a01 = float(self.tail_alpha_grid[i0, j1])
            a10 = float(self.tail_alpha_grid[i1, j0])
            a11 = float(self.tail_alpha_grid[i1, j1])
            b00 = float(self.tail_beta_grid[i0, j0])
            b01 = float(self.tail_beta_grid[i0, j1])
            b10 = float(self.tail_beta_grid[i1, j0])
            b11 = float(self.tail_beta_grid[i1, j1])
            alpha = w00 * a00 + w01 * a01 + w10 * a10 + w11 * a11
            beta  = w00 * b00 + w01 * b01 + w10 * b10 + w11 * b11

        return cdf, self.tail_mode, alpha, beta

    def _interp_freq_adj(self, A_lookup: float, X1: float) -> np.ndarray:
        """
        双线性插值 freq_adj（长度 K，元素非负，和为 1）。
        如当前表无 freq_adj_grid，则回退为均匀权重。
        """
        K = int(self.rank_K)
        if (self.freq_adj_grid is None) or (K <= 0):
            # 旧表兼容：均匀
            return np.full(K, 1.0 / max(K, 1), dtype=np.float64)
    
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)
    
        def fa(i, j):
            v = np.asarray(self.freq_adj_grid[i, j], dtype=np.float64)
            if v.size != K:
                # 形状异常时兜底
                return np.full(K, 1.0 / max(K, 1), dtype=np.float64)
            v = np.maximum(v, 0.0)
            s = float(v.sum())
            return (v / s) if s > 0.0 else np.full(K, 1.0 / max(K, 1), dtype=np.float64)
    
        f00 = fa(i0, j0)
        f01 = fa(i0, j1)
        f10 = fa(i1, j0)
        f11 = fa(i1, j1)
    
        f = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
        f = np.maximum(f, 0.0)
        s = float(f.sum())
        if s > 0.0:
            f /= s
        else:
            f[:] = 1.0 / max(K, 1)
        return f

    # ------- Public: rank-wise 表查询 -------
    def get_rank_tables(self, A: float, X1: float):
        """
        返回：
          rel1, rel3,
          rowsum_cdf_list[K], row_cdf_list[K],
          zmin1, zmin3,
          EN （期望碎片数）,
          tail_T_cdf, tail_mode, tail_alpha, tail_beta
        """
        # 先做 cache（按 A_lookup + X1 + interp）
        A_lookup = self._A_lookup(A)
        key_base = (round(float(np.log(max(A_lookup, self.eps))), 6), round(float(X1), 6), self.interp)

        # 判定是否单质（用体积分母更稳）
        denom1 = A * X1
        denom3 = A * (1.0 - X1)
        pure_A = (X1 > 1 - self.eps)   # X1≈1
        pure_B = (X1 < self.eps) # X1≈0

        if pure_A or pure_B:
            # 尝试命中“单质伪2D”的缓存
            tag = 201 if pure_A else 202
            key = key_base + (tag,)
            hit = self._cache_get(key)
            if hit is not None:
                return hit

            # 插值 rank-wise 1D：对每个 rank 得到 1D CDF（长度 N）
            i0,i1,j0,j1,w00,w01,w10,w11 = self._neighbors_weights(A_lookup, X1)
            N = self.N
            cdf1d_list = []
            for r in range(self.rank_K):
                def cdf_at(i, j):
                    return np.asarray(self.cdf1d_rank[r][i, j], dtype=np.float64)
                c00 = cdf_at(i0, j0)
                c01 = cdf_at(i0, j1)
                c10 = cdf_at(i1, j0)
                c11 = cdf_at(i1, j1)
                cdf = w00*c00 + w01*c01 + w10*c10 + w11*c11
                cdf = np.clip(cdf, 0.0, 1.0)
                cdf[-1] = 1.0
                cdf1d_list.append(cdf)

            # 伪 2D 的 rel1/rel3 与 rowsum/rowcdf
            if pure_A:
                # 行方向承载 1D（rA=z_tot），列方向固定 j=0（rB=0）
                rel1 = self.rel1d.copy()
                rel3 = np.zeros_like(self.rel1d)
                rowsum_list = [cdf.copy() for cdf in cdf1d_list]            # shape (N,)
                rowcdf_list = [np.ones((N, N), dtype=float) for _ in range(self.rank_K)]  # 每行全 1 → j=0
                zmin1 = float(np.clip(self.A0_run / max(denom1, 1e-20), 0.0, 1.0)) if denom1 > self.eps else 0.0
                zmin3 = 0.0
            else:
                # 列方向承载 1D（rB=z_tot），行方向固定 i=0（rA=0）
                rel1 = np.zeros_like(self.rel1d)
                rel3 = self.rel1d.copy()
                rowsum_list = [np.ones(N, dtype=float) for _ in range(self.rank_K)]      # 全 1 → i=0
                rowcdf_list = []
                for cdf in cdf1d_list:
                    rc = np.tile(cdf, (N, 1))  # N 行相同的CDF
                    rowcdf_list.append(rc)
                zmin1 = 0.0
                zmin3 = float(np.clip(self.A0_run / max(denom3, 1e-20), 0.0, 1.0)) if denom3 > self.eps else 0.0

            # EN & 尾部依旧从全表插值（不依赖维度，含义是“期望碎片数/尾部总量占比”）
            EN = self._interp_EN(A, X1)
            tail_T_cdf, tail_mode, tail_alpha, tail_beta = self._interp_tail(A_lookup, X1)

            out = (rel1, rel3, rowsum_list, rowcdf_list, zmin1, zmin3, float(EN), tail_T_cdf, tail_mode, tail_alpha, tail_beta)
            self._cache_set(key, out)
            return out

        # —— 混合体系：按原 2D 联合表插值 ——
        key = key_base + (200,)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        rowsum_list, rowcdf_list = self._interp_rank_tables(A_lookup, X1)
        EN = self._interp_EN(A, X1)
        tail_T_cdf, tail_mode, tail_alpha, tail_beta = self._interp_tail(A_lookup, X1)

        zmin1 = float(np.clip(self.A0_run / max(denom1, 1e-20), 0.0, 1.0)) if denom1 > self.eps else 0.0
        zmin3 = float(np.clip(self.A0_run / max(denom3, 1e-20), 0.0, 1.0)) if denom3 > self.eps else 0.0

        out = (self.rel1, self.rel3, rowsum_list, rowcdf_list, zmin1, zmin3, float(EN), tail_T_cdf, tail_mode, tail_alpha, tail_beta)
        self._cache_set(key, out)
        return out

    # ------- Optional: 一次性采样（Top-K + 尾部） -------
    def sample_one_shot(
        self,
        A: float,
        X1: float,
        rng: np.random.Generator,
        N: Optional[int] = None,
        K_use: Optional[int] = None,
        tail_strategy: str = "equal"  # 'equal' | 'beta'
    ) -> Tuple[List[float], List[float]]:
        """
        生成 N 个碎片的两相相对体积分布 rA/rB（长度 N），最后一块为余量，严格守恒。
        - 若 N is None：使用 self.NO_FRAG（与在线 LMC 语义一致，碎片数为定值）
        - K_use 默认 min(K, N-1)，仅显式采样这 K_use 个 rank；其余进入尾部 + 余量
        - 接入 freq_adj：当 K_use < N-1 时，用插值得到的 freq_adj 选择“更重要”的 rank
          进行显式采样（按 freq_adj 从大到小选 K_use 个秩次）。
        """
        rel1, rel3, rowsum_list, rowcdf_list, zmin1, zmin3, EN, tail_T_cdf, tmode, a_tail, b_tail = self.get_rank_tables(A, X1)
        Nbins = self.N
    
        # 与在线 LMC 对齐：碎片数为定值（如未指定 N）
        if N is None:
            N = max(2, int(self.NO_FRAG))
    
        pick = max(0, N - 1)  # 最后一块为余量
        K_use = min(self.rank_K, pick) if (K_use is None) else min(int(K_use), pick)
    
        # ==== 接入 freq_adj：当 K_use < pick 时，挑选要显式采样的 rank ====
        # 在表坐标里插值 freq_adj
        A_lookup = self._A_lookup(A)
        freq = self._interp_freq_adj(A_lookup, X1)  # 长度 K，和为1
        # 选择前 K_use 大的秩次（保持秩次从小到大顺序便于阅读/复现）
        selected = np.argsort(-freq)[:K_use]
        selected = np.sort(selected)  # 可选：避免输出顺序随权重波动
    
        rA: List[float] = []
        rB: List[float] = []
    
        # ==== 1. 显式采样所选的 K_use 个 rank ====
        for ridx in selected.tolist():
            rs = rowsum_list[ridx]
            rc = rowcdf_list[ridx]
            u1 = float(rng.random())
            i = int(np.searchsorted(rs, u1, side="right"))
            i = min(i, Nbins - 1)
            row = rc[i]
            u2 = float(rng.random())
            j = int(np.searchsorted(row, u2, side="right"))
            j = min(j, Nbins - 1)
    
            # 映射到相对比例
            rAi = float(rel1[i])
            rBj = float(rel3[j])
    
            # --- 最小体积阈值保护（局部） ---
            vt_rel = rAi * X1 + rBj * (1.0 - X1)
            vt_rel_min = self.A0_run / max(A, 1e-20)
            if vt_rel < vt_rel_min:
                denom1 = A * X1
                denom3 = A * (1.0 - X1)
                if denom1 > self.eps and rAi <= 0.0:
                    rAi = max(rAi, zmin1)
                elif denom3 > self.eps and rBj <= 0.0:
                    rBj = max(rBj, zmin3)
                else:
                    need = vt_rel_min - vt_rel
                    if need > 0.0:
                        if denom1 > self.eps and (X1 >= 0.5 or denom3 <= self.eps):
                            rAi += need / max(X1, 1e-20)
                        elif denom3 > self.eps:
                            rBj += need / max(1.0 - X1, 1e-20)
    
            rA.append(rAi)
            rB.append(rBj)
    
        # ==== 2. 采样尾部（其余 pick-K_use 个） ====
        tail_count = pick - K_use
        if tail_count > 0:
            uT = float(rng.random())
            kT = int(np.searchsorted(tail_T_cdf, uT, side="right"))
            kT = min(kT, Nbins - 1)
            T = (kT + 0.5) / Nbins
            T = max(0.0, min(1.0, T))
    
            if tail_strategy == "beta" and tmode == "beta" and (a_tail is not None) and (b_tail is not None) and a_tail > 0 and b_tail > 0:
                pA = float(rng.beta(a_tail, b_tail))
            else:
                pA = 0.5
    
            tiny = T / max(1, tail_count)
            for _ in range(tail_count):
                rA.append(pA * tiny)
                rB.append((1.0 - pA) * tiny)
    
        # ==== 3. 守恒与归一化 ====
        sumA = float(np.sum(rA))
        sumB = float(np.sum(rB))
        if (sumA > 1.0) or (sumB > 1.0):
            gA = 1.0 / max(sumA, 1e-20)
            gB = 1.0 / max(sumB, 1e-20)
            g = min(gA, gB)
            for t in range(len(rA)):
                rA[t] *= g
                rB[t] *= g
    
        # ==== 4. 计算余量（严格守恒） ====
        sA = float(np.sum(rA))
        sB = float(np.sum(rB))
        rA_last = max(0.0, 1.0 - sA)
        rB_last = max(0.0, 1.0 - sB)
        rA.append(rA_last)
        rB.append(rB_last)
    
        # ==== 5. clip 清理 ====
        rA = [max(0.0, min(1.0, x)) for x in rA]
        rB = [max(0.0, min(1.0, y)) for y in rB]
    
        # ==== 6. 最小体积兜底（全局） ====
        vt_rel_min = self.A0_run / max(A, 1e-20)
        X3 = 1.0 - X1
        vt_rel = np.array(rA) * X1 + np.array(rB) * X3
        too_small = vt_rel < vt_rel_min
        if np.any(too_small):
            for idx in np.where(too_small)[0]:
                need = vt_rel_min - vt_rel[idx]
                if X1 >= 0.5:
                    rA[idx] += need / max(X1, 1e-20)
                else:
                    rB[idx] += need / max(X3, 1e-20)
            # 再次归一化守恒
            sA = float(np.sum(rA))
            sB = float(np.sum(rB))
            gA = 1.0 / max(sA, 1e-20)
            gB = 1.0 / max(sB, 1e-20)
            g = min(gA, gB)
            for t in range(len(rA)):
                rA[t] *= g
                rB[t] *= g
    
        # ==== 7. 相内严格守恒校正（保持 sum(rA)=1、sum(rB)=1） ====
        sumA_final = float(np.sum(rA))
        sumB_final = float(np.sum(rB))
        scaleA = (1.0 / sumA_final) if sumA_final > 0 else 0.0
        scaleB = (1.0 / sumB_final) if sumB_final > 0 else 0.0
        for t in range(len(rA)):
            rA[t] *= scaleA
            rB[t] *= scaleB
    
        return rA, rB


# 小粒子策略的信号（由 solver 捕获）
class LMCLiveFallback(Exception):
    """当前母体可用单元数不足，不适合在线 LMC；请回退到 rank/tables。"""
    pass

class LMCLiveDisable(Exception):
    """当前母体可用单元数不足，按策略将其标记为不可破碎。"""
    pass


class LMCLiveAdapter:
    """
    一次采样一套碎片的在线 LMC 适配器：
      - 动态收敛 NO_FRAG：NO_FRAG_eff = min(NO_FRAG, floor(A/A0_run - delta)), 不低于2
      - 若 A/A0_run < 2：
          * small_particle_policy='fallback' -> 抛 LMCLiveFallback，让 solver 回退
          * small_particle_policy='disable'  -> 抛 LMCLiveDisable，让 solver 将该粒子破碎率置零
    """

    def __init__(self) -> None:
        # LMC 参数（默认可改）
        self.STR = np.array([1.0, 1.0, 1.0], dtype=float)
        self.NO_FRAG = 4
        self.gamma = 1.0
        self.allow_loops = False
        self.accept_all_cracks = False
        self.use_weighted_start = True
        self.aspect_ratio = 1.0
        self.int_bre = 0.0
        self.A0_run = 1.0  # 网格单元面积（把体积数值当面积用）

        # 小颗粒处理策略与安全边际
        self.small_particle_policy = "fallback"  # or "disable"
        self.delta_cells = 0.1  # A/A0_run 的安全边际

        self._sim: Optional[LMCSimulator] = None

    def configure_simulator(
        self,
        *,
        STR: Optional[np.ndarray] = None,
        NO_FRAG: Optional[int] = None,
        gamma: Optional[float] = None,
        allow_loops: Optional[bool] = None,
        accept_all_cracks: Optional[bool] = None,
        use_weighted_start: Optional[bool] = None,
        aspect_ratio: Optional[float] = None,
        int_bre: Optional[float] = None,
        A0_run: Optional[float] = None,
        small_particle_policy: Optional[str] = None,  # 'fallback' | 'disable'
        delta_cells: Optional[float] = None,
        rebuild: bool = True,
    ) -> None:
        if STR is not None: self.STR = np.asarray(STR, dtype=float)
        if NO_FRAG is not None: self.NO_FRAG = int(NO_FRAG)
        if gamma is not None: self.gamma = float(gamma)
        if allow_loops is not None: self.allow_loops = bool(allow_loops)
        if accept_all_cracks is not None: self.accept_all_cracks = bool(accept_all_cracks)
        if use_weighted_start is not None: self.use_weighted_start = bool(use_weighted_start)
        if aspect_ratio is not None: self.aspect_ratio = float(aspect_ratio)
        if int_bre is not None: self.int_bre = float(int_bre)
        if A0_run is not None: self.A0_run = float(A0_run)
        if small_particle_policy is not None:
            if small_particle_policy not in ("fallback", "disable"):
                raise ValueError("small_particle_policy must be 'fallback' or 'disable'")
            self.small_particle_policy = small_particle_policy
        if delta_cells is not None:
            self.delta_cells = float(delta_cells)

        if rebuild or (self._sim is None):
            self._sim = LMCSimulator(
                STR=self.STR,
                NO_FRAG=self.NO_FRAG,
                gamma=self.gamma,
                allow_loops=self.allow_loops,
                accept_all_cracks=self.accept_all_cracks,
                use_weighted_start=self.use_weighted_start,
                plotter=None,
            )

    def _AX1_from_Vparent(self, V_parent: np.ndarray) -> Tuple[float, float]:
        if V_parent.size == 1:
            A = float(V_parent[0])
            X1 = 1.0
        else:
            VA, VB = float(V_parent[0]), float(V_parent[1])
            A = VA + VB
            X1 = (VA / A) if A > 0 else 0.5
        return A, X1

    def sample_one_shot(
        self,
        V_parent: np.ndarray,
        rng: np.random.Generator,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], float]:
        if self._sim is None:
            self.configure_simulator(rebuild=True)

        V_parent = np.asarray(V_parent, dtype=float)
        if V_parent.ndim != 1 or V_parent.size not in (1, 2):
            raise ValueError("V_parent must be 1D array of length 1 or 2.")

        A, X1 = self._AX1_from_Vparent(V_parent)
        
        if A <= 0.0:
            return [V_parent.copy()], 0.0

        # 动态 NO_FRAG 收敛
        n_cells = A / self.A0_run
        NO_FRAG_raw = int(np.floor(max(n_cells - self.delta_cells, 0.0)))
        
        # 先看是否不足 2（用于 fallback/disable 策略）
        if NO_FRAG_raw < self.NO_FRAG:
        # if NO_FRAG_raw < 2:
            if self.small_particle_policy == "fallback":
                raise LMCLiveFallback()
            else:
                raise LMCLiveDisable()
        
        NO_FRAG_eff = min(self.NO_FRAG, NO_FRAG_raw)

        # 临时覆写 NO_FRAG
        old_nf = self._sim.NO_FRAG
        self._sim.NO_FRAG = int(NO_FRAG_eff)

        seed_use = int(seed) if (seed is not None) else int(rng.integers(0, 2**31 - 1))
        F = self._sim.mc_breakage_repeat(
            A=A, X1=X1, X2=(1.0 - X1),
            N_GRIDS=1, N_FRACS=1,
            A0=self.A0_run,
            aspect_ratio=self.aspect_ratio,
            int_bre=self.int_bre,
            seed=seed_use,
            plot_each=False,
        )

        # 还原 NO_FRAG
        self._sim.NO_FRAG = old_nf

        if F.size == 0:
            return [V_parent.copy()], 0.0

        VT = F[:, 0]
        valid = VT > 0.0
        energy = float(F[0, 3])

        frags: List[np.ndarray] = []
        if V_parent.size == 1:
            parts = VT[valid].astype(float)
            s = float(np.sum(parts))
            if s <= 0.0:
                return [V_parent.copy()], energy
            parts *= (V_parent[0] / s)
            for p in parts:
                frags.append(np.array([float(p)], dtype=float))
            # 最终逐相对齐（保险）
            # sumV = float(np.sum([f[0] for f in frags]))
            # if sumV > 0:
            #     g = V_parent[0] / sumV
            #     for f in frags:
            #         f[0] *= g
            return frags, energy

        else:
            VA_arr = F[valid, 1].astype(float)
            VB_arr = F[valid, 2].astype(float)
            if VA_arr.size == 0:
                return [V_parent.copy()], energy

            # # 可行域缩放 + 逐相精确拉回
            # sumA = float(np.sum(VA_arr))
            # sumB = float(np.sum(VB_arr))
            # if sumA > 0: VA_arr *= (V_parent[0] / sumA) * 0.999999
            # if sumB > 0: VB_arr *= (V_parent[1] / sumB) * 0.999999

            # sumA = float(np.sum(VA_arr))
            # sumB = float(np.sum(VB_arr))
            # if sumA > 0: VA_arr *= (V_parent[0] / sumA)
            # if sumB > 0: VB_arr *= (V_parent[1] / sumB)

            for a, b in zip(VA_arr, VB_arr):
                frags.append(np.array([float(a), float(b)], dtype=float))

            # # 保险再对齐一次
            # gotA = float(np.sum([f[0] for f in frags]))
            # gotB = float(np.sum([f[1] for f in frags]))
            # if gotA > 0:
            #     sA = V_parent[0] / gotA
            #     for f in frags: f[0] *= sA
            # if gotB > 0:
            #     sB = V_parent[1] / gotB
            #     for f in frags: f[1] *= sB

            return frags, energy


# ==============================
# Copula-based (Top-K stick-breaking) 适配器
# ==============================
try:
    import pyvinecopulib as pv
    _PV_OK = True
except Exception:
    _PV_OK = False

try:
    from scipy.stats import beta as sp_beta
    from scipy.stats import norm as sp_norm
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

_C_EPS = 1e-12
_C_CLIP = 1e-6


def _c_clip01(x: np.ndarray, eps: float = _C_CLIP) -> np.ndarray:
    return np.clip(x, eps, 1.0 - eps)

class _C_ECDFMarginal:
    def __init__(self, xs: np.ndarray, ps: np.ndarray):
        self.xs = np.asarray(xs, dtype=float)
        self.ps = np.asarray(ps, dtype=float)
        if self.xs.ndim != 1 or self.ps.ndim != 1:
            raise ValueError("ECDF marginal expects 1D xs, ps.")
        if self.xs.size != self.ps.size:
            raise ValueError("ECDF marginal: xs and ps must have same length.")
        # 保底
        if self.ps[0] > 0.0 or self.ps[-1] < 1.0:
            # 可选：扩一下头尾
            pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.xs, self.ps, left=self.ps[0], right=self.ps[-1])

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, self.ps[0], self.ps[-1])
        return np.interp(u, self.ps, self.xs)

class _C_BetaMarginal:
    """和训练脚本里一致的边缘，用来做 inverse-CDF。"""
    def __init__(self, alpha: float, beta: float):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if _SCIPY_OK:
            return sp_beta.cdf(x, self.alpha, self.beta, loc=0.0, scale=1.0)
        # fallback
        xs = np.linspace(0.0, 1.0, 2049)
        pdf = self.pdf(xs)
        c = np.cumsum(pdf)
        c /= c[-1]
        idx = np.searchsorted(xs, np.clip(x, 0.0, 1.0), side="right")
        idx = np.clip(idx, 1, xs.size - 1)
        w = (x - xs[idx-1]) / (xs[idx] - xs[idx-1] + _C_EPS)
        return c[idx-1] * (1 - w) + c[idx] * w

    def ppf(self, u: np.ndarray) -> np.ndarray:
        if _SCIPY_OK:
            return sp_beta.ppf(_c_clip01(u), self.alpha, self.beta, loc=0.0, scale=1.0)
        # bisection
        u = _c_clip01(u)
        lo = np.zeros_like(u); hi = np.ones_like(u)
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            cm = self.cdf(mid)
            lo = np.where(cm < u, mid, lo)
            hi = np.where(cm >= u, mid, hi)
        return 0.5 * (lo + hi)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if _SCIPY_OK:
            return sp_beta.pdf(x, self.alpha, self.beta, loc=0.0, scale=1.0)
        import math
        from math import lgamma
        a, b = self.alpha, self.beta
        x = np.clip(x, 0.0, 1.0)
        B = math.exp(lgamma(a) + lgamma(b) - lgamma(a + b))
        return np.where((x > 0) & (x < 1), x**(a-1) * (1 - x)**(b-1) / (B + _C_EPS), 0.0)


class _C_GaussianCopula:
    def __init__(self, R: np.ndarray):
        R = np.asarray(R, dtype=float)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-6, None)
        self.R = (V * w) @ V.T
        self.L = np.linalg.cholesky(self.R)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        Z = rng.standard_normal(size=(int(n), self.R.shape[0])) @ self.L.T
        if _SCIPY_OK:
            U = sp_norm.cdf(Z)
        else:
            U = 0.5 * (1.0 + np.erf(Z / np.sqrt(2.0)))
        return _c_clip01(U, 1e-12)


class _C_VineCopula:
    def __init__(self, json_blob: str):
        if not _PV_OK:
            raise RuntimeError("pyvinecopulib not available.")
        self.model = pv.Vinecop.from_json(json_blob)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        seed = int(rng.integers(0, 2**31 - 1))
        return self.model.simulate(int(n), seeds=[seed])

class LMCCopulaAdapter(LMCBaseAdapter):
    """
    读取新版 copula 预处理脚本生成的 lmc_copula_grid.npz，并根据
    - 纯相模型 (is_pure=True): 只对 Y_1..Y_K 做 K 维 copula，采样 Y→stick-breaking→碎片
    - 混相模型 (is_pure=False): 对 Y_1..Y_K 做 K 维 copula，pA_1..pA_K 不进 copula，只各自用 Beta 边缘抽样
    采样出一次破碎的 (rA, rB)，接口与 LMCRankAdapter.sample_one_shot 对齐。
    """

    def __init__(self, npz_path: str, interp: str = "bilinear",
                 A0_run: float = None, cache_enabled: bool = True):
        d = np.load(npz_path, allow_pickle=True)
        A_grid = np.asarray(d["A_grid"], dtype=float)
        X1_grid = np.asarray(d["X1_grid"], dtype=float)
        meta = d["meta"].item() if isinstance(d["meta"], np.ndarray) else dict(d["meta"])
        super()._init_common(
            A_grid, X1_grid, meta,
            interp=interp, A0_run=A0_run, cache_enabled=cache_enabled
        )

        # object[nA, nX] of dict or None
        self.models = d["models"]
        # 全局 K 只是默认值，实际每个 cell 里也会存 K
        self.K = int(self.meta.get("stick_breaking_K", 4))

    # ---------- helpers ----------
    @staticmethod
    def _inv_stick_breaking(Y: np.ndarray) -> Tuple[np.ndarray, float]:
        K = Y.size
        z = np.zeros(K, dtype=float)
        remain = 1.0
        for k in range(K):
            yk = float(np.clip(Y[k], 0.0, 1.0))
            z[k] = yk * remain
            remain = max(0.0, remain - z[k])
        return z, float(remain)

    def _pick_cell_model(self, A_lookup: float, X1: float,
                         want_pure: bool,
                         rng: np.random.Generator):
        """
        按四邻权重挑一个模型，优先挑类型一致（纯/混）的；
        若没有同类型，则退而求其次。
        返回 (i, j, w, model_obj) 或 None
        """
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)
        cand = [
            (i0, j0, w00),
            (i0, j1, w01),
            (i1, j0, w10),
            (i1, j1, w11),
        ]

        same_type: List[Tuple[int, int, float, Dict[str, Any]]] = []
        other_type: List[Tuple[int, int, float, Dict[str, Any]]] = []

        for (i, j, w) in cand:
            obj = self.models[i, j]
            if obj is None:
                continue
            is_pure_cell = bool(obj.get("is_pure", False))
            if is_pure_cell == want_pure:
                same_type.append((i, j, w, obj))
            else:
                other_type.append((i, j, w, obj))

        def _pick(lst):
            ws = np.array([x[2] for x in lst], dtype=float)
            ws = np.maximum(ws, 0.0)
            if ws.sum() <= 0.0:
                ws[:] = 1.0
            ws /= ws.sum()
            idx = int(rng.choice(len(lst), p=ws))
            return lst[idx]

        if same_type:
            return _pick(same_type)
        if other_type:
            return _pick(other_type)
        return None

    def _sample_from_cell(self, model_obj: Dict[str, Any],
                          rng: np.random.Generator) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        使用新版存储结构采样：
          - Y: 先从 y_copula 采样一条 U_y，再用 y_marginals.ppf 变回 Y
          - pA: 若有 pA_marginals，则对每一维单独抽一个 u~U(0,1)，走 pA_marginal.ppf
          - 返回 (Y, pA or None, is_pure_cell)
        """
        is_pure_cell = bool(model_obj.get("is_pure", False))
        K = int(model_obj.get("K", self.K))

        # 1) 还原 Y 的边缘
        y_mj = model_obj["y_marginals"]
        y_marginals = []
        for m in y_mj:
            mtype = m.get("type", "beta")
            if mtype == "ecdf":
                xs = np.asarray(m["xs"], dtype=float)
                ps = np.asarray(m["ps"], dtype=float)
                y_marginals.append(_C_ECDFMarginal(xs, ps))
            else:
                # 兼容以前的 beta 格式
                y_marginals.append(_C_BetaMarginal(float(m["alpha"]), float(m["beta"])))
        if len(y_marginals) != K:
            raise RuntimeError(f"cell: y_marginals length {len(y_marginals)} != K {K}")

        # 2) 还原 Y 的 copula
        ycop = model_obj["y_copula"]
        if ycop.get("type") == "vine":
            if not _PV_OK:
                raise RuntimeError("vine model present but pyvinecopulib not installed.")
            backend = _C_VineCopula(ycop["json"])
            U_y = backend.sample(1, rng).reshape(-1)
        else:
            R = np.asarray(ycop["R"], dtype=float)
            backend = _C_GaussianCopula(R)
            U_y = backend.sample(1, rng).reshape(-1)

        # 3) 反变换得到 Y
        Y = np.zeros(K, dtype=float)
        for d in range(K):
            Y[d] = y_marginals[d].ppf(np.array([U_y[d]], dtype=float))[0]
        Y = _c_clip01(Y)

        # 4) pA 部分（只在混相时有，而且是独立的一维 Beta）
        if is_pure_cell:
            return Y, None, True

        pA_list_raw = model_obj.get("pA_marginals", None)
        if pA_list_raw is None:
            # 没有存，就交给上层用 X1、可行域去兜底
            return Y, None, False

        pA = np.zeros(K, dtype=float)
        for d, m in enumerate(pA_list_raw):
            mtype = m.get("type", "beta")
            u = rng.random()
            if mtype == "ecdf":
                xs = np.asarray(m["xs"], dtype=float)
                ps = np.asarray(m["ps"], dtype=float)
                mm = _C_ECDFMarginal(xs, ps)
                pA[d] = mm.ppf(np.array([u], dtype=float))[0]
            else:
                bm = _C_BetaMarginal(float(m["alpha"]), float(m["beta"]))
                pA[d] = bm.ppf(np.array([u], dtype=float))[0]
        pA = _c_clip01(pA)

        return Y, pA, False

    # ---------- public API ----------
    def sample_one_shot(
        self,
        A: float,
        X1: float,
        rng: np.random.Generator,
        N: Optional[int] = None,
        K_use: Optional[int] = None,
        tail_strategy: str = "equal",
    ) -> Tuple[List[float], List[float]]:

        if N is None:
            N = max(2, int(self.NO_FRAG))
        pick = max(0, N - 1)

        X1 = float(np.clip(X1, 0.0, 1.0))
        pure_A_req = (X1 >= 1.0 - self.eps)
        pure_B_req = (X1 <= self.eps)
        want_pure = pure_A_req or pure_B_req

        # 小颗粒策略
        if self.small_particle_policy == "disable":
            if not self.eligible_for_tables(A):
                if pure_A_req:
                    return [1.0], [0.0]
                elif pure_B_req:
                    return [0.0], [1.0]
                else:
                    return [X1], [1.0 - X1]

        A_lookup = self._A_lookup(A)
        picked = self._pick_cell_model(A_lookup, X1, want_pure, rng)
        if picked is None:
            # 全无 → 均分回退
            z = np.full(pick, 1.0 / max(N, 1), dtype=float)
            if pure_A_req:
                rA = z.tolist(); rB = [0.0] * pick
            elif pure_B_req:
                rA = [0.0] * pick; rB = z.tolist()
            else:
                rA = (X1 * z / max(X1, 1e-12)).tolist()
                rB = ((1.0 - X1) * z / max(1.0 - X1, 1e-12)).tolist()
            rA.append(max(0.0, 1.0 - float(np.sum(rA))))
            rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB

        i_sel, j_sel, w_sel, model_obj = picked

        try:
            Y, pA, is_pure_cell = self._sample_from_cell(model_obj, rng)
        except Exception:
            # cell 内部失败 → 均分回退
            z = np.full(pick, 1.0 / max(N, 1), dtype=float)
            if pure_A_req:
                rA = z.tolist(); rB = [0.0] * pick
            elif pure_B_req:
                rA = [0.0] * pick; rB = z.tolist()
            else:
                rA = (X1 * z / max(X1, 1e-12)).tolist()
                rB = ((1.0 - X1) * z / max(1.0 - X1, 1e-12)).tolist()
            rA.append(max(0.0, 1.0 - float(np.sum(rA))))
            rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB

        # 逆 stick-breaking 得到前 K 的体积分配
        z_all, T = self._inv_stick_breaking(Y)
        if K_use is None:
            K_use = min(self.K, pick)
        else:
            K_use = min(int(K_use), pick)
        z = z_all[:K_use]

        rA: List[float] = []
        rB: List[float] = []

        if is_pure_cell or pure_A_req or pure_B_req:
            # 单相
            if pure_A_req or (is_pure_cell and X1 >= 0.5):
                for zk in z:
                    rA.append(float(zk)); rB.append(0.0)
            else:
                for zk in z:
                    rA.append(0.0); rB.append(float(zk))
        else:
            # 混相：pA 是独立的 Beta 边缘
            X3 = 1.0 - X1
            for idx in range(K_use):
                zk = float(np.clip(z[idx], 0.0, 1.0))
                if zk <= 0.0:
                    rA.append(0.0); rB.append(0.0); continue
                if pA is None:
                    # 训练端没给，就用全局 X1
                    p = float(np.clip(X1, 0.0, 1.0))
                else:
                    p = float(np.clip(pA[idx], 0.0, 1.0))

                # 可行域投影
                Lk = max(0.0, 1.0 - X3 / max(zk, _C_EPS))
                Uk = min(1.0, X1 / max(zk, _C_EPS))
                p = float(np.clip(p, Lk, Uk))

                rA.append(zk * p / max(X1, _C_EPS))
                rB.append(zk * (1.0 - p) / max(X3, _C_EPS))

        # 尾部均分
        tail_count = pick - K_use
        if tail_count > 0:
            t = float(T) / max(1, tail_count)
            if pure_A_req:
                rA.extend([t] * tail_count); rB.extend([0.0] * tail_count)
            elif pure_B_req:
                rA.extend([0.0] * tail_count); rB.extend([t] * tail_count)
            elif is_pure_cell:
                rA.extend([t] * tail_count); rB.extend([0.0] * tail_count)
            else:
                for _ in range(tail_count):
                    rA.append(0.5 * t / max(X1, 1e-12))
                    rB.append(0.5 * t / max(1.0 - X1, 1e-12))

        # 追加余量
        sA = float(np.sum(rA)); sB = float(np.sum(rB))
        rA.append(max(0.0, 1.0 - sA))
        rB.append(max(0.0, 1.0 - sB))

        # 最小体积兜底 + 归一
        vt_rel_min = self.A0_run / max(A, 1e-20)
        X3 = 1.0 - X1
        vt_rel = np.array(rA) * X1 + np.array(rB) * X3
        too_small = vt_rel < vt_rel_min
        if np.any(too_small):
            for idx in np.where(too_small)[0]:
                need = vt_rel_min - vt_rel[idx]
                if X1 >= 0.5:
                    rA[idx] += need / max(X1, 1e-12)
                else:
                    rB[idx] += need / max(X3, 1e-12)
            sA = float(np.sum(rA)); sB = float(np.sum(rB))
            gA = 1.0 / max(sA, 1e-20); gB = 1.0 / max(sB, 1e-20)
            for t in range(len(rA)):
                rA[t] *= gA; rB[t] *= gB

        # 最终严格守恒
        sA = float(np.sum(rA)); sB = float(np.sum(rB))
        if abs(sA - 1.0) > 1e-12:
            gA = 1.0 / max(sA, 1e-20)
            for t in range(len(rA)):
                rA[t] *= gA
        if abs(sB - 1.0) > 1e-12:
            gB = 1.0 / max(sB, 1e-20)
            for t in range(len(rB)):
                rB[t] *= gB

        return rA, rB

# -----------------------------------------------
#  Flow-based (conditional RealNVP) 适配器（pure / mix 双模型 + K-1 维）
# -----------------------------------------------
try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


class LMCFlowAdapter(LMCBaseAdapter):
    """
    使用离线训练好的条件流模型，分别针对纯净物和混合物建立两套模型：
      - pure 模型：target_dim = K-1，只预测前 K-1 块的 stick-breaking 体积分布
      - mix  模型：target_dim = 2*(K-1)，预测前 K-1 块的 stick-breaking + 各块的 pA
    adapter 负责：
      1) 根据 X1 判定用哪套模型
      2) 用剩余量补第 K 块
      3) 对混合物做可行域投影并拆成 rA / rB
    调用接口保持和 rank / copula 一致：
        rA, rB = flow.sample_one_shot(A, X1, rng, N=None)
    """

    def __init__(self,
                 pure_model_path: str = None,
                 mix_model_path: str = None,
                 *,
                 interp: str = "bilinear",
                 A0_run: float = None,
                 cache_enabled: bool = True):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch is required for LMCFlowAdapter.")

        # 我们仍然需要网格信息（A_grid / X1_grid）来取 meta 和 NO_FRAG，
        # 但 flow 训练是全局的，这里就做一个最小网格
        A_grid = np.array([1.0], dtype=float)
        X1_grid = np.array([0.0, 1.0], dtype=float)
        meta = {"NO_FRAG": 4, "A0": 1.0}
        super()._init_common(A_grid, X1_grid, meta,
                             interp=interp, A0_run=A0_run, cache_enabled=cache_enabled)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pure_model = None
        self.pure_meta = None
        self.mix_model = None
        self.mix_meta = None

        if pure_model_path is not None:
            self.pure_model, self.pure_meta = self._load_flow_model(pure_model_path)
        if mix_model_path is not None:
            self.mix_model, self.mix_meta = self._load_flow_model(mix_model_path)

        if self.pure_meta is None and self.mix_meta is None:
            raise ValueError("LMCFlowAdapter needs at least one of pure_model_path / mix_model_path.")

        # 取一个 K 基准
        if self.pure_meta is not None:
            self.K = int(self.pure_meta["K"])
        else:
            self.K = int(self.mix_meta["K"])

    # ====== 下面是和训练脚本同构的几个小模块 ======
    class _CondMLP(torch.nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, n_layers: int = 3):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(n_layers - 1):
                layers.append(torch.nn.Linear(d, hidden))
                layers.append(torch.nn.ReLU())
                d = hidden
            layers.append(torch.nn.Linear(d, out_dim))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class _RealNVPCoupling(torch.nn.Module):
        def __init__(self, dim: int, cond_dim: int, mask: torch.Tensor, hidden: int = 128):
            super().__init__()
            self.dim = dim
            self.cond_dim = cond_dim
            self.register_buffer("mask", mask)
            in_net = dim + cond_dim
            self.s_net = LMCFlowAdapter._CondMLP(in_net, dim, hidden=hidden)
            self.t_net = LMCFlowAdapter._CondMLP(in_net, dim, hidden=hidden)
            self.max_s = 2.0  # 可以和训练时的保持一致

        def forward(self, x, cond):
            m = self.mask
            x_masked = x * m
            inp = torch.cat([x_masked, cond], dim=1)
            s = self.s_net(inp).tanh() * self.max_s
            t = self.t_net(inp)
            s = s * (1.0 - m)
            t = t * (1.0 - m)
            y = x_masked + (1.0 - m) * (x * torch.exp(s) + t)
            logdet = ((1.0 - m) * s).sum(dim=1)
            return y, logdet

        def inverse(self, y, cond):
            m = self.mask
            y_masked = y * m
            inp = torch.cat([y_masked, cond], dim=1)
            s = self.s_net(inp).tanh() * self.max_s
            t = self.t_net(inp)
            s = s * (1.0 - m)
            t = t * (1.0 - m)
            x = y_masked + (1.0 - m) * ((y - t) * torch.exp(-s))
            logdet = -((1.0 - m) * s).sum(dim=1)
            return x, logdet

    class _CondRealNVP(torch.nn.Module):
        def __init__(self, dim: int, cond_dim: int, n_flows: int = 6, hidden: int = 128):
            super().__init__()
            masks = []
            for i in range(n_flows):
                if i % 2 == 0:
                    m = torch.cat([torch.ones(dim // 2), torch.zeros(dim - dim // 2)])
                else:
                    m = torch.cat([torch.zeros(dim // 2), torch.ones(dim - dim // 2)])
                masks.append(m)
            self.flows = torch.nn.ModuleList([
                LMCFlowAdapter._RealNVPCoupling(dim, cond_dim, mask=m, hidden=hidden) for m in masks
            ])
            self.dim = dim
            self.cond_dim = cond_dim
    
            # base dist = N(0,1)
            self.register_buffer("base_mu", torch.zeros(dim))
            self.register_buffer("base_logstd", torch.zeros(dim))
    
        def fwd(self, x, cond):
            logdet_sum = torch.zeros(x.size(0), device=x.device)
            h = x
            for flow in self.flows:
                h, logdet = flow(h, cond)
                logdet_sum = logdet_sum + logdet
            return h, logdet_sum
    
        def inv(self, z, cond):
            h = z
            logdet_sum = torch.zeros(z.size(0), device=z.device)
            for flow in reversed(self.flows):
                h, logdet = flow.inverse(h, cond)
                logdet_sum = logdet_sum + logdet
            return h, logdet_sum
    
        def log_prob(self, x, cond):
            z, logdet = self.fwd(x, cond)
            log_base = -0.5 * ((z - self.base_mu) ** 2 / torch.exp(self.base_logstd * 2) + math.log(2 * math.pi)).sum(dim=1)
            return log_base + logdet
    
        def sample(self, n: int, cond: torch.Tensor):
            # cond: (n, cond_dim)
            z = torch.randn(n, self.dim, device=cond.device)
            x, _ = self.inv(z, cond)
            return x

    # ====== 加载模型 ======
    def _load_flow_model(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        mk = ck["model_kwargs"]
        model = LMCFlowAdapter._CondRealNVP(
            dim=int(meta["target_dim"]),
            cond_dim=int(meta["cond_dim"]),
            n_flows=int(mk.get("n_flows", 6)),
            hidden=int(mk.get("hidden", 256)),
        ).to(self.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, meta

    @staticmethod
    def _is_pure_x1(x1: float, eps: float = 1e-6) -> bool:
        return (x1 <= eps) or (x1 >= 1.0 - eps)

    # ====== 公共的采样入口 ======
    def sample_one_shot(
        self,
        A: float,
        X1: float,
        rng: np.random.Generator,
        N: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        一次性返回 N 个碎片（前 N-1 由模型预测，最后 1 个为余量）。
        兼容 pure/mix 双模型（K-1 / 2*(K-1) 维）。
        """
        if N is None:
            N = max(2, int(self.NO_FRAG))
        pick = max(1, N - 1)  # 只预测前 N-1
    
        A = float(A)
        A_lookup = self._A_lookup(A)
        X1 = float(np.clip(X1, 0.0, 1.0))
        is_pure_req = (X1 <= self.eps) or (X1 >= 1.0 - self.eps)
    
        # 选模型：纯净优先 pure，否则 mix；缺哪套就用另一套兜底
        if is_pure_req:
            model, meta = (self.pure_model, self.pure_meta) if (self.pure_model is not None) else (self.mix_model, self.mix_meta)
        else:
            model, meta = (self.mix_model, self.mix_meta) if (self.mix_model is not None) else (self.pure_model, self.pure_meta)
    
        if model is None:
            # 极端兜底：均分 + 余量
            z = np.full(pick, 1.0 / max(N, 1), dtype=float)
            rA = (X1 * z / max(X1, 1e-12)).tolist()
            rB = ((1.0 - X1) * z / max(1.0 - X1, 1e-12)).tolist()
            rA.append(max(0.0, 1.0 - float(np.sum(rA))))
            rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB
    
        # 条件向量（与训练完全一致）：[logA, X1]
        cond_np = np.array([np.log(max(A_lookup, 1e-8)), X1], dtype=np.float32)[None, :]  # (1, 2)
        cond_t  = torch.from_numpy(cond_np).to(self.device)                         # (1, cond_dim)
    
        # 采样无界变量：一次只要 1 条（一个 K-1 或 2*(K-1) 向量）
        with torch.no_grad():
            x_u = model.sample(n=1, cond=cond_t)   # (1, target_dim)
            x_u = x_u[0]                           # (target_dim,)
    
        # 从无界空间映回 (0,1)
        if meta.get("support_transform") == "logit":
            x = torch.sigmoid(x_u).cpu().numpy()
        else:
            x = x_u.cpu().numpy()
            x = np.clip(x, 1e-6, 1.0 - 1e-6)
    
        K   = int(meta["K"])
        Km1 = K - 1
    
        # 纯净物：只学 Y[:K-1]，stick-breaking 得到 z，再按相别放到 rA 或 rB；末块用余量补齐
        if meta.get("model_kind") == "pure":
            Y = np.zeros(K, dtype=float)
            Y[:Km1] = x[:Km1]
            z_all= self._inv_stick_breaking(Y)   # 正确解包
            z_use = z_all[:pick]
    
            rA, rB = [], []
            if X1 >= 0.5:  # 纯 A
                rA.extend([float(zk) for zk in z_use])
                rB.extend([0.0] * len(z_use))
                rA.append(max(0.0, 1.0 - float(np.sum(rA))))  # 末块余量
                rB.append(0.0)
            else:          # 纯 B
                rA.extend([0.0] * len(z_use))
                rB.extend([float(zk) for zk in z_use])
                rA.append(0.0)
                rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB
    
        # 混合物：学 [Y[:K-1], pA[:K-1]]
        Y = np.zeros(K, dtype=float)
        Y[:Km1] = x[:Km1]
        z_all= self._inv_stick_breaking(Y)
        z_use = z_all[:pick]
    
        pA = np.zeros(K, dtype=float)
        pA[:Km1] = x[Km1: Km1 + Km1]
        pA[Km1] = X1  # 末块材料分数就用母粒的 X1
    
        rA: List[float] = []
        rB: List[float] = []
        X3 = 1.0 - X1
    
        # 可行域投影 + 相内归一
        for k in range(pick):
            zk = float(np.clip(z_use[k], 0.0, 1.0))
            if zk <= 0.0:
                rA.append(0.0); rB.append(0.0); continue
            pk = float(np.clip(pA[k], 0.0, 1.0))
            # 可行域：VA ≤ A*X1, VB ≤ A*(1-X1)  ⇒  p ∈ [max(0,1 - X3/zk), min(1, X1/zk)]
            Lk = max(0.0, 1.0 - X3 / max(zk, 1e-12))
            Uk = min(1.0, X1 / max(zk, 1e-12))
            pk = float(np.clip(pk, Lk, Uk))
            rA.append(zk * pk / max(X1, 1e-12))
            rB.append(zk * (1.0 - pk) / max(X3, 1e-12))
    
        # 末块严格守恒（各相单独补余量），保持与其它 adapter 一致
        sA = float(np.sum(rA)); sB = float(np.sum(rB))
        rA.append(max(0.0, 1.0 - sA))
        rB.append(max(0.0, 1.0 - sB))
    
        # 再做一次相内归一，避免累计误差
        sA = float(np.sum(rA)); sB = float(np.sum(rB))
        if sA > 0:
            gA = 1.0 / sA
            rA = [x * gA for x in rA]
        if sB > 0:
            gB = 1.0 / sB
            rB = [y * gB for y in rB]
    
        return rA, rB


    # ------- stick-breaking 还原，返回长度 K 的 z 向量 -------
    @staticmethod
    def _inv_stick_breaking(Y: np.ndarray) -> np.ndarray:
        K = Y.size
        z = np.zeros(K, dtype=float)
        remain = 1.0
        for k in range(K):
            yk = float(np.clip(Y[k], 0.0, 1.0))
            z[k] = yk * remain
            remain = max(0.0, remain - z[k])
        # 为了数值保险，把最后一块再对齐一下
        if remain > 1e-10:
            z[-1] += remain
        return z

