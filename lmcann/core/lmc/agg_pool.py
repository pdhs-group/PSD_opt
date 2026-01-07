# -*- coding: utf-8 -*-
import os, math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import h5py

def _format_pool_filename(Df: float, MAS: float) -> str:
    df_str = str(Df).replace(".", "p")
    mas_str = f"{MAS:.2f}".replace(".", "p")
    return f"aggregate_pool_Df{df_str}_MAS{mas_str}.h5"


class AggPool:
    """
    管理离线 aggregate 池子的读取缓存与插值抽样。
    - 持有 h5py.File 句柄（只读长期复用）
    - 维护 groups 索引（Np_target, frac_A_target, samples）
    - 提供两种插值模式：knn / bilinear
    """

    def __init__(self, pool_dir: str):
        self.pool_dir = pool_dir
        self._pool_cache: Dict[str, Dict[str, Any]] = {}
        self._pool_last_key: Optional[str] = None

    def close_pool_cache(self):
        """手动关闭所有缓存文件句柄。"""
        for c in self._pool_cache.values():
            h5 = c.get("h5", None)
            try:
                if h5 is not None:
                    h5.close()
            except Exception:
                pass
        self._pool_cache.clear()
        self._pool_last_key = None

    # ---------- internal: open & index ----------
    def _get_cache(self, Df: float, MAS: float) -> Dict[str, Any]:
        fname = _format_pool_filename(Df, MAS)
        pool_path = os.path.join(self.pool_dir, fname)
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"[AggPool] Pool file not found: {pool_path}")

        cache = self._pool_cache.get(pool_path, None)
        if cache is not None and cache.get("h5", None) is not None:
            self._pool_last_key = pool_path
            return cache

        h5 = h5py.File(pool_path, "r")
        groups = []
        for gname in h5.keys():
            grp = h5[gname]
            Np_t = float(grp.attrs.get("Np_target", np.nan))
            fracA_t = float(grp.attrs.get("frac_A_target", np.nan))
            samples = [s for s in grp.keys() if s.startswith("sample_")]
            if len(samples) == 0 or (not np.isfinite(Np_t)) or (not np.isfinite(fracA_t)):
                continue
            groups.append({
                "gname": gname,
                "Np_target": Np_t,
                "frac_A_target": fracA_t,
                "samples": samples,
            })

        if len(groups) == 0:
            h5.close()
            raise RuntimeError(f"[AggPool] No valid groups in pool file: {pool_path}")

        # 抽取所有 bin 值，便于 bilinear
        Np_vals = np.array(sorted({g["Np_target"] for g in groups}), dtype=float)
        XA_vals = np.array(sorted({g["frac_A_target"] for g in groups}), dtype=float)

        # 给 KNN 用的归一化尺度（logNp + fracA）
        logNp = np.array([math.log(max(g["Np_target"], 1e-9)) for g in groups], dtype=float)
        fracA = np.array([g["frac_A_target"] for g in groups], dtype=float)
        logNp_range = float(logNp.max() - logNp.min()) if logNp.size > 1 else 1.0
        fracA_range = float(fracA.max() - fracA.min()) if fracA.size > 1 else 1.0

        cache = dict(
            h5=h5,
            groups=groups,
            pool_path=pool_path,
            Np_vals=Np_vals,
            XA_vals=XA_vals,
            logNp=logNp,
            fracA=fracA,
            logNp_range=logNp_range,
            fracA_range=fracA_range,
        )
        self._pool_cache[pool_path] = cache
        self._pool_last_key = pool_path
        return cache

    # ---------- selection helpers ----------
    @staticmethod
    def _find_bracketing(vals: np.ndarray, x: float) -> Tuple[float, float]:
        """给定升序 vals，找 x 的左右夹逼点（超出则 clip）"""
        if x <= vals[0]:
            return float(vals[0]), float(vals[0])
        if x >= vals[-1]:
            return float(vals[-1]), float(vals[-1])
        idx = int(np.searchsorted(vals, x))
        lo = float(vals[idx - 1])
        hi = float(vals[idx])
        return lo, hi

    def _pick_group_knn(
        self, cache: Dict[str, Any], A_norm: float, X1: float,
        rng: np.random.Generator, KNN: int, sigma: float
    ) -> Dict[str, Any]:
        """KNN + 高斯权重抽样一个 group"""
        logA = math.log(max(A_norm, 1e-9))
        d_logNp = (logA - cache["logNp"]) / cache["logNp_range"]
        d_fracA = (X1 - cache["fracA"]) / cache["fracA_range"]
        dist2 = d_logNp * d_logNp + d_fracA * d_fracA

        K = min(int(KNN), dist2.size)
        nn_idx = np.argpartition(dist2, K-1)[:K]
        nn_dist2 = dist2[nn_idx]

        w = np.exp(-nn_dist2 / (2.0 * sigma * sigma))
        w_sum = float(w.sum())
        probs = (w / w_sum) if (np.isfinite(w_sum) and w_sum > 0) else np.ones(K)/K

        k_pick = int(rng.choice(K, p=probs))
        return cache["groups"][int(nn_idx[k_pick])]

    def _pick_group_bilinear(
        self,
        cache: Dict[str, Any],
        A_norm: float,
        X1: float,
        *,
        log_bilinear: bool = False,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        返回四邻点候选小池子 + 双线性权重（方案A需要多次抽样，因此这里不直接抽一个）。

        Parameters
        ----------
        log_bilinear : bool
            True 时在 log(Np) 空间做“夹逼+线性插值”，更适合 Np 倍增网格。
        """
        Np_vals = cache["Np_vals"]
        XA_vals = cache["XA_vals"]

        # ---- 1) 在 Np 维度找夹逼点（可选 log 空间） ----
        if log_bilinear:
            logNp_vals = np.log(np.maximum(Np_vals, 1e-9))
            logA = math.log(max(A_norm, 1e-9))
            logNp_lo, logNp_hi = self._find_bracketing(logNp_vals, logA)
            Np_lo = float(np.exp(logNp_lo))
            Np_hi = float(np.exp(logNp_hi))
            # 插值坐标 tx 在 log 空间计算
            def t_log(x_log, x0_log, x1_log):
                if x1_log == x0_log:
                    return 0.0
                return (x_log - x0_log) / (x1_log - x0_log)
            tx = t_log(logA, logNp_lo, logNp_hi)
        else:
            Np_lo, Np_hi = self._find_bracketing(Np_vals, A_norm)
            def t_lin(x, x0, x1):
                if x1 == x0:
                    return 0.0
                return (x - x0) / (x1 - x0)
            tx = t_lin(A_norm, Np_lo, Np_hi)

        # ---- 2) 在 XA 维度找夹逼点（线性） ----
        XA_lo, XA_hi = self._find_bracketing(XA_vals, X1)
        def t_lin(x, x0, x1):
            if x1 == x0:
                return 0.0
            return (x - x0) / (x1 - x0)
        ty = t_lin(X1, XA_lo, XA_hi)

        # ---- 3) 4 个角点及其双线性权重 ----
        corners = [
            (Np_lo, XA_lo, (1-tx)*(1-ty)),
            (Np_lo, XA_hi, (1-tx)*ty),
            (Np_hi, XA_lo, tx*(1-ty)),
            (Np_hi, XA_hi, tx*ty),
        ]

        cand_groups: List[Dict[str, Any]] = []
        cand_w: List[float] = []

        for Np_c, XA_c, w in corners:
            if w <= 0:
                continue
            g = next(
                (gg for gg in cache["groups"]
                 if float(gg["Np_target"]) == float(Np_c)
                 and float(gg["frac_A_target"]) == float(XA_c)),
                None
            )
            if g is not None:
                cand_groups.append(g)
                cand_w.append(float(w))

        if not cand_groups:
            # 极端情况下回退为最近邻（用 knn 1邻点）
            g_nn = self._pick_group_knn(cache, A_norm, X1,
                                        np.random.default_rng(), KNN=1, sigma=1.0)
            return [g_nn], np.array([1.0], dtype=float)

        w_arr = np.array(cand_w, dtype=float)
        w_arr /= w_arr.sum()
        return cand_groups, w_arr


    def sample_grid(
        self,
        Df: float,
        MAS: float,
        A_norm: float,
        X1: float,
        rng: np.random.Generator,
        *,
        interp: str = "knn",
        KNN: int = 4,
        sigma: float = 0.35,
        # ---- 方案A新增参数 ----
        max_draws: int = 15,
        tau_A: float | None = None,
        tau_X: float | None = None,
        log_bilinear: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        按 (A_norm, X1) 从池子中抽一个样本，返回 (M, Hbond, Vbond)。

        interp:
          - "knn"      : KNN + 高斯权重抽样 group（不做阈值软约束）
          - "bilinear" : 四邻点双线性（支持方案A：有限次重抽+最近回退）
        max_draws:
          bilinear 模式下最多尝试次数；超过则返回“尝试中最接近的样本”，保证不死循环。
        tau_A, tau_X:
          软阈值（相对误差）：
            |A_norm_pool - A_norm|/A_norm <= tau_A
            |X1_pool - X1| <= tau_X
          若任一为 None，则不启用阈值检查（直接一次抽样）。
        log_bilinear:
          bilinear 模式下是否在 log(Np) 空间插值。
        """
        cache = self._get_cache(Df, MAS)

        # -------- knn：保持原逻辑（一次选池+抽样） --------
        if interp != "bilinear":
            g_pick = self._pick_group_knn(cache, A_norm, X1, rng, KNN=KNN, sigma=sigma)
            subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"])))]
            sub = cache["h5"][g_pick["gname"]][subname]
            M = sub["M"][()]
            Hbond = sub["Hbond"][()]
            Vbond = sub["Vbond"][()]
            return M, Hbond, Vbond

        # -------- bilinear / log-bilinear：方案A --------
        cand_groups, cand_probs = self._pick_group_bilinear(
            cache, A_norm, X1, log_bilinear=log_bilinear
        )

        # 若未启用阈值，直接抽一次返回
        if tau_A is None or tau_X is None:
            g_pick = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
            subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"])))]
            sub = cache["h5"][g_pick["gname"]][subname]
            M = sub["M"][()]
            Hbond = sub["Hbond"][()]
            Vbond = sub["Vbond"][()]
            return M, Hbond, Vbond

        # 记录尝试中“最接近的样本”
        best_dist2 = float("inf")
        best_triplet = None  # (M, Hbond, Vbond)

        for _ in range(int(max_draws)):
            g_pick = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
            subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"])))]
            sub = cache["h5"][g_pick["gname"]][subname]

            M = sub["M"][()]
            Hbond = sub["Hbond"][()]
            Vbond = sub["Vbond"][()]

            # 计算该样本的实际 A_norm 与 X1（从 M 便宜得到）
            n1 = int((M == 1).sum())
            n2 = int((M == 2).sum())
            occ = n1 + n2
            if occ <= 0:
                continue
            A_norm_pool = float(occ)
            X1_pool = float(n1) / float(occ)

            dA = abs(A_norm_pool - A_norm) / max(A_norm, 1e-9)
            dX = abs(X1_pool - X1)

            dist2 = dA*dA + dX*dX
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_triplet = (M, Hbond, Vbond)

            if dA <= tau_A and dX <= tau_X:
                return M, Hbond, Vbond

        # 超过 max_draws 仍不满足：回退到最接近样本（保证不死循环）
        if best_triplet is not None:
            return best_triplet

        # 极端兜底：随便返回一个
        g_pick = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
        subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"])))]
        sub = cache["h5"][g_pick["gname"]][subname]
        M = sub["M"][()]
        Hbond = sub["Hbond"][()]
        Vbond = sub["Vbond"][()]
        return M, Hbond, Vbond