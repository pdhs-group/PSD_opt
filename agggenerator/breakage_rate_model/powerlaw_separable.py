# -*- coding: utf-8 -*-
"""
两段平滑幂律 (powerlaw + plateau [+ optional tail]) surrogate：

    基础形式（原模型）:
        E_base(V, θ) = Emax(θ) * ((V/Vc(θ))^σ(θ) / (1 + (V/Vc(θ))^σ(θ)))

    扩展形式（侵蚀尾巴，仅对侵蚀类 group 可选）:
        对于 int_bre == 0 的侵蚀型 group，在 enable_tail=True 时使用：

            E(V, θ) = E_base(V, θ) * exp( -α(θ) * max(log V - log Vc(θ), 0) )

        对于 int_bre != 0 的普通破碎 group，不使用 α（始终是纯 plateau）。

    - group 级别：
        * 每个 group 拟合 (σ_g, Vc_g, Emax_g[, α_g])
          · int_bre == 0 且 enable_tail=True → 带 α_g
          · 其他情况 → α_g=0，不拟合尾巴
    - θ 维度：
        * 按 int_bre==0 / !=0 将 group 分为两簇
        * 各自用线性回归拟合 θ -> 参数
            · 侵蚀簇: θ -> [σ, logVc, logEmax, (α)]
            · 普通簇: θ -> [σ, logVc, logEmax]
    - 样本级：
        * 输入 X = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
        * 根据 int_bre 选择侵蚀簇/普通簇的线性模型，再计算 logE_pred
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Tuple, List

import numpy as np

from .data_io import EnergyGroupRecord
from .base import BaseEnergyModel


# =============================================================================
# 工具：局部斜率、plateau 检测
# =============================================================================


@dataclass
class LocalSlopeProfile:
    """
    logE–logV 局部斜率剖面：
        sigma_loc[i] = ΔlogE / ΔlogV（相邻两点）
        V_mid[i]     = 对应两点的几何中点
    """
    V_mid: np.ndarray        # shape (n-1,)
    sigma_loc: np.ndarray    # shape (n-1,)

# ------------------------------------------------------------------
# 工具：按 int_bre 分簇
# ------------------------------------------------------------------
def split_groups_by_int_bre(
    groups: Sequence[EnergyGroupRecord],
    tol: float = 1e-12,
) -> Tuple[List[EnergyGroupRecord], List[EnergyGroupRecord]]:
    """
    按 int_bre=0 / !=0 将 group 分成侵蚀簇和普通簇。

    返回:
        erosion_groups, normal_groups
    """
    erosion: List[EnergyGroupRecord] = []
    normal: List[EnergyGroupRecord] = []

    for rec in groups:
        if abs(rec.int_bre) <= tol:
            erosion.append(rec)
        else:
            normal.append(rec)
    return erosion, normal

def compute_local_slopes(V: np.ndarray, E: np.ndarray) -> LocalSlopeProfile:
    """
    对相邻点计算 logE–logV 的局部斜率:
        sigma_loc[i] = ΔlogE / ΔlogV
        V_mid[i]     = sqrt(V_i * V_{i+1}) （几何中点，更适合 log 坐标）
    """
    mask = (V > 0.0) & (E > 0.0)
    V = V[mask]
    E = E[mask]

    if V.size < 2:
        return LocalSlopeProfile(V_mid=np.array([]), sigma_loc=np.array([]))

    logV = np.log(V)
    logE = np.log(E)

    dlogV = np.diff(logV)
    dlogE = np.diff(logE)

    sigma_loc = dlogE / dlogV
    V_mid = np.sqrt(V[:-1] * V[1:])

    return LocalSlopeProfile(V_mid=V_mid, sigma_loc=sigma_loc)


def find_plateau_transition(
    profile: LocalSlopeProfile,
    *,
    frac_threshold: float = 0.3,
    abs_threshold: float = 0.1,
) -> Optional[float]:
    """
    基于局部斜率 profile 估计 plateau 起点 V_c。

    思路（heuristic）：
    - 取中间 50% 的点，用它们的 |sigma_loc| 的中位数作为典型幂律斜率 sigma_mid；
    - 从大 V 端往回扫描，找到第一个满足:
          |sigma_loc| < max(frac_threshold * sigma_mid, abs_threshold)
      的点，把对应的 V_mid 当成 plateau 过渡的开始 V_c。

    若未找到，返回 None。
    """
    V_mid = profile.V_mid
    sigma_loc = profile.sigma_loc

    n = sigma_loc.size
    if n == 0:
        return None

    # 中间区间：大致 25%~75% 作为“典型幂律”估计
    i1 = n // 4
    i2 = 3 * n // 4
    if i2 <= i1:
        i1, i2 = 0, n

    sigma_mid = np.median(np.abs(sigma_loc[i1:i2]))
    if not np.isfinite(sigma_mid) or sigma_mid <= 0:
        return None

    thresh = max(frac_threshold * sigma_mid, abs_threshold)

    # 从 large-V 端往回找第一个 “斜率很小” 的位置
    for i in range(n - 1, -1, -1):
        if abs(sigma_loc[i]) < thresh:
            return float(V_mid[i])

    return None


# =============================================================================
# 平滑两段模型 + 可选尾巴，带 log-space & clip（数值稳定）
# =============================================================================


def smooth_powerlaw_plateau_with_tail(
    V: np.ndarray,
    sigma: np.ndarray,
    Vc: np.ndarray,
    Emax: np.ndarray,
    alpha: np.ndarray,
    *,
    enable_tail: bool = True,
    clip_logx: float = 50.0,
) -> np.ndarray:
    """
    扩展两段模型：

        x = (V/Vc)^σ
        E_base = Emax * x / (1 + x)

        tail: g(V) = exp( -α * max(log V - log Vc, 0) )
        E = E_base * g(V)

    - 在 log 空间计算 (V/Vc)^σ，使用 clip 防止 overflow。
    - alpha >= 0，若 enable_tail=False 或 alpha 全 0，则退化为原模型。

    参数可以是标量或向量（自动广播）。
    """
    V = np.asarray(V, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    Vc = np.asarray(Vc, dtype=float)
    Emax = np.asarray(Emax, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    # 避免 log(0)
    V_safe = np.maximum(V, 1e-16)
    Vc_safe = np.maximum(Vc, 1e-16)
    Emax_safe = np.maximum(Emax, 1e-16)

    logV = np.log(V_safe)
    logVc = np.log(Vc_safe)

    # 计算 (V/Vc)^sigma 的 log 形式，并进行 clip
    logx = sigma * (logV - logVc)
    logx = np.clip(logx, -clip_logx, clip_logx)
    x = np.exp(logx)

    E_base = Emax_safe * x / (1.0 + x)

    if not enable_tail:
        return E_base

    # 若 alpha 全为 0，直接返回
    if np.all(alpha == 0.0):
        return E_base

    # tail: g(V) = exp(-alpha * max(logV - logVc, 0))
    tail_arg = np.maximum(logV - logVc, 0.0)
    log_g = -alpha * tail_arg
    log_g = np.clip(log_g, -clip_logx, clip_logx)
    g = np.exp(log_g)

    return E_base * g


def smooth_powerlaw_plateau(
    V: np.ndarray,
    sigma: np.ndarray,
    Vc: np.ndarray,
    Emax: np.ndarray,
) -> np.ndarray:
    """
    原来的平滑两段模型接口（无尾巴），现在是
    smooth_powerlaw_plateau_with_tail 的安全包装：
        alpha=0, enable_tail=False
    """
    alpha = 0.0
    return smooth_powerlaw_plateau_with_tail(
        V,
        sigma,
        Vc,
        Emax,
        alpha,
        enable_tail=False,
    )


# =============================================================================
# 非线性最小二乘拟合：带 log 参数化 + bounds
# =============================================================================


def fit_smooth_two_segment(
    V: np.ndarray,
    E: np.ndarray,
    Vc_init: Optional[float] = None,
    *,
    fit_mode: str = "logE",
    enable_tail: bool = True,
    sigma_bounds: Tuple[float, float] = (0.0, 10.0),
    logVc_margin: float = 2.0,
    logEmax_margin: float = 2.0,
    alpha_bounds: Tuple[float, float] = (0.0, 0.5),
    clip_logx: float = 50.0,
) -> Tuple[float, float, float, float]:
    """
    使用非线性最小二乘拟合“平滑两段 + 可选尾巴”模型的参数：

        若 enable_tail=True:  拟合 [sigma, logVc, logEmax, alpha]
        若 enable_tail=False: 实际上 alpha 固定为 0，仍返回 alpha=0

    参数空间：
        - sigma      ∈ sigma_bounds
        - logVc      ∈ [log(Vmin) - logVc_margin, log(Vmax) + logVc_margin]
        - logEmax    ∈ [log(Emin), log(Emax) + logEmax_margin]
        - alpha      ∈ alpha_bounds (若启用 tail)

    fit_mode:
        - "E"    : 在 E 空间做最小二乘
        - "logE" : 在 log(E) 空间做最小二乘
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError(
            "fit_smooth_two_segment requires scipy.optimize.least_squares. "
            "Please install scipy."
        ) from e

    mask = (V > 0.0) & (E > 0.0)
    V = V[mask]
    E = E[mask]

    if V.size < 3:
        raise RuntimeError("Not enough points to fit smooth two-segment model.")

    logV = np.log(V)
    logE = np.log(E)

    # 初始 guess：全局线性拟合提供 sigma0, b0
    A = np.vstack([np.ones_like(logV), logV]).T
    coef, *_ = np.linalg.lstsq(A, logE, rcond=None)
    b0, sigma0 = coef

    Emax0 = float(np.max(E))
    if Vc_init is None:
        Vc0 = float(np.median(V))
    else:
        Vc0 = float(Vc_init)

    logVc0 = float(np.log(max(Vc0, 1e-8)))
    logEmax0 = float(np.log(max(Emax0, 1e-12)))
    alpha0 = 0.0  # 默认无侵蚀尾巴

    if enable_tail:
        x0 = np.array([sigma0, logVc0, logEmax0, alpha0], dtype=float)
    else:
        x0 = np.array([sigma0, logVc0, logEmax0], dtype=float)

    # bounds
    sigma_lo, sigma_hi = sigma_bounds
    logVmin = float(np.log(np.min(V)))
    logVmax = float(np.log(np.max(V)))
    logVc_lo = logVmin - logVc_margin
    logVc_hi = logVmax + logVc_margin

    logEmin = float(np.log(np.min(E)))
    logEmax = float(np.log(np.max(E))) + logEmax_margin

    if enable_tail:
        alpha_lo, alpha_hi = alpha_bounds
        lower = np.array([sigma_lo, logVc_lo, logEmin, alpha_lo], dtype=float)
        upper = np.array([sigma_hi, logVc_hi, logEmax, alpha_hi], dtype=float)
    else:
        lower = np.array([sigma_lo, logVc_lo, logEmin], dtype=float)
        upper = np.array([sigma_hi, logVc_hi, logEmax], dtype=float)

    def residual_E(x: np.ndarray, V: np.ndarray, E: np.ndarray) -> np.ndarray:
        if enable_tail:
            sigma, logVc, logEmax, alpha = x
        else:
            sigma, logVc, logEmax = x
            alpha = 0.0
        Vc = np.exp(logVc)
        Emax_ = np.exp(logEmax)
        E_pred = smooth_powerlaw_plateau_with_tail(
            V,
            sigma,
            Vc,
            Emax_,
            alpha,
            enable_tail=enable_tail,
            clip_logx=clip_logx,
        )
        return E_pred - E

    def residual_logE(x: np.ndarray, V: np.ndarray, E: np.ndarray) -> np.ndarray:
        if enable_tail:
            sigma, logVc, logEmax, alpha = x
        else:
            sigma, logVc, logEmax = x
            alpha = 0.0
        Vc = np.exp(logVc)
        Emax_ = np.exp(logEmax)
        E_pred = smooth_powerlaw_plateau_with_tail(
            V,
            sigma,
            Vc,
            Emax_,
            alpha,
            enable_tail=enable_tail,
            clip_logx=clip_logx,
        )
        E_pred = np.maximum(E_pred, 1e-12)
        return np.log(E_pred) - np.log(E)

    if fit_mode == "E":
        fun = lambda x: residual_E(x, V, E)
    elif fit_mode == "logE":
        fun = lambda x: residual_logE(x, V, E)
    else:
        raise ValueError(f"Unknown fit_mode '{fit_mode}', must be 'E' or 'logE'.")

    res = least_squares(fun, x0, bounds=(lower, upper), method="trf")

    if enable_tail:
        sigma_fit, logVc_fit, logEmax_fit, alpha_fit = res.x
    else:
        sigma_fit, logVc_fit, logEmax_fit = res.x
        alpha_fit = 0.0

    Vc_fit = float(np.exp(logVc_fit))
    Emax_fit = float(np.exp(logEmax_fit))

    return float(sigma_fit), Vc_fit, Emax_fit, float(alpha_fit)


def _fit_global_powerlaw(V: np.ndarray, E: np.ndarray) -> Tuple[float, float, float]:
    """
    单段幂律拟合：logE ≈ b + sigma logV

    返回:
        sigma, b, pearson_r
    """
    mask = (V > 0.0) & (E > 0.0)
    V_valid = V[mask]
    E_valid = E[mask]
    if V_valid.size < 2:
        return np.nan, np.nan, np.nan

    logV = np.log(V_valid)
    logE = np.log(E_valid)

    if logV.size > 1:
        r = np.corrcoef(logV, logE)[0, 1]
    else:
        r = np.nan

    A = np.vstack([np.ones_like(logV), logV]).T
    coef, *_ = np.linalg.lstsq(A, logE, rcond=None)
    b, sigma = coef
    return float(sigma), float(b), float(r)


def fit_two_segment_params(
    V: np.ndarray,
    E: np.ndarray,
    pearson_r: float,
    *,
    frac_threshold: float = 0.3,
    abs_threshold: float = 0.1,
    fit_mode: str = "logE",
    pure_powerlaw_if_no_plateau: bool = True,
    enable_tail: bool = False,
    sigma_bounds: Tuple[float, float] = (0.0, 10.0),
    logVc_margin: float = 2.0,
    logEmax_margin: float = 2.0,
    alpha_bounds: Tuple[float, float] = (0.0, 0.5),
    max_V: Optional[float] = None,
) -> Dict[str, float]:
    """
    对单条 (V, E) 曲线做“两段模型(+可选尾巴)”拟合，返回参数字典：

        {
          'sigma': sigma_fit,
          'Vc': Vc_fit,
          'Emax': Emax_fit,
          'alpha': alpha_fit,   # 若 enable_tail=False，则恒为 0
          'Vc_local': Vc_local,
          'r_global': pearson_r,
        }

    行为：
    - 先通过局部斜率 profile 尝试找到 plateau 起点 Vc_local；
    - 若能找到，则使用平滑两段(+可选尾巴)模型拟合；
    - 若找不到且 pure_powerlaw_if_no_plateau=True，则退化成单段幂律：
        * 在 logE–logV 上拟合 sigma, b
        * 设 Vc_fit = 10 * max(V)，Emax_fit = exp(b + sigma * log(Vc_fit))
          这样在当前 V 范围内等价于纯幂律（plateau 在观测范围之外），
          alpha_fit = 0。
    """
    mask = (V > 0.0) & (E > 0.0)
    if max_V is not None:
        mask &= (V <= max_V)
    V = V[mask]
    E = E[mask]

    if V.size < 3:
        # 点太少则直接返回 NaN（上层会跳过这个 group）
        return dict(
            sigma=np.nan,
            Vc=np.nan,
            Emax=np.nan,
            alpha=0.0,
            Vc_local=np.nan,
            r_global=pearson_r,
        )

    profile = compute_local_slopes(V, E)
    Vc_local = find_plateau_transition(
        profile,
        frac_threshold=frac_threshold,
        abs_threshold=abs_threshold,
    )

    # 无 plateau 情况：可以退化为单段幂律
    if Vc_local is None and pure_powerlaw_if_no_plateau:
        sigma_pl, b_pl, r_pl = _fit_global_powerlaw(V, E)
        if not np.isfinite(sigma_pl) or not np.isfinite(b_pl):
            # 再不行就返回 NaN
            sigma_fit = np.nan
            Vc_fit = np.nan
            Emax_fit = np.nan
            alpha_fit = np.nan
        else:
            Vc_fit = float(np.max(V) * 10.0)
            logVc = np.log(Vc_fit)
            # E(V) = exp(b + sigma log V)；希望在 V=Vc 时连续：
            #   Emax = exp(b + sigma logVc)
            Emax_fit = float(np.exp(b_pl + sigma_pl * logVc))
            sigma_fit = sigma_pl
            alpha_fit = 0.0
    else:
        # 找到了 plateau 或者我们强制要两段模型
        try:
            sigma_fit, Vc_fit, Emax_fit, alpha_fit = fit_smooth_two_segment(
                V,
                E,
                Vc_init=Vc_local,
                fit_mode=fit_mode,
                enable_tail=enable_tail,
                sigma_bounds=sigma_bounds,
                logVc_margin=logVc_margin,
                logEmax_margin=logEmax_margin,
                alpha_bounds=alpha_bounds,
            )
        except Exception:
            sigma_fit = np.nan
            Vc_fit = np.nan
            Emax_fit = np.nan
            alpha_fit = np.nan

    return dict(
        sigma=sigma_fit,
        Vc=Vc_fit,
        Emax=Emax_fit,
        alpha=alpha_fit,
        Vc_local=(Vc_local if Vc_local is not None else np.nan),
        r_global=pearson_r,
    )


# =============================================================================
# PowerLawSeparableModel（按 int_bre 分簇）
# =============================================================================


class PowerLawSeparableModel(BaseEnergyModel):
    """
    两段平滑幂律 + 可选侵蚀尾巴 的 surrogate：

        对 int_bre != 0 的普通破碎 group:
            E(V, θ) = Emax(θ) * ((V/Vc(θ))^σ(θ) / (1 + (V/Vc(θ))^σ(θ)))

        对 int_bre == 0 的侵蚀型 group（当 enable_tail=True 时）:
            在上述基础上乘一个尾巴衰减:
                g(V, θ) = exp( -α(θ) * max(log V - log Vc(θ), 0) )

    训练时：
        - 按 int_bre == 0 / !=0 将 group 分为两簇
        - 每簇分别：
            · 对每个 group 拟合 (σ_g, Vc_g, Emax_g[, α_g])
            · 构造 θ = [log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
            · 用线性回归 θ -> 参数

    预测时：
        - 根据样本的 int_bre 选择对应簇的线性模型，
          再根据 V 和预测的参数计算 logE。
    """

    def __init__(
        self,
        *,
        pearson_min: Optional[float] = None,
        frac_threshold: float = 0.3,
        abs_threshold: float = 0.1,
        fit_mode: str = "logE",
        pure_powerlaw_if_no_plateau: bool = True,
        enable_tail: bool = False,
        sigma_bounds: Tuple[float, float] = (0.0, 10.0),
        logVc_margin: float = 2.0,
        logEmax_margin: float = 2.0,
        alpha_bounds: Tuple[float, float] = (0.0, 0.5),
        max_V: Optional[float] = None,
        name: Optional[str] = None,
        plateau_weight: float = 3.0,
        regress_type: str = "linear",   # "linear" or "ridge"
        ridge_lambda: float = 1e-2,
    ):
        super().__init__(name=name or "PowerLawSeparableModel")

        # group 内拟合与 plateau 检测相关参数
        self.pearson_min = pearson_min
        self.frac_threshold = frac_threshold
        self.abs_threshold = abs_threshold
        self.fit_mode = fit_mode
        self.pure_powerlaw_if_no_plateau = pure_powerlaw_if_no_plateau
        self.enable_tail = enable_tail
        self.sigma_bounds = sigma_bounds
        self.logVc_margin = logVc_margin
        self.logEmax_margin = logEmax_margin
        self.alpha_bounds = alpha_bounds
        self.max_V = max_V
        self.plateau_weight = plateau_weight
        self.regress_type = regress_type
        self.ridge_lambda = ridge_lambda

        # θ 维度固定为 6: [log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
        self._theta_dim: int = 6

        # 两簇各自线性回归系数
        self._coef_erosion: Optional[np.ndarray] = None   # shape (d+1, k_e)
        self._coef_normal: Optional[np.ndarray] = None    # shape (d+1, k_n)

    # ------------------------------------------------------------------
    # 内部：从一簇 groups 中收集 θ 和 (σ, logVc, logEmax[, α])，并标记 plateau
    # ------------------------------------------------------------------
    def _collect_group_params(
        self,
        groups: Sequence[EnergyGroupRecord],
        *,
        enable_tail_for_subset: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        对给定的一簇 group：
            - 过滤 pearson_min
            - 对每个 group 进行两段(+可选尾巴)拟合
            - 返回:
                Theta       : shape (n_groups, d)
                Params      : shape (n_groups, n_param)
                plateau_mask: shape (n_groups,), bool
                               True 表示该 group 检测到了 plateau
                               （即 fit_two_segment_params 的 Vc_local 为有限值）
        """
        theta_list: List[np.ndarray] = []
        param_list: List[np.ndarray] = []
        plateau_flags: List[bool] = []

        for rec in groups:
            # pearson 过滤（若有设置）
            if self.pearson_min is not None:
                r_g = rec.pearson_r_fit
                if np.isfinite(r_g) and r_g < self.pearson_min:
                    continue

            params = fit_two_segment_params(
                rec.V,
                rec.E_mean,
                rec.pearson_r_fit,
                frac_threshold=self.frac_threshold,
                abs_threshold=self.abs_threshold,
                fit_mode=self.fit_mode,
                pure_powerlaw_if_no_plateau=self.pure_powerlaw_if_no_plateau,
                enable_tail=enable_tail_for_subset,
                sigma_bounds=self.sigma_bounds,
                logVc_margin=self.logVc_margin,
                logEmax_margin=self.logEmax_margin,
                alpha_bounds=self.alpha_bounds,
                max_V=self.max_V,
            )
            sigma_g = params["sigma"]
            Vc_g = params["Vc"]
            Emax_g = params["Emax"]
            alpha_g = params.get("alpha", 0.0)
            Vc_local = params.get("Vc_local", np.nan)

            # 拟合失败或参数不合理就跳过
            if not (np.isfinite(sigma_g) and np.isfinite(Vc_g) and np.isfinite(Emax_g)):
                continue
            if Vc_g <= 0.0 or Emax_g <= 0.0:
                continue
            if not np.isfinite(alpha_g):
                alpha_g = 0.0

            # θ 特征: [log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
            log_gamma = np.log(rec.gamma)
            log_NOFRAG = np.log(rec.NO_FRAG)
            theta_vec = np.array(
                [log_gamma, log_NOFRAG, rec.int_bre, rec.Df, rec.MAS, rec.X1],
                dtype=float,
            )

            theta_list.append(theta_vec)

            if enable_tail_for_subset:
                param_vec = np.array(
                    [sigma_g, np.log(Vc_g), np.log(Emax_g), alpha_g],
                    dtype=float,
                )
            else:
                param_vec = np.array(
                    [sigma_g, np.log(Vc_g), np.log(Emax_g)],
                    dtype=float,
                )

            param_list.append(param_vec)

            # plateau 标记：如果 Vc_local 是有限值，就认为该 group 需要 plateau
            plateau_flags.append(np.isfinite(Vc_local))

        if not theta_list:
            return None, None, None

        Theta = np.vstack(theta_list)
        Params = np.vstack(param_list)
        plateau_mask = np.array(plateau_flags, dtype=bool)
        return Theta, Params, plateau_mask


    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def fit_from_groups(
        self,
        groups: Sequence[EnergyGroupRecord],
    ) -> "PowerLawSeparableModel":
        """
        从 EnergyGroupRecord 列表中拟合模型参数。

        步骤：
            1. 按 int_bre==0 / !=0 将 group 分成侵蚀簇 / 普通簇；
            2. 对每簇：
                - 进行 group 内两段(+可选尾巴)拟合
                - 线性/岭回归 θ -> 参数
                  · 侵蚀簇：按需拟合尾巴，不加权
                  · 普通簇：不带尾巴，对 plateau 组加权 plateau_weight
            3. 保存两簇的回归系数，用于预测。

        回归类型由 self.regress_type 决定：
            - "linear" : 普通最小二乘 (np.linalg.lstsq)
            - "ridge"  : 岭回归 (闭式解)，使用 self.ridge_lambda（若存在）或默认 1e-2
        """
        erosion_groups, normal_groups = split_groups_by_int_bre(groups)

        # 侵蚀簇：按需拟合尾巴，但不做权重
        Theta_e, Y_e, _ = self._collect_group_params(
            erosion_groups,
            enable_tail_for_subset=self.enable_tail,  # 侵蚀簇只有在 enable_tail=True 时才拟合 α
        )

        # 普通簇：不带尾巴，但对 plateau 组做权重
        Theta_n, Y_n, plateau_n = self._collect_group_params(
            normal_groups,
            enable_tail_for_subset=False,             # 普通簇始终不带尾巴
        )

        if Theta_e is None and Theta_n is None:
            raise RuntimeError(
                "PowerLawSeparableModel.fit_from_groups: no valid groups to train on."
            )

        # θ 维度（两个簇应该一致）
        self._theta_dim = 6

        # 选择回归类型 & λ
        regress_type = getattr(self, "regress_type", "linear").lower()
        ridge_lambda = float(getattr(self, "ridge_lambda", 1e-2))

        # ------------------------------------------------------------------
        # 侵蚀簇回归：不加权，可选 linear / ridge
        # ------------------------------------------------------------------
        if Theta_e is not None:
            n_e, d_e = Theta_e.shape
            A_e = np.column_stack([np.ones(n_e), Theta_e])   # (n_e, d_e+1)

            if regress_type == "linear":
                # 普通最小二乘
                coef_e, *_ = np.linalg.lstsq(A_e, Y_e, rcond=None)
            elif regress_type == "ridge":
                # 岭回归： (A^T A + λI)W = A^T Y
                ATA = A_e.T @ A_e
                reg = ridge_lambda * np.eye(ATA.shape[0])
                # 一般不对偏置项做正则
                reg[0, 0] = 0.0
                coef_e = np.linalg.solve(ATA + reg, A_e.T @ Y_e)
            else:
                raise ValueError(f"Unknown regress_type '{self.regress_type}'")

            self._coef_erosion = coef_e
        else:
            self._coef_erosion = None

        # ------------------------------------------------------------------
        # 普通簇回归：对 plateau 组加权，可选 linear / ridge
        # ------------------------------------------------------------------
        if Theta_n is not None:
            n_n, d_n = Theta_n.shape
            A_n = np.column_stack([np.ones(n_n), Theta_n])   # (n_n, d_n+1)

            if plateau_n is not None and plateau_n.size == n_n:
                K = getattr(self, "plateau_weight", 3.0)
                # 权重向量：plateau 组权重 K，其余为 1
                w = np.ones(n_n, dtype=float)
                w[plateau_n] = K
                sqrt_w = np.sqrt(w)[:, None]       # (n_n, 1)

                # 通过对行乘 sqrt_w 来实现加权 least-squares / ridge
                A_w = A_n * sqrt_w                 # (n_n, d_n+1)
                Y_w = Y_n * sqrt_w                 # (n_n, n_param)

                if regress_type == "linear":
                    coef_n, *_ = np.linalg.lstsq(A_w, Y_w, rcond=None)
                elif regress_type == "ridge":
                    ATA = A_w.T @ A_w
                    reg = ridge_lambda * np.eye(ATA.shape[0])
                    reg[0, 0] = 0.0
                    coef_n = np.linalg.solve(ATA + reg, A_w.T @ Y_w)
                else:
                    raise ValueError(f"Unknown regress_type '{self.regress_type}'")
            else:
                # 如果 plateau 标记不可用，就退化为普通 least-squares / ridge
                if regress_type == "linear":
                    coef_n, *_ = np.linalg.lstsq(A_n, Y_n, rcond=None)
                elif regress_type == "ridge":
                    ATA = A_n.T @ A_n
                    reg = ridge_lambda * np.eye(ATA.shape[0])
                    reg[0, 0] = 0.0
                    coef_n = np.linalg.solve(ATA + reg, A_n.T @ Y_n)
                else:
                    raise ValueError(f"Unknown regress_type '{self.regress_type}'")

            self._coef_normal = coef_n
        else:
            self._coef_normal = None

        if self._coef_erosion is None and self._coef_normal is None:
            raise RuntimeError(
                "PowerLawSeparableModel.fit_from_groups: both erosion and normal fits failed."
            )

        self._is_fitted = True
        return self



    def fit(
        self,
        X_train: np.ndarray | None,
        y_train: np.ndarray | None = None,
        **kwargs,
    ) -> "PowerLawSeparableModel":
        """
        与 BaseEnergyModel 接口兼容的 fit：

        约定用法：
            model.fit(None, None, groups=groups)
        """
        groups: Optional[Sequence[EnergyGroupRecord]] = kwargs.get("groups", None)
        if groups is None:
            raise ValueError("PowerLawSeparableModel.fit requires 'groups' argument.")
        return self.fit_from_groups(groups)

    # ------------------------------------------------------------------
    # 子簇预测工具
    # ------------------------------------------------------------------
    def _predict_subset(
        self,
        V: np.ndarray,
        theta: np.ndarray,
        coef: np.ndarray,
        *,
        enable_tail_for_subset: bool,
    ) -> np.ndarray:
        """
        给定 V、theta 和某个簇的线性回归系数 coef，
        计算该簇下的 E_pred(V, θ)。
        """
        if coef is None:
            raise RuntimeError("Internal error: coef is None in _predict_subset.")

        n, d = theta.shape
        A = np.column_stack([np.ones(n), theta])   # (n, d+1)
        Y = A @ coef                               # (n, n_param)

        sigma = Y[:, 0]
        Vc = np.exp(Y[:, 1])
        Emax = np.exp(Y[:, 2])

        if enable_tail_for_subset and Y.shape[1] >= 4:
            alpha = Y[:, 3]
        else:
            alpha = np.zeros_like(sigma)

        E_pred = smooth_powerlaw_plateau_with_tail(
            V,
            sigma,
            Vc,
            Emax,
            alpha,
            enable_tail=enable_tail_for_subset,
        )
        return E_pred

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        输入特征矩阵 X：

            X[i] = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1]

        输出：
            y_pred[i] = log(E_pred(V_i, θ_i))
        """
        if not self._is_fitted:
            raise RuntimeError("PowerLawSeparableModel is not fitted.")

        if X.ndim != 2 or X.shape[1] < 1 + self._theta_dim:
            raise ValueError(
                f"X shape not compatible: got {X.shape}, "
                f"expect (n_samples, {1 + self._theta_dim})"
            )

        logV = X[:, 0]
        V = np.exp(logV)
        theta = X[:, 1:1 + self._theta_dim]

        int_bre = theta[:, 2]
        erosion_mask = np.isclose(int_bre, 0.0)
        normal_mask = ~erosion_mask

        E_pred = np.empty_like(V)

        # erosion subset (int_bre == 0)
        if np.any(erosion_mask):
            if self._coef_erosion is not None:
                E_pred[erosion_mask] = self._predict_subset(
                    V[erosion_mask],
                    theta[erosion_mask],
                    self._coef_erosion,
                    enable_tail_for_subset=self.enable_tail,
                )
            elif self._coef_normal is not None:
                # fallback：若没有 erosion 模型，则用 normal 模型
                E_pred[erosion_mask] = self._predict_subset(
                    V[erosion_mask],
                    theta[erosion_mask],
                    self._coef_normal,
                    enable_tail_for_subset=False,
                )
            else:
                raise RuntimeError("No valid coef for erosion subset.")

        # normal subset (int_bre != 0)
        if np.any(normal_mask):
            if self._coef_normal is not None:
                E_pred[normal_mask] = self._predict_subset(
                    V[normal_mask],
                    theta[normal_mask],
                    self._coef_normal,
                    enable_tail_for_subset=False,
                )
            elif self._coef_erosion is not None:
                # fallback：若没有 normal 模型，则用 erosion 模型但禁用 tail
                E_pred[normal_mask] = self._predict_subset(
                    V[normal_mask],
                    theta[normal_mask],
                    self._coef_erosion,
                    enable_tail_for_subset=False,
                )
            else:
                raise RuntimeError("No valid coef for normal subset.")

        E_pred = np.maximum(E_pred, 1e-12)  # 避免 log(0)
        return np.log(E_pred)

    def predict_energy(self, X: np.ndarray) -> np.ndarray:
        """
        与 predict 相同，但返回的是 E_pred 而非 log(E_pred)。
        """
        if not self._is_fitted:
            raise RuntimeError("PowerLawSeparableModel is not fitted.")

        if X.ndim != 2 or X.shape[1] < 1 + self._theta_dim:
            raise ValueError(
                f"X shape not compatible: got {X.shape}, "
                f"expect (n_samples, {1 + self._theta_dim})"
            )

        logV = X[:, 0]
        V = np.exp(logV)
        theta = X[:, 1:1 + self._theta_dim]

        int_bre = theta[:, 2]
        erosion_mask = np.isclose(int_bre, 0.0)
        normal_mask = ~erosion_mask

        E_pred = np.empty_like(V)

        if np.any(erosion_mask):
            if self._coef_erosion is not None:
                E_pred[erosion_mask] = self._predict_subset(
                    V[erosion_mask],
                    theta[erosion_mask],
                    self._coef_erosion,
                    enable_tail_for_subset=self.enable_tail,
                )
            elif self._coef_normal is not None:
                E_pred[erosion_mask] = self._predict_subset(
                    V[erosion_mask],
                    theta[erosion_mask],
                    self._coef_normal,
                    enable_tail_for_subset=False,
                )
            else:
                raise RuntimeError("No valid coef for erosion subset.")

        if np.any(normal_mask):
            if self._coef_normal is not None:
                E_pred[normal_mask] = self._predict_subset(
                    V[normal_mask],
                    theta[normal_mask],
                    self._coef_normal,
                    enable_tail_for_subset=False,
                )
            elif self._coef_erosion is not None:
                E_pred[normal_mask] = self._predict_subset(
                    V[normal_mask],
                    theta[normal_mask],
                    self._coef_erosion,
                    enable_tail_for_subset=False,
                )
            else:
                raise RuntimeError("No valid coef for normal subset.")

        return E_pred
