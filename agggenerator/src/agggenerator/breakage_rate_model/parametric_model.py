# -*- coding: utf-8 -*-
"""
Parametric + correction 模型（带 erosion / normal 分簇 & plateau 加权）

    log E(V, θ) = log E_trend(V, θ) + δ(V, θ)

其中：
  - E_trend 使用与 PowerLawSeparableModel 相同的“两段平滑幂律 + 可选尾巴 α”结构；
  - trend 的参数 (σ, Vc, Emax[, α]) 由 θ 通过线性映射得到；
  - δ(V, θ) 为残差校正项，按簇（erosion / normal）分别训练线性 / Ridge 模型，
    在 normal 簇中对 plateau 组进行加权。

特征约定：
    X[i] = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
"""

from __future__ import annotations

from typing import Optional, Sequence, List, Tuple, Dict

import numpy as np

from .data_io import EnergyGroupRecord
from .base import BaseEnergyModel
from .powerlaw_separable import (
    fit_two_segment_params,
    smooth_powerlaw_plateau_with_tail,
    compute_local_slopes,
    find_plateau_transition,
    split_groups_by_int_bre,
)

# =============================================================================
# ParametricEnergyModel
# =============================================================================


class ParametricEnergyModel(BaseEnergyModel):
    """
    Parametric + correction 模型：

        log E(V, θ) = log E_trend(V, θ) + δ(V, θ)

    - trend 使用两段平滑幂律：
        erosion 簇 (int_bre == 0):
            · 可以在 enable_tail=True 时带侵蚀尾巴 α
        normal 簇 (int_bre != 0):
            · 始终不带尾巴（α=0）

    - θ→参数 线性回归：
        erosion / normal 两簇分别回归；
        normal 簇中 plateau 组在回归中使用权重 plateau_weight (>1)。

    - 残差 δ：
        erosion / normal 两簇分别训练；
        normal 簇中 plateau 组同样使用 plateau_weight 加权。
    """

    def __init__(
        self,
        *,
        pearson_min: Optional[float] = None,
        frac_threshold: float = 0.3,
        abs_threshold: float = 0.1,
        fit_mode: str = "logE",
        pure_powerlaw_if_no_plateau: bool = True,
        residual_type: str = "ridge",   # "linear" / "ridge" / "none"
        residual_lambda: float = 1e-2,
        enable_tail: bool = True,       # 仅对 erosion 簇有效
        plateau_weight: float = 3.0,    # normal 簇 plateau 组的权重 (>=1)
        tol_int_bre: float = 1e-12,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "ParametricEnergyModel")

        # ---- 主趋势（two-segment）相关超参数 ----
        self.pearson_min = pearson_min
        self.frac_threshold = frac_threshold
        self.abs_threshold = abs_threshold
        self.fit_mode = fit_mode
        self.pure_powerlaw_if_no_plateau = pure_powerlaw_if_no_plateau

        self.enable_tail = enable_tail
        self.plateau_weight = max(float(plateau_weight), 1.0)
        self.tol_int_bre = tol_int_bre

        # ---- 残差模型相关超参数 ----
        self.residual_type = residual_type  # "linear", "ridge", "none"
        self.residual_lambda = residual_lambda

        # θ 维度固定为 6: [log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
        self._theta_dim: int = 6

        # ---- trend: θ→参数 的线性回归系数（erosion / normal 两簇） ----
        # erosion: params = [σ, logVc, logEmax, (α)]
        self._coef_trend_erosion: Optional[np.ndarray] = None  # (d+1, k_e)
        # normal:  params = [σ, logVc, logEmax]
        self._coef_trend_normal: Optional[np.ndarray] = None   # (d+1, 3)

        # 记录 plateau 标记（按 group key）
        self._plateau_flag_erosion: Dict[object, bool] = {}
        self._plateau_flag_normal: Dict[object, bool] = {}

        # ---- 残差模型 δ: erosion / normal 两套系数 ----
        # residual 特征维度: p = 1 + θ_dim  (logV + θ)
        self._residual_dim: int = 1 + self._theta_dim
        self._coef_residual_erosion: Optional[np.ndarray] = None  # (p+1,)
        self._coef_residual_normal: Optional[np.ndarray] = None   # (p+1,)

        # 残差特征标准化参数： (mean, std)，按簇分别记录
        self._residual_scaler_erosion: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._residual_scaler_normal: Optional[Tuple[np.ndarray, np.ndarray]] = None


    # =====================================================================
    #  主趋势：group-level 拟合 + θ→参数 线性回归（按簇 & plateau 加权）
    # =====================================================================

    def _collect_trend_params(
        self,
        groups: Sequence[EnergyGroupRecord],
        *,
        enable_tail_for_subset: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[object]]]:
        """
        对给定一簇 groups:
            - 过滤 pearson_min
            - 对每个 group 调用 fit_two_segment_params 获取 (σ, Vc, Emax[, α], Vc_local)
            - 生成:
                Theta        : (n_groups, d)
                Params       : (n_groups, n_param)
                plateau_mask : (n_groups,), bool (Vc_local 有效视为 plateau)
                keys         : group.key 列表
        """
        theta_list: List[np.ndarray] = []
        param_list: List[np.ndarray] = []
        plateau_flags: List[bool] = []
        keys: List[object] = []

        for rec in groups:
            # pearson 过滤
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

            # θ 特征
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

            plateau_flags.append(np.isfinite(Vc_local))
            keys.append(rec.key)

        if not theta_list:
            return None, None, None, None

        Theta = np.vstack(theta_list)
        Params = np.vstack(param_list)
        plateau_mask = np.array(plateau_flags, dtype=bool)
        return Theta, Params, plateau_mask, keys

    def _fit_trend_from_groups(
        self,
        groups: Sequence[EnergyGroupRecord],
    ) -> None:
        """
        trend 拟合：
            1. 按 int_bre 分为 erosion / normal 簇；
            2. 各簇内：
               - 对每个 group 用两段(+尾巴)模型拟合参数；
               - 用 θ→参数 做线性回归；
               - normal 簇中 plateau 组使用权重 plateau_weight。
        """
        erosion_groups, normal_groups = split_groups_by_int_bre(
            groups, tol=self.tol_int_bre
        )

        # erosion 簇：只有在 enable_tail=True 时拟合 α
        Theta_e, Y_e, plateau_e, keys_e = self._collect_trend_params(
            erosion_groups,
            enable_tail_for_subset=self.enable_tail,
        )

        # normal 簇：不带尾巴
        Theta_n, Y_n, plateau_n, keys_n = self._collect_trend_params(
            normal_groups,
            enable_tail_for_subset=False,
        )

        # 记录 plateau 标记字典
        self._plateau_flag_erosion = {}
        self._plateau_flag_normal = {}
        if keys_e is not None and plateau_e is not None:
            self._plateau_flag_erosion = {
                k: bool(f) for k, f in zip(keys_e, plateau_e)
            }
        if keys_n is not None and plateau_n is not None:
            self._plateau_flag_normal = {
                k: bool(f) for k, f in zip(keys_n, plateau_n)
            }

        if Theta_e is None and Theta_n is None:
            raise RuntimeError("ParametricEnergyModel: no valid groups to train trend on.")

        self._theta_dim = 6  # 固定

        # ---- erosion 簇：普通 least-squares ----
        if Theta_e is not None:
            n_e, d_e = Theta_e.shape
            A_e = np.column_stack([np.ones(n_e), Theta_e])  # (n_e, d_e+1)
            coef_e, *_ = np.linalg.lstsq(A_e, Y_e, rcond=None)
            self._coef_trend_erosion = coef_e
        else:
            self._coef_trend_erosion = None

        # ---- normal 簇：plateau 组加权 ----
        if Theta_n is not None:
            n_n, d_n = Theta_n.shape
            A_n = np.column_stack([np.ones(n_n), Theta_n])  # (n_n, d_n+1)
            if plateau_n is not None and plateau_n.size == n_n and self.plateau_weight > 1.0:
                w = np.ones(n_n, dtype=float)
                w[plateau_n] = self.plateau_weight
                sqrt_w = np.sqrt(w)[:, None]  # (n_n,1)

                A_w = A_n * sqrt_w
                Y_w = Y_n * sqrt_w
                coef_n, *_ = np.linalg.lstsq(A_w, Y_w, rcond=None)
            else:
                coef_n, *_ = np.linalg.lstsq(A_n, Y_n, rcond=None)
            self._coef_trend_normal = coef_n
        else:
            self._coef_trend_normal = None

        if self._coef_trend_erosion is None and self._coef_trend_normal is None:
            raise RuntimeError("ParametricEnergyModel: trend fit failed for both erosion and normal subsets.")

    # ---------------------------------------------------------------------
    # trend 参数预测
    # ---------------------------------------------------------------------

    def _predict_trend_params(
        self,
        theta: np.ndarray,
        is_erosion: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        给定 theta (n,d) 和是否 erosion，预测 trend 参数:
            sigma, Vc, Emax, alpha
        其中：
            - 对 normal 簇 alpha ≡ 0
            - 对 erosion 簇，在 enable_tail=True 且模型包含 α 时 alpha 才非零。
        """
        n, d = theta.shape
        if d != self._theta_dim:
            raise ValueError(f"theta dimension mismatch: expected {self._theta_dim}, got {d}")

        if is_erosion:
            coef = self._coef_trend_erosion
        else:
            coef = self._coef_trend_normal

        if coef is None:
            raise RuntimeError("Trend coefficients for the requested subset are not fitted.")

        A = np.column_stack([np.ones(n), theta])  # (n, d+1)
        Y = A @ coef                              # (n, n_param)

        sigma = Y[:, 0]
        Vc = np.exp(Y[:, 1])
        Emax = np.exp(Y[:, 2])

        if is_erosion and self.enable_tail and Y.shape[1] >= 4:
            alpha = Y[:, 3]
        else:
            alpha = np.zeros_like(sigma)

        return sigma, Vc, Emax, alpha

    # =====================================================================
    #  残差模型拟合：δ = logE_true - logE_trend（按簇 & plateau 加权）
    # =====================================================================

    def _fit_residual_model(
        self,
        groups: Sequence[EnergyGroupRecord],
    ) -> None:
        """
        在当前 trend 模型下构建残差数据集，并按 erosion / normal 两簇
        分别拟合 δ([logV, θ])。normal 簇中 plateau 组使用 plateau_weight 加权。

        若 residual_type 为 "none"，则不拟合残差模型。
        """
        rtype = (self.residual_type or "none").lower()
        if rtype == "none":
            self._coef_residual_erosion = None
            self._coef_residual_normal = None
            self._residual_scaler_erosion = None
            self._residual_scaler_normal = None
            return

        if self._coef_trend_erosion is None and self._coef_trend_normal is None:
            raise RuntimeError("Trend must be fitted before fitting residual model.")

        X_e_list: List[np.ndarray] = []
        y_e_list: List[float] = []
        w_e_list: List[float] = []

        X_n_list: List[np.ndarray] = []
        y_n_list: List[float] = []
        w_n_list: List[float] = []

        for rec in groups:
            # pearson 过滤
            if self.pearson_min is not None:
                r_g = rec.pearson_r_fit
                if np.isfinite(r_g) and r_g < self.pearson_min:
                    continue

            is_erosion = abs(rec.int_bre) <= self.tol_int_bre

            # θ 特征
            log_gamma = np.log(rec.gamma)
            log_NOFRAG = np.log(rec.NO_FRAG)
            theta_vec = np.array(
                [log_gamma, log_NOFRAG, rec.int_bre, rec.Df, rec.MAS, rec.X1],
                dtype=float,
            )

            # trend 参数
            sigma_g, Vc_g, Emax_g, alpha_g = self._predict_trend_params(
                theta_vec[None, :],
                is_erosion=is_erosion,
            )
            sigma_g = sigma_g[0]
            Vc_g = Vc_g[0]
            Emax_g = Emax_g[0]
            alpha_g = alpha_g[0]

            # 当前 group 是否被判为 plateau（用于 normal 簇加权）
            if is_erosion:
                plateau_flag = self._plateau_flag_erosion.get(rec.key, False)
            else:
                plateau_flag = self._plateau_flag_normal.get(rec.key, False)

            V = rec.V
            E_mean = rec.E_mean
            mask = (V > 0.0) & (E_mean > 0.0)
            V_sel = V[mask]
            E_sel = E_mean[mask]
            if V_sel.size == 0:
                continue

            # trend 预测
            E_trend = smooth_powerlaw_plateau_with_tail(
                V_sel,
                sigma_g,
                Vc_g,
                Emax_g,
                alpha_g,
                enable_tail=(self.enable_tail and is_erosion),
            )
            E_trend = np.maximum(E_trend, 1e-12)

            logE_true = np.log(E_sel)
            logE_trend = np.log(E_trend)
            delta = logE_true - logE_trend  # 残差

            logV_sel = np.log(V_sel)
            for lv, dlt in zip(logV_sel, delta):
                feat = np.array(
                    [lv, log_gamma, log_NOFRAG, rec.int_bre, rec.Df, rec.MAS, rec.X1],
                    dtype=float,
                )
                if is_erosion:
                    X_e_list.append(feat)
                    y_e_list.append(float(dlt))
                    w_e_list.append(1.0)  # 暂时 erosion 不区分 plateau
                else:
                    X_n_list.append(feat)
                    y_n_list.append(float(dlt))
                    # normal 簇 plateau 组加权
                    w = self.plateau_weight if plateau_flag else 1.0
                    w_n_list.append(w)

        # ----- erosion 簇残差模型 -----
        self._residual_scaler_erosion = None
        if X_e_list:
            X_e = np.vstack(X_e_list)              # (n_e, p)
            y_e = np.array(y_e_list, dtype=float)  # (n_e,)
            w_e = np.array(w_e_list, dtype=float)  # (n_e,)

            n_e, p = X_e.shape
            self._residual_dim = p

            # 标准化特征：mean/std 存入 scaler
            mu_e = X_e.mean(axis=0)
            std_e = X_e.std(axis=0)
            std_e[std_e < 1e-12] = 1.0
            X_e_scaled = (X_e - mu_e) / std_e
            self._residual_scaler_erosion = (mu_e, std_e)

            A_e = np.column_stack([np.ones(n_e), X_e_scaled])  # (n_e, p+1)

            if np.all(w_e == 1.0):
                # 无加权
                if rtype == "linear":
                    coef_e, *_ = np.linalg.lstsq(A_e, y_e, rcond=None)
                elif rtype == "ridge":
                    lam = float(self.residual_lambda)
                    ATA = A_e.T @ A_e
                    cond = np.linalg.cond(ATA)
                    print(cond)
                    reg = np.eye(p + 1)
                    reg[0, 0] = 0.0  # 不正则化偏置
                    coef_e = np.linalg.solve(ATA + lam * reg, A_e.T @ y_e)
                else:
                    raise ValueError(f"Unknown residual_type '{self.residual_type}'")
            else:
                # 加权 least-squares / ridge
                sqrt_w = np.sqrt(w_e)[:, None]
                A_w = A_e * sqrt_w
                y_w = y_e * sqrt_w[:, 0]
                if rtype == "linear":
                    coef_e, *_ = np.linalg.lstsq(A_w, y_w, rcond=None)
                elif rtype == "ridge":
                    lam = float(self.residual_lambda)
                    ATA = A_w.T @ A_w
                    reg = np.eye(p + 1)
                    reg[0, 0] = 0.0
                    coef_e = np.linalg.solve(ATA + lam * reg, A_w.T @ y_w)
                else:
                    raise ValueError(f"Unknown residual_type '{self.residual_type}'")

            self._coef_residual_erosion = coef_e
        else:
            self._coef_residual_erosion = None

        # ----- normal 簇残差模型 -----
        self._residual_scaler_normal = None
        if X_n_list:
            X_n = np.vstack(X_n_list)              # (n_n, p)
            y_n = np.array(y_n_list, dtype=float)  # (n_n,)
            w_n = np.array(w_n_list, dtype=float)  # (n_n,)

            n_n, p = X_n.shape
            self._residual_dim = p

            # 标准化特征
            mu_n = X_n.mean(axis=0)
            std_n = X_n.std(axis=0)
            std_n[std_n < 1e-12] = 1.0
            X_n_scaled = (X_n - mu_n) / std_n
            self._residual_scaler_normal = (mu_n, std_n)

            A_n = np.column_stack([np.ones(n_n), X_n_scaled])  # (n_n, p+1)

            if np.all(w_n == 1.0):
                if rtype == "linear":
                    coef_n, *_ = np.linalg.lstsq(A_n, y_n, rcond=None)
                elif rtype == "ridge":
                    lam = float(self.residual_lambda)
                    ATA = A_n.T @ A_n
                    reg = np.eye(p + 1)
                    reg[0, 0] = 0.0
                    coef_n = np.linalg.solve(ATA + lam * reg, A_n.T @ y_n)
                else:
                    raise ValueError(f"Unknown residual_type '{self.residual_type}'")
            else:
                sqrt_w = np.sqrt(w_n)[:, None]
                A_w = A_n * sqrt_w
                y_w = y_n * sqrt_w[:, 0]
                if rtype == "linear":
                    coef_n, *_ = np.linalg.lstsq(A_w, y_w, rcond=None)
                elif rtype == "ridge":
                    lam = float(self.residual_lambda)
                    ATA = A_w.T @ A_w
                    cond = np.linalg.cond(ATA)
                    print(cond)
                    reg = np.eye(p + 1)
                    reg[0, 0] = 0.0
                    coef_n = np.linalg.solve(ATA + lam * reg, A_w.T @ y_w)
                else:
                    raise ValueError(f"Unknown residual_type '{self.residual_type}'")

            self._coef_residual_normal = coef_n
        else:
            self._coef_residual_normal = None


    # =====================================================================
    #  公共训练接口
    # =====================================================================

    def fit_from_groups(
        self,
        groups: Sequence[EnergyGroupRecord],
    ) -> "ParametricEnergyModel":
        """
        两阶段训练：
            1) trend：按 erosion / normal 分簇，两段(+尾巴)模型 + plateau 加权；
            2) residual：在 logE 空间构建残差 δ，并按簇拟合线性 / Ridge 模型。
        """
        self._fit_trend_from_groups(groups)
        self._fit_residual_model(groups)
        self._is_fitted = True
        return self

    def fit(
        self,
        X_train: np.ndarray | None,
        y_train: np.ndarray | None = None,
        **kwargs,
    ) -> "ParametricEnergyModel":
        """
        为了兼容 BaseEnergyModel 抽象接口，提供 fit 包装。

        使用方式：
            model.fit(None, None, groups=groups)
        """
        groups: Optional[Sequence[EnergyGroupRecord]] = kwargs.get("groups", None)
        if groups is None:
            raise ValueError("ParametricEnergyModel.fit requires 'groups' argument.")
        return self.fit_from_groups(groups)

    # =====================================================================
    #  预测接口
    # =====================================================================

    def _predict_residual_subset(
        self,
        X_res: np.ndarray,
        is_erosion: bool,
    ) -> np.ndarray:
        """
        给定 residual 特征 (logV, θ) 和是否 erosion，预测 δ。
        在这里对特征使用训练阶段记录的 mean/std 做标准化。
        """
        if X_res.ndim != 2 or X_res.shape[1] != self._residual_dim:
            raise ValueError(
                f"Residual feature dimension mismatch: expected {self._residual_dim}, got {X_res.shape[1]}"
            )

        # 选择对应簇的系数与 scaler
        if is_erosion:
            coef = self._coef_residual_erosion
            scaler = self._residual_scaler_erosion
        else:
            coef = self._coef_residual_normal
            scaler = self._residual_scaler_normal

        if coef is None or scaler is None:
            # 没有残差模型或没有 scaler：返回 0
            return np.zeros(X_res.shape[0], dtype=float)

        mu, std = scaler
        std_safe = np.where(std < 1e-12, 1.0, std)
        X_scaled = (X_res - mu) / std_safe

        n = X_scaled.shape[0]
        A = np.column_stack([np.ones(n), X_scaled])  # (n, p+1)
        return A @ coef


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        输入特征矩阵 X：

            X[i] = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1]

        输出：
            y_pred[i] = log(E_pred(V_i, θ_i))
        """
        if not self._is_fitted:
            raise RuntimeError("ParametricEnergyModel is not fitted.")

        if X.ndim != 2 or X.shape[1] < 1 + self._theta_dim:
            raise ValueError(
                f"X shape not compatible: got {X.shape}, "
                f"expect (n_samples, {1 + self._theta_dim})"
            )

        logV = X[:, 0]
        V = np.exp(logV)
        theta = X[:, 1:1 + self._theta_dim]
        int_bre = theta[:, 2]

        erosion_mask = np.isclose(int_bre, 0.0, atol=self.tol_int_bre)
        normal_mask = ~erosion_mask

        logE_pred = np.empty_like(logV)

        # erosion subset
        if np.any(erosion_mask):
            theta_e = theta[erosion_mask]
            V_e = V[erosion_mask]
            sigma_e, Vc_e, Emax_e, alpha_e = self._predict_trend_params(
                theta_e,
                is_erosion=True,
            )
            E_trend_e = smooth_powerlaw_plateau_with_tail(
                V_e,
                sigma_e,
                Vc_e,
                Emax_e,
                alpha_e,
                enable_tail=self.enable_tail,
            )
            E_trend_e = np.maximum(E_trend_e, 1e-12)
            logE_trend_e = np.log(E_trend_e)

            X_res_e = X[erosion_mask, :1 + self._theta_dim]
            delta_e = self._predict_residual_subset(X_res_e, is_erosion=True)
            logE_pred[erosion_mask] = logE_trend_e + delta_e

        # normal subset
        if np.any(normal_mask):
            theta_n = theta[normal_mask]
            V_n = V[normal_mask]
            sigma_n, Vc_n, Emax_n, alpha_n = self._predict_trend_params(
                theta_n,
                is_erosion=False,
            )
            E_trend_n = smooth_powerlaw_plateau_with_tail(
                V_n,
                sigma_n,
                Vc_n,
                Emax_n,
                alpha_n,
                enable_tail=False,  # normal 簇无尾巴
            )
            E_trend_n = np.maximum(E_trend_n, 1e-12)
            logE_trend_n = np.log(E_trend_n)

            X_res_n = X[normal_mask, :1 + self._theta_dim]
            delta_n = self._predict_residual_subset(X_res_n, is_erosion=False)
            logE_pred[normal_mask] = logE_trend_n + delta_n

        return logE_pred

    def predict_energy(self, X: np.ndarray) -> np.ndarray:
        """
        与 predict 相同，但返回 E_pred 而非 log(E_pred)。
        """
        logE_pred = self.predict(X)
        return np.exp(logE_pred)
