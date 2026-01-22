# -*- coding: utf-8 -*-
"""
Adapter: using MLPEnergyModel to provide breakage rates for MCPBE.

核心接口：
    - MLPBreakageRateAdapter.compute_rates_full(pbe)  -> np.ndarray, shape (a,)
    - MLPBreakageRateAdapter.compute_rate_single(pbe, i) -> float

其中 pbe 应该是 MCPBEBreak 或兼容的对象，至少要暴露：
    - pbe.a_tot              : 当前活跃颗粒数 a
    - pbe.dim                : 1 或 2（目前支持 1D/2D）
    - pbe.V_flat             : shape (dim, max_a)，最后一行为总体积 V
      · 若 dim == 1: V_flat[-1, :a] 为颗粒体积
      · 若 dim == 2: V_flat[0,:a] 为 v1, V_flat[1,:a] 为 v3, V = v1+v3
    - （可选）pbe.lmc_gamma, pbe.lmc_NO_FRAG, pbe.lmc_int_bre,
              pbe.lmc_Df, pbe.lmc_MAS
      若没有，则使用 adapter 初始化时传入的参数或默认值。

MLP 模型必须是 MLPEnergyModel（或兼容接口），即：
    - model.predict(X: np.ndarray) -> np.ndarray (logE_need),
      内部已经做好标准化等处理。
"""

from __future__ import annotations

from typing import Optional, Callable, Sequence

import numpy as np
import torch

from agggenerator.breakage_rate_model.base import BaseEnergyModel
from agggenerator.breakage_rate_model.mlp_model import MLPEnergyModel


class MLPBreakageRateAdapter:
    """
    用训练好的 MLPEnergyModel 给 MCPBE 提供破碎率（breakage rate）。

    使用方式（示例）：

        adapter = MLPBreakageRateAdapter(
            model_path="mlp_energy_model.pkl",
            lambda_E=1.0,          # E_in(V) = lambda_E * V^energy_exp
            energy_exp=1.0,
            # 可选：指定 LMC 参数默认值（若 pbe 上没有对应属性）
            gamma=1.0,
            NO_FRAG=4,
            int_bre=0.0,
            Df=2.5,
            MAS=0.0,
        )

        mcpbe.break_rate_adapter = adapter
        mcpbe.use_mlp_break_rate = True

    然后在 MCPBEBreak._calc_break_rates_full / _break_rate_single 中
    调用 adapter.compute_rates_full / compute_rate_single 即可。
    """

    def __init__(
        self,
        *,
        model: Optional[MLPEnergyModel] = None,
        model_path: Optional[str] = None,
        # 入射能量 E_in(V) = lambda_E * V^energy_exp  或 自定义 energy_in_fn
        lambda_E: float = 1.0,
        energy_exp: float = 1.0,
        energy_in_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        # LMC 参数的默认值（如果 pbe 没挂对应属性，就用这些）
        gamma: float = 1.0,
        NO_FRAG: float = 4.0,
        int_bre: float = 0.0,
        Df: float = 2.5,
        MAS: float = 0.0,
        # 破碎率裁剪（避免极端值/数值不稳定）
        rate_min: float = 0.0,
        rate_max: Optional[float] = None,
        eps_E: float = 1e-30,
        A0_run: float = 1.0,
    ):
        """
        参数
        ----
        model:
            已经在外部构建并训练好的 MLPEnergyModel 实例（可选）。
        model_path:
            若 model 为 None，则通过 BaseEnergyModel.load(model_path) 载入。
        lambda_E, energy_exp:
            默认入射能量模型：E_in(V) = lambda_E * V^energy_exp。
            若提供了 energy_in_fn，则忽略这两个。
        energy_in_fn:
            自定义能量输入函数，签名：energy_in_fn(V: np.ndarray) -> np.ndarray。
        gamma, NO_FRAG, int_bre, Df, MAS:
            LMC 参数的默认值，用于从 pbe 中取不到对应属性时的 fallback。
        rate_min, rate_max:
            对最终速率做 clip；rate_max 为 None 时只做下截断。
        eps_E:
            避免 E_need 为 0 时的除零。
        """
        if model is None:
            if model_path is None:
                raise ValueError("Either 'model' or 'model_path' must be provided.")
            loaded = BaseEnergyModel.load(model_path, device="cpu")
            if not isinstance(loaded, MLPEnergyModel):
                raise TypeError(
                    f"Loaded model from {model_path} is not MLPEnergyModel (got {type(loaded)})"
                )
            # 双保险：确保 device 和 _net 都在 CPU
            loaded.device = torch.device("cpu")
            if getattr(loaded, "_net", None) is not None:
                loaded._net.to("cpu")
            self.model: MLPEnergyModel = loaded
        else:
            # 外部传进来的模型也统一迁移到 CPU
            model.device = torch.device("cpu")
            if getattr(model, "_net", None) is not None:
                model._net.to("cpu")
            self.model = model

        self.lambda_E = float(lambda_E)
        self.energy_exp = float(energy_exp)
        self.energy_in_fn = energy_in_fn

        # LMC 默认参数
        self.gamma_default = float(gamma)
        self.NO_FRAG_default = float(NO_FRAG)
        self.int_bre_default = float(int_bre)
        self.Df_default = float(Df)
        self.MAS_default = float(MAS)

        self.rate_min = float(rate_min)
        self.rate_max = None if rate_max is None else float(rate_max)
        self.eps_E = float(eps_E)
        self.A0_run = A0_run

    # ------------------------------------------------------------------
    # 内部工具：从 pbe 获取 LMC 参数（若没有则用默认）
    # ------------------------------------------------------------------
    def _get_lmc_params_from_pbe(self, pbe) -> tuple[float, float, float, float, float]:
        gamma = float(getattr(pbe, "lmc_gamma", self.gamma_default))
        NO_FRAG = float(getattr(pbe, "lmc_NO_FRAG", self.NO_FRAG_default))
        int_bre = float(getattr(pbe, "lmc_int_bre", self.int_bre_default))
        Df = float(getattr(pbe, "lmc_Df", self.Df_default))
        MAS = float(getattr(pbe, "lmc_MAS", self.MAS_default))
        return gamma, NO_FRAG, int_bre, Df, MAS

    # ------------------------------------------------------------------
    # 内部工具：入射能量 E_in(V)
    # ------------------------------------------------------------------
    def _energy_in(self, V: np.ndarray) -> np.ndarray:
        """
        入射能量模型：

            若 energy_in_fn 不为 None：
                E_in(V) = energy_in_fn(V)
            否则：
                E_in(V) = lambda_E * V^energy_exp
        """
        V = np.asarray(V, dtype=float)
        if self.energy_in_fn is not None:
            E_in = self.energy_in_fn(V)
            return np.asarray(E_in, dtype=float)
        # 默认：power law in V
        return self.lambda_E * (V**self.energy_exp)

    # ------------------------------------------------------------------
    # 内部工具：为一批粒子构造特征 X
    # ------------------------------------------------------------------
    def _build_features_batch(
        self,
        pbe,
        indices: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        为给定 pbe 中的若干粒子（indices）构造特征矩阵 X。

        返回:
            X   : shape (n, 7) 特征
            V   : shape (n,)  对应颗粒体积（便于后续算 E_in）
        """
        dim = int(getattr(pbe, "dim", 1))
        a = int(getattr(pbe, "a_tot", 0))
        if a <= 0:
            return np.zeros((0, 7), dtype=float), np.zeros((0,), dtype=float)

        V_flat = np.asarray(pbe.V_flat, dtype=float)

        if indices is None:
            idx = np.arange(a, dtype=int)
        else:
            idx = np.asarray(indices, dtype=int)
            # 简单 sanity check
            idx = idx[(idx >= 0) & (idx < a)]
        if idx.size == 0:
            return np.zeros((0, 7), dtype=float), np.zeros((0,), dtype=float)

        # V & 组分体积分数 X1
        if dim == 1:
            V = V_flat[-1, idx]
            X1 = np.ones_like(V)
        elif dim == 2:
            v1 = V_flat[0, idx]
            v3 = V_flat[1, idx]
            A = v1 + v3
            V = A
            X1 = np.where(A > 0.0, v1 / A, 0.5)
        else:
            raise NotImplementedError(f"MLPBreakageRateAdapter currently only supports dim=1 or 2 (got dim={dim})")
        
        V /= self.A0_run
        gamma, NO_FRAG, int_bre, Df, MAS = self._get_lmc_params_from_pbe(pbe)

        # 构造特征
        logV = np.log(np.maximum(V, 1e-30))
        log_gamma = np.log(gamma)
        log_NOFRAG = np.log(NO_FRAG)

        # 标量 → 向量广播
        log_gamma_v = np.full_like(logV, log_gamma, dtype=float)
        log_nf_v = np.full_like(logV, log_NOFRAG, dtype=float)
        int_bre_v = np.full_like(logV, int_bre, dtype=float)
        Df_v = np.full_like(logV, Df, dtype=float)
        MAS_v = np.full_like(logV, MAS, dtype=float)

        X = np.stack(
            [logV, log_gamma_v, log_nf_v, int_bre_v, Df_v, MAS_v, X1],
            axis=1,
        )  # shape (n, 7)

        return X, V

    # ------------------------------------------------------------------
    # 公共接口：全表破碎率
    # ------------------------------------------------------------------
    def compute_rates_full(self, pbe) -> np.ndarray:
        """
        计算当前 pbe 中所有活跃颗粒的破碎率数组，shape = (a_tot,)。

        步骤：
          1. 根据 pbe.V_flat, pbe.dim 等构造特征 X；
          2. 调用 MLP 模型预测 logE_need；
          3. 计算 E_in(V)；
          4. 计算 rate = E_in / E_need，并做裁剪。
        """
        X, V = self._build_features_batch(pbe, indices=None)
        n = X.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=float)

        # 预测 logE_need
        logE_need = self.model.predict(X)           # shape (n,)
        E_need = np.exp(np.asarray(logE_need, dtype=float))
        E_need = np.maximum(E_need, self.eps_E)

        # 入射能量
        E_in = self._energy_in(V)

        # 破碎率
        rates = E_in / E_need
        rates = np.maximum(rates, self.rate_min)
        if self.rate_max is not None:
            rates = np.minimum(rates, self.rate_max)

        return rates

    # ------------------------------------------------------------------
    # 公共接口：单颗粒破碎率
    # ------------------------------------------------------------------
    def compute_rate_single(self, pbe, i: int) -> float:
        """
        计算 pbe 中第 i 个颗粒的破碎率。

        通常用于：
          - 破碎事件之后，新生成颗粒的局部更新
          - 某个颗粒体积/组分发生变化后的局部更新
        """
        a = int(getattr(pbe, "a_tot", 0))
        if i < 0 or i >= a:
            return 0.0

        X, V = self._build_features_batch(pbe, indices=[i])
        if X.shape[0] == 0:
            return 0.0

        logE_need = self.model.predict(X[0:1, :])   # shape (1,)
        E_need = float(np.exp(logE_need[0]))
        if E_need < self.eps_E:
            E_need = self.eps_E

        E_in = float(self._energy_in(np.array([V[0]], dtype=float))[0])
        rate = E_in / E_need
        if rate < self.rate_min:
            rate = self.rate_min
        if self.rate_max is not None and rate > self.rate_max:
            rate = self.rate_max

        return float(rate)
