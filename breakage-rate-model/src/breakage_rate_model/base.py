# -*- coding: utf-8 -*-
"""
负责：
- 通用的度量函数 (MSE/MAE/MAPE/R2)
- 抽象基类 BaseEnergyModel:
    * fit
    * predict
    * validate
    * analyze_one_group：给定 group_index，把模型预测和原始数据画在一起
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional, Callable, List

import numpy as np
import pickle
import matplotlib.pyplot as plt

from .data_io import EnergyGroupRecord


# =============================================================================
# 度量函数
# =============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


# =============================================================================
# BaseEnergyModel
# =============================================================================

class BaseEnergyModel(ABC):
    """
    所有 E_need surrogate model 的统一接口。

    子类至少需要实现:
        - fit(X_train, y_train, **kwargs)
        - predict(X)

    validate() 提供一个统一的验证入口，返回常见指标。
    analyze_one_group() 用于对某个参数组合的曲线做“数据 vs 模型”的可视化。
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._is_fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "BaseEnergyModel":
        """
        训练模型，返回 self。

        对于像 PowerLawSeparableModel 这种更多依赖 group 数据的模型，
        可以约定：
            model.fit(None, None, groups=groups)
        并在内部走自定义逻辑。
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        输入特征矩阵 X，返回预测 y_pred（与 y_true 对齐）。

        一般约定：
            - 若模型拟合的是 log(E)，则 predict 返回 log(E_pred)；
            - 若拟合的是 E，则 predict 返回 E_pred。
        """
        ...

    def validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        metrics: Sequence[str] = ("mse", "mae", "mape", "r2"),
    ) -> Dict[str, float]:
        """
        在验证集上评估模型表现，返回 {metric_name: value} 字典。

        默认支持:
            - mse
            - mae
            - mape
            - r2
        """
        if not self._is_fitted:
            raise RuntimeError(f"Model {self._name} is not fitted yet.")

        y_pred = self.predict(X_val)

        out: Dict[str, float] = {}
        for m in metrics:
            if m == "mse":
                out["mse"] = mse(y_val, y_pred)
            elif m == "mae":
                out["mae"] = mae(y_val, y_pred)
            elif m == "mape":
                out["mape"] = mape(y_val, y_pred)
            elif m == "r2":
                out["r2"] = r2(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric '{m}'")

        return out

    # -------------------------------------------------------------------------
    # 通用的“单组可视化对比”接口
    # -------------------------------------------------------------------------
    def analyze_one_group(
        self,
        groups: Sequence[EnergyGroupRecord],
        group_index: int = 0,
        *,
        feature_fn: Optional[Callable[[EnergyGroupRecord, float], np.ndarray]] = None,
        target: str = "log_mean",
        quantile: Optional[float] = None,
        show: bool = True,
    ) -> None:
        """
        通用的“单组可视化对比”接口：
        - 从 groups 中选第 group_index 个参数组合；
        - 基于指定 target 构造 y_true（默认 log(E_mean)）；
        - 用 feature_fn 构造 X，然后调用模型 predict；
        - 在 log–log 坐标下画数据点和模型曲线。

        参数
        ----
        groups: EnergyGroupRecord 列表
        group_index: 要分析的组号（0-based）
        feature_fn:   给定 (rec, V) 返回特征向量的函数；
                      若为 None，则使用默认特征：
                          [log V, log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
        target:
            - "log_mean": log(E_mean)
            - "mean":     E_mean
            - "log_quantile": log(quantile(E_samples))
            - "quantile":     quantile(E_samples)
        quantile:
            target 为 quantile/log_quantile 时使用的分位数（0~1）。
        """
        if not self._is_fitted:
            raise RuntimeError(f"Model {self._name} is not fitted yet.")
        if not groups:
            raise RuntimeError("Empty groups list.")

        if group_index < 0 or group_index >= len(groups):
            raise IndexError(f"group_index out of range: 0 <= idx < {len(groups)}")

        rec = groups[group_index]
        V = rec.V
        E_mean = rec.E_mean

        print(f"[ANALYZE] Model={self._name}, group={group_index}, key={rec.key}")
        print(f"  NO_FRAG={rec.NO_FRAG}, int_bre={rec.int_bre}, gamma={rec.gamma}, "
              f"Df={rec.Df}, MAS={rec.MAS}, X1={rec.X1}")

        # 默认特征函数（与 datasets.build_energy_dataset 中一致）
        def default_feature_fn(r: EnergyGroupRecord, V_val: float) -> np.ndarray:
            logV = np.log(V_val)
            log_gamma = np.log(r.gamma)
            log_NOFRAG = np.log(r.NO_FRAG)
            return np.array(
                [logV, log_gamma, log_NOFRAG, r.int_bre, r.Df, r.MAS, r.X1],
                dtype=float,
            )

        feat_fn = feature_fn or default_feature_fn

        # 构造 y_true 和 X
        mask = (V > 0.0) & (E_mean > 0.0)
        V_plot = V[mask]
        if V_plot.size == 0:
            print("No positive V/E_mean data points in this group.")
            return

        if target == "log_mean":
            y_true = np.log(E_mean[mask])
        elif target == "mean":
            y_true = E_mean[mask]
        elif target in ("quantile", "log_quantile"):
            if quantile is None:
                raise ValueError("quantile must be provided for quantile targets.")
            idx_all = np.where(mask)[0]
            q_vals: List[float] = []
            for vi in idx_all:
                samples = rec.E_samples[vi, :]
                q_vals.append(float(np.quantile(samples, quantile)))
            q_vals = np.array(q_vals, dtype=float)
            y_true = np.log(q_vals) if target == "log_quantile" else q_vals
        else:
            raise ValueError(f"Unknown target '{target}'")

        X_list = [feat_fn(rec, float(v)) for v in V_plot]
        X_plot = np.vstack(X_list)

        # 模型预测
        y_pred = self.predict(X_plot)

        # 绘图：默认在 log–log 空间展示
        if show:
            fig, ax = plt.subplots(figsize=(6, 5))

            logV = np.log(V_plot)

            if target.startswith("log"):
                ax.scatter(logV, y_true, label="data (target)", alpha=0.8)
                ax.plot(logV, y_pred, label="model prediction", alpha=0.8)
                ax.set_ylabel("log E")
            else:
                ax.scatter(logV, np.log(y_true), label="data (log target)", alpha=0.8)
                ax.plot(logV, np.log(y_pred), label="model prediction (log)", alpha=0.8)
                ax.set_ylabel("log E")

            ax.set_xlabel("log V")
            ax.set_title(f"Group {group_index}: data vs model ({self._name})")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # 通用的保存 / 加载接口（pickle 整个模型实例）
    # -------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        将整个模型对象保存到磁盘。包括：
          - 已拟合的参数
          - 归一化 scaler（若有）
          - torch 子模块（MLP / ANN 的 _net）
        之后可以用 BaseEnergyModel.load(path) 或对应子类来恢复。

        注意：要求当前 Python 环境中已 import 对应模型类。
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str, device: str | None = "cpu") -> "BaseEnergyModel":
        with open(path, "rb") as f:
            model = pickle.load(f)

        # 对 torch 模型做一下设备修正（若可用）
        try:
            import torch
            if hasattr(model, "_net") and getattr(model, "_net") is not None:
                if device is None:
                    # 不强制覆盖，使用模型内部自带的 device（一般已经是 cpu）
                    dev = getattr(model, "device", torch.device("cpu"))
                else:
                    dev = torch.device(device)
                    setattr(model, "device", dev)

                model._net.to(dev)
        except Exception:
            # 没装 torch 或其它错误，忽略即可
            pass

        return model
