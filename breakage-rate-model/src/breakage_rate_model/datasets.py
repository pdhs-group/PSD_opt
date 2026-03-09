# -*- coding: utf-8 -*-
"""
负责：
- 基于 EnergyGroupRecord 构造用于不同拟合任务的 X, y 数据集
  * SigmaDataset：用于 σ(θ), b(θ) 的模型（幂律/可分离结构）
  * EnergyDataset：用于直接拟合 E_need 的模型（方案 2/3/4）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Callable, Tuple

import numpy as np

from .data_io import EnergyGroupRecord


# =============================================================================
# 1. 方案 1：σ(θ) 模型 – 每个 group → 一个样本
# =============================================================================


@dataclass
class SigmaDataset:
    """
    用于方案 1（幂律可分离结构）的数据集:
        每个 group 一条样本，特征只包含 θ（不含 V）
    """
    X: np.ndarray          # shape (n_groups, n_features)
    y_sigma: np.ndarray    # shape (n_groups,)
    y_b: np.ndarray        # shape (n_groups,)
    meta: List[EnergyGroupRecord]


def _theta_features(rec: EnergyGroupRecord) -> np.ndarray:
    """
    从 EnergyGroupRecord 提取 θ 特征：
        θ = [log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
    """
    log_gamma = np.log(rec.gamma)
    log_NOFRAG = np.log(rec.NO_FRAG)
    return np.array(
        [log_gamma, log_NOFRAG, rec.int_bre, rec.Df, rec.MAS, rec.X1],
        dtype=float,
    )


def build_sigma_dataset(
    groups: Sequence[EnergyGroupRecord],
    pearson_min: float = 0.95,
    use_attr_if_available: bool = True,
) -> SigmaDataset:
    """
    构建用于 σ(θ), b(θ) 拟合的数据集。

    - 每个 group -> 一条样本
    - 特征 θ = [log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
    - 目标:
        - sigma: 优先用 attrs['sigma'] (如果 use_attr_if_available=True)，否则用 sigma_fit
        - b:     使用 b_fit
    - 过滤: 只保留 pearson_r >= pearson_min 的 group
      （优先用 attr['pearson_r']，否则使用 pearson_r_fit）
    """
    X_list: List[np.ndarray] = []
    y_sigma_list: List[float] = []
    y_b_list: List[float] = []
    meta_list: List[EnergyGroupRecord] = []

    for rec in groups:
        # 选择用于判断幂律质量的 r
        r_use = rec.pearson_r_attr if (use_attr_if_available and rec.pearson_r_attr is not None) else rec.pearson_r_fit
        if np.isnan(r_use) or r_use < pearson_min:
            # 这个参数组合下 logE ~ logV 幂律不好，丢掉
            continue

        theta_vec = _theta_features(rec)

        # 目标 sigma
        if use_attr_if_available and rec.sigma_attr is not None:
            sigma = rec.sigma_attr
        else:
            sigma = rec.sigma_fit

        b = rec.b_fit  # 截距

        X_list.append(theta_vec)
        y_sigma_list.append(float(sigma))
        y_b_list.append(float(b))
        meta_list.append(rec)

    if not X_list:
        raise RuntimeError("No groups passed the pearson_min filter for sigma dataset.")

    X = np.vstack(X_list)
    y_sigma = np.array(y_sigma_list, dtype=float)
    y_b = np.array(y_b_list, dtype=float)

    return SigmaDataset(X=X, y_sigma=y_sigma, y_b=y_b, meta=meta_list)


# =============================================================================
# 2. 方案 2/3/4：直接拟合 E_need – 每个 (group, Np) 或 (group, Np, sample) → 一条样本
# =============================================================================


@dataclass
class EnergyDataset:
    """
    用于方案 2/3/4（直接拟合 E）的数据集：
        - 若 per_sample=False: 每个 group 的每个 Np -> 一条样本
        - 若 per_sample=True:  每个 MC 样本 -> 一条样本（展平 E_samples）
    """
    X: np.ndarray          # 特征矩阵 shape (n_samples, n_features)
    y: np.ndarray          # 目标向量 shape (n_samples,)
    meta_idx: np.ndarray   # shape (n_samples, 3)，记录 (group_idx, np_index, sample_index or -1)
    groups: List[EnergyGroupRecord]


def _energy_features(rec: EnergyGroupRecord, V_val: float) -> np.ndarray:
    """
    ???????? E ???:
        [log V, log gamma, log NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]
    """
    logV = np.log(V_val)
    log_gamma = np.log(rec.gamma)
    log_NOFRAG = np.log(rec.NO_FRAG)
    str_values = np.asarray(rec.STR, dtype=float).reshape(-1)
    if str_values.size != 3:
        raise ValueError(
            f"EnergyDataset expects STR to have length 3, got shape={np.asarray(rec.STR).shape}"
        )
    return np.array(
        [
            logV,
            log_gamma,
            log_NOFRAG,
            rec.int_bre,
            rec.Df,
            rec.MAS,
            rec.X1,
            str_values[0],
            str_values[1],
            str_values[2],
        ],
        dtype=float,
    )
def build_energy_dataset(
    groups: Sequence[EnergyGroupRecord],
    *,
    per_sample: bool = False,
    target: str = "log_mean",
    quantile: Optional[float] = None,
    feature_fn: Optional[Callable[[EnergyGroupRecord, float], np.ndarray]] = None,
) -> EnergyDataset:
    """
    构建用于直接拟合 E_need 的数据集。

    参数
    ----
    groups:
        EnergyGroupRecord 列表
    per_sample:
        False: 每个 (group, Np) -> 使用 E_mean 的一条样本；
        True:  每个 (group, Np, sample) -> 使用 E_samples 展平，样本更多。
    target:
        指定 y 的含义：
        - "log_mean": log(E_mean)
        - "mean":     E_mean
        - "log_quantile": log(quantile(E_samples))
        - "quantile":     quantile(E_samples)
    quantile:
        当 target 为 quantile/log_quantile 时使用的分位数（0~1）。
    feature_fn:
        自定义特征构造函数：feature_fn(record, V) -> feature_vector
        如果为 None，则使用默认特征：
            [log V, log gamma, log NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]
    返回
    ----
    EnergyDataset(X, y, meta_idx, groups)
    """
    if target in ("quantile", "log_quantile") and quantile is None:
        raise ValueError("quantile must be provided when target is 'quantile' or 'log_quantile'.")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meta_idx_list: List[Tuple[int, int, int]] = []  # (group_idx, np_index, sample_index or -1)

    feat_fn = feature_fn or _energy_features

    for gi, rec in enumerate(groups):
        n_Np = rec.V.shape[0]

        if not per_sample:
            # 每个 Np 一条样本，使用 E_mean 或分位数
            for vi in range(n_Np):
                V_val = float(rec.V[vi])
                if V_val <= 0.0:
                    continue

                if target == "log_mean":
                    y_val = np.log(rec.E_mean[vi])
                elif target == "mean":
                    y_val = float(rec.E_mean[vi])
                elif target in ("quantile", "log_quantile"):
                    samples = rec.E_samples[vi, :]
                    q_val = float(np.quantile(samples, quantile))
                    y_val = np.log(q_val) if target == "log_quantile" else q_val
                else:
                    raise ValueError(f"Unknown target '{target}'")

                X_vec = feat_fn(rec, V_val)
                X_list.append(X_vec)
                y_list.append(y_val)
                meta_idx_list.append((gi, vi, -1))

        else:
            # 每个 MC 样本一条数据，使用 E_samples 展平
            for vi in range(n_Np):
                V_val = float(rec.V[vi])
                if V_val <= 0.0:
                    continue
                samples = rec.E_samples[vi, :]
                for si, e_val in enumerate(samples):
                    if target == "log_mean":
                        # per_sample 时 "log_mean" 就是 log(E_sample)
                        y_val = np.log(e_val)
                    elif target == "mean":
                        y_val = float(e_val)
                    elif target == "quantile":
                        y_val = float(e_val)
                    elif target == "log_quantile":
                        y_val = np.log(e_val)
                    else:
                        raise ValueError(f"Unknown target '{target}'")

                    X_vec = feat_fn(rec, V_val)
                    X_list.append(X_vec)
                    y_list.append(y_val)
                    meta_idx_list.append((gi, vi, si))

    if not X_list:
        raise RuntimeError("No samples were generated in build_energy_dataset.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    meta_idx = np.array(meta_idx_list, dtype=int)

    return EnergyDataset(X=X, y=y, meta_idx=meta_idx, groups=list(groups))

