# -*- coding: utf-8 -*-
"""
负责：
- 从 energy_scan_results.h5 读取数据
- 封装为 EnergyGroupRecord 结构，便于后续 dataset 构造和模型使用
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import h5py


@dataclass
class EnergyGroupRecord:
    """
    表示 H5 文件中 /runs/<key> 下的一组参数组合及其能量曲线数据。
    """
    key: str

    # attrs: 参数
    NO_FRAG: int
    int_bre: float
    gamma: float
    Df: float
    MAS: float
    X1: float
    A0: float
    N_GRIDS: int
    N_FRACS: int
    base_seed: int
    workers: int
    STR: np.ndarray

    # attrs: 预先计算或写入的幂律拟合结果（旧数据可能不存在）
    sigma_attr: Optional[float]
    pearson_r_attr: Optional[float]

    # datasets: 曲线数据
    Np: np.ndarray       # shape: [n_Np]
    V: np.ndarray        # shape: [n_Np]
    E_mean: np.ndarray   # shape: [n_Np]
    E_std: np.ndarray    # shape: [n_Np]
    E_samples: np.ndarray  # shape: [n_Np, n_runs_per_np]

    # 重新计算得到的 log–log 直线拟合结果
    sigma_fit: float
    b_fit: float
    pearson_r_fit: float


def _fit_line_logV_logE(V: np.ndarray, E: np.ndarray) -> Tuple[float, float, float]:
    """
    在 log–log 空间对 E_mean vs V 做线性拟合:
        logE ≈ b + sigma * logV

    返回:
        sigma, b, pearson_r
    """
    mask = (V > 0.0) & (E > 0.0)
    V_valid = V[mask]
    E_valid = E[mask]
    if V_valid.size < 2:
        # 数据太少，返回 NaN
        return np.nan, np.nan, np.nan

    logV = np.log(V_valid)
    logE = np.log(E_valid)

    # Pearson r
    if logV.size > 1:
        r = np.corrcoef(logV, logE)[0, 1]
    else:
        r = np.nan

    # 线性回归 logE = b + sigma * logV
    A = np.vstack([np.ones_like(logV), logV]).T
    coef, *_ = np.linalg.lstsq(A, logE, rcond=None)
    b, sigma = coef  # 注意顺序：b 是截距，sigma 是斜率

    return float(sigma), float(b), float(r)


def load_energy_groups_from_h5(h5_path: str) -> List[EnergyGroupRecord]:
    """
    扫描 energy_scan_results.h5 下的所有 /runs/<key>，
    读取参数 attrs + Np/V/E_mean/E_samples，返回 EnergyGroupRecord 列表。

    要求 H5 结构（每个 group）包含：
        attrs:
            NO_FRAG, int_bre, gamma, Df, MAS, X1, A0,
            N_GRIDS, N_FRACS, base_seed, workers, STR, [sigma], [pearson_r]
        datasets:
            Np, V, E_mean, E_std, E_samples
    """
    records: List[EnergyGroupRecord] = []

    with h5py.File(h5_path, "r") as f:
        if "runs" not in f:
            raise RuntimeError("HDF5 file has no '/runs' group.")

        runs_grp = f["runs"]

        for key in runs_grp.keys():
            grp = runs_grp[key]

            # ---- attrs ----
            NO_FRAG = int(grp.attrs["NO_FRAG"])
            int_bre = float(grp.attrs["int_bre"])
            gamma = float(grp.attrs["gamma"])
            Df = float(grp.attrs["Df"])
            MAS = float(grp.attrs["MAS"])
            X1 = float(grp.attrs["X1"])
            A0 = float(grp.attrs["A0"])
            N_GRIDS = int(grp.attrs["N_GRIDS"])
            N_FRACS = int(grp.attrs["N_FRACS"])
            base_seed = int(grp.attrs["base_seed"])
            workers = int(grp.attrs["workers"])
            STR = np.array(grp.attrs["STR"], dtype=float)

            sigma_attr = float(grp.attrs["sigma"]) if "sigma" in grp.attrs else None
            pearson_r_attr = float(grp.attrs["pearson_r"]) if "pearson_r" in grp.attrs else None

            # ---- datasets ----
            Np = grp["Np"][:]
            V = grp["V"][:]
            E_mean = grp["E_mean"][:]
            E_std = grp["E_std"][:]
            E_samples = grp["E_samples"][:]

            # ---- 重新拟合 log–log 直线（作为“真实” sigma / b / r）----
            sigma_fit, b_fit, r_fit = _fit_line_logV_logE(V, E_mean)

            rec = EnergyGroupRecord(
                key=key,
                NO_FRAG=NO_FRAG,
                int_bre=int_bre,
                gamma=gamma,
                Df=Df,
                MAS=MAS,
                X1=X1,
                A0=A0,
                N_GRIDS=N_GRIDS,
                N_FRACS=N_FRACS,
                base_seed=base_seed,
                workers=workers,
                STR=STR,
                sigma_attr=sigma_attr,
                pearson_r_attr=pearson_r_attr,
                Np=Np,
                V=V,
                E_mean=E_mean,
                E_std=E_std,
                E_samples=E_samples,
                sigma_fit=sigma_fit,
                b_fit=b_fit,
                pearson_r_fit=r_fit,
            )
            records.append(rec)

    return records
