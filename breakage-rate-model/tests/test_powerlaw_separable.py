# -*- coding: utf-8 -*-
"""
用于快速测试 PowerLawSeparableModel：

- 从 HDF5 读入 EnergyGroupRecord
- 构建 EnergyDataset (target=log_mean)
- 训练 PowerLawSeparableModel
- 在整体数据上做一个简单的验证
- 对指定 group_index 做可视化对比

新增：
- debug_overflow_groups: 按 group 遍历，捕捉哪些 group 在预测时触发
  数值溢出（FloatingPointError），用于定位 overflow 问题。
"""

from __future__ import annotations

import numpy as np

from breakage_rate_model.data_io import load_energy_groups_from_h5
from breakage_rate_model.datasets import build_energy_dataset
from breakage_rate_model.powerlaw_separable import (
    PowerLawSeparableModel,
    fit_two_segment_params,
    smooth_powerlaw_plateau,
    smooth_powerlaw_plateau_with_tail,
)
import matplotlib.pyplot as plt


def main():
    # 修改为你的 HDF5 路径
    h5_file = "energy_scan_results.h5"

    print(f"Loading groups from {h5_file} ...")
    groups = load_energy_groups_from_h5(h5_file)
    print(f"Loaded {len(groups)} groups.")

    # 构建用于评估的 dataset：直接拟合 log(E_mean)
    print("Building energy dataset (log_mean).")
    energy_ds = build_energy_dataset(
        groups,
        per_sample=False,
        target="log_mean",
    )
    X = energy_ds.X
    y = energy_ds.y
    print(f"EnergyDataset: X.shape={X.shape}, y.shape={y.shape}")

    # 简单切一刀 train/val（这里只是 sanity check，用整集也可以）
    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)

    split = int(0.8 * n_samples)
    idx_train = idx[:split]
    idx_val = idx[split:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]

    # 训练 PowerLawSeparableModel
    print("Fitting PowerLawSeparableModel from groups ...")
    model = PowerLawSeparableModel(
        pearson_min=None,   # 或者比如 0.8
        fit_mode="logE",
        pure_powerlaw_if_no_plateau=True,
        frac_threshold=0.3,
        abs_threshold=0.1,
        enable_tail=True,
        max_V=None,
    )
    # 注意：fit 忽略 X_train/y_train，直接用 groups 进行 group-level 拟合
    model.fit(None, None, groups=groups)

    # 在 val 集上评估（注意：目标是 log_mean，predict 也返回 logE）
    metrics = model.validate(X_val, y_val, metrics=("mse", "mae", "mape", "r2"))
    print("Validation metrics on log(E_mean):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g}")

    # 可视化某一组
    group_index = 1  # 换成你想看的组号
    model.analyze_one_group(
        groups,
        group_index=group_index,
        target="log_mean",  # 模型返回的是 logE
        show=True,
    )

def debug_overflow_in_fit(h5_file: str = "energy_scan_results.h5"):
    """
    按 group 遍历，在 *拟合两段模型* 的阶段检测溢出：

        对每个 group 调用一次 fit_two_segment_params(rec.V, rec.E_mean, ...)

    并且用 numpy.seterr(over='raise', invalid='raise', divide='raise') 把
    RuntimeWarning 升级为 FloatingPointError，捕捉并打印出出问题的 group。
    """
    print(f"[DEBUG-FIT] Loading groups from {h5_file} ...")
    groups = load_energy_groups_from_h5(h5_file)
    print(f"[DEBUG-FIT] Loaded {len(groups)} groups.")

    # 打开 "overflow/invalid/divide -> raise" 模式
    old_settings = np.seterr(over="raise", invalid="raise", divide="raise")

    bad_groups = []

    try:
        for gi, rec in enumerate(groups):
            V = rec.V
            E_mean = rec.E_mean

            # 只用正的 V/E
            mask = (V > 0.0) & (E_mean > 0.0)
            V_pos = V[mask]
            E_pos = E_mean[mask]
            if V_pos.size < 3:
                continue

            try:
                params = fit_two_segment_params(
                    V_pos,
                    E_pos,
                    rec.pearson_r_fit,
                    frac_threshold=0.3,
                    abs_threshold=0.1,
                    fit_mode="logE",
                    pure_powerlaw_if_no_plateau=True,
                )
                # 可选：检查一下返回的参数是否已经是 nan/inf
                if not np.all(np.isfinite([params["sigma"], params["Vc"], params["Emax"]])):
                    raise FloatingPointError("params contain NaN/Inf")

            except FloatingPointError as e:
                print(f"[OVERFLOW-FIT] group {gi}, key={rec.key}: {e}")
                bad_groups.append(gi)
            except Exception as e:
                # 其它异常也先打印出来
                print(f"[ERROR-FIT] group {gi}, key={rec.key}: {type(e).__name__}: {e}")

    finally:
        # 恢复原本的 numpy 错误处理配置
        np.seterr(**old_settings)

    if not bad_groups:
        print("[DEBUG-FIT] No groups triggered overflow/invalid during fit_two_segment_params.")
    else:
        print(f"[DEBUG-FIT] Groups with overflow/invalid in fit_two_segment_params: {bad_groups}")
        
def inspect_one_group(
    group_index: int,
    h5_file: str = "energy_scan_results.h5",
    enable_tail: bool = True,
    max_V: float | None = None,
    tol_int_bre: float = 1e-12,
):
    """
    对某个 group 做详细检查（适配当前按 int_bre 分簇的新算法）：
      - 根据 int_bre 和 enable_tail 决定本组是否拟合尾巴 α：
          * int_bre == 0 且 enable_tail=True → 带 α（侵蚀型）
          * 其他情况 → 不带 α（普通破碎）
      - 打印拟合得到的参数 (sigma, Vc, Emax, alpha, Vc_local, r_global)
      - 画出原始数据和拟合曲线（有尾巴则使用 smooth_powerlaw_plateau_with_tail）
    """
    groups = load_energy_groups_from_h5(h5_file)
    rec = groups[group_index]

    print(f"=== Inspect group {group_index}, key={rec.key} ===")
    print(f"  NO_FRAG={rec.NO_FRAG}, int_bre={rec.int_bre}, gamma={rec.gamma}")
    print(f"  Df={rec.Df}, MAS={rec.MAS}, A0={rec.A0}, X1={rec.X1}")

    V = rec.V
    E = rec.E_mean
    mask = (V > 0.0) & (E > 0.0)
    if max_V is not None:
        mask &= (V <= max_V)
    V = V[mask]
    E = E[mask]

    print(f"  V range: [{V.min():.3g}, {V.max():.3g}], "
          f"E range: [{E.min():.3g}, {E.max():.3g}]")

    # 当前算法：只有 int_bre==0 且 enable_tail=True 的 group 才拟尾巴
    is_erosion = abs(rec.int_bre) <= tol_int_bre
    enable_tail_group = (enable_tail and is_erosion)
    print(f"  is_erosion(int_bre≈0)? {is_erosion}, "
          f"enable_tail_for_group={enable_tail_group}")

    # 拟合两段模型（和 PowerLawSeparableModel 中保持一致）
    params = fit_two_segment_params(
        V,
        E,
        rec.pearson_r_fit,
        frac_threshold=0.3,
        abs_threshold=0.1,
        fit_mode="logE",
        pure_powerlaw_if_no_plateau=True,
        enable_tail=enable_tail_group,
        max_V=max_V,
    )
    print("  fitted params:")
    for k, v in params.items():
        print(f"    {k}: {v}")

    sigma = params["sigma"]
    Vc = params["Vc"]
    Emax = params["Emax"]
    alpha = params.get("alpha", 0.0)

    # 画拟合曲线
    V_plot = np.linspace(V.min(), V.max(), 200)
    if np.isfinite(sigma) and np.isfinite(Vc) and np.isfinite(Emax) and Vc > 0 and Emax > 0:
        if enable_tail_group:
            E_fit = smooth_powerlaw_plateau_with_tail(
                V_plot,
                sigma,
                Vc,
                Emax,
                alpha,
                enable_tail=True,
            )
        else:
            E_fit = smooth_powerlaw_plateau(V_plot, sigma, Vc, Emax)
        E_fit = np.maximum(E_fit, 1e-12)
    else:
        E_fit = None

    plt.figure(figsize=(6, 5))
    plt.title(f"Group {group_index}: raw vs two-seg fit "
              f"(tail={'on' if enable_tail_group else 'off'})")
    plt.scatter(np.log(V), np.log(E), label="data (log)", s=20)
    if E_fit is not None:
        plt.plot(np.log(V_plot), np.log(E_fit), label="two-seg fit")
    plt.xlabel("log V")
    plt.ylabel("log E")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # main()
    # 如需专门做溢出排查，可以在这里调用：
    # debug_overflow_in_fit("energy_scan_results.h5")
    bad_indices = [0,1,2]
    for gi in bad_indices:
        inspect_one_group(gi)