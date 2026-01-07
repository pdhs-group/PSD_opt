# -*- coding: utf-8 -*-
"""
统一测试脚本：

- 从 HDF5 读入 EnergyGroupRecord
- 构建 EnergyDataset (target=log_mean)
- 根据参数选择测试哪种模型：
    * PowerLawSeparableModel
    * ParametricEnergyModel
    * MLPEnergyModel
- 在整体数据上做一个简单的验证
- 对指定 group_index 做可视化对比

同时提供统一的函数接口，方便在 Spyder 中复用训练结果：
    - load_data
    - split_train_val_by_group
    - fit_powerlaw_model
    - fit_parametric_model
    - fit_mlp_model
    - evaluate_model
    - run_experiment
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import time

from agggenerator.breakage_rate_model.data_io import load_energy_groups_from_h5
from agggenerator.breakage_rate_model.datasets import build_energy_dataset
from agggenerator.breakage_rate_model.powerlaw_separable import PowerLawSeparableModel
from agggenerator.breakage_rate_model.parametric_model import ParametricEnergyModel
from agggenerator.breakage_rate_model.mlp_model import MLPEnergyModel
from agggenerator.breakage_rate_model.ann_model import ANNEnergyModel
from agggenerator.breakage_rate_model.base import mse, mae, mape, r2


# =============================================================================
# 数据加载与拆分
# =============================================================================

def load_data(h5_file: str):
    """
    从 HDF5 载入 groups，并构建用于 log(E_mean) 拟合的 EnergyDataset。

    返回:
        groups, X, y
    """
    print(f"Loading groups from {h5_file} ...")
    groups = load_energy_groups_from_h5(h5_file)
    print(f"Loaded {len(groups)} groups.")

    print("Building energy dataset (log_mean)...")
    energy_ds = build_energy_dataset(
        groups,
        per_sample=False,
        target="log_mean",
    )
    X = energy_ds.X
    y = energy_ds.y
    print(f"EnergyDataset: X.shape={X.shape}, y.shape={y.shape}")

    return groups, X, y


def _compute_group_spans(groups):
    """
    根据 groups 计算每个 group 在拼接后的 X/y 中的起止索引。
    返回:
        group_starts, group_ends
    """
    group_sizes = []
    for g in groups:
        n = len(g.Np)   # 每个 group 有多少个 V 点
        group_sizes.append(n)

    group_starts = np.cumsum([0] + group_sizes[:-1])
    group_ends = np.cumsum(group_sizes)
    return group_starts, group_ends


def split_train_val_by_group(
    groups,
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    按 group 拆分 train / val —— 每个 group 的全部样本保持在一起。

    参数
    ----
    groups : List[EnergyGroupRecord]
    X, y   : build_energy_dataset 的输出（按 group 顺序拼接）
    val_ratio: 验证集占 group 的比例
    seed  : 随机种子

    返回
    ----
    (X_train, y_train, X_val, y_val)
    """

    rng = np.random.default_rng(seed)

    group_starts, group_ends = _compute_group_spans(groups)
    n_groups = len(groups)
    group_indices = np.arange(n_groups)
    rng.shuffle(group_indices)

    # 按 group 随机划分 train / val
    n_val = int(np.ceil(val_ratio * n_groups))
    val_groups = group_indices[:n_val]
    train_groups = group_indices[n_val:]

    train_idx = []
    val_idx = []

    for gi in train_groups:
        s = group_starts[gi]
        e = group_ends[gi]
        train_idx.extend(range(s, e))

    for gi in val_groups:
        s = group_starts[gi]
        e = group_ends[gi]
        val_idx.extend(range(s, e))

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
    )


# =============================================================================
# 不同模型的拟合函数
# =============================================================================

def fit_powerlaw_model(
    groups,
    pearson_min=None,
    fit_mode="logE",
    pure_powerlaw_if_no_plateau=True,
    frac_threshold=0.3,
    abs_threshold=0.1,
    max_V=None,
    enable_tail=True,
    plateau_weight=3.0,
    regress_type="linear",      # "linear" / "ridge"
    ridge_lambda=1e-2,
) -> PowerLawSeparableModel:
    """
    拟合 PowerLawSeparableModel。

    注意：此模型不使用 X_train / y_train，而是直接基于 groups 拟合。
    """
    print("Fitting PowerLawSeparableModel from groups ...")
    model = PowerLawSeparableModel(
        pearson_min=pearson_min,
        fit_mode=fit_mode,
        pure_powerlaw_if_no_plateau=pure_powerlaw_if_no_plateau,
        frac_threshold=frac_threshold,
        abs_threshold=abs_threshold,
        max_V=max_V,
        enable_tail=enable_tail,
        plateau_weight=plateau_weight,
        regress_type=regress_type,
        ridge_lambda=ridge_lambda,
    )
    model.fit(None, None, groups=groups)
    return model


def fit_parametric_model(
    groups,
    pearson_min=None,
    fit_mode="logE",
    pure_powerlaw_if_no_plateau=True,
    frac_threshold=0.3,
    abs_threshold=0.1,
    residual_type="ridge",      # "linear" / "ridge" / "none"
    residual_lambda=1e-2,
    enable_tail=True,
    plateau_weight=3.0,
) -> ParametricEnergyModel:
    """
    拟合 ParametricEnergyModel（两段主趋势 + 残差校正）。
    """
    print("Fitting ParametricEnergyModel from groups ...")
    model = ParametricEnergyModel(
        pearson_min=pearson_min,
        fit_mode=fit_mode,
        pure_powerlaw_if_no_plateau=pure_powerlaw_if_no_plateau,
        frac_threshold=frac_threshold,
        abs_threshold=abs_threshold,
        residual_type=residual_type,
        residual_lambda=residual_lambda,
        enable_tail=enable_tail,
        plateau_weight=plateau_weight,
    )
    model.fit(None, None, groups=groups)
    return model


def fit_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    input_dim: int = 7,
    hidden_sizes=(64, 64),
    activation="relu",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    batch_size: int = 128,
    patience: int = 20,
    seed: int = 0,
    device: str | None = None,
) -> MLPEnergyModel:
    """
    拟合 MLPEnergyModel（轻量级 MLP surrogate）:

        X, y 都是样本级别的数据：
            X[i] = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
            y[i] = log(E_mean)
    """
    print("Fitting MLPEnergyModel on (X_train, y_train) ...")
    model = MLPEnergyModel(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
        seed=seed,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return model

def fit_ann_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    input_dim: int = 7,
    hidden_sizes=(256, 256, 128),
    activation="silu",
    dropout=0.1,
    lr=3e-4,
    weight_decay=1e-4,
    max_epochs=300,
    batch_size=256,
    patience=30,
    seed=0,
    device=None,
) -> MLPEnergyModel:
    """
    拟合 MLPEnergyModel（轻量级 MLP surrogate）:

        X, y 都是样本级别的数据：
            X[i] = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1]
            y[i] = log(E_mean)
    """
    print("Fitting MLPEnergyModel on (X_train, y_train) ...")
    model = ANNEnergyModel(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
        seed=seed,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return model
# =============================================================================
# 统一评估函数
# =============================================================================

def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, name: str = ""):
    """
    在验证集上评估模型表现，返回一个 metrics 字典，并打印结果。

    默认假设：
        y_val 是 log(E_mean)
        model.predict(X_val) 返回的也是 log(E_pred)
    """
    if not name:
        name = getattr(model, "name", model.__class__.__name__)

    y_pred = model.predict(X_val)
    metrics = {
        "mse": mse(y_val, y_pred),
        "mae": mae(y_val, y_pred),
        "mape": mape(y_val, y_pred),
        "r2": r2(y_val, y_pred),
    }

    print(f"Validation metrics on log(E_mean) for {name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g}")

    return metrics


# =============================================================================
# MLP 专用：某个 group 的预测可视化
# =============================================================================

def plot_group_mlp_prediction(
    model: MLPEnergyModel,
    groups,
    X: np.ndarray,
    y: np.ndarray,
    group_index: int,
    title: str | None = None,
):
    """
    对 MLP 模型，绘制指定 group 的 logE–logV 对比图。

    假设：
        - X, y 来自 build_energy_dataset(per_sample=False, target="log_mean")
        - group 顺序和 X, y 的拼接顺序一致
    """
    group_starts, group_ends = _compute_group_spans(groups)
    if group_index < 0 or group_index >= len(groups):
        raise IndexError(f"group_index {group_index} out of range (0..{len(groups)-1})")

    s = group_starts[group_index]
    e = group_ends[group_index]

    Xg = X[s:e]
    yg = y[s:e]

    logV = Xg[:, 0]
    y_pred = model.predict(Xg)

    if title is None:
        title = f"MLP prediction vs true, group {group_index}"

    plt.figure(figsize=(6, 5))
    plt.title(title)
    plt.scatter(logV, yg, label="true logE", s=25)
    plt.plot(logV, y_pred, label="MLP pred", linewidth=2)
    plt.xlabel("log V")
    plt.ylabel("log E")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sweep_V_for_group(
    model,
    groups,
    X: np.ndarray,
    y: np.ndarray,
    group_index: int,
    *,
    V_min: float = 100.0,
    V_max: float = 50000.0,
    n_points: int = 100,
    title: str | None = None,
):
    """
    在“同一个 group、不同 V 网格”上测试模型的外插/插值能力。

    思路：
      - 找到指定 group 在拼接后的 X, y 中的区间 [s:e)
      - 取该 group 的 θ 参数（假设对该 group 内所有样本相同）：
            θ = Xg[0, 1:]
      - 在 log(V_min) ~ log(V_max) 上等距采样 n_points 个 logV_new
      - 构造新特征：
            X_new[i] = [logV_new[i], *θ]
      - 用模型预测 y_new = model.predict(X_new)
      - 同时把原始数据 (logV_orig, y_orig) 和 X_new 上的预测画在一张图上

    参数
    ----
    model      : 已经训练好的某个模型（PowerLaw/Parametric/MLP/ANN）
    groups     : EnergyGroupRecord 列表
    X, y       : build_energy_dataset(per_sample=False, target="log_mean") 的输出
    group_index: 要测试的 group 编号
    V_min, V_max: V 的实际范围，对应 logV_min/logV_max
    n_points   : 在 [log(V_min), log(V_max)] 上采样多少个点
    """

    group_starts, group_ends = _compute_group_spans(groups)
    if group_index < 0 or group_index >= len(groups):
        raise IndexError(f"group_index {group_index} out of range (0..{len(groups)-1})")

    # 原始 group 的数据
    s = group_starts[group_index]
    e = group_ends[group_index]

    Xg = X[s:e]
    yg = y[s:e]

    # 原始的 logV（X 第一列）
    logV_orig = Xg[:, 0]

    # ---- 提取该 group 的 θ 特征（假设 group 内 θ 不变） ----
    theta = Xg[0, 1:].copy()   # 形状: (input_dim-1,)

    # 检查一下 group 内 θ 是否一致（不是必须，但有助于 sanity check）
    if not np.allclose(Xg[:, 1:], theta[None, :], atol=1e-8):
        print(f"[WARN] group {group_index} 内 θ 特征并非完全常数，但仍使用第一行的 θ 作为代表。")

    input_dim = X.shape[1]
    if theta.shape[0] != input_dim - 1:
        raise ValueError(
            f"theta dim mismatch: theta has {theta.shape[0]}, but X has dim={input_dim}"
        )

    # ---- 构造新的 logV 网格 ----
    logV_min = np.log(V_min)
    logV_max = np.log(V_max)
    logV_new = np.linspace(logV_min, logV_max, n_points)

    # 构造新的特征矩阵 X_new: [logV_new, θ]
    X_new = np.zeros((n_points, input_dim), dtype=np.float32)
    X_new[:, 0] = logV_new
    X_new[:, 1:] = theta[None, :]

    # ---- 在原始点和新网格上分别做预测 ----
    y_pred_orig = model.predict(Xg)
    y_pred_new = model.predict(X_new)

    # ---- 画图：原始样本 vs 新 V 网格预测 ----
    if title is None:
        title = f"V-sweep test for group {group_index}"

    plt.figure(figsize=(7, 5))
    plt.title(title)

    # 原始数据（真值）
    plt.scatter(logV_orig, yg, label="true (orig grid)", s=25)

    # 原始 grid 上的模型预测（可选，看你想不想画，先画出来方便对比）
    plt.plot(logV_orig, y_pred_orig, label="model on orig grid", linewidth=2, alpha=0.7)

    # 新 V 网格上的预测
    plt.plot(logV_new, y_pred_new, label="model on new V-grid", linewidth=2, linestyle="--")

    plt.xlabel("log V")
    plt.ylabel("log E")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return logV_new, y_pred_new

# =============================================================================
# 统一的实验入口
# =============================================================================

def run_experiment(
    model_kind: str = "powerlaw",
    h5_file: str = "energy_scan_results.h5",
):
    """
    统一的实验入口：

    参数
    ----
    model_kind:
        "powerlaw"   -> 使用 PowerLawSeparableModel
        "parametric" -> 使用 ParametricEnergyModel
        "mlp"        -> 使用 MLPEnergyModel
        "ann"        -> 使用 ANNEnergyModel
        "all"        -> 依次训练并对比上述所有模型
    h5_file:
        HDF5 路径

    返回
    ----
    若 model_kind != "all":
        model, groups, X, y, (X_train, y_train, X_val, y_val), metrics

    若 model_kind == "all":
        results, groups, X, y, (X_train, y_train, X_val, y_val)

        其中 results 是:
            {
              "powerlaw": {"time": ..., "mse": ..., "mae": ..., "mape": ..., "r2": ...},
              "parametric": {...},
              "mlp": {...},
              "ann": {...},
            }
    """
    # 1. 载入数据
    groups, X, y = load_data(h5_file)

    # 2. 按 group 拆分 train / val
    X_train, y_train, X_val, y_val = split_train_val_by_group(
        groups,
        X, y,
        val_ratio=0.2,
        seed=42,
    )

    mk = model_kind.lower()

    # ------------------------------------------------------------------
    # 单模型模式：保持原来的行为
    # ------------------------------------------------------------------
    if mk in ("powerlaw", "parametric", "mlp", "ann"):
        if mk == "powerlaw":
            model = fit_powerlaw_model(
                groups,
                pearson_min=None,
                fit_mode="logE",
                pure_powerlaw_if_no_plateau=True,
                frac_threshold=0.3,
                abs_threshold=0.1,
                max_V=None,
                enable_tail=True,
                plateau_weight=4.0,
                regress_type="ridge",     # 或 "linear"
                ridge_lambda=1e2,
            )

        elif mk == "parametric":
            model = fit_parametric_model(
                groups,
                pearson_min=None,
                fit_mode="logE",
                pure_powerlaw_if_no_plateau=True,
                frac_threshold=0.3,
                abs_threshold=0.1,
                residual_type="ridge",    # "linear" / "ridge" / "none"
                residual_lambda=1e1,
                enable_tail=True,
                plateau_weight=3.0,
            )

        elif mk == "mlp":
            model = fit_mlp_model(
                X_train,
                y_train,
                X_val,
                y_val,
                input_dim=X_train.shape[1],
                hidden_sizes=(256, 256),
                activation="relu",
                lr=1e-3,
                weight_decay=1e-4,
                max_epochs=200,
                batch_size=128,
                patience=20,
                seed=42,
                device=None,
            )

        elif mk == "ann":
            model = fit_ann_model(
                X_train,
                y_train,
                X_val,
                y_val,
                input_dim=X_train.shape[1],
                hidden_sizes=(256, 256, 128),
                activation="silu",
                dropout=0.1,
                lr=3e-4,
                weight_decay=1e-4,
                max_epochs=300,
                batch_size=256,
                patience=50,
                seed=42,
                device=None,
            )
        else:
            raise ValueError("unreachable")

        metrics = evaluate_model(model, X_val, y_val, name=mk)
        return model, groups, X, y, (X_train, y_train, X_val, y_val), metrics

    # ------------------------------------------------------------------
    # "all" 模式：训练所有模型 -> 保存 -> 从文件加载 -> 评估并计时
    # ------------------------------------------------------------------
    if mk == "all":
        results = {}
        model_specs = ["powerlaw", "parametric", "mlp", "ann"]

        for spec in model_specs:
            print("=" * 80)
            print(f"[ALL] Training model: {spec}")

            if spec == "powerlaw":
                model = fit_powerlaw_model(
                    groups,
                    pearson_min=None,
                    fit_mode="logE",
                    pure_powerlaw_if_no_plateau=True,
                    frac_threshold=0.3,
                    abs_threshold=0.1,
                    max_V=None,
                    enable_tail=True,
                    plateau_weight=4.0,
                    regress_type="ridge",
                    ridge_lambda=1e2,
                )
            elif spec == "parametric":
                model = fit_parametric_model(
                    groups,
                    pearson_min=None,
                    fit_mode="logE",
                    pure_powerlaw_if_no_plateau=True,
                    frac_threshold=0.3,
                    abs_threshold=0.1,
                    residual_type="ridge",
                    residual_lambda=1e1,
                    enable_tail=True,
                    plateau_weight=3.0,
                )
            elif spec == "mlp":
                model = fit_mlp_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    input_dim=X_train.shape[1],
                    hidden_sizes=(256, 256),
                    activation="relu",
                    lr=1e-3,
                    weight_decay=1e-4,
                    max_epochs=200,
                    batch_size=128,
                    patience=20,
                    seed=42,
                    device=None,
                )
            elif spec == "ann":
                model = fit_ann_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    input_dim=X_train.shape[1],
                    hidden_sizes=(256, 256, 128),
                    activation="silu",
                    dropout=0.1,
                    lr=3e-4,
                    weight_decay=1e-4,
                    max_epochs=300,
                    batch_size=256,
                    patience=50,
                    seed=42,
                    device=None,
                )
            else:
                raise ValueError(f"Unknown spec '{spec}'")

            # 保存模型到文件
            model_path = f"{spec}_model.pkl"
            print(f"[ALL] Saving {spec} to {model_path}")
            model.save(model_path)

            # 计时：从文件加载 + 在验证集上预测并评估
            print(f"[ALL] Loading {spec} from {model_path} and evaluating ...")
            t0 = time.perf_counter()
            loaded_model = model.load(model_path)
            metrics = evaluate_model(loaded_model, X_val, y_val, name=spec)
            t1 = time.perf_counter()
            elapsed = t1 - t0

            results[spec] = {"time": elapsed}
            results[spec].update(metrics)

        # 返回结果和数据，方便在 main 或 Spyder 里做绘图
        return results, groups, X, y, (X_train, y_train, X_val, y_val)

    # 其它非法输入
    raise ValueError(
        f"Unknown model_kind '{model_kind}', must be 'powerlaw', 'parametric', 'mlp', 'ann' or 'all'."
    )



# =============================================================================
# main: 只负责选择模型类型 + 可视化某组
# =============================================================================

if __name__ == "__main__":
    # 这里改这几个参数，就可以用 Spyder 重复测试不同组合
    H5_FILE = "energy_scan_results.h5"
    MODEL_KIND = "all"   # "powerlaw" / "parametric" / "mlp" / "ann" / "all"
    GROUP_INDEX = 0      # 想看的组号（非 all 时）

    if MODEL_KIND.lower() == "all":
        # 运行 all 模式：训练+保存+加载+评估
        results, groups, X, Y, split_data = run_experiment(
            model_kind="all",
            h5_file=H5_FILE,
        )

        GLOBAL_RESULTS = results
        GLOBAL_GROUPS = groups
        GLOBAL_X = X
        GLOBAL_Y = Y
        GLOBAL_SPLIT = split_data

        # 画 5 张柱状图：time, mse, mae, mape, r2
        metrics_to_plot = ["time", "mse", "mae", "mape", "r2"]
        model_labels = list(results.keys())

        for metric in metrics_to_plot:
            plt.figure(figsize=(6, 4))
            vals = [results[m][metric] for m in model_labels]
            x = np.arange(len(model_labels))
            plt.bar(x, vals)
            plt.xticks(x, model_labels)
            plt.ylabel(metric)
            plt.title(f"Comparison of {metric} across models")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()

    else:
        # 单模型模式：保持原来的行为
        model, groups, X, y, split_data, metrics = run_experiment(
            model_kind=MODEL_KIND,
            h5_file=H5_FILE,
        )

        # 把结果挂到全局变量，方便 Spyder 中直接访问
        GLOBAL_MODEL = model
        GLOBAL_GROUPS = groups
        GLOBAL_X = X
        GLOBAL_Y = y
        GLOBAL_SPLIT = split_data
        GLOBAL_METRICS = metrics

        # 在当前脚本执行时直接画一组
        mk = MODEL_KIND.lower()
        if mk in ("powerlaw", "parametric"):
            model.analyze_one_group(
                groups,
                group_index=GROUP_INDEX,
                target="log_mean",  # 模型返回的是 logE
                show=True,
            )
        elif mk in ("mlp", "ann"):
            # plot_group_mlp_prediction(
            #     model,
            #     groups,
            #     split_data[2],   # X_val
            #     split_data[3],   # y_val
            #     group_index=GROUP_INDEX,
            # )
            logV_new, y_new = sweep_V_for_group(
                model,
                groups,
                X, y,
                group_index=0,       # 想看的 group
                V_min=100.0,
                V_max=50000.0,
                n_points=100,
            )

    
    