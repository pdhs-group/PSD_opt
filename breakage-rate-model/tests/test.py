# -*- coding: utf-8 -*-
"""
Overview of the unified test script:

1. The script reads `/runs/<key>` from the HDF5 file and loads each `<key>` as
   one `EnergyGroupRecord`. Here a group means one full energy curve with a fixed
   set of input parameters. Within one group, `NO_FRAG`, `int_bre`, `gamma`, `Df`,
   `MAS`, `X1`, and `STR` stay constant, while `Np / V` varies.
2. `build_energy_dataset(...)` expands the groups into supervised samples `(X, y)`.
   With `per_sample=False`, each `V` point inside a group becomes one sample, so
   a single group usually contributes multiple rows.
3. The train/validation split is performed by group rather than by shuffled rows
   to avoid leakage between points from the same physical curve.
4. The total number of groups is determined by the actual `/runs/<key>` entries
   in the HDF5 file; it is not hard-coded in this script.
5. `split_train_val_by_group(...)` uses `val_ratio=0.2`, so about 20% of groups
   go to validation and the remaining groups go to training.
6. Strictly speaking, this script uses a train/validation split only; it does
   not build a separate third test set by default.
7. Under the same data-loading interface, the script can compare four models:
   `PowerLawSeparableModel`, `ParametricEnergyModel`, `MLPEnergyModel`, and
   `ANNEnergyModel`.

"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import time

from breakage_rate_model.data_io import load_energy_groups_from_h5
from breakage_rate_model.datasets import build_energy_dataset
from breakage_rate_model.powerlaw_separable import PowerLawSeparableModel
from breakage_rate_model.parametric_model import ParametricEnergyModel
from breakage_rate_model.mlp_model import MLPEnergyModel
from breakage_rate_model.ann_model import ANNEnergyModel
from breakage_rate_model.base import mse, mae, mape, r2


# =============================================================================
# Data Loading and Splitting
# =============================================================================

def load_data(h5_file: str):
    """
    Load groups from HDF5 and build an EnergyDataset for fitting log(E_mean).

    Returns:
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
    Compute the start/end indices of each group in the concatenated X/y arrays.
    Returns:
        group_starts, group_ends
    """
    group_sizes = []
    for g in groups:
        n = len(g.Np)   # number of V points in this group
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
    Split train / val by group so that all samples from one group stay together.

    Parameters
    ----
    groups : List[EnergyGroupRecord]
    X, y   : output of build_energy_dataset (concatenated in group order)
    val_ratio: fraction of groups assigned to validation
    seed  : random seed

    Returns
    ----
    (X_train, y_train, X_val, y_val)
    """

    rng = np.random.default_rng(seed)

    group_starts, group_ends = _compute_group_spans(groups)
    n_groups = len(groups)
    group_indices = np.arange(n_groups)
    rng.shuffle(group_indices)

    # Random group-wise train / val split
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
# Model Fitting Helpers
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
    Fit PowerLawSeparableModel.

    Note: this model does not use X_train / y_train and is fitted directly from groups.
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
    Fit ParametricEnergyModel (two-stage trend + residual correction).
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
    input_dim: int = 10,
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
    Fit MLPEnergyModel (lightweight MLP surrogate):

        X and y are sample-level arrays:
            X[i] = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]
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
    input_dim: int = 10,
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
    Fit ANNEnergyModel (ANN surrogate):

        X and y are sample-level arrays:
            X[i] = [logV, log gamma, log NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]
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
# Unified Evaluation Helper
# =============================================================================

def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, name: str = ""):
    """
    Evaluate model performance on the validation set, return a metrics dict, and print it.

    Default assumptions:
        y_val æ˜¯ log(E_mean)
        model.predict(X_val) also returns log(E_pred)
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
# MLP-Specific Visualization for One Group
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
    Plot logE-logV comparison for a selected group using an MLP-like model.

    Assumptions:
        - X and y come from build_energy_dataset(per_sample=False, target="log_mean")
        - group order matches the concatenation order of X and y
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
    Test interpolation/extrapolation ability on one fixed group over a new V grid.

    Procedure:
      - locate the slice [s:e) of the selected group in the concatenated X/y arrays
      - extract the shared parameter vector `theta = Xg[0, 1:]`
            theta = Xg[0, 1:]
      - sample `n_points` values uniformly in log(V) from `V_min` to `V_max`
      - build new features:
            X_new[i] = [logV_new[i], *theta]
      - predict `y_new = model.predict(X_new)`
      - plot original data and predictions on both the original and new grids

    Parameters
    ----
    model      : trained model (PowerLaw / Parametric / MLP / ANN)
    groups     : list of EnergyGroupRecord
    X, y       : output of build_energy_dataset(per_sample=False, target="log_mean")
    group_index: target group index
    V_min, V_max: physical V range mapped to logV_min/logV_max
    n_points   : number of sampled points on the new log(V) grid
    """

    group_starts, group_ends = _compute_group_spans(groups)
    if group_index < 0 or group_index >= len(groups):
        raise IndexError(f"group_index {group_index} out of range (0..{len(groups)-1})")

    # Original data of the selected group
    s = group_starts[group_index]
    e = group_ends[group_index]

    Xg = X[s:e]
    yg = y[s:e]

    # Original logV values (first column of X)
    logV_orig = Xg[:, 0]

    # ---- Extract the group theta features (assumed constant within the group) ----
    theta = Xg[0, 1:].copy()   # shape: (input_dim - 1,)

    # Check whether theta is constant inside the group (not required, but useful)
    if not np.allclose(Xg[:, 1:], theta[None, :], atol=1e-8):
        print(f"[WARN] theta features are not perfectly constant inside group {group_index}; using the first row as the representative theta.")

    input_dim = X.shape[1]
    if theta.shape[0] != input_dim - 1:
        raise ValueError(
            f"theta dim mismatch: theta has {theta.shape[0]}, but X has dim={input_dim}"
        )

    # ---- Build the new logV grid ----
    logV_min = np.log(V_min)
    logV_max = np.log(V_max)
    logV_new = np.linspace(logV_min, logV_max, n_points)

    # Build the new feature matrix X_new: [logV_new, theta]
    X_new = np.zeros((n_points, input_dim), dtype=np.float32)
    X_new[:, 0] = logV_new
    X_new[:, 1:] = theta[None, :]

    # ---- Predict on both the original points and the new grid ----
    y_pred_orig = model.predict(Xg)
    y_pred_new = model.predict(X_new)

    # ---- Plot: original samples vs predictions on the new V grid ----
    if title is None:
        title = f"V-sweep test for group {group_index}"

    plt.figure(figsize=(7, 5))
    plt.title(title)

    # Original data (ground truth)
    plt.scatter(logV_orig, yg, label="true (orig grid)", s=25)

    # Model prediction on the original grid
    plt.plot(logV_orig, y_pred_orig, label="model on orig grid", linewidth=2, alpha=0.7)

    # Prediction on the new V grid
    plt.plot(logV_new, y_pred_new, label="model on new V-grid", linewidth=2, linestyle="--")

    plt.xlabel("log V")
    plt.ylabel("log E")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return logV_new, y_pred_new

# =============================================================================
# Unified Experiment Entry
# =============================================================================

def run_experiment(
    model_kind: str = "powerlaw",
    h5_file: str = "energy_scan_results.h5",
    only_analyze: bool = False,
):
    """
    Unified experiment entry point:

    Parameters
    ----
    model_kind:
        "powerlaw"   -> use PowerLawSeparableModel
        "parametric" -> use ParametricEnergyModel
        "mlp"        -> use MLPEnergyModel
        "ann"        -> use ANNEnergyModel
        "all"        -> train and compare all models in sequence
    h5_file:
        HDF5 path

    Returns
    ----
    If model_kind != "all":
        model, groups, X, y, (X_train, y_train, X_val, y_val), metrics

    If model_kind == "all":
        results, groups, X, y, (X_train, y_train, X_val, y_val)

        Here `results` has the form:
            {
              "powerlaw": {"time": ..., "mse": ..., "mae": ..., "mape": ..., "r2": ...},
              "parametric": {...},
              "mlp": {...},
              "ann": {...},
            }
    """
    # 1. Load data
    groups, X, y = load_data(h5_file)

    # 2. Split train / val by group
    X_train, y_train, X_val, y_val = split_train_val_by_group(
        groups,
        X, y,
        val_ratio=0.2,
        seed=42,
    )

    mk = model_kind.lower()
    model_dir = os.path.dirname(h5_file) or "."

    def _default_model_path(kind: str) -> str:
        return os.path.join(model_dir, f"{kind}_model.pkl")

    def _load_saved_model(kind: str):
        model_path = _default_model_path(kind)
        print(f"Loading {kind} from {model_path} ...")
        if kind == "powerlaw":
            return PowerLawSeparableModel.load(model_path)
        if kind == "parametric":
            return ParametricEnergyModel.load(model_path)
        if kind == "mlp":
            return MLPEnergyModel.load(model_path)
        if kind == "ann":
            return ANNEnergyModel.load(model_path)
        raise ValueError(f"Unknown model kind '{kind}'")

    # ------------------------------------------------------------------
    # Single-model mode: keep the original behavior
    # ------------------------------------------------------------------
    if mk in ("powerlaw", "parametric", "mlp", "ann"):
        if only_analyze:
            model = _load_saved_model(mk)
        elif mk == "powerlaw":
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
                regress_type="ridge",     # or "linear"
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
    # "all" mode: train all models -> save -> reload -> evaluate and time
    # ------------------------------------------------------------------
    if mk == "all":
        results = {}
        model_specs = ["powerlaw", "parametric", "mlp", "ann"]

        for spec in model_specs:
            print("=" * 80)
            if only_analyze:
                print(f"[ALL] Loading pre-trained model: {spec}")
                t0 = time.perf_counter()
                loaded_model = _load_saved_model(spec)
                metrics = evaluate_model(loaded_model, X_val, y_val, name=spec)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                results[spec] = {"time": elapsed}
                results[spec].update(metrics)
                continue

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

            # Save the model to disk
            model_path = _default_model_path(spec)
            print(f"[ALL] Saving {spec} to {model_path}")
            model.save(model_path)

            # Timing: load from disk and evaluate on the validation set
            print(f"[ALL] Loading {spec} from {model_path} and evaluating ...")
            t0 = time.perf_counter()
            loaded_model = model.load(model_path)
            metrics = evaluate_model(loaded_model, X_val, y_val, name=spec)
            t1 = time.perf_counter()
            elapsed = t1 - t0

            results[spec] = {"time": elapsed}
            results[spec].update(metrics)

        # Return results and data for plotting in main or Spyder
        return results, groups, X, y, (X_train, y_train, X_val, y_val)

    # Invalid input branch
    raise ValueError(
        f"Unknown model_kind '{model_kind}', must be 'powerlaw', 'parametric', 'mlp', 'ann' or 'all'."
    )



# =============================================================================
# main: choose the model type and visualize one group
# =============================================================================

if __name__ == "__main__":
    # Edit these parameters here to quickly test different setups in Spyder
    data_path = r"C:\Users\px2030\Code\PSD_opt\breakage-rate-model\data"
    # data_path = os.environ.get('STORAGE_PATH')
    H5_FILE = os.path.join(data_path, "energy_scan_results.h5")
    MODEL_KIND = "mlp"   # "powerlaw" / "parametric" / "mlp" / "ann" / "all"
    ONLY_ANALYZE = True # True -> load an existing saved model and only run analysis
    GROUP_INDEX = 0      # group index to inspect in non-"all" mode

    if MODEL_KIND.lower() == "all":
        # Run all mode: train + save + reload + evaluate
        results, groups, X, Y, split_data = run_experiment(
            model_kind="all",
            h5_file=H5_FILE,
            only_analyze=ONLY_ANALYZE,
        )

        GLOBAL_RESULTS = results
        GLOBAL_GROUPS = groups
        GLOBAL_X = X
        GLOBAL_Y = Y
        GLOBAL_SPLIT = split_data

        # Plot 5 bar charts: time, mse, mae, mape, r2
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
        # Single-model mode: keep the original behavior
        model, groups, X, y, split_data, metrics = run_experiment(
            model_kind=MODEL_KIND,
            h5_file=H5_FILE,
            only_analyze=ONLY_ANALYZE,
        )

        # Expose results as globals for direct access in Spyder
        GLOBAL_MODEL = model
        GLOBAL_GROUPS = groups
        GLOBAL_X = X
        GLOBAL_Y = y
        GLOBAL_SPLIT = split_data
        GLOBAL_METRICS = metrics

        # Plot one group directly when running this script
        mk = MODEL_KIND.lower()
        if mk in ("powerlaw", "parametric"):
            model.analyze_one_group(
                groups,
                group_index=GROUP_INDEX,
                target="log_mean",  # the model returns logE
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
                group_index=5,       # group to inspect
                V_min=10.0,
                V_max=100000.0,
                n_points=100,
            )

    
    


