# -*- coding: utf-8 -*-
"""Spyder-friendly training and validation entry point for energy surrogates.

The HDF5 reader deliberately uses ``load_samples=False`` here: fitting the
default ``log_mean`` target only needs the completed mean energy curves.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Dict, Sequence

import numpy as np

from breakage_rate_model.ann_model import ANNEnergyModel
from breakage_rate_model.data_io import EnergyGroupRecord, load_energy_groups_from_h5
from breakage_rate_model.datasets import EnergyDataset, build_energy_dataset
from breakage_rate_model.features import (
    DEFAULT_ACTIVE_FEATURE_NAMES,
    FULL_ENERGY_FEATURE_NAMES,
    full_energy_features,
    normalize_strength_normalization,
    preprocess_log_energy_targets,
    strength_log_scale,
)
from breakage_rate_model.mlp_model import MLPEnergyModel
from breakage_rate_model.parametric_model import ParametricEnergyModel
from breakage_rate_model.powerlaw_separable import PowerLawSeparableModel


@dataclass(frozen=True)
class GroupSplit:
    """A group-disjoint split, with sample rows selected through ``meta_idx``."""

    train_group_indices: np.ndarray
    val_group_indices: np.ndarray
    train_groups: list[EnergyGroupRecord]
    val_groups: list[EnergyGroupRecord]
    X_train: np.ndarray
    y_train: np.ndarray
    meta_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    meta_val: np.ndarray


PURE_ACTIVE_FEATURE_NAMES = ("logV", "log_gamma")


def load_data(h5_file: str) -> tuple[list[EnergyGroupRecord], EnergyDataset]:
    """Load only required curve means and build the primary log-mean dataset."""
    print(f"Loading completed v2 energy groups from {h5_file} ...")
    groups = load_energy_groups_from_h5(h5_file, load_samples=False)
    dataset = build_energy_dataset(groups, per_sample=False, target="log_mean")
    print(f"Loaded {len(groups)} groups and {dataset.X.shape[0]} log-mean samples.")
    return groups, dataset


def split_train_val_by_group(
    groups: Sequence[EnergyGroupRecord],
    dataset: EnergyDataset,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> GroupSplit:
    """Randomly split group ids and select samples using ``EnergyDataset.meta_idx``."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1.")
    if len(groups) != len(dataset.groups):
        raise ValueError("groups and dataset.groups must describe the same group list.")
    if len(groups) < 2:
        raise ValueError("At least two groups are required for a train/validation split.")
    if dataset.meta_idx.ndim != 2 or dataset.meta_idx.shape[1] != 3:
        raise ValueError("EnergyDataset.meta_idx must have shape (n_samples, 3).")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(groups))
    n_val = int(np.ceil(val_ratio * len(groups)))
    n_val = min(max(n_val, 1), len(groups) - 1)
    val_group_indices = np.sort(shuffled[:n_val])
    train_group_indices = np.sort(shuffled[n_val:])

    sample_group_indices = dataset.meta_idx[:, 0]
    train_mask = np.isin(sample_group_indices, train_group_indices)
    val_mask = np.isin(sample_group_indices, val_group_indices)
    if np.any(train_mask & val_mask) or not np.all(train_mask | val_mask):
        raise RuntimeError("Sample selection is not a disjoint and exhaustive group split.")
    if not np.any(train_mask) or not np.any(val_mask):
        raise RuntimeError("Group split produced an empty sample partition.")

    return GroupSplit(
        train_group_indices=train_group_indices,
        val_group_indices=val_group_indices,
        train_groups=[groups[index] for index in train_group_indices],
        val_groups=[groups[index] for index in val_group_indices],
        X_train=dataset.X[train_mask],
        y_train=dataset.y[train_mask],
        meta_train=dataset.meta_idx[train_mask],
        X_val=dataset.X[val_mask],
        y_val=dataset.y[val_mask],
        meta_val=dataset.meta_idx[val_mask],
    )


def fit_powerlaw_model(
    train_groups: Sequence[EnergyGroupRecord],
    *,
    active_feature_names: Sequence[str] = DEFAULT_ACTIVE_FEATURE_NAMES,
    strength_normalization: str = "none",
) -> PowerLawSeparableModel:
    """Fit the structural PowerLaw model from training groups only."""
    model = PowerLawSeparableModel(
        fit_mode="logE",
        pure_powerlaw_if_no_plateau=True,
        enable_tail=True,
        plateau_weight=4.0,
        regress_type="ridge",
        ridge_lambda=1e2,
        active_feature_names=active_feature_names,
        strength_normalization=strength_normalization,
    )
    model.fit(None, None, groups=train_groups)
    return model


def fit_parametric_model(
    train_groups: Sequence[EnergyGroupRecord],
    *,
    active_feature_names: Sequence[str] = DEFAULT_ACTIVE_FEATURE_NAMES,
    strength_normalization: str = "none",
) -> ParametricEnergyModel:
    """Fit the trend-plus-residual Parametric model from training groups only."""
    model = ParametricEnergyModel(
        fit_mode="logE",
        pure_powerlaw_if_no_plateau=True,
        residual_type="ridge",
        residual_lambda=1e1,
        enable_tail=True,
        plateau_weight=3.0,
        active_feature_names=active_feature_names,
        strength_normalization=strength_normalization,
    )
    model.fit(None, None, groups=train_groups)
    return model


def fit_mlp_model(
    split: GroupSplit,
    *,
    active_feature_names: Sequence[str] = DEFAULT_ACTIVE_FEATURE_NAMES,
    strength_normalization: str = "none",
) -> MLPEnergyModel:
    """Fit the MLP on training rows while reserving validation rows for stopping."""
    model = MLPEnergyModel(
        hidden_sizes=(256, 256),
        activation="relu",
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=200,
        batch_size=128,
        patience=20,
        seed=42,
        active_feature_names=active_feature_names,
        strength_normalization=strength_normalization,
    )
    model.fit(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)
    return model


def fit_ann_model(
    split: GroupSplit,
    *,
    active_feature_names: Sequence[str] = DEFAULT_ACTIVE_FEATURE_NAMES,
    strength_normalization: str = "none",
) -> ANNEnergyModel:
    """Fit the ANN on training rows while reserving validation rows for stopping."""
    model = ANNEnergyModel(
        hidden_sizes=(256, 256, 128),
        activation="silu",
        dropout=0.1,
        lr=3e-4,
        weight_decay=1e-4,
        max_epochs=300,
        batch_size=256,
        patience=50,
        seed=42,
        active_feature_names=active_feature_names,
        strength_normalization=strength_normalization,
    )
    model.fit(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)
    return model


def evaluate_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    meta_idx: np.ndarray,
) -> Dict[str, float]:
    """Report log-scale, energy-scale, and equal-group-weighted validation errors."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    meta_idx = np.asarray(meta_idx, dtype=int)
    if y_true_log.ndim != 1 or y_pred_log.ndim != 1:
        raise ValueError("Both targets must be one-dimensional.")
    if y_true_log.shape != y_pred_log.shape or y_true_log.size == 0:
        raise ValueError("Targets must be non-empty and have identical shapes.")
    if meta_idx.shape != (y_true_log.size, 3):
        raise ValueError("meta_idx must align with targets and have three columns.")
    if not np.all(np.isfinite(y_true_log)) or not np.all(np.isfinite(y_pred_log)):
        raise ValueError("Evaluation targets and predictions must be finite.")

    error_log = y_pred_log - y_true_log
    energy_true = np.exp(y_true_log)
    energy_pred = np.exp(y_pred_log)
    if not np.all(np.isfinite(energy_true)) or not np.all(np.isfinite(energy_pred)):
        raise FloatingPointError("Energy-scale evaluation overflowed or produced non-finite values.")
    relative_error = (energy_pred - energy_true) / energy_true
    ape = np.abs(relative_error)

    ss_tot = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
    r2_log = np.nan if ss_tot == 0.0 else float(1.0 - np.sum(error_log**2) / ss_tot)
    group_mae_values = []
    group_mape_values = []
    for group_index in np.unique(meta_idx[:, 0]):
        mask = meta_idx[:, 0] == group_index
        group_mae_values.append(float(np.mean(np.abs(error_log[mask]))))
        group_mape_values.append(float(np.mean(ape[mask])))

    return {
        "rmse_log": float(np.sqrt(np.mean(error_log**2))),
        "mae_log": float(np.mean(np.abs(error_log))),
        "r2_log": r2_log,
        "mape_E": float(np.mean(ape)),
        "median_ape_E": float(np.median(ape)),
        "rmse_rel_E": float(np.sqrt(np.mean(relative_error**2))),
        "group_mae_log": float(np.mean(group_mae_values)),
        "group_mape_E": float(np.mean(group_mape_values)),
    }


def evaluate_model(model, split: GroupSplit, name: str) -> Dict[str, float]:
    """Predict and evaluate one model only on held-out validation groups."""
    started = time.perf_counter()
    prediction = model.predict(split.X_val)
    predict_seconds = time.perf_counter() - started
    y_val_model_unit = preprocess_log_energy_targets(
        split.X_val, split.y_val, model.strength_normalization
    )
    metrics = evaluate_predictions(y_val_model_unit, prediction, split.meta_val)
    metrics["predict_seconds"] = predict_seconds
    print(f"Validation metrics for {name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6g}")
    return metrics


def _model_filename(kind: str, model_prefix: str = "") -> str:
    """Return one unambiguous model filename for a dataset-specific model set."""
    if not isinstance(model_prefix, str):
        raise TypeError("model_prefix must be a string.")
    prefix = model_prefix.strip().lower()
    if not prefix:
        return f"{kind}_model.pkl"
    if not prefix.replace("_", "").isalnum():
        raise ValueError("model_prefix must contain only letters, digits, and underscores.")
    return f"{prefix}_{kind}_model.pkl"


def _load_saved_model(kind: str, model_path: str, *, strength_normalization: str | None = None):
    classes = {
        "powerlaw": PowerLawSeparableModel,
        "parametric": ParametricEnergyModel,
        "mlp": MLPEnergyModel,
        "ann": ANNEnergyModel,
    }
    if kind not in classes:
        raise ValueError(f"Unknown model kind {kind!r}.")
    expected_type = classes[kind]
    model = expected_type.load(model_path)
    if not isinstance(model, expected_type):
        raise TypeError(
            f"Model file {model_path!r} was requested as {kind!r}, but contains "
            f"{type(model).__name__} instead of {expected_type.__name__}."
        )
    if not model.is_fitted:
        raise RuntimeError(f"Saved {kind!r} model at {model_path!r} is not fitted.")
    if strength_normalization is not None:
        expected_mode = normalize_strength_normalization(strength_normalization)
        if model.strength_normalization != expected_mode:
            raise ValueError(
                f"Saved {kind!r} model at {model_path!r} has strength_normalization="
                f"{model.strength_normalization!r}, expected {expected_mode!r}."
            )
    return model


def _report_group_features(group_index: int, group: EnergyGroupRecord) -> None:
    """Print the canonical ten-feature contract for one energy curve."""
    features_at_first_volume = full_energy_features(group, float(group.V[0]))
    log_volumes = np.log(group.V)

    print(f"Selected group index={group_index}, key={group.key!r}")
    print("Canonical model features:")
    for feature_index, feature_name in enumerate(FULL_ENERGY_FEATURE_NAMES):
        if feature_name == "logV":
            print(
                f"  {feature_name}: variable across the curve; "
                f"V=[{group.V[0]:.6g}, {group.V[-1]:.6g}], "
                f"logV=[{log_volumes[0]:.6g}, {log_volumes[-1]:.6g}]"
            )
        else:
            print(f"  {feature_name}: {features_at_first_volume[feature_index]:.6g}")
    print("Group curve metadata:")
    print(
        f"  A0={group.A0:.6g}, Np=[{group.Np[0]}, {group.Np[-1]}], "
        f"n_volume_points={group.V.size}"
    )


def compare_models_on_group(
    h5_file: str,
    group_index: int,
    *,
    model_prefix: str = "",
    strength_normalization: str = "none",
    show: bool = True,
):
    """Plot one group mean-energy curve against all four saved model predictions.

    The input file is read with ``load_samples=False``.  The four model files
    are expected beside the HDF5 file and must be named
    ``<kind>_model.pkl`` for ``powerlaw``, ``parametric``, ``mlp`` and ``ann``.
    The observed curve and all predictions use the model's persisted energy
    unit.  Mixed models use ``E/S0``; pure reference models use ``E``.
    """
    if isinstance(group_index, bool) or not isinstance(group_index, (int, np.integer)):
        raise TypeError("group_index must be a zero-based integer.")

    groups = load_energy_groups_from_h5(h5_file, load_samples=False)
    group_index = int(group_index)
    if group_index < 0 or group_index >= len(groups):
        raise IndexError(
            f"group_index must satisfy 0 <= group_index < {len(groups)}, got {group_index}."
        )
    group = groups[group_index]
    if group.E_samples is not None:
        raise RuntimeError("Group comparison must not load raw E_samples.")

    _report_group_features(group_index, group)

    strength_normalization = normalize_strength_normalization(strength_normalization)
    X_group = np.vstack(
        [full_energy_features(group, float(volume)) for volume in group.V]
    )
    log_scale = strength_log_scale(np.asarray(group.STR, dtype=float))[0]
    energy_scale = float(np.exp(log_scale)) if strength_normalization == "geometric_mean_relative" else 1.0
    observed_energy = group.E_mean / energy_scale
    model_kinds = ("powerlaw", "parametric", "mlp", "ann")
    model_directory = os.path.dirname(os.path.abspath(h5_file))
    prediction_energy: dict[str, np.ndarray] = {}
    for kind in model_kinds:
        model_path = os.path.join(model_directory, _model_filename(kind, model_prefix))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Required saved {kind!r} model is missing: {model_path}"
            )
        model = _load_saved_model(
            kind,
            model_path,
            strength_normalization=strength_normalization,
        )
        prediction_log = np.asarray(model.predict(X_group), dtype=float)
        if prediction_log.shape != (group.V.size,):
            raise RuntimeError(
                f"{kind} prediction must have shape ({group.V.size},), "
                f"got {prediction_log.shape}."
            )
        if not np.all(np.isfinite(prediction_log)):
            raise FloatingPointError(f"{kind} produced non-finite log-energy predictions.")
        with np.errstate(over="raise", invalid="raise"):
            prediction = np.exp(prediction_log)
        if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
            raise FloatingPointError(f"{kind} produced invalid energy predictions.")
        prediction_energy[kind] = prediction

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        group.V,
        observed_energy,
        "o",
        color="black",
        label="E_mean/S0" if strength_normalization == "geometric_mean_relative" else "E_mean",
        zorder=3,
    )
    for kind in model_kinds:
        axis.plot(group.V, prediction_energy[kind], linewidth=2.0, label=kind)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("particle volume V")
    axis.set_ylabel(
        "relative breakage energy E/S0"
        if strength_normalization == "geometric_mean_relative"
        else "reference breakage energy E"
    )
    axis.set_title(f"Energy curve comparison: group {group_index} ({group.key})")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def _fit_model(
    kind: str,
    split: GroupSplit,
    active_feature_names: Sequence[str],
    strength_normalization: str,
):
    if kind == "powerlaw":
        return fit_powerlaw_model(
            split.train_groups,
            active_feature_names=active_feature_names,
            strength_normalization=strength_normalization,
        )
    if kind == "parametric":
        return fit_parametric_model(
            split.train_groups,
            active_feature_names=active_feature_names,
            strength_normalization=strength_normalization,
        )
    if kind == "mlp":
        return fit_mlp_model(
            split,
            active_feature_names=active_feature_names,
            strength_normalization=strength_normalization,
        )
    if kind == "ann":
        return fit_ann_model(
            split,
            active_feature_names=active_feature_names,
            strength_normalization=strength_normalization,
        )
    raise ValueError(f"Unknown model kind {kind!r}.")


def _validate_pure_groups(groups: Sequence[EnergyGroupRecord]) -> None:
    """Require one unit-strength, single-composition pure reference dataset."""
    if not groups:
        raise ValueError("Pure-model training requires at least one energy group.")
    unit_strength = np.ones(3, dtype=float)
    pure_x1 = float(groups[0].X1)
    if pure_x1 not in (0.0, 1.0):
        raise ValueError("Pure-model groups must have X1 exactly 0.0 or 1.0.")

    reference = groups[0]
    fixed_reference = (
        reference.NO_FRAG,
        reference.int_bre,
        reference.Df,
        reference.MAS,
        reference.A0,
        reference.X1,
    )
    for group in groups:
        if not np.array_equal(np.asarray(group.STR, dtype=float), unit_strength):
            raise ValueError("Pure-model groups must use STR=(1.0, 1.0, 1.0).")
        fixed_values = (
            group.NO_FRAG,
            group.int_bre,
            group.Df,
            group.MAS,
            group.A0,
            group.X1,
        )
        if fixed_values != fixed_reference:
            raise ValueError("Pure-model groups must keep NO_FRAG, int_bre, Df, MAS, A0, and X1 fixed.")


def run_experiment(
    model_kind: str = "powerlaw",
    h5_file: str = "energy_scan_results.h5",
    only_analyze: bool = False,
    active_feature_names: Sequence[str] = DEFAULT_ACTIVE_FEATURE_NAMES,
    strength_normalization: str = "none",
    model_prefix: str = "",
    dataset_kind: str | None = None,
):
    """Train or load one/all models and evaluate only validation-group samples."""
    strength_normalization = normalize_strength_normalization(strength_normalization)
    if dataset_kind not in (None, "mixed", "pure"):
        raise ValueError("dataset_kind must be None, 'mixed', or 'pure'.")
    groups, dataset = load_data(h5_file)
    if dataset_kind == "mixed" and strength_normalization != "geometric_mean_relative":
        raise ValueError("Mixed-model training requires geometric_mean_relative normalization.")
    if dataset_kind == "pure":
        _validate_pure_groups(groups)
        if tuple(active_feature_names) != PURE_ACTIVE_FEATURE_NAMES:
            raise ValueError(
                f"Pure-model training requires active features {PURE_ACTIVE_FEATURE_NAMES}."
            )
        if strength_normalization != "none":
            raise ValueError("Pure reference-model training requires strength_normalization='none'.")
    split = split_train_val_by_group(groups, dataset, val_ratio=0.2, seed=42)
    kinds = ("powerlaw", "parametric", "mlp", "ann")
    requested = model_kind.lower()
    if requested != "all" and requested not in kinds:
        raise ValueError(f"Unknown model_kind {model_kind!r}.")
    run_kinds = kinds if requested == "all" else (requested,)
    model_dir = os.path.dirname(h5_file) or "."
    results: Dict[str, Dict[str, float]] = {}
    models = {}

    for kind in run_kinds:
        model_path = os.path.join(model_dir, _model_filename(kind, model_prefix))
        if only_analyze:
            started = time.perf_counter()
            model = _load_saved_model(
                kind,
                model_path,
                strength_normalization=strength_normalization,
            )
            load_seconds = time.perf_counter() - started
        else:
            started = time.perf_counter()
            model = _fit_model(kind, split, active_feature_names, strength_normalization)
            fit_seconds = time.perf_counter() - started
            model.save(model_path)
            started = time.perf_counter()
            model = _load_saved_model(
                kind,
                model_path,
                strength_normalization=strength_normalization,
            )
            load_seconds = time.perf_counter() - started
        metrics = evaluate_model(model, split, kind)
        metrics["load_seconds"] = load_seconds
        if not only_analyze:
            metrics["fit_seconds"] = fit_seconds
        results[kind] = metrics
        models[kind] = model

    if requested == "all":
        return results, models, groups, dataset, split
    return models[requested], groups, dataset, split, results[requested]


if __name__ == "__main__":
    # Edit these values directly when debugging through Spyder.
    DATA_PATH = r"D:\Codex_tem\LMC\energy_pool"
    MIXED_H5_FILE = os.path.join(DATA_PATH, "energy_scan_results.h5")
    PURE_H5_FILE = os.path.join(DATA_PATH, "psd_data_pure.h5")
    RUN_MODE = "train"  # "train" or "compare_group"
    DATASET_MODE = "both"  # "mixed" / "pure" / "both"; used only by RUN_MODE="train".
    COMPARE_DATASET = "mixed"  # "mixed" / "pure"; used only by RUN_MODE="compare_group".
    MODEL_KIND = "all"  # powerlaw / parametric / mlp / ann / all
    ONLY_ANALYZE = True
    GROUP_INDEX = 0  # Zero-based group index, used only by RUN_MODE="compare_group".

    DATASET_SETTINGS = {
        "mixed": {
            "h5_file": MIXED_H5_FILE,
            "active_feature_names": DEFAULT_ACTIVE_FEATURE_NAMES,
            "strength_normalization": "geometric_mean_relative",
            "model_prefix": "mixed",
        },
        "pure": {
            "h5_file": PURE_H5_FILE,
            "active_feature_names": PURE_ACTIVE_FEATURE_NAMES,
            "strength_normalization": "none",
            "model_prefix": "pure",
        },
    }

    if RUN_MODE == "train":
        if DATASET_MODE not in ("mixed", "pure", "both"):
            raise ValueError(
                "DATASET_MODE must be 'mixed', 'pure', or 'both', "
                f"got {DATASET_MODE!r}."
            )
        selected_datasets = ("mixed", "pure") if DATASET_MODE == "both" else (DATASET_MODE,)
        EXPERIMENT_RESULT = {}
        for selected_dataset in selected_datasets:
            settings = DATASET_SETTINGS[selected_dataset]
            EXPERIMENT_RESULT[selected_dataset] = run_experiment(
                model_kind=MODEL_KIND,
                h5_file=settings["h5_file"],
                only_analyze=ONLY_ANALYZE,
                active_feature_names=settings["active_feature_names"],
                strength_normalization=settings["strength_normalization"],
                model_prefix=settings["model_prefix"],
                dataset_kind=selected_dataset,
            )
    elif RUN_MODE == "compare_group":
        if COMPARE_DATASET not in DATASET_SETTINGS:
            raise ValueError(
                f"COMPARE_DATASET must be 'mixed' or 'pure', got {COMPARE_DATASET!r}."
            )
        settings = DATASET_SETTINGS[COMPARE_DATASET]
        GROUP_COMPARISON_FIGURE = compare_models_on_group(
            settings["h5_file"],
            GROUP_INDEX,
            model_prefix=settings["model_prefix"],
            strength_normalization=settings["strength_normalization"],
        )
    else:
        raise ValueError(
            f"RUN_MODE must be 'train' or 'compare_group', got {RUN_MODE!r}."
        )
