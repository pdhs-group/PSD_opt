# -*- coding: utf-8 -*-
"""Spyder-friendly CMA-ES hyperparameter searches for energy surrogate models.

This module intentionally imports the data-loading, group-splitting, and
evaluation helpers from the adjacent ``train_4_models.py`` entry point.  Therefore a
search uses exactly the same completed-v2 HDF5 contract, group-disjoint split,
and validation metrics as a normal training run, without writing a model for
every Optuna trial.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler

from breakage_rate_model.ann_model import ANNEnergyModel
from breakage_rate_model.features import DEFAULT_ACTIVE_FEATURE_NAMES
from breakage_rate_model.mlp_model import MLPEnergyModel
from breakage_rate_model.parametric_model import ParametricEnergyModel
from breakage_rate_model.powerlaw_separable import PowerLawSeparableModel


# =============================================================================
# Spyder configuration
# =============================================================================

DATA_DIRECTORY = Path(r"D:\LMC\energy_pool")
H5_FILENAME = "energy_scan_results.h5"
SEARCH_KIND = "powerlaw"  # powerlaw / parametric / mlp / ann
N_TRIALS = 400  # Number of *new* trials appended by each invocation.
SAMPLER_SEED = 42
VALIDATION_RATIO = 0.2
SPLIT_SEED = 42
OBJECTIVE_METRIC = "group_mae_log"
ACTIVE_FEATURE_NAMES = DEFAULT_ACTIVE_FEATURE_NAMES
SHOW_TRAINING_OUTPUT = False

_VALID_SEARCH_KINDS = ("powerlaw", "parametric", "mlp", "ann")
_VALID_METRICS = {
    "rmse_log",
    "mae_log",
    "r2_log",
    "mape_E",
    "median_ape_E",
    "rmse_rel_E",
    "group_mae_log",
    "group_mape_E",
}
_CONFIG_ATTR = "hyperparameter_search_config_json"


@dataclass(frozen=True)
class SearchContext:
    """Invariant data and settings shared by all trials in one search."""

    h5_file: Path
    output_directory: Path
    split: Any
    training_module: ModuleType
    n_trials: int
    sampler_seed: int
    objective_metric: str
    show_training_output: bool


@dataclass(frozen=True)
class SearchResult:
    """Persistent artifacts and final re-fit results of one search."""

    kind: str
    study_name: str
    best_trial_number: int
    best_params: dict[str, Any]
    best_objective: float
    validation_metrics: dict[str, float]
    fit_seconds: float
    predict_seconds: float
    database_path: Path
    trials_csv_path: Path
    best_result_path: Path
    best_model_path: Path


def _load_training_module() -> ModuleType:
    """Load the adjacent training entry point without relying on the working directory."""
    path = Path(__file__).with_name("train_4_models.py")
    spec = importlib.util.spec_from_file_location("energy_training_entry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import specification for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_search_context(
    h5_file: str | Path,
    *,
    output_directory: str | Path,
    n_trials: int,
    sampler_seed: int,
    validation_ratio: float,
    split_seed: int,
    objective_metric: str,
    show_training_output: bool,
) -> SearchContext:
    """Load curve means once and create the one fixed group-disjoint split."""
    h5_path = Path(h5_file).resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"Energy-pool HDF5 file does not exist: {h5_path}")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if objective_metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown objective_metric {objective_metric!r}; expected one of {sorted(_VALID_METRICS)}."
        )

    training_module = _load_training_module()
    groups, dataset = training_module.load_data(str(h5_path))
    split = training_module.split_train_val_by_group(
        groups, dataset, val_ratio=validation_ratio, seed=split_seed
    )
    return SearchContext(
        h5_file=h5_path,
        output_directory=Path(output_directory).resolve(),
        split=split,
        training_module=training_module,
        n_trials=int(n_trials),
        sampler_seed=int(sampler_seed),
        objective_metric=objective_metric,
        show_training_output=bool(show_training_output),
    )


def _validate_float_range(
    name: str,
    value_range: tuple[float, float],
    *,
    positive: bool = False,
    minimum: float | None = None,
) -> tuple[float, float]:
    if len(value_range) != 2:
        raise ValueError(f"{name} must be a two-element (low, high) range.")
    low, high = (float(value_range[0]), float(value_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"{name} must contain finite values satisfying low < high.")
    if positive and low <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    if minimum is not None and low < minimum:
        raise ValueError(f"{name} must have low >= {minimum}.")
    return low, high


def _validate_int_range(name: str, value_range: tuple[int, int], *, minimum: int) -> tuple[int, int]:
    if len(value_range) != 2:
        raise ValueError(f"{name} must be a two-element (low, high) range.")
    low, high = int(value_range[0]), int(value_range[1])
    if low >= high or low < minimum:
        raise ValueError(f"{name} must satisfy {minimum} <= low < high.")
    return low, high


@contextmanager
def _training_output(enabled: bool):
    """Silence only normal model progress lines; exceptions still propagate unchanged."""
    if enabled:
        with nullcontext():
            yield
    else:
        with redirect_stdout(io.StringIO()):
            yield


def _json_value(value: Any) -> Any:
    """Convert numerical metadata to strict JSON values without changing model data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _split_metadata(split: Any) -> dict[str, list[int]]:
    return {
        "train_group_indices": [int(index) for index in split.train_group_indices],
        "val_group_indices": [int(index) for index in split.val_group_indices],
    }


def _release_study_storage(study: optuna.study.Study) -> None:
    """Release SQLite sessions and pooled connections so Windows can unlock the database."""
    study._storage.remove_session()
    study._storage._backend.engine.dispose()


def _create_or_load_study(
    context: SearchContext,
    *,
    kind: str,
    configuration: Mapping[str, Any],
) -> tuple[optuna.study.Study, Path, str]:
    context.output_directory.mkdir(parents=True, exist_ok=True)
    database_path = context.output_directory / f"{kind}_study.sqlite3"
    storage_url = f"sqlite:///{database_path.as_posix()}"
    study_name = f"energy_surrogate_{kind}_cmaes"
    sampler = CmaEsSampler(seed=context.sampler_seed, n_startup_trials=10)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    try:
        fingerprint = _canonical_json(configuration)
        previous_fingerprint = study.user_attrs.get(_CONFIG_ATTR)
        if previous_fingerprint is None:
            if study.trials:
                raise ValueError(
                    f"Existing study {study_name!r} has trials but no configuration fingerprint. "
                    "Use a new output directory rather than mixing incomparable trials."
                )
            study.set_user_attr(_CONFIG_ATTR, fingerprint)
        elif previous_fingerprint != fingerprint:
            raise ValueError(
                f"Existing study {study_name!r} was created with a different data, split, "
                "search-space, or fixed-parameter configuration. Use a new output directory."
            )
    except BaseException:
        # RDBStorage keeps a thread-local SQLite session on Windows.  Re-raise
        # the original failure after releasing its file handle.
        _release_study_storage(study)
        raise
    return study, database_path, study_name


def _fit_and_evaluate(
    model: Any,
    context: SearchContext,
    fit_model: Callable[[Any], None],
) -> tuple[dict[str, float], float, float]:
    with _training_output(context.show_training_output):
        started = time.perf_counter()
        fit_model(model)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict(context.split.X_val)
        predict_seconds = time.perf_counter() - started
    metrics = context.training_module.evaluate_predictions(
        context.split.y_val, prediction, context.split.meta_val
    )
    return {name: float(value) for name, value in metrics.items()}, fit_seconds, predict_seconds


def _run_search(
    context: SearchContext,
    *,
    kind: str,
    configuration: Mapping[str, Any],
    build_trial_model: Callable[[optuna.trial.Trial], Any],
    build_best_model: Callable[[Mapping[str, Any]], Any],
    fit_model: Callable[[Any], None],
) -> SearchResult:
    if kind not in _VALID_SEARCH_KINDS:
        raise ValueError(f"Unknown search kind {kind!r}.")
    if context.objective_metric not in _VALID_METRICS:
        raise ValueError(f"Unknown objective metric {context.objective_metric!r}.")

    study, database_path, study_name = _create_or_load_study(
        context, kind=kind, configuration=configuration
    )

    def objective(trial: optuna.trial.Trial) -> float:
        model = build_trial_model(trial)
        metrics, fit_seconds, predict_seconds = _fit_and_evaluate(model, context, fit_model)
        objective_value = metrics[context.objective_metric]
        if not np.isfinite(objective_value):
            raise FloatingPointError(
                f"Trial {trial.number} produced a non-finite {context.objective_metric}."
            )
        trial.set_user_attr("validation_metrics", _json_value(metrics))
        trial.set_user_attr("fit_seconds", float(fit_seconds))
        trial.set_user_attr("predict_seconds", float(predict_seconds))
        return float(objective_value)

    try:
        study.optimize(objective, n_trials=context.n_trials)
        best_trial = study.best_trial
        best_model = build_best_model(best_trial.params)
        final_metrics, fit_seconds, predict_seconds = _fit_and_evaluate(best_model, context, fit_model)

        trials_csv_path = context.output_directory / f"{kind}_trials.csv"
        study.trials_dataframe().to_csv(trials_csv_path, index=False)

        best_model_path = context.output_directory / f"best_{kind}_model.pkl"
        best_model.save(str(best_model_path))
        best_result_path = context.output_directory / f"{kind}_best.json"
        result_payload = {
            "study_name": study_name,
            "kind": kind,
            "objective_metric": context.objective_metric,
            "best_trial_number": int(best_trial.number),
            "best_objective": float(best_trial.value),
            "best_params": dict(best_trial.params),
            "best_trial_validation_metrics": best_trial.user_attrs["validation_metrics"],
            "refit_validation_metrics": final_metrics,
            "refit_fit_seconds": float(fit_seconds),
            "refit_predict_seconds": float(predict_seconds),
            "database_path": str(database_path),
            "trials_csv_path": str(trials_csv_path),
            "best_model_path": str(best_model_path),
            "configuration": configuration,
        }
        with best_result_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_value(result_payload), handle, indent=2, sort_keys=True, allow_nan=False)

        return SearchResult(
            kind=kind,
            study_name=study_name,
            best_trial_number=int(best_trial.number),
            best_params=dict(best_trial.params),
            best_objective=float(best_trial.value),
            validation_metrics=final_metrics,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            database_path=database_path,
            trials_csv_path=trials_csv_path,
            best_result_path=best_result_path,
            best_model_path=best_model_path,
        )
    finally:
        _release_study_storage(study)


def _base_configuration(
    context: SearchContext,
    *,
    kind: str,
    active_feature_names: Sequence[str],
    search_space: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "source_h5": _source_metadata(context.h5_file),
        "group_split": _split_metadata(context.split),
        "objective_metric": context.objective_metric,
        "sampler": {"name": "CmaEsSampler", "seed": context.sampler_seed, "n_startup_trials": 10},
        "active_feature_names": tuple(active_feature_names),
        "search_space": dict(search_space),
        "fixed_parameters": dict(fixed_parameters),
    }


def search_powerlaw(
    context: SearchContext,
    *,
    ridge_lambda_range: tuple[float, float],
    plateau_weight_range: tuple[float, float],
    frac_threshold_range: tuple[float, float],
    abs_threshold_range: tuple[float, float],
    pearson_min: float | None,
    fit_mode: str,
    pure_powerlaw_if_no_plateau: bool,
    enable_tail: bool,
    sigma_bounds: tuple[float, float],
    log_vc_margin: float,
    log_emax_margin: float,
    alpha_bounds: tuple[float, float],
    max_v: float | None,
    regress_type: str,
    active_feature_names: Sequence[str],
) -> SearchResult:
    """Search continuous PowerLaw parameters while exposing all fixed parameters."""
    ridge_lambda_range = _validate_float_range("ridge_lambda_range", ridge_lambda_range, positive=True)
    plateau_weight_range = _validate_float_range(
        "plateau_weight_range", plateau_weight_range, minimum=1.0
    )
    frac_threshold_range = _validate_float_range("frac_threshold_range", frac_threshold_range, positive=True)
    abs_threshold_range = _validate_float_range("abs_threshold_range", abs_threshold_range, positive=True)
    search_space = {
        "ridge_lambda": ridge_lambda_range,
        "plateau_weight": plateau_weight_range,
        "frac_threshold": frac_threshold_range,
        "abs_threshold": abs_threshold_range,
    }
    fixed_parameters = {
        "pearson_min": pearson_min,
        "fit_mode": fit_mode,
        "pure_powerlaw_if_no_plateau": pure_powerlaw_if_no_plateau,
        "enable_tail": enable_tail,
        "sigma_bounds": sigma_bounds,
        "logVc_margin": log_vc_margin,
        "logEmax_margin": log_emax_margin,
        "alpha_bounds": alpha_bounds,
        "max_V": max_v,
        "regress_type": regress_type,
    }
    configuration = _base_configuration(
        context,
        kind="powerlaw",
        active_feature_names=active_feature_names,
        search_space=search_space,
        fixed_parameters=fixed_parameters,
    )

    def make_model(parameters: Mapping[str, Any]) -> PowerLawSeparableModel:
        return PowerLawSeparableModel(
            pearson_min=pearson_min,
            frac_threshold=float(parameters["frac_threshold"]),
            abs_threshold=float(parameters["abs_threshold"]),
            fit_mode=fit_mode,
            pure_powerlaw_if_no_plateau=pure_powerlaw_if_no_plateau,
            enable_tail=enable_tail,
            sigma_bounds=sigma_bounds,
            logVc_margin=log_vc_margin,
            logEmax_margin=log_emax_margin,
            alpha_bounds=alpha_bounds,
            max_V=max_v,
            plateau_weight=float(parameters["plateau_weight"]),
            regress_type=regress_type,
            ridge_lambda=float(parameters["ridge_lambda"]),
            active_feature_names=active_feature_names,
        )

    def build_trial_model(trial: optuna.trial.Trial) -> PowerLawSeparableModel:
        return make_model(
            {
                "ridge_lambda": trial.suggest_float("ridge_lambda", *ridge_lambda_range, log=True),
                "plateau_weight": trial.suggest_float("plateau_weight", *plateau_weight_range),
                "frac_threshold": trial.suggest_float("frac_threshold", *frac_threshold_range),
                "abs_threshold": trial.suggest_float("abs_threshold", *abs_threshold_range),
            }
        )

    return _run_search(
        context,
        kind="powerlaw",
        configuration=configuration,
        build_trial_model=build_trial_model,
        build_best_model=make_model,
        fit_model=lambda model: model.fit(None, None, groups=context.split.train_groups),
    )


def search_parametric(
    context: SearchContext,
    *,
    residual_lambda_range: tuple[float, float],
    plateau_weight_range: tuple[float, float],
    frac_threshold_range: tuple[float, float],
    abs_threshold_range: tuple[float, float],
    pearson_min: float | None,
    fit_mode: str,
    pure_powerlaw_if_no_plateau: bool,
    residual_type: str,
    enable_tail: bool,
    tol_int_bre: float,
    active_feature_names: Sequence[str],
) -> SearchResult:
    """Search Parametric residual/trend settings with explicit fixed parameters."""
    residual_lambda_range = _validate_float_range(
        "residual_lambda_range", residual_lambda_range, positive=True
    )
    plateau_weight_range = _validate_float_range(
        "plateau_weight_range", plateau_weight_range, minimum=1.0
    )
    frac_threshold_range = _validate_float_range("frac_threshold_range", frac_threshold_range, positive=True)
    abs_threshold_range = _validate_float_range("abs_threshold_range", abs_threshold_range, positive=True)
    search_space = {
        "residual_lambda": residual_lambda_range,
        "plateau_weight": plateau_weight_range,
        "frac_threshold": frac_threshold_range,
        "abs_threshold": abs_threshold_range,
    }
    fixed_parameters = {
        "pearson_min": pearson_min,
        "fit_mode": fit_mode,
        "pure_powerlaw_if_no_plateau": pure_powerlaw_if_no_plateau,
        "residual_type": residual_type,
        "enable_tail": enable_tail,
        "tol_int_bre": tol_int_bre,
    }
    configuration = _base_configuration(
        context,
        kind="parametric",
        active_feature_names=active_feature_names,
        search_space=search_space,
        fixed_parameters=fixed_parameters,
    )

    def make_model(parameters: Mapping[str, Any]) -> ParametricEnergyModel:
        return ParametricEnergyModel(
            pearson_min=pearson_min,
            frac_threshold=float(parameters["frac_threshold"]),
            abs_threshold=float(parameters["abs_threshold"]),
            fit_mode=fit_mode,
            pure_powerlaw_if_no_plateau=pure_powerlaw_if_no_plateau,
            residual_type=residual_type,
            residual_lambda=float(parameters["residual_lambda"]),
            enable_tail=enable_tail,
            plateau_weight=float(parameters["plateau_weight"]),
            tol_int_bre=tol_int_bre,
            active_feature_names=active_feature_names,
        )

    def build_trial_model(trial: optuna.trial.Trial) -> ParametricEnergyModel:
        return make_model(
            {
                "residual_lambda": trial.suggest_float(
                    "residual_lambda", *residual_lambda_range, log=True
                ),
                "plateau_weight": trial.suggest_float("plateau_weight", *plateau_weight_range),
                "frac_threshold": trial.suggest_float("frac_threshold", *frac_threshold_range),
                "abs_threshold": trial.suggest_float("abs_threshold", *abs_threshold_range),
            }
        )

    return _run_search(
        context,
        kind="parametric",
        configuration=configuration,
        build_trial_model=build_trial_model,
        build_best_model=make_model,
        fit_model=lambda model: model.fit(None, None, groups=context.split.train_groups),
    )


def search_mlp(
    context: SearchContext,
    *,
    hidden_1_range: tuple[int, int],
    hidden_2_range: tuple[int, int],
    lr_range: tuple[float, float],
    weight_decay_range: tuple[float, float],
    patience_range: tuple[int, int],
    activation: str,
    max_epochs: int,
    batch_size: int,
    device: str | None,
    seed: int | None,
    active_feature_names: Sequence[str],
) -> SearchResult:
    """Search two-layer MLP capacity and optimiser settings."""
    hidden_1_range = _validate_int_range("hidden_1_range", hidden_1_range, minimum=1)
    hidden_2_range = _validate_int_range("hidden_2_range", hidden_2_range, minimum=1)
    lr_range = _validate_float_range("lr_range", lr_range, positive=True)
    weight_decay_range = _validate_float_range("weight_decay_range", weight_decay_range, positive=True)
    patience_range = _validate_int_range("patience_range", patience_range, minimum=1)
    if max_epochs < 1 or batch_size < 1:
        raise ValueError("max_epochs and batch_size must both be positive.")
    search_space = {
        "hidden_1": hidden_1_range,
        "hidden_2": hidden_2_range,
        "lr": lr_range,
        "weight_decay": weight_decay_range,
        "patience": patience_range,
    }
    fixed_parameters = {
        "activation": activation,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
    }
    configuration = _base_configuration(
        context,
        kind="mlp",
        active_feature_names=active_feature_names,
        search_space=search_space,
        fixed_parameters=fixed_parameters,
    )

    def make_model(parameters: Mapping[str, Any]) -> MLPEnergyModel:
        return MLPEnergyModel(
            hidden_sizes=(int(parameters["hidden_1"]), int(parameters["hidden_2"])),
            activation=activation,
            lr=float(parameters["lr"]),
            weight_decay=float(parameters["weight_decay"]),
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=int(parameters["patience"]),
            device=device,
            seed=seed,
            active_feature_names=active_feature_names,
        )

    def build_trial_model(trial: optuna.trial.Trial) -> MLPEnergyModel:
        return make_model(
            {
                "hidden_1": trial.suggest_int("hidden_1", *hidden_1_range),
                "hidden_2": trial.suggest_int("hidden_2", *hidden_2_range),
                "lr": trial.suggest_float("lr", *lr_range, log=True),
                "weight_decay": trial.suggest_float("weight_decay", *weight_decay_range, log=True),
                "patience": trial.suggest_int("patience", *patience_range),
            }
        )

    return _run_search(
        context,
        kind="mlp",
        configuration=configuration,
        build_trial_model=build_trial_model,
        build_best_model=make_model,
        fit_model=lambda model: model.fit(
            context.split.X_train,
            context.split.y_train,
            X_val=context.split.X_val,
            y_val=context.split.y_val,
        ),
    )


def search_ann(
    context: SearchContext,
    *,
    hidden_1_range: tuple[int, int],
    hidden_2_range: tuple[int, int],
    hidden_3_range: tuple[int, int],
    dropout_range: tuple[float, float],
    lr_range: tuple[float, float],
    weight_decay_range: tuple[float, float],
    patience_range: tuple[int, int],
    activation: str,
    max_epochs: int,
    batch_size: int,
    device: str | None,
    seed: int | None,
    active_feature_names: Sequence[str],
) -> SearchResult:
    """Search BatchNorm/Dropout ANN capacity and optimiser settings."""
    hidden_1_range = _validate_int_range("hidden_1_range", hidden_1_range, minimum=1)
    hidden_2_range = _validate_int_range("hidden_2_range", hidden_2_range, minimum=1)
    hidden_3_range = _validate_int_range("hidden_3_range", hidden_3_range, minimum=1)
    dropout_range = _validate_float_range("dropout_range", dropout_range, minimum=0.0)
    if dropout_range[1] >= 1.0:
        raise ValueError("dropout_range must remain strictly below 1.0.")
    lr_range = _validate_float_range("lr_range", lr_range, positive=True)
    weight_decay_range = _validate_float_range("weight_decay_range", weight_decay_range, positive=True)
    patience_range = _validate_int_range("patience_range", patience_range, minimum=1)
    if max_epochs < 1 or batch_size < 2:
        raise ValueError("max_epochs must be positive and ANN batch_size must be at least 2.")
    search_space = {
        "hidden_1": hidden_1_range,
        "hidden_2": hidden_2_range,
        "hidden_3": hidden_3_range,
        "dropout": dropout_range,
        "lr": lr_range,
        "weight_decay": weight_decay_range,
        "patience": patience_range,
    }
    fixed_parameters = {
        "activation": activation,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
    }
    configuration = _base_configuration(
        context,
        kind="ann",
        active_feature_names=active_feature_names,
        search_space=search_space,
        fixed_parameters=fixed_parameters,
    )

    def make_model(parameters: Mapping[str, Any]) -> ANNEnergyModel:
        return ANNEnergyModel(
            hidden_sizes=(
                int(parameters["hidden_1"]),
                int(parameters["hidden_2"]),
                int(parameters["hidden_3"]),
            ),
            activation=activation,
            dropout=float(parameters["dropout"]),
            lr=float(parameters["lr"]),
            weight_decay=float(parameters["weight_decay"]),
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=int(parameters["patience"]),
            device=device,
            seed=seed,
            active_feature_names=active_feature_names,
        )

    def build_trial_model(trial: optuna.trial.Trial) -> ANNEnergyModel:
        return make_model(
            {
                "hidden_1": trial.suggest_int("hidden_1", *hidden_1_range),
                "hidden_2": trial.suggest_int("hidden_2", *hidden_2_range),
                "hidden_3": trial.suggest_int("hidden_3", *hidden_3_range),
                "dropout": trial.suggest_float("dropout", *dropout_range),
                "lr": trial.suggest_float("lr", *lr_range, log=True),
                "weight_decay": trial.suggest_float("weight_decay", *weight_decay_range, log=True),
                "patience": trial.suggest_int("patience", *patience_range),
            }
        )

    return _run_search(
        context,
        kind="ann",
        configuration=configuration,
        build_trial_model=build_trial_model,
        build_best_model=make_model,
        fit_model=lambda model: model.fit(
            context.split.X_train,
            context.split.y_train,
            X_val=context.split.X_val,
            y_val=context.split.y_val,
        ),
    )


def main() -> SearchResult:
    """Run one selected search. Edit the explicit ranges and fixed values below in Spyder."""
    h5_file = DATA_DIRECTORY / H5_FILENAME
    output_directory = DATA_DIRECTORY / "hyperparameter_search"
    context = build_search_context(
        h5_file,
        output_directory=output_directory,
        n_trials=N_TRIALS,
        sampler_seed=SAMPLER_SEED,
        validation_ratio=VALIDATION_RATIO,
        split_seed=SPLIT_SEED,
        objective_metric=OBJECTIVE_METRIC,
        show_training_output=SHOW_TRAINING_OUTPUT,
    )
    kind = SEARCH_KIND.lower()

    # PowerLaw: searchable continuous parameters and all fixed model settings.
    powerlaw_search_space = {
        "ridge_lambda_range": (1e-4, 1e4),
        "plateau_weight_range": (1.0, 8.0),
        "frac_threshold_range": (0.05, 8.0),
        "abs_threshold_range": (0.01, 5.0),
    }
    powerlaw_fixed_parameters = {
        "pearson_min": None,
        "fit_mode": "logE",
        "pure_powerlaw_if_no_plateau": True,
        "enable_tail": True,
        "sigma_bounds": (0.0, 10.0),
        "log_vc_margin": 2.0,
        "log_emax_margin": 2.0,
        "alpha_bounds": (0.0, 0.5),
        "max_v": None,
        "regress_type": "ridge",
        "active_feature_names": ACTIVE_FEATURE_NAMES,
    }

    # Parametric: searchable residual/trend parameters and all fixed settings.
    parametric_search_space = {
        "residual_lambda_range": (1e-4, 1e4),
        "plateau_weight_range": (1.0, 8.0),
        "frac_threshold_range": (0.05, 8.0),
        "abs_threshold_range": (0.01, 5.0),
    }
    parametric_fixed_parameters = {
        "pearson_min": None,
        "fit_mode": "logE",
        "pure_powerlaw_if_no_plateau": True,
        "residual_type": "ridge",
        "enable_tail": True,
        "tol_int_bre": 1e-12,
        "active_feature_names": ACTIVE_FEATURE_NAMES,
    }

    # MLP: searchable capacity/optimiser settings and all fixed settings.
    mlp_search_space = {
        "hidden_1_range": (32, 512),
        "hidden_2_range": (32, 512),
        "lr_range": (1e-5, 1e-2),
        "weight_decay_range": (1e-8, 1e-2),
        "patience_range": (10, 100),
    }
    mlp_fixed_parameters = {
        "activation": "relu",
        "max_epochs": 400,
        "batch_size": 128,
        "device": None,
        "seed": 42,
        "active_feature_names": ACTIVE_FEATURE_NAMES,
    }

    # ANN: searchable capacity/dropout/optimiser settings and all fixed settings.
    ann_search_space = {
        "hidden_1_range": (32, 512),
        "hidden_2_range": (32, 512),
        "hidden_3_range": (16, 256),
        "dropout_range": (0.0, 0.5),
        "lr_range": (1e-5, 1e-2),
        "weight_decay_range": (1e-8, 1e-2),
        "patience_range": (10, 100),
    }
    ann_fixed_parameters = {
        "activation": "silu",
        "max_epochs": 800,
        "batch_size": 256,
        "device": None,
        "seed": 42,
        "active_feature_names": ACTIVE_FEATURE_NAMES,
    }

    if kind == "powerlaw":
        return search_powerlaw(context, **powerlaw_search_space, **powerlaw_fixed_parameters)
    if kind == "parametric":
        return search_parametric(context, **parametric_search_space, **parametric_fixed_parameters)
    if kind == "mlp":
        return search_mlp(context, **mlp_search_space, **mlp_fixed_parameters)
    if kind == "ann":
        return search_ann(context, **ann_search_space, **ann_fixed_parameters)
    raise ValueError(f"SEARCH_KIND must be one of {_VALID_SEARCH_KINDS}, got {SEARCH_KIND!r}.")


if __name__ == "__main__":
    SEARCH_RESULT = main()
    print(f"Best {SEARCH_RESULT.kind} trial: {SEARCH_RESULT.best_trial_number}")
    print(f"{OBJECTIVE_METRIC}: {SEARCH_RESULT.best_objective:.6g}")
    print(f"Results: {SEARCH_RESULT.best_result_path}")
