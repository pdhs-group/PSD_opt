"""Canonical feature contract for breakage-energy surrogate models."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


FULL_ENERGY_FEATURE_NAMES: Tuple[str, ...] = (
    "logV",
    "log_gamma",
    "log_NO_FRAG",
    "int_bre",
    "Df",
    "MAS",
    "X1",
    "STR0",
    "STR1",
    "STR2",
)

DEFAULT_ACTIVE_FEATURE_NAMES: Tuple[str, ...] = (
    "logV",
    "log_gamma",
    "MAS",
    "X1",
    "STR0",
    "STR1",
    "STR2",
)

FEATURE_INDEX = {name: index for index, name in enumerate(FULL_ENERGY_FEATURE_NAMES)}
INT_BRE_FEATURE_INDEX = FEATURE_INDEX["int_bre"]


def normalize_active_feature_names(
    active_feature_names: Sequence[str] | None,
) -> Tuple[str, ...]:
    """Validate and normalize the model's internally active full-vector columns."""
    if active_feature_names is None:
        names = DEFAULT_ACTIVE_FEATURE_NAMES
    else:
        names = tuple(active_feature_names)

    if not names:
        raise ValueError("active_feature_names must not be empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"active_feature_names contains duplicates: {names}")
    unknown = tuple(name for name in names if name not in FEATURE_INDEX)
    if unknown:
        raise ValueError(
            f"active_feature_names contains unknown features {unknown}; "
            f"valid names are {FULL_ENERGY_FEATURE_NAMES}"
        )
    if "logV" not in names:
        raise ValueError("active_feature_names must include 'logV'.")
    return names


def active_feature_indices(active_feature_names: Sequence[str] | None) -> np.ndarray:
    """Return full-vector column indices for validated active features."""
    names = normalize_active_feature_names(active_feature_names)
    return np.array([FEATURE_INDEX[name] for name in names], dtype=int)


def theta_feature_names(active_feature_names: Sequence[str] | None) -> Tuple[str, ...]:
    """Return active curve-parameter features, excluding the independent variable logV."""
    return tuple(
        name for name in normalize_active_feature_names(active_feature_names) if name != "logV"
    )


def theta_feature_indices(active_feature_names: Sequence[str] | None) -> np.ndarray:
    """Return full-vector column indices for active curve-parameter features."""
    return np.array(
        [FEATURE_INDEX[name] for name in theta_feature_names(active_feature_names)],
        dtype=int,
    )


def require_full_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Require the stable external 10-column surrogate-model interface."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FULL_ENERGY_FEATURE_NAMES):
        raise ValueError(
            f"X must have shape (n_samples, {len(FULL_ENERGY_FEATURE_NAMES)}) in the "
            f"canonical order {FULL_ENERGY_FEATURE_NAMES}; got {X.shape}"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")
    return X


def full_energy_features(record, V_value: float) -> np.ndarray:
    """Build the canonical external feature vector from one energy-group record."""
    V_value = float(V_value)
    if not np.isfinite(V_value) or V_value <= 0.0:
        raise ValueError(f"V_value must be finite and positive, got {V_value}")
    gamma = float(record.gamma)
    no_frag = float(record.NO_FRAG)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and positive for record {record.key!r}.")
    if not np.isfinite(no_frag) or no_frag <= 0.0:
        raise ValueError(f"NO_FRAG must be finite and positive for record {record.key!r}.")

    str_values = np.asarray(record.STR, dtype=float).reshape(-1)
    if str_values.shape != (3,) or not np.all(np.isfinite(str_values)):
        raise ValueError(f"STR must be a finite vector of length 3, got {np.asarray(record.STR).shape}")

    return np.array(
        [
            np.log(V_value),
            np.log(gamma),
            np.log(no_frag),
            float(record.int_bre),
            float(record.Df),
            float(record.MAS),
            float(record.X1),
            str_values[0],
            str_values[1],
            str_values[2],
        ],
        dtype=float,
    )


def group_theta_features(record, active_feature_names: Sequence[str] | None) -> np.ndarray:
    """Build active non-volume features from a group without inventing a V value."""
    names = theta_feature_names(active_feature_names)
    if not names:
        return np.empty((0,), dtype=float)

    gamma = float(record.gamma)
    no_frag = float(record.NO_FRAG)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and positive for record {record.key!r}.")
    if not np.isfinite(no_frag) or no_frag <= 0.0:
        raise ValueError(f"NO_FRAG must be finite and positive for record {record.key!r}.")

    str_values = np.asarray(record.STR, dtype=float).reshape(-1)
    if str_values.shape != (3,) or not np.all(np.isfinite(str_values)):
        raise ValueError(f"STR must be a finite vector of length 3, got {np.asarray(record.STR).shape}")

    values = {
        "log_gamma": np.log(gamma),
        "log_NO_FRAG": np.log(no_frag),
        "int_bre": float(record.int_bre),
        "Df": float(record.Df),
        "MAS": float(record.MAS),
        "X1": float(record.X1),
        "STR0": str_values[0],
        "STR1": str_values[1],
        "STR2": str_values[2],
    }
    theta = np.array([values[name] for name in names], dtype=float)
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"Non-finite active group features for record {record.key!r}.")
    return theta
