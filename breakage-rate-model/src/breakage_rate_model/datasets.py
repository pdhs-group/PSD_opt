# -*- coding: utf-8 -*-
"""Dataset builders for completed energy-scan groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .data_io import EnergyGroupRecord
from .features import full_energy_features


@dataclass
class SigmaDataset:
    X: np.ndarray
    y_sigma: np.ndarray
    y_b: np.ndarray
    meta: List[EnergyGroupRecord]


def _theta_features(rec: EnergyGroupRecord) -> np.ndarray:
    return np.array(
        [np.log(rec.gamma), np.log(rec.NO_FRAG), rec.int_bre, rec.Df, rec.MAS, rec.X1],
        dtype=float,
    )


def build_sigma_dataset(
    groups: Sequence[EnergyGroupRecord],
    pearson_min: float = 0.95,
    use_attr_if_available: bool = True,
) -> SigmaDataset:
    X_list: List[np.ndarray] = []
    y_sigma_list: List[float] = []
    y_b_list: List[float] = []
    meta_list: List[EnergyGroupRecord] = []

    for rec in groups:
        r_use = rec.pearson_r_attr if use_attr_if_available and rec.pearson_r_attr is not None else rec.pearson_r_fit
        if np.isnan(r_use) or r_use < pearson_min:
            continue
        sigma = rec.sigma_attr if use_attr_if_available and rec.sigma_attr is not None else rec.sigma_fit
        X_list.append(_theta_features(rec))
        y_sigma_list.append(float(sigma))
        y_b_list.append(float(rec.b_fit))
        meta_list.append(rec)

    if not X_list:
        raise RuntimeError("No groups passed the pearson_min filter for sigma dataset.")
    return SigmaDataset(
        X=np.vstack(X_list),
        y_sigma=np.asarray(y_sigma_list, dtype=float),
        y_b=np.asarray(y_b_list, dtype=float),
        meta=meta_list,
    )


@dataclass
class EnergyDataset:
    X: np.ndarray
    y: np.ndarray
    meta_idx: np.ndarray
    groups: List[EnergyGroupRecord]


def _require_samples(groups: Sequence[EnergyGroupRecord]) -> None:
    missing = [rec.key for rec in groups if rec.E_samples is None]
    if missing:
        raise ValueError(
            "E_samples are required for per_sample or quantile targets; reload with "
            f"load_samples=True. Missing sample arrays for groups {missing[:5]}"
        )


def build_energy_dataset(
    groups: Sequence[EnergyGroupRecord],
    *,
    per_sample: bool = False,
    target: str = "log_mean",
    quantile: Optional[float] = None,
    feature_fn: Optional[Callable[[EnergyGroupRecord, float], np.ndarray]] = None,
) -> EnergyDataset:
    """Build a canonical full-feature energy dataset.

    ``per_sample=False`` emits one target per ``(group, V)``. ``per_sample=True``
    flattens the v2 ``(N_GRIDS, N_FRACS)`` sample plane for every V value.
    """
    if not groups:
        raise ValueError("groups must not be empty.")
    if target not in ("log_mean", "mean", "log_quantile", "quantile"):
        raise ValueError(f"Unknown target {target!r}")
    if target in ("quantile", "log_quantile"):
        if quantile is None or not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile targets require quantile in [0, 1].")
    if per_sample or target in ("quantile", "log_quantile"):
        _require_samples(groups)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meta_idx_list: List[Tuple[int, int, int]] = []
    feat_fn = feature_fn or full_energy_features

    for group_index, rec in enumerate(groups):
        for volume_index, V_value in enumerate(rec.V):
            V_value = float(V_value)
            if not np.isfinite(V_value) or V_value <= 0.0:
                raise ValueError(f"Group {rec.key!r} has invalid V at index {volume_index}.")
            X_value = np.asarray(feat_fn(rec, V_value), dtype=float)
            if X_value.ndim != 1 or not np.all(np.isfinite(X_value)):
                raise ValueError(f"Feature function returned invalid features for group {rec.key!r}.")

            if not per_sample:
                if target in ("log_mean", "mean"):
                    energy_value = float(rec.E_mean[volume_index])
                else:
                    samples = rec.E_samples
                    assert samples is not None
                    energy_value = float(np.quantile(samples[volume_index], quantile))
                if not np.isfinite(energy_value) or energy_value <= 0.0:
                    raise ValueError(
                        f"Group {rec.key!r} has invalid target energy at V index {volume_index}."
                    )
                y_value = np.log(energy_value) if target.startswith("log_") else energy_value
                X_list.append(X_value)
                y_list.append(float(y_value))
                meta_idx_list.append((group_index, volume_index, -1))
                continue

            samples = rec.E_samples
            assert samples is not None
            flat_samples = np.asarray(samples[volume_index], dtype=float).reshape(-1)
            if flat_samples.size != rec.N_GRIDS * rec.N_FRACS:
                raise ValueError(f"Group {rec.key!r} has inconsistent E_samples size.")
            if not np.all(np.isfinite(flat_samples)) or np.any(flat_samples <= 0.0):
                raise ValueError(f"Group {rec.key!r} has invalid E_samples at V index {volume_index}.")
            log_target = target in ("log_mean", "log_quantile")
            for sample_index, energy_value in enumerate(flat_samples):
                X_list.append(X_value)
                y_list.append(float(np.log(energy_value) if log_target else energy_value))
                meta_idx_list.append((group_index, volume_index, sample_index))

    if not X_list:
        raise RuntimeError("No samples were generated in build_energy_dataset.")
    return EnergyDataset(
        X=np.vstack(X_list),
        y=np.asarray(y_list, dtype=float),
        meta_idx=np.asarray(meta_idx_list, dtype=int),
        groups=list(groups),
    )
