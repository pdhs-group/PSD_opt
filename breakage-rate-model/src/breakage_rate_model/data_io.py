# -*- coding: utf-8 -*-
"""Strict reader for completed v2 energy-scan HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np


REQUIRED_GROUP_ATTRS = (
    "NO_FRAG",
    "int_bre",
    "gamma",
    "Df",
    "MAS",
    "X1",
    "A0",
    "N_GRIDS",
    "N_FRACS",
    "base_seed",
    "workers",
    "STR",
    "checkpoint_version",
    "completed_tasks",
    "expected_tasks",
)
REQUIRED_GROUP_DATASETS = ("Np", "V", "E_mean", "E_std", "E_samples", "completed_mask")


@dataclass
class EnergyGroupRecord:
    """One completed energy curve stored under ``/runs/<key>``."""

    key: str
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
    sigma_attr: Optional[float]
    pearson_r_attr: Optional[float]
    Np: np.ndarray
    V: np.ndarray
    E_mean: np.ndarray
    E_std: np.ndarray
    E_samples: Optional[np.ndarray]
    sigma_fit: float
    b_fit: float
    pearson_r_fit: float


def _fit_line_logV_logE(V: np.ndarray, E: np.ndarray) -> Tuple[float, float, float]:
    """Fit ``log(E) = b + sigma * log(V)`` for a validated positive curve."""
    if V.size < 2:
        return np.nan, np.nan, np.nan

    logV = np.log(V)
    logE = np.log(E)
    pearson_r = float(np.corrcoef(logV, logE)[0, 1])
    A = np.column_stack([np.ones_like(logV), logV])
    b_fit, sigma_fit = np.linalg.lstsq(A, logE, rcond=None)[0]
    return float(sigma_fit), float(b_fit), pearson_r


def _require_scalar_finite(value: float, name: str, group_path: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{group_path}: attribute {name!r} must be finite.")
    return value


def _validate_completed_group(group: h5py.Group, group_path: str, *, load_samples: bool) -> None:
    """Validate v2 checkpoint completion and the model-facing data contract."""
    missing_attrs = [name for name in REQUIRED_GROUP_ATTRS if name not in group.attrs]
    if missing_attrs:
        raise ValueError(f"{group_path}: missing required attributes {missing_attrs}")
    missing_datasets = [name for name in REQUIRED_GROUP_DATASETS if name not in group]
    if missing_datasets:
        raise ValueError(f"{group_path}: missing required datasets {missing_datasets}")

    if int(group.attrs["checkpoint_version"]) != 2:
        raise ValueError(f"{group_path}: checkpoint_version must be 2.")

    NO_FRAG = int(group.attrs["NO_FRAG"])
    gamma = _require_scalar_finite(group.attrs["gamma"], "gamma", group_path)
    A0 = _require_scalar_finite(group.attrs["A0"], "A0", group_path)
    if NO_FRAG <= 0:
        raise ValueError(f"{group_path}: NO_FRAG must be positive.")
    if gamma <= 0.0:
        raise ValueError(f"{group_path}: gamma must be positive.")
    if A0 <= 0.0:
        raise ValueError(f"{group_path}: A0 must be positive.")
    for name in ("int_bre", "Df", "MAS", "X1"):
        _require_scalar_finite(group.attrs[name], name, group_path)

    n_grids = int(group.attrs["N_GRIDS"])
    n_fracs = int(group.attrs["N_FRACS"])
    if n_grids <= 0 or n_fracs <= 0:
        raise ValueError(f"{group_path}: N_GRIDS and N_FRACS must be positive.")

    str_values = np.asarray(group.attrs["STR"], dtype=float)
    if str_values.shape != (3,) or not np.all(np.isfinite(str_values)) or np.any(str_values <= 0.0):
        raise ValueError(f"{group_path}: STR must be a finite positive vector of shape (3,).")

    Np_ds = group["Np"]
    if Np_ds.ndim != 1 or not np.issubdtype(Np_ds.dtype, np.integer):
        raise ValueError(f"{group_path}: Np must be a one-dimensional integer dataset.")
    Np = np.asarray(Np_ds[...])
    n_np = int(Np.size)
    if n_np == 0 or np.any(Np <= 0):
        raise ValueError(f"{group_path}: Np must contain positive values.")

    for name in ("V", "E_mean", "E_std"):
        dataset = group[name]
        if dataset.ndim != 1 or dataset.shape != (n_np,):
            raise ValueError(f"{group_path}: {name} must have shape ({n_np},).")
    V = np.asarray(group["V"][...], dtype=float)
    E_mean = np.asarray(group["E_mean"][...], dtype=float)
    E_std = np.asarray(group["E_std"][...], dtype=float)
    if not np.all(np.isfinite(V)) or np.any(V <= 0.0) or np.any(np.diff(V) <= 0.0):
        raise ValueError(f"{group_path}: V must be finite, positive, and strictly increasing.")
    if not np.array_equal(V, Np.astype(float) * A0):
        raise ValueError(f"{group_path}: V must equal Np * A0 exactly.")
    if not np.all(np.isfinite(E_mean)) or np.any(E_mean <= 0.0):
        raise ValueError(f"{group_path}: E_mean must be finite and positive.")
    if not np.all(np.isfinite(E_std)) or np.any(E_std < 0.0):
        raise ValueError(f"{group_path}: E_std must be finite and non-negative.")

    samples_ds = group["E_samples"]
    expected_samples_shape = (n_np, n_grids, n_fracs)
    if samples_ds.ndim != 3 or samples_ds.shape != expected_samples_shape:
        raise ValueError(
            f"{group_path}: E_samples must have shape {expected_samples_shape}, got {samples_ds.shape}."
        )
    completed_mask = np.asarray(group["completed_mask"][...], dtype=bool)
    expected_mask_shape = (n_np, n_grids)
    if completed_mask.shape != expected_mask_shape:
        raise ValueError(
            f"{group_path}: completed_mask must have shape {expected_mask_shape}, got {completed_mask.shape}."
        )
    expected_tasks = n_np * n_grids
    if int(group.attrs["expected_tasks"]) != expected_tasks:
        raise ValueError(f"{group_path}: expected_tasks must equal {expected_tasks}.")
    if int(group.attrs["completed_tasks"]) != expected_tasks or not np.all(completed_mask):
        raise ValueError(f"{group_path}: checkpoint is incomplete.")

    if load_samples:
        samples = np.asarray(samples_ds[...], dtype=float)
        if not np.all(np.isfinite(samples)) or np.any(samples <= 0.0):
            raise ValueError(f"{group_path}: E_samples must be finite and positive.")


def load_energy_groups_from_h5(
    h5_path: str,
    *,
    load_samples: bool = False,
) -> List[EnergyGroupRecord]:
    """Load completed v2 energy-scan groups without reading raw samples by default."""
    records: List[EnergyGroupRecord] = []

    with h5py.File(h5_path, "r") as h5_file:
        if "runs" not in h5_file or not isinstance(h5_file["runs"], h5py.Group):
            raise ValueError("HDF5 file must contain a '/runs' group.")

        runs_group = h5_file["runs"]
        if len(runs_group) == 0:
            raise ValueError("HDF5 '/runs' group is empty.")

        for key in sorted(runs_group.keys()):
            group = runs_group[key]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"/runs/{key}: expected an HDF5 group.")
            group_path = f"/runs/{key}"
            _validate_completed_group(group, group_path, load_samples=load_samples)

            V = np.asarray(group["V"][...], dtype=float)
            E_mean = np.asarray(group["E_mean"][...], dtype=float)
            sigma_fit, b_fit, pearson_r_fit = _fit_line_logV_logE(V, E_mean)
            E_samples = np.asarray(group["E_samples"][...], dtype=float) if load_samples else None

            records.append(
                EnergyGroupRecord(
                    key=key,
                    NO_FRAG=int(group.attrs["NO_FRAG"]),
                    int_bre=float(group.attrs["int_bre"]),
                    gamma=float(group.attrs["gamma"]),
                    Df=float(group.attrs["Df"]),
                    MAS=float(group.attrs["MAS"]),
                    X1=float(group.attrs["X1"]),
                    A0=float(group.attrs["A0"]),
                    N_GRIDS=int(group.attrs["N_GRIDS"]),
                    N_FRACS=int(group.attrs["N_FRACS"]),
                    base_seed=int(group.attrs["base_seed"]),
                    workers=int(group.attrs["workers"]),
                    STR=np.asarray(group.attrs["STR"], dtype=float),
                    sigma_attr=float(group.attrs["sigma"]) if "sigma" in group.attrs else None,
                    pearson_r_attr=(
                        float(group.attrs["pearson_r"]) if "pearson_r" in group.attrs else None
                    ),
                    Np=np.asarray(group["Np"][...]),
                    V=V,
                    E_mean=E_mean,
                    E_std=np.asarray(group["E_std"][...], dtype=float),
                    E_samples=E_samples,
                    sigma_fit=sigma_fit,
                    b_fit=b_fit,
                    pearson_r_fit=pearson_r_fit,
                )
            )

    return records
