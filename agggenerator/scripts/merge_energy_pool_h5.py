# -*- coding: utf-8 -*-
"""Merge completed v2 energy-pool HDF5 files without materializing datasets.

Edit the configuration variables below when running from Spyder.  The script
first validates all sources, then copies complete ``/runs/<key>`` groups using
HDF5-native copying.  It never overwrites an existing output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# =============================================================================
# Spyder configuration
# =============================================================================

INPUT_DIRECTORY = Path(os.environ["STORAGE_PATH"]) / "energy_pool"
INPUT_FILENAMES = tuple(f"psd_data{index}.h5" for index in range(1, 10))
OUTPUT_FILENAME = "energy_scan_results.h5"
TEMPORARY_PREFIX = ".energy_scan_results.h5.merge-"

REQUIRED_ATTRS = (
    "NO_FRAG", "int_bre", "gamma", "Df", "MAS", "X1", "A0",
    "N_GRIDS", "N_FRACS", "base_seed", "workers", "STR",
    "checkpoint_version", "completed_tasks", "expected_tasks",
)
REQUIRED_DATASETS = ("Np", "V", "E_samples", "E_mean", "E_std", "completed_mask")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_completed_v2_run(group: h5py.Group, group_path: str) -> None:
    """Fail fast unless ``group`` is a complete version-2 energy checkpoint."""
    for name in REQUIRED_ATTRS:
        _require(name in group.attrs, f"{group_path} is missing required attribute {name!r}.")
    for name in REQUIRED_DATASETS:
        _require(name in group, f"{group_path} is missing required dataset {name!r}.")

    _require(
        int(group.attrs["checkpoint_version"]) == 2,
        f"{group_path} has checkpoint_version != 2.",
    )
    n_grids = int(group.attrs["N_GRIDS"])
    n_fracs = int(group.attrs["N_FRACS"])
    _require(n_grids > 0 and n_fracs > 0, f"{group_path} has invalid grid dimensions.")
    _require(float(group.attrs["NO_FRAG"]) > 0.0, f"{group_path} has non-positive NO_FRAG.")
    _require(float(group.attrs["gamma"]) > 0.0, f"{group_path} has non-positive gamma.")
    a0 = float(group.attrs["A0"])
    _require(np.isfinite(a0) and a0 > 0.0, f"{group_path} has invalid A0.")
    str_values = np.asarray(group.attrs["STR"], dtype=float)
    _require(str_values.shape == (3,), f"{group_path} STR must have shape (3,).")
    _require(np.all(np.isfinite(str_values)) and np.all(str_values > 0.0), f"{group_path} has invalid STR.")
    for name in ("int_bre", "Df", "MAS", "X1"):
        _require(np.isfinite(float(group.attrs[name])), f"{group_path} has non-finite {name}.")

    np_values = np.asarray(group["Np"][...])
    _require(np_values.ndim == 1 and np_values.size > 0, f"{group_path}/Np must be non-empty 1-D.")
    _require(np.issubdtype(np_values.dtype, np.integer), f"{group_path}/Np must have integer dtype.")
    _require(np.all(np_values > 0), f"{group_path}/Np must be positive.")
    n_np = np_values.size

    v_values = np.asarray(group["V"][...], dtype=float)
    mean_values = np.asarray(group["E_mean"][...], dtype=float)
    std_values = np.asarray(group["E_std"][...], dtype=float)
    for name, values in (("V", v_values), ("E_mean", mean_values), ("E_std", std_values)):
        _require(values.shape == (n_np,), f"{group_path}/{name} has incorrect shape.")
        _require(np.all(np.isfinite(values)), f"{group_path}/{name} has non-finite values.")
    _require(np.all(v_values > 0.0) and np.all(np.diff(v_values) > 0.0), f"{group_path}/V must be positive and strictly increasing.")
    _require(np.array_equal(v_values, np_values.astype(float) * a0), f"{group_path}/V must equal Np * A0 exactly.")
    _require(np.all(mean_values > 0.0), f"{group_path}/E_mean must be positive.")
    _require(np.all(std_values >= 0.0), f"{group_path}/E_std must be non-negative.")

    samples = group["E_samples"]
    completion = np.asarray(group["completed_mask"][...], dtype=bool)
    _require(samples.shape == (n_np, n_grids, n_fracs), f"{group_path}/E_samples has incorrect v2 shape.")
    _require(completion.shape == (n_np, n_grids), f"{group_path}/completed_mask has incorrect shape.")
    _require(np.all(completion), f"{group_path}/completed_mask contains unfinished tasks.")
    expected_tasks = n_np * n_grids
    _require(int(group.attrs["expected_tasks"]) == expected_tasks, f"{group_path} expected_tasks is inconsistent.")
    _require(int(group.attrs["completed_tasks"]) == expected_tasks, f"{group_path} completed_tasks is inconsistent.")


def scan_sources(input_paths: Iterable[Path]) -> list[tuple[Path, tuple[str, ...]]]:
    """Validate all sources and ensure global uniqueness of run keys before writing."""
    source_info: list[tuple[Path, tuple[str, ...]]] = []
    seen_keys: set[str] = set()
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Energy-pool input does not exist: {path}")
        with h5py.File(path, "r") as source:
            if "runs" not in source or not isinstance(source["runs"], h5py.Group):
                raise ValueError(f"{path} does not contain a /runs group.")
            keys = tuple(sorted(source["runs"].keys()))
            if not keys:
                raise ValueError(f"{path} contains no /runs entries.")
            for key in keys:
                if key in seen_keys:
                    raise ValueError(f"Duplicate /runs key across inputs: {key!r}.")
                group = source["runs"][key]
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"{path}:/runs/{key} is not an HDF5 group.")
                validate_completed_v2_run(group, f"{path}:/runs/{key}")
                seen_keys.add(key)
            source_info.append((path, keys))
    return source_info


def merge_energy_pool(
    input_directory: Path = INPUT_DIRECTORY,
    input_filenames: tuple[str, ...] = INPUT_FILENAMES,
    output_filename: str = OUTPUT_FILENAME,
) -> Path:
    """Merge validated files atomically, returning the newly created output path."""
    input_paths = tuple(input_directory / name for name in input_filenames)
    output_path = input_directory / output_filename
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    source_info = scan_sources(input_paths)
    temporary_path = input_directory / f"{TEMPORARY_PREFIX}{os.getpid()}.tmp"
    if temporary_path.exists():
        raise FileExistsError(f"Temporary merge path already exists: {temporary_path}")

    try:
        with h5py.File(temporary_path, "x") as destination:
            runs_destination = destination.create_group("runs")
            destination.attrs["merge_format_version"] = 1
            destination.attrs["merged_source_filenames"] = json.dumps(
                [path.name for path, _ in source_info], separators=(",", ":")
            )
            destination.attrs["merged_run_keys"] = json.dumps(
                [key for _, keys in source_info for key in keys], separators=(",", ":")
            )
            for path, keys in source_info:
                with h5py.File(path, "r") as source:
                    for key in keys:
                        source.copy(source["runs"][key], runs_destination, name=key)
            destination.flush()
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return output_path


if __name__ == "__main__":
    OUTPUT_PATH = merge_energy_pool()
    print(f"Merged energy pool written to {OUTPUT_PATH}")
