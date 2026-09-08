# -*- coding: utf-8 -*-
"""Plot selected mean-result comparisons from a completed phase-1 scan.

This Spyder-friendly script only reads ``phase1_results.h5``.  It never
restarts a simulation or modifies the HDF5 result file.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
import numpy as np


matplotlib.use("Agg")  # This batch post-processing script only writes PNG files.
import matplotlib.pyplot as plt


# =============================================================================
# Editable configuration
# =============================================================================

RESULT_H5 = Path(r"D:\Codex_tem\LMC\pbe_parameter_scan_phase1\phase1_results.h5")
OUTPUT_DIRECTORY = RESULT_H5.parent / "analysis_plots"

RUN_PRIMARY_COMPARISONS = True       # MAS/X1/gamma: final diameters, D50(t), moments
RUN_STR_MOMENT_COMPARISONS = True    # Requested STR combinations: high-order moments

BASE_MAS = 0.5
BASE_X1 = 0.5
BASE_STR = (1.0, 1.0, 1.0)
BASE_GAMMA = 1.0

MAS_VALUES = (0.1, 0.5, 0.9)
X1_VALUES = (0.1, 0.5, 0.9)
GAMMA_VALUES = (1.0e-3, 1.0, 1.0e3)
STR_COMPARISONS = ((1.0, 1.0, 1.0), (1.0, 1.0e3, 1.0), (1.0e3, 1.0e3, 1.0))


_MOMENT_ORDERS = ((2, 0), (0, 2), (1, 1))
_REQUIRED_DATASETS = ("psd_Q_mean", "x50_mean", "moments_mean")


def _as_str_tuple(values: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not np.all(np.isfinite(result)) or any(value <= 0.0 for value in result):
        raise ValueError(f"STR must contain three positive finite values, got {result}.")
    return result


def _case_key(MAS: float, X1: float, STR: tuple[float, float, float], gamma: float) -> tuple[float, ...]:
    return (float(MAS), float(X1), *_as_str_tuple(STR), float(gamma))


def _format_number(value: float) -> str:
    return format(float(value), ".4g")


def _format_str(STR: tuple[float, float, float]) -> str:
    return "(" + ", ".join(_format_number(value) for value in STR) + ")"


def _validate_configuration() -> None:
    for name, values in (("MAS_VALUES", MAS_VALUES), ("X1_VALUES", X1_VALUES), ("GAMMA_VALUES", GAMMA_VALUES)):
        if not values or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a non-empty finite sequence.")
        if len(set(float(value) for value in values)) != len(values):
            raise ValueError(f"{name} contains duplicate values.")
    if not np.all((0.0 <= np.asarray(MAS_VALUES)) & (np.asarray(MAS_VALUES) <= 1.0)):
        raise ValueError("MAS_VALUES must lie in [0, 1].")
    if not np.all((0.0 <= np.asarray(X1_VALUES)) & (np.asarray(X1_VALUES) <= 1.0)):
        raise ValueError("X1_VALUES must lie in [0, 1].")
    if any(value <= 0.0 for value in GAMMA_VALUES):
        raise ValueError("GAMMA_VALUES must be strictly positive.")
    if not (np.isfinite(BASE_MAS) and np.isfinite(BASE_X1) and np.isfinite(BASE_GAMMA) and BASE_GAMMA > 0.0):
        raise ValueError("Base MAS, X1, and gamma must be finite and base gamma positive.")
    _as_str_tuple(BASE_STR)
    for STR in STR_COMPARISONS:
        _as_str_tuple(STR)


def _read_root_arrays(h5_file: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    t_vec = np.asarray(h5_file["t_vec"], dtype=float)
    psd_x_grid = np.asarray(h5_file["psd_x_grid"], dtype=float)
    if t_vec.ndim != 1 or t_vec.size < 2 or not np.all(np.isfinite(t_vec)) or np.any(np.diff(t_vec) <= 0.0):
        raise ValueError("t_vec must be a finite, strictly increasing one-dimensional array.")
    if (psd_x_grid.ndim != 1 or psd_x_grid.size < 2 or not np.all(np.isfinite(psd_x_grid))
            or np.any(psd_x_grid <= 0.0) or np.any(np.diff(psd_x_grid) <= 0.0)):
        raise ValueError("psd_x_grid must be finite, positive, and strictly increasing.")
    return t_vec, psd_x_grid


def _condition_index(h5_file: h5py.File) -> dict[tuple[float, ...], h5py.Group]:
    conditions = h5_file["conditions"]
    index: dict[tuple[float, ...], h5py.Group] = {}
    for case_id in conditions:
        group = conditions[case_id]
        key = _case_key(group.attrs["MAS"], group.attrs["X1"], group.attrs["STR"], group.attrs["gamma"])
        if key in index:
            raise ValueError(f"Duplicate scan condition for {key}.")
        index[key] = group
    return index


def _read_condition(
    index: dict[tuple[float, ...], h5py.Group],
    MAS: float,
    X1: float,
    STR: tuple[float, float, float],
    gamma: float,
    t_count: int,
    grid_count: int,
) -> dict[str, np.ndarray]:
    key = _case_key(MAS, X1, STR, gamma)
    if key not in index:
        raise KeyError(f"Requested completed condition is absent: MAS={MAS}, X1={X1}, STR={STR}, gamma={gamma}.")
    group = index[key]
    if not bool(group.attrs["complete"]):
        raise ValueError(f"Requested condition is incomplete: {group.name}.")
    for name in _REQUIRED_DATASETS:
        if name not in group:
            raise KeyError(f"Required dataset {name!r} is absent from {group.name}.")

    psd_Q = np.asarray(group["psd_Q_mean"], dtype=float)
    x50 = np.asarray(group["x50_mean"], dtype=float)
    moments = np.asarray(group["moments_mean"], dtype=float)
    if psd_Q.shape != (t_count, grid_count):
        raise ValueError(f"Unexpected psd_Q_mean shape in {group.name}: {psd_Q.shape}.")
    if x50.shape != (t_count,):
        raise ValueError(f"Unexpected x50_mean shape in {group.name}: {x50.shape}.")
    if moments.ndim != 3 or moments.shape[0] < 3 or moments.shape[1] < 3 or moments.shape[2] != t_count:
        raise ValueError(f"Unexpected moments_mean shape in {group.name}: {moments.shape}.")
    if not (np.all(np.isfinite(psd_Q)) and np.all(np.isfinite(moments))):
        raise ValueError(f"Non-finite result data in {group.name}.")
    if np.any((psd_Q < 0.0) | (psd_Q > 1.0)) or np.any(np.diff(psd_Q, axis=1) < 0.0):
        raise ValueError(f"psd_Q_mean is not a monotone CDF in {group.name}.")
    # The solver's initial monodisperse snapshot stores x50_mean[0]=NaN.
    # D50(t) is therefore obtained directly from the valid stored PSD CDF.
    if not np.all(np.isfinite(x50[1:])) or np.any(x50[1:] <= 0.0):
        raise ValueError(f"x50_mean after t=0 must be finite and strictly positive in {group.name}.")
    return {"psd_Q": psd_Q, "x50": x50, "moments": moments}


def _canonical_str_request(
    STR: tuple[float, float, float], X1: float
) -> tuple[tuple[float, float, float], bool]:
    """Map the omitted phase-label mirror to its stored canonical STR ordering."""
    STR = _as_str_tuple(STR)
    if STR[0] <= STR[2]:
        return STR, False
    if float(X1) != 0.5:
        raise ValueError(
            "A reversed STR request needs phase-label exchange, which this analysis only "
            "uses at X1=0.5. Choose the stored canonical STR0 <= STR2 condition instead."
        )
    return (STR[2], STR[1], STR[0]), True


def _read_requested_str_condition(
    index: dict[tuple[float, ...], h5py.Group],
    STR: tuple[float, float, float],
    t_count: int,
    grid_count: int,
) -> dict[str, np.ndarray]:
    stored_STR, phase_swapped = _canonical_str_request(STR, BASE_X1)
    data = _read_condition(index, BASE_MAS, BASE_X1, stored_STR, BASE_GAMMA, t_count, grid_count)
    if phase_swapped:
        data["moments"] = np.swapaxes(data["moments"], 0, 1)
        print(f"STR={_format_str(STR)} uses stored phase-label mirror STR={_format_str(stored_STR)}.")
    return data


def _invert_cdf(psd_x_grid: np.ndarray, cdf: np.ndarray, quantile: float) -> float:
    if quantile < cdf[0] or quantile > cdf[-1]:
        raise ValueError(f"CDF does not cover requested quantile {quantile}: [{cdf[0]}, {cdf[-1]}].")
    right = int(np.searchsorted(cdf, quantile, side="left"))
    if right == 0:
        return float(psd_x_grid[0])
    if cdf[right] == quantile:
        return float(psd_x_grid[right])
    left = right - 1
    denominator = cdf[right] - cdf[left]
    if denominator <= 0.0:
        raise ValueError("CDF plateau prevents interpolation of the requested quantile.")
    return float(psd_x_grid[left] + (quantile - cdf[left]) * (psd_x_grid[right] - psd_x_grid[left]) / denominator)


def _set_scan_x_axis(axis: plt.Axes, parameter: str) -> None:
    if parameter == "gamma":
        axis.set_xscale("log")


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_final_diameters(
    parameter: str, values: tuple[float, ...], data: list[dict[str, np.ndarray]], psd_x_grid: np.ndarray, output_directory: Path
) -> Path:
    d10 = np.array([_invert_cdf(psd_x_grid, item["psd_Q"][-1], 0.1) for item in data])
    d90 = np.array([_invert_cdf(psd_x_grid, item["psd_Q"][-1], 0.9) for item in data])
    d50 = np.array([item["x50"][-1] for item in data])
    figure, left = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    right = left.twinx()
    first = left.plot(values, d90 / d10, "o-", color="tab:blue", label=r"$D_{90}/D_{10}$")
    second = right.plot(values, d50, "s-", color="tab:orange", label=r"$D_{50}$")
    _set_scan_x_axis(left, parameter)
    left.set_xlabel(parameter)
    left.set_ylabel(r"$D_{90}/D_{10}$", color="tab:blue")
    right.set_ylabel(r"$D_{50}$", color="tab:orange")
    left.ticklabel_format(axis="y", style="plain", useOffset=False)
    right.ticklabel_format(axis="y", style="plain", useOffset=False)
    left.set_title(f"Final size metrics: {parameter} scan")
    left.grid(True)
    left.legend(first + second, [line.get_label() for line in first + second])
    return _save_figure(figure, output_directory / f"final_size_metrics_vs_{parameter}.png")


def _plot_time_series(
    parameter: str,
    values: tuple[float, ...],
    data: list[dict[str, np.ndarray]],
    t_vec: np.ndarray,
    output_directory: Path,
    psd_x_grid: np.ndarray,
    moment: tuple[int, int] | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for value, item in zip(values, data, strict=True):
        y_values = (
            np.array([_invert_cdf(psd_x_grid, cdf, 0.5) for cdf in item["psd_Q"]])
            if moment is None else item["moments"][moment[0], moment[1]]
        )
        axis.plot(t_vec, y_values, "o-", label=f"{parameter}={_format_number(value)}")
    axis.set_xlabel("time")
    axis.set_ylabel(r"$D_{50}$" if moment is None else rf"$\mu_{{{moment[0]},{moment[1]}}}$")
    axis.ticklabel_format(
        axis="y", style="plain" if moment is None else "sci", scilimits=(0, 0), useOffset=False
    )
    title_quantity = r"$D_{50}(t)$" if moment is None else rf"$\mu_{{{moment[0]},{moment[1]}}}(t)$"
    axis.set_title(f"{title_quantity}: {parameter} scan")
    axis.grid(True)
    axis.legend()
    suffix = "d50" if moment is None else f"mu{moment[0]}{moment[1]}"
    return _save_figure(figure, output_directory / f"{suffix}_vs_{parameter}.png")


def _plot_str_moments(
    data: list[dict[str, np.ndarray]], t_vec: np.ndarray, output_directory: Path
) -> list[Path]:
    outputs: list[Path] = []
    for moment in _MOMENT_ORDERS:
        figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for STR, item in zip(STR_COMPARISONS, data, strict=True):
            axis.plot(t_vec, item["moments"][moment[0], moment[1]], "o-", label=f"STR={_format_str(STR)}")
        axis.set_xlabel("time")
        axis.set_ylabel(rf"$\mu_{{{moment[0]},{moment[1]}}}$")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
        axis.set_title(rf"$\mu_{{{moment[0]},{moment[1]}}}(t)$: STR comparison")
        axis.grid(True)
        axis.legend()
        outputs.append(_save_figure(figure, output_directory / f"mu{moment[0]}{moment[1]}_vs_str.png"))
    return outputs


def run_analysis(
    result_h5: Path = RESULT_H5,
    output_directory: Path = OUTPUT_DIRECTORY,
    run_primary_comparisons: bool = RUN_PRIMARY_COMPARISONS,
    run_str_moment_comparisons: bool = RUN_STR_MOMENT_COMPARISONS,
) -> list[Path]:
    """Validate a completed HDF5 scan and write the enabled comparison PNG files."""
    _validate_configuration()
    result_h5 = Path(result_h5)
    output_directory = Path(output_directory)
    if not result_h5.is_file():
        raise FileNotFoundError(f"Result HDF5 does not exist: {result_h5}")

    with h5py.File(result_h5, "r") as h5_file:
        t_vec, psd_x_grid = _read_root_arrays(h5_file)
        index = _condition_index(h5_file)
        primary_data: dict[str, tuple[tuple[float, ...], list[dict[str, np.ndarray]]]] = {}
        if run_primary_comparisons:
            scan_definitions = {
                "MAS": (MAS_VALUES, lambda value: (value, BASE_X1, BASE_STR, BASE_GAMMA)),
                "X1": (X1_VALUES, lambda value: (BASE_MAS, value, BASE_STR, BASE_GAMMA)),
                "gamma": (GAMMA_VALUES, lambda value: (BASE_MAS, BASE_X1, BASE_STR, value)),
            }
            for parameter, (values, coordinates) in scan_definitions.items():
                primary_data[parameter] = (
                    tuple(float(value) for value in values),
                    [_read_condition(index, *coordinates(value), t_vec.size, psd_x_grid.size) for value in values],
                )
        str_data = (
            [_read_requested_str_condition(index, STR, t_vec.size, psd_x_grid.size) for STR in STR_COMPARISONS]
            if run_str_moment_comparisons else []
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for parameter, (values, data) in primary_data.items():
        outputs.append(_plot_final_diameters(parameter, values, data, psd_x_grid, output_directory))
        outputs.append(_plot_time_series(parameter, values, data, t_vec, output_directory, psd_x_grid))
        outputs.extend(
            _plot_time_series(parameter, values, data, t_vec, output_directory, psd_x_grid, moment)
            for moment in _MOMENT_ORDERS
        )
    if run_str_moment_comparisons:
        outputs.extend(_plot_str_moments(str_data, t_vec, output_directory))
    return outputs


if __name__ == "__main__":
    OUTPUTS = run_analysis()
    print(f"Wrote {len(OUTPUTS)} PNG files to {Path(OUTPUT_DIRECTORY)}")
