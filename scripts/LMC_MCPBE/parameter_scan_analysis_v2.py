# -*- coding: utf-8 -*-
"""Discrete four-factor sensitivity analysis of an existing PBE scan.

Edit the configuration below in Spyder. No PBE simulation is run here.
STR is ONE joint categorical factor, with the stored STR0 <= STR2 combinations.
All factor levels have equal probability. Bootstrap intervals describe Monte
Carlo repeat uncertainty, not physical/model uncertainty or simultaneous bands.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import csv
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from PIL import Image

from parameter_scan_analysis import _invert_cdf


# =============================================================================
# Spyder configuration
# =============================================================================
RESULT_H5 = Path(r"D:\Codex_tem\LMC\pbe_parameter_scan_phase3\phase1_results.h5")
OUTPUT_DIRECTORY = RESULT_H5.parent / "analysis_v2"
STRICT_CHECKS = False
OVERWRITE_OUTPUTS = False
RESPONSE_NAMES = ("mu00", "mu20", "mu02", "mu11", "M2_total", "D50", "D90_D10", "fine_volume")
BOOTSTRAP_N = 500
ANALYSIS_SEED = 42
FINE_DIAMETER_FRACTION = 0.9
INTERACTION_TIMES = (10.0, 20.0, 40.0)
PLOT_CONDITIONAL_RESPONSES = True
PLOT_MARGINAL_RESPONSES = True
PLOT_SENSITIVITY = True
PLOT_INTERACTIONS = True
STR_CURVES_PER_PAGE = 10
RESPONSE_YSCALE = "symlog"  # or "linear"
SYMLOG_LINTHRESH = 1.0e-6
FIGURE_DPI = 140
N_PLOT_WORKERS = 4  # Only PNG rendering is parallel; statistical RNG stays serial.

FACTORS = ("MAS", "X1", "STR", "gamma")
SUBSETS = tuple(s for size in range(1, 5) for s in combinations(range(4), size))
PAIRS = tuple(s for s in SUBSETS if len(s) == 2)
ALL_RESPONSES = ("mu00", "mu20", "mu02", "mu11", "M2_total", "D50", "D90_D10", "fine_volume")
RESPONSE_LABELS = {
    "mu00": r"$\Delta\mu_{00}/\mu_{00}(0)$ [%]",
    "mu20": r"$\Delta\mu_{20}/\mu_{20}(0)$ [%]",
    "mu02": r"$\Delta\mu_{02}/\mu_{02}(0)$ [%]",
    "mu11": r"$\Delta\mu_{11}/\mu_{11}(0)$ [%]",
    "M2_total": r"$\Delta M_2/M_2(0)$ [%]",
    "D50": r"$\Delta D_{50}/D_{50}(0)$ [%]",
    "D90_D10": r"$\Delta(D_{90}/D_{10})/(D_{90}/D_{10})_0$ [%; grid-based]",
    "fine_volume": "Fine volume-fraction change [percentage points; grid-based]",
}
EVENT_NAMES = ("sim_agg", "sim_break", "real_agg", "real_break")
# Roundoff tolerances only verify saved numerical results; they never alter data.
CHECK_RTOL = 1.0e-10
CDF_ROUNDOFF = 1.0e-12


@dataclass(frozen=True)
class AnalysisConfig:
    result_h5: Path = RESULT_H5
    output_directory: Path = OUTPUT_DIRECTORY
    strict_checks: bool = STRICT_CHECKS
    overwrite_outputs: bool = OVERWRITE_OUTPUTS
    response_names: tuple[str, ...] = RESPONSE_NAMES
    bootstrap_n: int = BOOTSTRAP_N
    analysis_seed: int = ANALYSIS_SEED
    fine_diameter_fraction: float = FINE_DIAMETER_FRACTION
    interaction_times: tuple[float, ...] = INTERACTION_TIMES
    plot_conditional_responses: bool = PLOT_CONDITIONAL_RESPONSES
    plot_marginal_responses: bool = PLOT_MARGINAL_RESPONSES
    plot_sensitivity: bool = PLOT_SENSITIVITY
    plot_interactions: bool = PLOT_INTERACTIONS
    str_curves_per_page: int = STR_CURVES_PER_PAGE
    response_yscale: str = RESPONSE_YSCALE
    symlog_linthresh: float = SYMLOG_LINTHRESH
    figure_dpi: int = FIGURE_DPI
    n_plot_workers: int = N_PLOT_WORKERS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config: AnalysisConfig) -> None:
    if not config.response_names or len(set(config.response_names)) != len(config.response_names):
        raise ValueError("response_names must be nonempty and unique.")
    if any(name not in ALL_RESPONSES for name in config.response_names):
        raise ValueError(f"Unknown response; choose from {ALL_RESPONSES}.")
    for name in ("bootstrap_n", "str_curves_per_page", "figure_dpi", "n_plot_workers"):
        value = asdict(config)[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config.bootstrap_n < 2 or not isinstance(config.analysis_seed, int) or config.analysis_seed < 0:
        raise ValueError("Use bootstrap_n >= 2 and a nonnegative integer analysis_seed.")
    if not 0 < config.fine_diameter_fraction < 1:
        raise ValueError("fine_diameter_fraction must be strictly between 0 and 1.")
    if config.response_yscale not in ("linear", "symlog") or not np.isfinite(config.symlog_linthresh) or config.symlog_linthresh <= 0:
        raise ValueError("Use linear/symlog and a positive finite symlog_linthresh.")
    if config.plot_interactions and (not config.interaction_times or len(set(config.interaction_times)) != len(config.interaction_times)):
        raise ValueError("interaction_times must be nonempty and unique.")


# =============================================================================
# Reading and response transformation
# =============================================================================
def compute_responses(moments, x50, cdf, grid, initial_diameter, config):
    """Transform each repeat before averaging; initial x50 NaN is not used."""
    raw = {
        "mu00": moments[:, 0, 0], "mu20": moments[:, 2, 0],
        "mu02": moments[:, 0, 2], "mu11": moments[:, 1, 1],
        "M2_total": moments[:, 2, 0] + 2 * moments[:, 1, 1] + moments[:, 0, 2],
        "D50": np.concatenate((np.full((len(x50), 1), initial_diameter), x50[:, 1:]), axis=1),
    }
    if "D90_D10" in config.response_names:
        flat_cdf = cdf.reshape(-1, len(grid))
        d10 = np.array([_invert_cdf(grid, q, .1) for q in flat_cdf]).reshape(cdf.shape[:2])
        d90 = np.array([_invert_cdf(grid, q, .9) for q in flat_cdf]).reshape(cdf.shape[:2])
        raw["D90_D10"] = d90 / d10
    if "fine_volume" in config.response_names:
        cutoff = config.fine_diameter_fraction * initial_diameter
        if cutoff < grid[0] or cutoff > grid[-1]:
            raise ValueError("PSD grid does not cover the configured fine-particle cutoff.")
        fine = np.array([np.interp(cutoff, grid, q) for q in cdf.reshape(-1, len(grid))]).reshape(cdf.shape[:2])
        raw["fine_volume"] = fine
    responses = []
    for name in config.response_names:
        values = raw[name]
        if name == "fine_volume":
            responses.append(100 * (values - values[:, :1]))
        else:
            if np.any(~np.isfinite(values[:, 0])) or np.any(values[:, 0] <= 0):
                raise ValueError(f"{name} needs a finite positive initial value for every repeat.")
            responses.append(100 * (values / values[:, :1] - 1))
    return np.stack(responses, axis=-1)


def read_results(config: AnalysisConfig) -> dict:
    path = Path(config.result_h5)
    input_hash = _sha256(path) if config.strict_checks else None
    with h5py.File(path, "r") as f:
        source_config = json.loads(f["config_json"][()].decode())
        if f.attrs["initial_state_kind"] != "monodisperse_2d_explicit":
            raise ValueError("This analysis requires the recorded explicit monodisperse initial state.")
        t, grid = f["t_vec"][()], f["psd_x_grid"][()]
        if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError("t_vec must start at zero and increase strictly.")
        if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
            raise ValueError("PSD diameter grid must be positive and increase strictly.")
        if config.plot_interactions and any(value not in t for value in config.interaction_times):
            raise ValueError("Every interaction time must be an explicitly saved time.")
        strengths = tuple(s for s in product(source_config["str_values"], repeat=3) if s[0] <= s[2])
        levels = (tuple(source_config["mas_values"]), tuple(source_config["x1_values"]),
                  strengths, tuple(source_config["gamma_values"]))
        if any(not values or len(set(values)) != len(values) for values in levels):
            raise ValueError("Factor levels must be nonempty and unique.")
        shape = tuple(len(values) for values in levels)
        n = int(source_config["n_repeats"])
        if n < 2:
            raise ValueError("At least two independent repeats per condition are required.")
        diameter = (6 * float(f.attrs["initial_particle_volume"]) / np.pi) ** (1 / 3)
        response = np.empty((*shape, n, len(t), len(config.response_names)))
        cases = np.empty(shape, dtype=object)
        auxiliaries = np.empty(shape, dtype=object)
        seen = set()
        for case_id, g in f["conditions"].items():
            if not bool(g.attrs["complete"]):
                raise ValueError(f"Incomplete condition: {case_id}")
            key = (float(g.attrs["MAS"]), float(g.attrs["X1"]), tuple(g.attrs["STR"]), float(g.attrs["gamma"]))
            idx = tuple(levels[i].index(key[i]) for i in range(4))
            if idx in seen:
                raise ValueError(f"Duplicate condition: {key}")
            seen.add(idx)
            m, x = g["moments"][()], g["x50"][()]
            q = g["psd_Q"][()] if any(k in config.response_names for k in ("D90_D10", "fine_volume")) else None
            if m.shape != (n, 3, 3, len(t)) or x.shape != (n, len(t)) or (q is not None and q.shape != (n, len(t), len(grid))):
                raise ValueError(f"Unexpected moments/x50/PSD dimensions: {case_id}")
            events = g["events"][()]
            if events.shape != (n, len(t), 4) or g["seed"].shape != (n,):
                raise ValueError(f"Unexpected event/seed dimensions: {case_id}")
            error, initial, final = g["phase_volume_relative_error"][()], g["initial_phase_volume"][()], g["final_phase_volume"][()]
            recon = g["reconstruction_count"][()]
            if error.shape != (n,) or initial.shape != (n, 2) or final.shape != (n, 2) or recon.shape != (n,):
                raise ValueError(f"Unexpected conservation/reconstruction dimensions: {case_id}")
            if config.strict_checks:
                for array in (m, x[:, 1:], events, error, initial, final, recon):
                    if not np.all(np.isfinite(array)):
                        raise ValueError(f"Nonfinite result data: {case_id}")
                if np.any(x[:, 1:] <= 0):
                    raise ValueError(f"Nonpositive saved D50: {case_id}")
                if q is not None and (not np.all(np.isfinite(q)) or np.any(np.diff(q, axis=-1) < -CDF_ROUNDOFF) or np.min(q) < -CDF_ROUNDOFF or np.max(q) > 1 + CDF_ROUNDOFF):
                    raise ValueError(f"Invalid PSD CDF: {case_id}")
                np.testing.assert_allclose(m.mean(axis=0), g["moments_mean"][()], rtol=CHECK_RTOL, atol=0)
                np.testing.assert_allclose(events.mean(axis=0), g["events_mean"][()], rtol=CHECK_RTOL, atol=0)
                np.testing.assert_allclose(g["psd_support"][:, 0], diameter, rtol=CHECK_RTOL, atol=0)
                np.testing.assert_allclose(initial, final, rtol=float(source_config["mass_conservation_rtol"]), atol=0)
                if len(set(g["seed"][()].tolist())) != n:
                    raise ValueError(f"Repeat seeds are not unique: {case_id}")
            response[idx] = compute_responses(m, x, q, grid, diameter, config)
            cases[idx] = dict(case_id=case_id, MAS=key[0], X1=key[1], STR0=key[2][0], STR1=key[2][1], STR2=key[2][2], gamma=key[3])
            auxiliaries[idx] = dict(events_mean=events.mean(axis=0), events_std=events.std(axis=0, ddof=1),
                initial_phase_volume=initial.mean(axis=0), final_phase_volume=final.mean(axis=0),
                conservation_error_max=float(np.max(error)), reconstruction_mean=float(recon.mean()),
                reconstruction_std=float(recon.std(ddof=1)))
        if len(seen) != math.prod(shape):
            raise ValueError(f"Incomplete full factorial grid: found {len(seen)}, expected {math.prod(shape)} conditions.")
        fingerprint = str(f.attrs["config_fingerprint"])
    if config.strict_checks and not np.all(np.isfinite(response)):
        raise ValueError("Response transformation produced nonfinite values.")
    return dict(levels=levels, t=t, grid=grid, responses=response, cases=cases, auxiliaries=auxiliaries,
                source_config=source_config, input_sha256=input_hash, source_fingerprint=fingerprint)


# =============================================================================
# Orthogonal functional ANOVA on the finite uniform product grid
# =============================================================================
def functional_anova(means: np.ndarray, strict_checks: bool = False) -> dict:
    """f_U=E[f|X_U]-sum_{V subsetneq U} f_V; V_U=E[f_U**2]."""
    axes = (0, 1, 2, 3)
    # Subtract an anchor first, so an exactly constant response has exact zero variance.
    shifted = means - means[0, 0, 0, 0]
    centered = shifted - shifted.mean(axis=axes, keepdims=True)
    variance = np.mean(centered ** 2, axis=axes)
    effects, components = {}, []
    for subset in SUBSETS:
        other = tuple(i for i in axes if i not in subset)
        effect = centered.mean(axis=other, keepdims=True) if other else centered.copy()
        for smaller, previous in effects.items():
            if set(smaller).issubset(subset):
                effect = effect - previous
        effects[subset] = effect
        components.append(np.mean(effect ** 2, axis=axes))
    components = np.stack(components)
    total = np.stack([components[[k for k, s in enumerate(SUBSETS) if i in s]].sum(axis=0) for i in axes])
    if strict_checks:
        np.testing.assert_allclose(components.sum(axis=0), variance, rtol=CHECK_RTOL, atol=0)
    return dict(variance=variance, components=components, total=total, effects=effects)


def defined_ratio(numerator, denominator):
    """NaN denotes an undefined statistic (zero variance), never repaired input."""
    out = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    return np.divide(numerator, denominator, out=out, where=np.asarray(denominator) > 0)


def percentile_interval(samples):
    """Only defined bootstrap statistics enter quantiles; report their counts."""
    flat = samples.reshape(len(samples), -1)
    counts = np.sum(np.isfinite(flat), axis=0)
    bounds = np.full((2, flat.shape[1]), np.nan)
    defined = counts > 0
    if np.any(defined):
        bounds[:, defined] = np.nanquantile(flat[:, defined], (.025, .975), axis=0)
    return bounds.T.reshape((*samples.shape[1:], 2)), counts.reshape(samples.shape[1:])


# =============================================================================
# Whole-trajectory bootstrap
# =============================================================================
def bootstrap_analysis(scan: dict, config: AnalysisConfig) -> dict:
    raw = scan["responses"]
    shape = raw.shape[:4]
    n, nt, nm = raw.shape[4:]
    flat = raw.reshape(-1, n, nt, nm)
    count = len(flat)
    means, sd = raw.mean(axis=4), raw.std(axis=4, ddof=1)
    point = functional_anova(means, config.strict_checks)
    b = config.bootstrap_n
    rng = np.random.default_rng(config.analysis_seed)
    indices = rng.integers(0, n, size=(b, count, n))
    curve_ci = np.empty((count, nt, nm, 2))
    for case in range(count):
        boot_mean = flat[case][indices[:, case]].mean(axis=1)
        curve_ci[case], _ = percentile_interval(boot_mean)
    print(f"Bootstrap: {count} condition response intervals prepared.", flush=True)
    component_samples = np.empty((b, len(SUBSETS), nt, nm))
    variance_samples = np.empty((b, nt, nm))
    total_samples = np.empty((b, 4, nt, nm))
    marginal_samples = [np.empty((b, size, nt, nm)) for size in shape]
    for draw in range(b):
        sample = flat[np.arange(count)[:, None], indices[draw]].mean(axis=1).reshape((*shape, nt, nm))
        result = functional_anova(sample, config.strict_checks)
        component_samples[draw], variance_samples[draw], total_samples[draw] = result["components"], result["variance"], result["total"]
        for axis in range(4):
            marginal_samples[axis][draw] = sample.mean(axis=tuple(i for i in range(4) if i != axis))
        if (draw + 1) % 50 == 0 or draw + 1 == b:
            print(f"Bootstrap variance decomposition: {draw+1}/{b}", flush=True)
    component_index = defined_ratio(point["components"], point["variance"])
    total_index = defined_ratio(point["total"], point["variance"])
    component_ci, component_valid = percentile_interval(defined_ratio(component_samples, variance_samples[:, None]))
    total_ci, total_valid = percentile_interval(defined_ratio(total_samples, variance_samples[:, None]))
    component_variance_ci, _ = percentile_interval(component_samples)
    integrated_var = np.trapezoid(point["variance"], scan["t"], axis=0)
    integrated_components = defined_ratio(np.trapezoid(point["components"], scan["t"], axis=1), integrated_var)
    integrated_total = defined_ratio(np.trapezoid(point["total"], scan["t"], axis=1), integrated_var)
    boot_integrated_var = np.trapezoid(variance_samples, scan["t"], axis=1)
    integrated_component_ci, integrated_component_valid = percentile_interval(defined_ratio(
        np.trapezoid(component_samples, scan["t"], axis=2), boot_integrated_var[:, None]))
    integrated_total_ci, integrated_total_valid = percentile_interval(defined_ratio(
        np.trapezoid(total_samples, scan["t"], axis=2), boot_integrated_var[:, None]))
    return dict(means=means, sd=sd, curve_ci=curve_ci.reshape((*shape, nt, nm, 2)), point=point,
        component_index=component_index, total_index=total_index, component_ci=component_ci,
        total_ci=total_ci, component_valid=component_valid, total_valid=total_valid,
        component_variance_ci=component_variance_ci,
        integrated_components=integrated_components, integrated_total=integrated_total,
        integrated_component_ci=integrated_component_ci, integrated_total_ci=integrated_total_ci,
        integrated_component_valid=integrated_component_valid, integrated_total_valid=integrated_total_valid,
        marginal_ci=[percentile_interval(a)[0] for a in marginal_samples],
        noise_repeat_variance=np.mean(sd ** 2, axis=(0, 1, 2, 3)),
        noise_mean_variance=np.mean(sd ** 2 / n, axis=(0, 1, 2, 3)))


# =============================================================================
# Output tables / output protection
# =============================================================================
def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def analysis_identity(scan, config):
    options = asdict(config)
    for key in ("result_h5", "output_directory", "strict_checks", "overwrite_outputs", "n_plot_workers"):
        options.pop(key)
    return _json_value(dict(source=str(Path(config.result_h5).resolve()), source_fingerprint=scan["source_fingerprint"],
        levels=scan["levels"], times=scan["t"], options=options, method_version=2))


def prepare_output(scan, config):
    directory = Path(config.output_directory)
    identity = analysis_identity(scan, config)
    if directory.exists() and any(directory.iterdir()):
        if not config.overwrite_outputs:
            raise FileExistsError(f"Analysis output exists; choose a new directory or set OVERWRITE_OUTPUTS=True: {directory}")
        previous = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Existing analysis uses a different input/configuration; use a new output directory.")
        if config.strict_checks and previous["input_sha256"] is not None and previous["input_sha256"] != scan["input_sha256"]:
            raise ValueError("Input contents changed; use a new output directory.")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(identity=identity, input_sha256=scan["input_sha256"], strict_checks=config.strict_checks, complete=False)
    _write_json(directory / "manifest.json", manifest)
    return manifest


def _write_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    count = 1
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _subset_label(subset):
    return " x ".join(FACTORS[i] for i in subset)


def _level_label(value):
    return "(" + ",".join(f"{v:g}" for v in value) + ")" if isinstance(value, tuple) else f"{value:g}"


def export_tables(scan, result, config):
    out, t, levels = Path(config.output_directory), scan["t"], scan["levels"]
    names = config.response_names
    shape = tuple(len(v) for v in levels)

    def condition_rows():
        for idx in np.ndindex(shape):
            for ti, instant in enumerate(t):
                row = dict(scan["cases"][idx], time=instant)
                for mi, name in enumerate(names):
                    row.update({f"{name}_mean": result["means"][idx][ti, mi], f"{name}_std": result["sd"][idx][ti, mi],
                        f"{name}_ci95_low": result["curve_ci"][idx][ti, mi, 0], f"{name}_ci95_high": result["curve_ci"][idx][ti, mi, 1]})
                for ei, event in enumerate(EVENT_NAMES):
                    row[f"{event}_mean"] = scan["auxiliaries"][idx]["events_mean"][ti, ei]
                    row[f"{event}_std"] = scan["auxiliaries"][idx]["events_std"][ti, ei]
                yield row

    def component_rows():
        for si, subset in enumerate(SUBSETS):
            for ti, instant in enumerate(t):
                for mi, name in enumerate(names):
                    yield dict(response=name, time=instant, component=_subset_label(subset), order=len(subset),
                        total_variance=result["point"]["variance"][ti, mi], variance=result["point"]["components"][si, ti, mi],
                        variance_ci95_low=result["component_variance_ci"][si, ti, mi, 0], variance_ci95_high=result["component_variance_ci"][si, ti, mi, 1],
                        sensitivity=result["component_index"][si, ti, mi],
                        ci95_low=result["component_ci"][si, ti, mi, 0], ci95_high=result["component_ci"][si, ti, mi, 1],
                        valid_bootstraps=result["component_valid"][si, ti, mi],
                        status="defined" if result["point"]["variance"][ti, mi] > 0 else "zero_variance")

    def sensitivity_rows():
        for axis, factor in enumerate(FACTORS):
            for ti, instant in enumerate(t):
                for mi, name in enumerate(names):
                    yield dict(response=name, time=instant, factor=factor,
                        first=result["component_index"][axis, ti, mi], first_ci95_low=result["component_ci"][axis, ti, mi, 0],
                        first_ci95_high=result["component_ci"][axis, ti, mi, 1], first_valid_bootstraps=result["component_valid"][axis, ti, mi],
                        total=result["total_index"][axis, ti, mi], total_ci95_low=result["total_ci"][axis, ti, mi, 0],
                        total_ci95_high=result["total_ci"][axis, ti, mi, 1], total_valid_bootstraps=result["total_valid"][axis, ti, mi],
                        status="defined" if result["point"]["variance"][ti, mi] > 0 else "zero_variance")

    def marginal_rows():
        for axis, factor in enumerate(FACTORS):
            values = result["means"].mean(axis=tuple(i for i in range(4) if i != axis))
            for li, level in enumerate(levels[axis]):
                for ti, instant in enumerate(t):
                    for mi, name in enumerate(names):
                        yield dict(response=name, factor=factor, level=_level_label(level), time=instant,
                            mean=values[li, ti, mi], ci95_low=result["marginal_ci"][axis][li, ti, mi, 0],
                            ci95_high=result["marginal_ci"][axis][li, ti, mi, 1])

    def interaction_rows():
        for pair in PAIRS:
            values = result["means"].mean(axis=tuple(i for i in range(4) if i not in pair))
            effect = np.squeeze(result["point"]["effects"][pair], axis=tuple(i for i in range(4) if i not in pair))
            for ia, ib in product(range(len(levels[pair[0]])), range(len(levels[pair[1]]))):
                for ti, instant in enumerate(t):
                    for mi, name in enumerate(names):
                        yield dict(response=name, factor_a=FACTORS[pair[0]], level_a=_level_label(levels[pair[0]][ia]),
                            factor_b=FACTORS[pair[1]], level_b=_level_label(levels[pair[1]][ib]), time=instant,
                            mean_response=values[ia, ib, ti, mi], pure_interaction=effect[ia, ib, ti, mi])

    def noise_rows():
        for ti, instant in enumerate(t):
            for mi, name in enumerate(names):
                yield dict(response=name, time=instant, repeat_variance=result["noise_repeat_variance"][ti, mi],
                    condition_mean_noise_variance=result["noise_mean_variance"][ti, mi],
                    observed_condition_variance=result["point"]["variance"][ti, mi],
                    noise_fraction=defined_ratio(result["noise_mean_variance"][ti, mi], result["point"]["variance"][ti, mi]).item())

    def auxiliary_rows():
        for idx in np.ndindex(shape):
            aux = scan["auxiliaries"][idx]
            yield dict(scan["cases"][idx], conservation_error_max=aux["conservation_error_max"],
                initial_phase1_volume=aux["initial_phase_volume"][0], initial_phase2_volume=aux["initial_phase_volume"][1],
                final_phase1_volume=aux["final_phase_volume"][0], final_phase2_volume=aux["final_phase_volume"][1],
                reconstruction_mean=aux["reconstruction_mean"], reconstruction_std=aux["reconstruction_std"])

    tables = {"condition_response_time.csv": condition_rows(), "variance_components_time.csv": component_rows(),
        "sensitivity_time.csv": sensitivity_rows(), "marginal_response_time.csv": marginal_rows(),
        "interaction_response_time.csv": interaction_rows(), "noise_time.csv": noise_rows(),
        "auxiliary_conditions.csv": auxiliary_rows()}
    counts = {name: _write_csv(out / name, rows) for name, rows in tables.items()}
    summary = dict(method="Finite uniform four-factor functional ANOVA of repeat means", factors=FACTORS,
        levels=levels, end_time=t[-1], n_conditions=math.prod(shape), n_repeats=scan["responses"].shape[4],
        bootstrap_n=config.bootstrap_n, analysis_seed=config.analysis_seed, confidence_level=.95,
        response_definition="100*(value(t)/value(0)-1); fine_volume is a percentage-point difference",
        total_m2_definition="mu20+2*mu11+mu02", fine_diameter_fraction=config.fine_diameter_fraction,
        D50_definition="Saved positive-time x50, analytical monodisperse initial diameter; gridded quantiles can introduce an initial discretization offset",
        PSD_limit="Saved diameter quantiles and fine_volume are grid-based; tiny changes can be resolution-limited",
        confidence_limit="Pointwise bootstrap intervals; only repeat uncertainty, not physical or model uncertainty",
        undefined_statistics="Zero-variance sensitivities are NaN in CSV/null in JSON; undefined bootstrap draws are counted and excluded only from quantiles",
        noise_limit="Indices describe observed condition means, without noise subtraction; high-order components can include Monte Carlo noise",
        total_effect_limit="Total effects share interactions and must not be summed",
        source=str(Path(config.result_h5).resolve()), input_sha256=scan["input_sha256"],
        source_fingerprint=scan["source_fingerprint"], configuration=asdict(config),
        source_configuration=scan["source_config"], strict_checks=config.strict_checks,
        max_conservation_error=max(a["conservation_error_max"] for a in scan["auxiliaries"].flat), responses={})
    rank_rows = []
    for mi, name in enumerate(names):
        summary["responses"][name] = dict(
            final_first=result["component_index"][:4, -1, mi], final_total=result["total_index"][:, -1, mi],
            final_first_ci95=result["component_ci"][:4, -1, mi], final_total_ci95=result["total_ci"][:, -1, mi],
            integrated_first=result["integrated_components"][:4, mi], integrated_total=result["integrated_total"][:, mi],
            integrated_first_ci95=result["integrated_component_ci"][:4, mi], integrated_total_ci95=result["integrated_total_ci"][:, mi],
            integrated_components={_subset_label(s): result["integrated_components"][si, mi] for si, s in enumerate(SUBSETS)},
            final_noise_fraction=defined_ratio(result["noise_mean_variance"][-1, mi], result["point"]["variance"][-1, mi]).item())
        response_summary = summary["responses"][name]
        response_summary["rankings"] = {
            f"{scope}_{kind}": [FACTORS[i] for i in np.argsort(-response_summary[f"{scope}_{kind}"])
                                if np.isfinite(response_summary[f"{scope}_{kind}"][i])]
            for scope, kind in product(("final", "integrated"), ("first", "total"))}
        for axis, factor in enumerate(FACTORS):
            for scope in ("final", "integrated"):
                for kind in ("first", "total"):
                    values = summary["responses"][name][f"{scope}_{kind}"]
                    interval = summary["responses"][name][f"{scope}_{kind}_ci95"][axis]
                    rank = None if not np.isfinite(values[axis]) else 1 + int(np.sum(values > values[axis]))
                    rank_rows.append(dict(response=name, scope=scope, kind=kind, factor=factor, rank=rank,
                        sensitivity=values[axis], ci95_low=interval[0], ci95_high=interval[1]))
    counts["sensitivity_ranking.csv"] = _write_csv(out / "sensitivity_ranking.csv", rank_rows)
    _write_json(out / "summary.json", summary)
    return counts


# =============================================================================
# PNG rendering; each worker receives immutable statistics once
# =============================================================================
_PLOT_CONTEXT = None


def _initialize_plot_worker(context):
    global _PLOT_CONTEXT
    _PLOT_CONTEXT = context


def _response_axes(config):
    columns = 4
    fig, axes = plt.subplots(math.ceil(len(config.response_names) / columns), columns,
                             figsize=(18, 8), squeeze=False)
    for ax in axes.flat[len(config.response_names):]:
        ax.set_visible(False)
    return fig, list(axes.flat[:len(config.response_names)])


def curve_jobs(levels, config):
    for axis in range(4):
        pages = [range(start, min(start + config.str_curves_per_page, len(levels[axis])))
                 for start in range(0, len(levels[axis]), config.str_curves_per_page)] if axis == 2 else [range(len(levels[axis]))]
        if config.plot_conditional_responses:
            other = tuple(i for i in range(4) if i != axis)
            for background in product(*(range(len(levels[i])) for i in other)):
                for page, indices in enumerate(pages):
                    yield ("conditional", axis, background, page, tuple(indices))
        if config.plot_marginal_responses:
            for page, indices in enumerate(pages):
                yield ("marginal", axis, (), page, tuple(indices))


def plot_jobs(scan, config):
    yield from curve_jobs(scan["levels"], config)
    if config.plot_sensitivity:
        yield from ((kind,) for kind in ("first", "total", "pair", "ranking"))
    if config.plot_interactions:
        for pair, mi in product(PAIRS, range(len(config.response_names))):
            yield ("interaction", pair, mi)


def _plot_curve(job, context):
    kind, axis, background, page, indices = job
    config, scan, result = context["config"], context["scan"], context["result"]
    levels, t = scan["levels"], scan["t"]
    fig, axes = _response_axes(config)
    if kind == "conditional":
        other = tuple(i for i in range(4) if i != axis)
        fixed = dict(zip(other, background))
        title = "; ".join(f"{FACTORS[i]}={_level_label(levels[i][fixed[i]])}" for i in other)
        filename = f"{FACTORS[axis]}_background_" + "_".join(str(i) for i in background) + f"_page_{page+1}.png"
        curves, bands = [], []
        for li in indices:
            idx = tuple(li if i == axis else fixed[i] for i in range(4))
            curves.append(result["means"][idx]); bands.append(result["curve_ci"][idx])
    else:
        title = "Other factors uniformly averaged; interval describes repeat uncertainty only"
        filename = f"{FACTORS[axis]}_marginal_page_{page+1}.png"
        marginal = result["means"].mean(axis=tuple(i for i in range(4) if i != axis))
        curves, bands = marginal[list(indices)], result["marginal_ci"][axis][list(indices)]
    colors = plt.get_cmap("tab10")(np.arange(len(indices)) % 10)
    for mi, ax in enumerate(axes):
        for index, (li, values, band) in enumerate(zip(indices, curves, bands)):
            ax.plot(t, values[:, mi], color=colors[index], linewidth=1.25, label=_level_label(levels[axis][li]))
            ax.fill_between(t, band[:, mi, 0], band[:, mi, 1], color=colors[index], alpha=.12)
        if config.response_yscale == "symlog":
            ax.set_yscale("symlog", linthresh=config.symlog_linthresh)
        ax.axhline(0, color=".5", lw=.5)
        ax.set_xlabel("time [s]"); ax.set_ylabel(RESPONSE_LABELS[config.response_names[mi]], fontsize=9)
        ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=FACTORS[axis], loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"{kind.title()} response: {FACTORS[axis]} | {title}", fontsize=11)
    fig.subplots_adjust(left=.065, right=.99, bottom=.15, top=.91, hspace=.43, wspace=.38)
    return fig, Path(kind) / FACTORS[axis] / filename


def _plot_sensitivity(job, context):
    kind = job[0]
    config, scan, result = context["config"], context["scan"], context["result"]
    fig, axes = _response_axes(config)
    if kind == "ranking":
        for mi, ax in enumerate(axes):
            positions = np.arange(4)
            for offset, values, label in (
                (-.27, result["component_index"][:4, -1, mi], "final first"),
                (-.09, result["total_index"][:, -1, mi], "final total"),
                (.09, result["integrated_components"][:4, mi], "integrated first"),
                (.27, result["integrated_total"][:, mi], "integrated total")):
                ax.bar(positions + offset, values, width=.18, label=label)
            ax.set_xticks(positions, FACTORS); ax.set_ylabel("variance fraction")
            ax.set_title(config.response_names[mi]); ax.grid(axis="y", alpha=.25)
    else:
        ids = [SUBSETS.index(pair) for pair in PAIRS] if kind == "pair" else list(range(4))
        values = result["total_index"] if kind == "total" else result["component_index"]
        bands = result["total_ci"] if kind == "total" else result["component_ci"]
        labels = [_subset_label(SUBSETS[i]) for i in ids] if kind == "pair" else list(FACTORS)
        for mi, ax in enumerate(axes):
            for index, label in zip(ids, labels):
                line, = ax.plot(scan["t"], values[index, :, mi], label=label, lw=1.4)
                ax.fill_between(scan["t"], bands[index, :, mi, 0], bands[index, :, mi, 1], color=line.get_color(), alpha=.12)
            ax.set_title(config.response_names[mi]); ax.set_xlabel("time [s]")
            ax.set_ylabel("variance fraction"); ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    description = "final and variance-integrated point estimates" if kind == "ranking" else "pointwise 95% bootstrap intervals"
    fig.suptitle(f"{kind.title()} sensitivity | finite uniform grid | {description}")
    fig.subplots_adjust(left=.06, right=.99, bottom=.14, top=.91, hspace=.4, wspace=.32)
    return fig, Path("sensitivity") / f"{kind}.png"


def _plot_interaction(job, context):
    _, pair, mi = job
    config, scan, result = context["config"], context["scan"], context["result"]
    other = tuple(i for i in range(4) if i not in pair)
    means = result["means"].mean(axis=other)
    effect = np.squeeze(result["point"]["effects"][pair], axis=other)
    times = [list(scan["t"]).index(t) for t in config.interaction_times]
    # Keep a 40-level STR axis vertical so labels remain readable.
    vertical, horizontal = (pair[1], pair[0]) if pair[1] == 2 else pair
    fig, axes = plt.subplots(2, len(times), figsize=(5 * len(times), 12 if 2 in pair else 8), squeeze=False)
    for row, source in enumerate((means, effect)):
        selected = source[:, :, times, mi]
        extent = float(np.max(np.abs(selected)))
        norm = SymLogNorm(config.symlog_linthresh, vmin=-extent, vmax=extent) if extent > 0 else None
        for col, ti in enumerate(times):
            ax = axes[row, col]
            values = source[:, :, ti, mi]
            if vertical == pair[1]:
                values = values.T  # Rows correspond to vertical; columns to horizontal.
            image = ax.imshow(values, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
            ax.set_xticks(range(len(scan["levels"][horizontal])), [_level_label(v) for v in scan["levels"][horizontal]], fontsize=8)
            ax.set_yticks(range(len(scan["levels"][vertical])), [_level_label(v) for v in scan["levels"][vertical]], fontsize=7)
            ax.set_xlabel(FACTORS[horizontal]); ax.set_ylabel(FACTORS[vertical])
            ax.set_title(f"{'Mean response' if row == 0 else 'Pure interaction'} | t={scan['t'][ti]:g} s")
            fig.colorbar(image, ax=ax, fraction=.045, pad=.04)
    fig.suptitle(f"{_subset_label(pair)} | {RESPONSE_LABELS[config.response_names[mi]]}")
    fig.subplots_adjust(left=.12 if 2 in pair else .06, right=.98, bottom=.07, top=.93, hspace=.3, wspace=.45)
    return fig, Path("interactions") / f"{FACTORS[pair[0]]}_{FACTORS[pair[1]]}_{config.response_names[mi]}.png"


def _render_job(job):
    context = _PLOT_CONTEXT
    if job[0] in ("conditional", "marginal"):
        fig, relative = _plot_curve(job, context)
    elif job[0] == "interaction":
        fig, relative = _plot_interaction(job, context)
    else:
        fig, relative = _plot_sensitivity(job, context)
    path = Path(context["config"].output_directory) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=context["config"].figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(relative)


def render_plots(scan, result, config):
    jobs = list(plot_jobs(scan, config))
    if not jobs:
        return []
    # Workers do not need raw repeats, input files or conservation records.
    plot_scan = {key: scan[key] for key in ("levels", "t")}
    context = dict(scan=plot_scan, result=result, config=config)
    outputs = []
    if config.n_plot_workers == 1:
        _initialize_plot_worker(context)
        for job in jobs:
            outputs.append(_render_job(job))
            if len(outputs) % 20 == 0 or len(outputs) == len(jobs):
                print(f"PNG rendering: {len(outputs)}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.n_plot_workers, initializer=_initialize_plot_worker, initargs=(context,)) as pool:
            for output in pool.map(_render_job, jobs, chunksize=1):
                outputs.append(output)
                if len(outputs) % 20 == 0 or len(outputs) == len(jobs):
                    print(f"PNG rendering: {len(outputs)}/{len(jobs)}", flush=True)
    return outputs


def run_analysis(config: AnalysisConfig | None = None) -> Path:
    if config is None:
        config = AnalysisConfig()
    _validate_config(config)
    started = time.perf_counter()
    print(f"Reading {config.result_h5}; strict_checks={config.strict_checks}", flush=True)
    scan = read_results(config)
    manifest = prepare_output(scan, config)
    print(f"Read {scan['cases'].size} conditions, {scan['responses'].shape[4]} repeats; expected PNG count={sum(1 for _ in plot_jobs(scan, config))}", flush=True)
    result = bootstrap_analysis(scan, config)
    tables = export_tables(scan, result, config)
    print("CSV/JSON exported; rendering grouped response and interaction figures.", flush=True)
    plots = render_plots(scan, result, config)
    if config.strict_checks:
        if _sha256(Path(config.result_h5)) != scan["input_sha256"]:
            raise RuntimeError("Input HDF5 changed during analysis.")
        for name, count in tables.items():
            with (Path(config.output_directory) / name).open(newline="", encoding="utf-8-sig") as stream:
                if sum(1 for _ in csv.reader(stream)) - 1 != count:
                    raise RuntimeError(f"Output CSV row-count mismatch: {name}")
        for name in plots:
            with Image.open(Path(config.output_directory) / name) as image:
                image.verify()
    manifest.update(complete=True, csv_rows=tables, png_count=len(plots), png_files=plots,
                    elapsed_seconds=time.perf_counter()-started)
    _write_json(Path(config.output_directory) / "manifest.json", manifest)
    print(f"Complete: {len(plots)} PNGs, {len(tables)} CSVs; output={config.output_directory}", flush=True)
    return Path(config.output_directory)


if __name__ == "__main__":
    ANALYSIS_OUTPUT = run_analysis()
