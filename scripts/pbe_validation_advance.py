"""Advanced 2D validation workflow for analytical moments, dPBE, and WMCPBE."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bootstrap_project_paths() -> None:
    root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        root,
        root / "scripts" / "pbe_validation" / "new",
        root / "dpbe" / "src",
        root / "mcpbe" / "src",
        root / "pbe-core" / "src",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_project_paths()

from validation import (  # noqa: E402
    MIN,
    CaseConfig,
    DPBEVariantConfig,
    MethodResult,
    ValidationConfig,
    ValidationResult,
    ValidationRunner,
    WMCPBEVariantConfig,
)
from pbe_core.plotter.plotter_new import PaperPlotter  # noqa: E402
from wmcpbe import MCPBESolver  # noqa: E402


@dataclass
class DirichletInitialCondition:
    alpha_x: float
    alpha_y: float
    alpha_rest: float
    x_min_scale: float = 2.0
    x_max_scale: float = 0.5
    y_min_scale: float = 2.0
    y_max_scale: float = 0.5
    total_number: Optional[float] = None
    volume_concentration: Optional[float] = None
    inverse: bool = False


@dataclass
class AdvancedValidationResult:
    base_result: ValidationResult
    reference_x_centers: np.ndarray
    reference_y_centers: np.ndarray
    reference_x_edges: np.ndarray
    reference_y_edges: np.ndarray
    psd_counts: Dict[str, np.ndarray]
    psd_repeat_samples: Dict[str, np.ndarray]
    moment_repeat_samples: Dict[str, np.ndarray]
    repeat_seed_info: Dict[str, List[Dict[str, object]]]
    variances: Dict[str, np.ndarray]
    l1_errors: Dict[str, np.ndarray]
    aggregated_errors: Dict[str, float]
    aggregated_error_variances: Dict[str, float]
    cpu_times: Dict[str, float]
    moment_error_summary: Dict[str, Dict[str, Dict[str, float]]]


@dataclass
class WMCPBERunArtifacts:
    result: MethodResult
    psd_mean: np.ndarray
    moment_variance: np.ndarray
    psd_repeat_samples: np.ndarray
    moment_repeat_samples: np.ndarray
    repeat_seed_info: List[Dict[str, object]]


class Dirichlet2DValidationRunner(ValidationRunner):
    """Validation runner with a truncated 2D Dirichlet-type initial condition."""

    def __init__(self, config: ValidationConfig, init_dist: DirichletInitialCondition):
        super().__init__(config)
        if self.config.case.dim != 2:
            raise ValueError("Advanced validation currently supports dim=2 only.")
        self.init_dist = init_dist

    def _initialize_dpbe_solver(self, solver, variant) -> None:
        super()._initialize_dpbe_solver(solver, variant)
        self._apply_dirichlet_initial_condition(solver)

    def _apply_dirichlet_initial_condition(self, solver) -> None:
        x_centers = np.asarray(solver.V[1:, 0], dtype=float)
        y_centers = np.asarray(solver.V[0, 1:], dtype=float)
        x_edges = np.asarray(solver.V_e1[1:], dtype=float)
        y_edges = np.asarray(solver.V_e3[1:], dtype=float)

        x_low = float(self.init_dist.x_min_scale * x_centers.min())
        x_high = float(self.init_dist.x_max_scale * x_centers.max())
        y_low = float(self.init_dist.y_min_scale * y_centers.min())
        y_high = float(self.init_dist.y_max_scale * y_centers.max())
        if not (x_low < x_high and y_low < y_high):
            raise ValueError("Dirichlet support bounds are invalid. Check min/max scaling factors.")

        x_range = max(x_high - x_low, MIN)
        y_range = max(y_high - y_low, MIN)
        u = (x_centers[:, None] - x_low) / x_range
        v = (y_centers[None, :] - y_low) / y_range
        u_grid = np.broadcast_to(u, (x_centers.size, y_centers.size))
        v_grid = np.broadcast_to(v, (x_centers.size, y_centers.size))
        w = 1.0 - u_grid - v_grid

        coef = math.gamma(self.init_dist.alpha_x + self.init_dist.alpha_y + self.init_dist.alpha_rest)
        coef /= (
            math.gamma(self.init_dist.alpha_x)
            * math.gamma(self.init_dist.alpha_y)
            * math.gamma(self.init_dist.alpha_rest)
        )

        pdf = np.zeros((x_centers.size, y_centers.size), dtype=float)
        mask = (u_grid > 0.0) & (v_grid > 0.0) & (w > 0.0)
        pdf[mask] = (
            coef
            * u_grid[mask] ** (self.init_dist.alpha_x - 1.0)
            * v_grid[mask] ** (self.init_dist.alpha_y - 1.0)
            * w[mask] ** (self.init_dist.alpha_rest - 1.0)
            / (x_range * y_range)
        )
        if bool(self.init_dist.inverse):
            pdf = np.flip(pdf, axis=(0, 1))

        dx = np.diff(x_edges)
        dy = np.diff(y_edges)
        cell_prob = pdf * dx[:, None] * dy[None, :]
        cell_prob = np.maximum(cell_prob, 0.0)
        prob_sum = float(np.sum(cell_prob))
        if prob_sum <= 0.0:
            raise ValueError("The Dirichlet initial condition has zero mass on the selected dPBE grid.")
        cell_prob /= prob_sum

        total_volume = x_centers[:, None] + y_centers[None, :]
        c_target = self.init_dist.volume_concentration
        n0_target = self.init_dist.total_number

        if c_target is not None and n0_target is not None:
            warnings.warn(
                "Both volume_concentration and total_number are provided; "
                "volume_concentration will be used.",
                RuntimeWarning,
            )

        if c_target is not None:
            scale = float(c_target) / max(float(np.sum(cell_prob * total_volume)), MIN)
        elif n0_target is not None:
            scale = float(n0_target)
        else:
            scale = float(self.config.case.c) / max(float(np.sum(cell_prob * total_volume)), MIN)

        field = cell_prob * scale
        solver.N[:, :, 0] = 0.0
        solver.N[1:, 1:, 0] = field

        solver.validation_total_number = float(np.sum(field))
        solver.validation_volume_concentration = float(np.sum(field * total_volume))
        self.config.case.c = solver.validation_volume_concentration


class PBEValidationAdvanced:
    MOMENT_KEYS = {
        "M00": (0, 0),
        "M01": (0, 1),
        "M11": (1, 1),
        "M02": (0, 2),
    }

    def __init__(
        self,
        config: ValidationConfig,
        init_dist: DirichletInitialCondition,
        plotter: Optional[PaperPlotter] = None,
        export_dir: Optional[Path] = None,
    ):
        if config.case.dim != 2:
            raise ValueError("PBEValidationAdvanced only supports 2D cases.")
        self.config = config
        self.init_dist = init_dist
        self.runner = Dirichlet2DValidationRunner(config, init_dist)
        self.plotter = plotter or PaperPlotter()
        self.export_dir = Path(export_dir) if export_dir is not None else Path(__file__).resolve().parent / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._export_counts: Dict[str, int] = {}

    def _next_export_stem(self, method_name: str, suffix: str = "") -> Path:
        base = self._slugify(method_name)
        if suffix:
            base = f"{base}_{self._slugify(suffix)}"
        index = self._export_counts.get(base, 0) + 1
        self._export_counts[base] = index
        return self.export_dir / f"{base}_{index:02d}"

    def _slugify(self, text: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "export"

    def _sheet_name(self, text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in text)
        cleaned = cleaned.strip() or "Sheet"
        return cleaned[:31]

    def _write_excel_book(
        self,
        method_name: str,
        metadata: Dict[str, object],
        sheets: Dict[str, pd.DataFrame],
        suffix: str = "",
        export_stem: Optional[Path] = None,
    ) -> Path:
        stem = export_stem if export_stem is not None else self._next_export_stem(method_name, suffix=suffix)
        path = stem.with_suffix(".xlsx")
        with pd.ExcelWriter(path) as writer:
            meta_df = pd.DataFrame(
                [{"key": key, "value": value if np.isscalar(value) or value is None else str(value)} for key, value in metadata.items()]
            )
            meta_df.to_excel(writer, sheet_name="metadata", index=False)
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=self._sheet_name(name), index=False)
        print(f"Saved Excel export: {path}")
        return path

    def _raw_repeat_artifacts_path(self, export_stem: Path) -> Path:
        return export_stem.parent / f"{export_stem.stem}_raw_repeats.h5"

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value, key=str)
        return str(value)

    def _create_hdf5_array_dataset(self, group: Any, name: str, data: np.ndarray) -> Any:
        array = np.asarray(data)
        if array.shape == ():
            return group.create_dataset(name, data=array)
        return group.create_dataset(
            name,
            data=array,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )

    def _unique_hdf5_group_name(self, parent: Any, name: str) -> str:
        base = self._slugify(name)
        candidate = base
        index = 2
        while candidate in parent:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _select_moment_curves(self, moments: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.asarray(moments[i, j, :], dtype=float) for i, j in self.MOMENT_KEYS.values()],
            axis=-1,
        )

    def _select_repeat_moment_samples(self, repeat_moments: List[np.ndarray]) -> np.ndarray:
        return np.asarray(
            [self._select_moment_curves(moments) for moments in repeat_moments],
            dtype=float,
        )

    def _write_raw_repeat_artifacts(
        self,
        result: AdvancedValidationResult,
        path: Path,
    ) -> Optional[Path]:
        method_names = [
            name
            for name in result.base_result.methods
            if name in result.moment_repeat_samples or name in result.psd_repeat_samples
        ]
        if not method_names:
            return None

        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Saving raw repeat artifacts requires h5py in the active Python environment."
            ) from exc

        string_dtype = h5py.string_dtype(encoding="utf-8")
        moment_keys = list(self.MOMENT_KEYS.keys())

        with h5py.File(path, "w") as h5:
            h5.attrs["schema_version"] = 1
            h5.attrs["process"] = str(result.base_result.process)
            h5.attrs["kernel"] = str(result.base_result.kernel)
            h5.attrs["compression"] = "gzip"
            h5.attrs["compression_opts"] = 4
            h5.attrs["moment_keys_json"] = json.dumps(moment_keys)
            h5.attrs["content"] = "Raw WMCPBE repeat moments and PSD counts"

            time_ds = self._create_hdf5_array_dataset(h5, "time_s", result.base_result.time)
            time_ds.attrs["axes"] = "time"

            reference_group = h5.create_group("reference")
            self._create_hdf5_array_dataset(reference_group, "x_centers", result.reference_x_centers)
            self._create_hdf5_array_dataset(reference_group, "y_centers", result.reference_y_centers)
            self._create_hdf5_array_dataset(reference_group, "x_edges", result.reference_x_edges)
            self._create_hdf5_array_dataset(reference_group, "y_edges", result.reference_y_edges)
            analytical = result.base_result.methods.get("Analytical Solution")
            if analytical is not None:
                ref_moments = self._create_hdf5_array_dataset(
                    reference_group,
                    "moments",
                    self._select_moment_curves(analytical.moments),
                )
                ref_moments.attrs["axes"] = "time,moment"
                ref_moments.attrs["moment_keys_json"] = json.dumps(moment_keys)

            methods_group = h5.create_group("methods")
            for method_name in method_names:
                method = result.base_result.methods[method_name]
                method_group = methods_group.create_group(
                    self._unique_hdf5_group_name(methods_group, method_name)
                )
                method_group.attrs["method_name"] = method_name
                method_group.attrs["family"] = method.family
                method_group.create_dataset(
                    "method_meta_json",
                    data=json.dumps(method.meta, default=self._json_default, ensure_ascii=False),
                    dtype=string_dtype,
                )

                if method_name in result.repeat_seed_info:
                    seed_values = np.asarray(
                        [
                            json.dumps(info, default=self._json_default, ensure_ascii=False)
                            for info in result.repeat_seed_info[method_name]
                        ],
                        dtype=object,
                    )
                    seed_ds = method_group.create_dataset(
                        "repeat_seed_info_json",
                        data=seed_values,
                        dtype=string_dtype,
                    )
                    seed_ds.attrs["axes"] = "repeat"

                if method_name in result.moment_repeat_samples:
                    moment_ds = self._create_hdf5_array_dataset(
                        method_group,
                        "moments",
                        result.moment_repeat_samples[method_name],
                    )
                    moment_ds.attrs["axes"] = "repeat,time,moment"
                    moment_ds.attrs["moment_keys_json"] = json.dumps(moment_keys)

                if method_name in result.psd_repeat_samples:
                    psd_ds = self._create_hdf5_array_dataset(
                        method_group,
                        "psd_counts",
                        result.psd_repeat_samples[method_name],
                    )
                    psd_ds.attrs["axes"] = "repeat,time,x_bin,y_bin"
                    psd_ds.attrs["x_axis"] = "/reference/x_centers"
                    psd_ds.attrs["y_axis"] = "/reference/y_centers"
                    psd_ds.attrs["x_edges"] = "/reference/x_edges"
                    psd_ds.attrs["y_edges"] = "/reference/y_edges"

        print(f"Saved raw repeat HDF5 export: {path}")
        return path

    def _save_figure(self, fig: plt.Figure, export_stem: Path, suffix: str = "") -> Path:
        filename = export_stem.stem
        if suffix:
            filename = f"{filename}_{self._slugify(suffix)}"
        path = export_stem.parent / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved figure export: {path}")
        return path

    def _curve_sheet(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str,
        x_label: str,
        y_label: str,
        std: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> pd.DataFrame:
        data: Dict[str, object] = {
            x_label: np.asarray(x, dtype=float),
            y_label: np.asarray(y, dtype=float),
            "label": [label] * len(x),
        }
        if std is not None:
            data[f"{y_label}_std"] = np.asarray(std, dtype=float)
        if extra is not None:
            for key, value in extra.items():
                data[key] = [value] * len(x)
        return pd.DataFrame(data)

    def _surface_sheet(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        label: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> pd.DataFrame:
        x_grid, y_grid = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), indexing="ij")
        data: Dict[str, object] = {
            "x": x_grid.ravel(),
            "y": y_grid.ravel(),
            "count": np.asarray(z, dtype=float).ravel(),
            "label": [label] * x_grid.size,
        }
        if extra is not None:
            for key, value in extra.items():
                data[key] = [value] * x_grid.size
        return pd.DataFrame(data)

    def run(self) -> AdvancedValidationResult:
        canonical = self.runner.build_canonical_initial_state()
        base_result = ValidationResult(
            time=self.config.case.t_vec.copy(),
            dim=self.config.case.dim,
            kernel=self.config.case.kernel,
            process=self.config.case.process,
            canonical_initial_state=canonical,
        )

        ref_solver = self.runner._build_reference_dpbe_initialized()
        x_centers = np.asarray(ref_solver.V[:, 0], dtype=float)
        y_centers = np.asarray(ref_solver.V[0, :], dtype=float)
        x_edges = np.asarray(ref_solver.V_e1, dtype=float).copy()
        y_edges = np.asarray(ref_solver.V_e3, dtype=float).copy()
        x_edges[0] = min(float(x_edges[0]), 0.0)
        y_edges[0] = min(float(y_edges[0]), 0.0)

        analytic_moments = self.runner.compute_analytical_moments(canonical.reference_initial_moments)
        base_result.add_method(
            MethodResult(
                name="Analytical Solution",
                family="analytical",
                moments=analytic_moments,
                meta={"elapsed_s": 0.0},
            )
        )

        psd_counts: Dict[str, np.ndarray] = {}
        psd_repeat_samples: Dict[str, np.ndarray] = {}
        moment_repeat_samples: Dict[str, np.ndarray] = {}
        repeat_seed_info: Dict[str, List[Dict[str, object]]] = {}
        variances: Dict[str, np.ndarray] = {}
        cpu_times: Dict[str, float] = {"Analytical Solution": 0.0}

        for variant in self.config.dpbe_variants:
            if not variant.enabled:
                continue
            dpbe_result, dpbe_psd = self._run_dpbe_variant(variant)
            base_result.add_method(dpbe_result)
            psd_counts[dpbe_result.name] = dpbe_psd
            cpu_times[dpbe_result.name] = float(dpbe_result.meta.get("elapsed_s", 0.0))

        for variant in self.config.wmcpbe_variants:
            if not variant.enabled:
                continue
            wm_artifacts = self._run_wmcpbe_variant(
                variant=variant,
                canonical=canonical,
                x_edges=x_edges,
                y_edges=y_edges,
                reference_moments=analytic_moments,
            )
            wm_result = wm_artifacts.result
            base_result.add_method(wm_result)
            psd_counts[wm_result.name] = wm_artifacts.psd_mean
            psd_repeat_samples[wm_result.name] = wm_artifacts.psd_repeat_samples
            moment_repeat_samples[wm_result.name] = wm_artifacts.moment_repeat_samples
            repeat_seed_info[wm_result.name] = wm_artifacts.repeat_seed_info
            variances[wm_result.name] = wm_artifacts.moment_variance
            cpu_times[wm_result.name] = float(wm_result.meta.get("elapsed_s", 0.0))

        aggregated_errors = self._compute_aggregated_errors(base_result)
        aggregated_error_variances = self._collect_aggregated_error_variances(base_result)
        moment_error_summary = self._compute_moment_error_summary(base_result)

        return AdvancedValidationResult(
            base_result=base_result,
            reference_x_centers=x_centers,
            reference_y_centers=y_centers,
            reference_x_edges=x_edges,
            reference_y_edges=y_edges,
            psd_counts=psd_counts,
            psd_repeat_samples=psd_repeat_samples,
            moment_repeat_samples=moment_repeat_samples,
            repeat_seed_info=repeat_seed_info,
            variances=variances,
            l1_errors={},
            aggregated_errors=aggregated_errors,
            aggregated_error_variances=aggregated_error_variances,
            cpu_times=cpu_times,
            moment_error_summary=moment_error_summary,
        )

    def print_moment_error_summary(self, result: AdvancedValidationResult) -> None:
        summary_rows: List[Dict[str, object]] = []
        export_stem = self._next_export_stem("print_moment_error_summary")
        raw_repeat_path = self._raw_repeat_artifacts_path(export_stem)
        for method_name, summary in result.moment_error_summary.items():
            if method_name == "Analytical Solution":
                continue
            method_meta = result.base_result.methods[method_name].meta
            cpu_time_total = float(result.cpu_times[method_name])
            cpu_time_mean = float(method_meta.get("elapsed_mean_s", cpu_time_total))
            sim_agg_events = float(method_meta.get("sim_agg_events_mean", np.nan))
            sim_break_events = float(method_meta.get("sim_break_events_mean", np.nan))
            sim_total_events = float(method_meta.get("sim_total_events_mean", np.nan))
            real_agg_events = float(method_meta.get("real_agg_events_mean", np.nan))
            real_break_events = float(method_meta.get("real_break_events_mean", np.nan))
            real_total_events = float(method_meta.get("real_total_events_mean", np.nan))
            aggregated_error_variance = float(result.aggregated_error_variances.get(method_name, np.nan))
            mean_events_cpu_time = (
                cpu_time_mean / real_total_events
                if np.isfinite(real_total_events) and real_total_events > 0.0
                else np.nan
            )
            mean_sim_events_cpu_time = (
                cpu_time_mean / sim_total_events
                if np.isfinite(sim_total_events) and sim_total_events > 0.0
                else np.nan
            )
            for key in self.MOMENT_KEYS:
                entry = summary[key]
                summary_rows.append(
                    {
                        "method": method_name,
                        "moment": key,
                        "max_rel_err": entry["max_rel_err"],
                        "final_rel_err": entry["final_rel_err"],
                        "aggregated_error": result.aggregated_errors[method_name],
                        "aggregated_error_variance": aggregated_error_variance,
                        "cpu_time_total_s": cpu_time_total,
                        "cpu_time_mean_s": cpu_time_mean,
                        "sim_agg_events_mean": sim_agg_events,
                        "sim_break_events_mean": sim_break_events,
                        "sim_total_events_mean": sim_total_events,
                        "mean_sim_events_cpu_time_s": mean_sim_events_cpu_time,
                        "real_agg_events_mean": real_agg_events,
                        "real_break_events_mean": real_break_events,
                        "real_total_events_mean": real_total_events,
                        "mean_events_cpu_time_s": mean_events_cpu_time,
                    }
                )
        self._write_excel_book(
            method_name="print_moment_error_summary",
            metadata={
                "process": result.base_result.process,
                "kernel": result.base_result.kernel,
                "time_points": len(result.base_result.time),
                "reference": "Analytical Solution",
                "content": "Moment relative error summary",
                "raw_repeat_hdf5": str(raw_repeat_path),
            },
            sheets={"summary": pd.DataFrame(summary_rows)},
            export_stem=export_stem,
        )
        self._write_raw_repeat_artifacts(result, raw_repeat_path)
        print("\nAdvanced moment error summary")
        print("-" * 72)
        for method_name, summary in result.moment_error_summary.items():
            if method_name == "Analytical Solution":
                continue
            method_meta = result.base_result.methods[method_name].meta
            cpu_time_total = float(result.cpu_times[method_name])
            cpu_time_mean = float(method_meta.get("elapsed_mean_s", cpu_time_total))
            sim_agg_events = float(method_meta.get("sim_agg_events_mean", np.nan))
            sim_break_events = float(method_meta.get("sim_break_events_mean", np.nan))
            sim_total_events = float(method_meta.get("sim_total_events_mean", np.nan))
            real_agg_events = float(method_meta.get("real_agg_events_mean", np.nan))
            real_break_events = float(method_meta.get("real_break_events_mean", np.nan))
            real_total_events = float(method_meta.get("real_total_events_mean", np.nan))
            aggregated_error_variance = float(result.aggregated_error_variances.get(method_name, np.nan))
            mean_events_cpu_time = (
                cpu_time_mean / real_total_events
                if np.isfinite(real_total_events) and real_total_events > 0.0
                else np.nan
            )
            mean_sim_events_cpu_time = (
                cpu_time_mean / sim_total_events
                if np.isfinite(sim_total_events) and sim_total_events > 0.0
                else np.nan
            )
            print(f"{method_name}")
            for key in self.MOMENT_KEYS:
                entry = summary[key]
                print(
                    f"  {key}: max rel err = {entry['max_rel_err']:.6e}, "
                    f"final rel err = {entry['final_rel_err']:.6e}"
                )
            print(f"  Aggregated error = {result.aggregated_errors[method_name]:.6e}")
            print(f"  Aggregated error variance = {aggregated_error_variance:.6e}")
            print(f"  CPU time (total) = {cpu_time_total:.3f} s")
            print(f"  CPU time (mean)  = {cpu_time_mean:.3f} s")
            print(f"  Mean sim agg     = {sim_agg_events:.6e}")
            print(f"  Mean sim break   = {sim_break_events:.6e}")
            print(f"  Mean sim-event CPU = {mean_sim_events_cpu_time:.6e} s/event")
            print(f"  Mean real agg    = {real_agg_events:.6e}")
            print(f"  Mean real break  = {real_break_events:.6e}")
            print(f"  Mean real-event CPU = {mean_events_cpu_time:.6e} s/event")
            print("")

    def plot_psd_snapshot(
        self,
        result: AdvancedValidationResult,
        t_index: int = -1,
        method_names: Optional[Iterable[str]] = None,
        two_d: bool = True,
        marginal: bool = False,
        total: bool = False,
        q0: bool = False,
        q3: bool = False,
    ) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}
        export_stem = self._next_export_stem(
            "plot_psd_snapshot",
            suffix=f"t{t_index}_2d{int(two_d)}_m{int(marginal)}_tot{int(total)}_q0{int(q0)}_q3{int(q3)}",
        )

        if t_index < 0:
            t_index = len(result.base_result.time) + t_index
        t_value = float(result.base_result.time[t_index])

        ordered = list(result.psd_counts.keys()) if method_names is None else list(method_names)
        ref_name = "WMCPBE (ref)" if "WMCPBE (ref)" in result.psd_counts else None
        ref_2d = None if ref_name is None else result.psd_counts[ref_name][t_index]
        export_sheets: Dict[str, pd.DataFrame] = {}
        export_metadata: Dict[str, object] = {
            "time_index": t_index,
            "time_value_s": t_value,
            "two_d": two_d,
            "marginal": marginal,
            "total": total,
            "q0": q0,
            "q3": q3,
            "reference_name": ref_name,
            "x_label_2d": "log10(x)",
            "y_label_2d": "log10(y)",
            "z_label_2d": "Discrete PSD count",
        }
        if two_d:
            fig, ax = self.plotter.figure(projection="3d", width_scale=1.15, height_scale=1.25)
            x_plot = self._axis_for_plot(result.reference_x_centers, log_scale=True)
            y_plot = self._axis_for_plot(result.reference_y_centers, log_scale=True)
            x_mesh, y_mesh = np.meshgrid(
                np.log10(x_plot),
                np.log10(y_plot),
                indexing="ij",
            )
            for name in ordered:
                if name not in result.psd_counts:
                    continue
                family = result.base_result.methods[name].family if name in result.base_result.methods else "dpbe"
                l1 = np.nan if ref_2d is None else self._relative_l1(result.psd_counts[name][t_index], ref_2d)
                self.plotter.plot_wireframe(
                    ax,
                    x_mesh,
                    y_mesh,
                    result.psd_counts[name][t_index],
                    key=name,
                    family=family,
                    linewidth=1.0,
                    alpha=0.65,
                )
                export_sheets[f"2d_{name}"] = self._surface_sheet(
                    x=result.reference_x_centers,
                    y=result.reference_y_centers,
                    z=result.psd_counts[name][t_index],
                    label=name,
                    extra={"family": family, "l1_to_ref": l1, "reference_name": ref_name},
                )
            ax.set_zlabel("Discrete PSD count")
            self.plotter.finalize_axes(
                ax,
                xlabel="log10(x)",
                ylabel="log10(y)",
                title=f"2D PSD snapshot at t = {t_value:.3f} s",
                grid=False,
            )
            handles = []
            labels = []
            for name in ordered:
                if name not in result.psd_counts:
                    continue
                family = result.base_result.methods[name].family if name in result.base_result.methods else "dpbe"
                label = name
                if ref_2d is not None:
                    l1 = self._relative_l1(result.psd_counts[name][t_index], ref_2d)
                    label = f"{name} | L1={l1:.3e}"
                handles.append(self.plotter.proxy_handle(name, family=family))
                labels.append(label)
            if handles:
                ax.legend(handles, labels, loc="upper right")
            self.plotter.tighten(fig)
            self._save_figure(fig, export_stem, suffix="2d")
            figures["2D"] = fig

        if marginal:
            fig, axes = self.plotter.subplots(1, 2, width_scale=1.65, height_scale=1.0)
            x_axis = result.reference_x_centers
            y_axis = result.reference_y_centers
            x_axis_plot = self._axis_for_plot(x_axis, log_scale=True)
            y_axis_plot = self._axis_for_plot(y_axis, log_scale=True)
            ref_mx = None if ref_2d is None else np.sum(ref_2d, axis=1)
            ref_my = None if ref_2d is None else np.sum(ref_2d, axis=0)
            for name in ordered:
                if name not in result.psd_counts:
                    continue
                family = result.base_result.methods[name].family if name in result.base_result.methods else "dpbe"
                mean_xy = result.psd_counts[name][t_index]
                mx = np.sum(mean_xy, axis=1)
                my = np.sum(mean_xy, axis=0)
                l1_x = np.nan if ref_mx is None else self._relative_l1(mx, ref_mx)
                l1_y = np.nan if ref_my is None else self._relative_l1(my, ref_my)
                label_x = name if ref_mx is None else f"{name} | L1={l1_x:.3e}"
                label_y = name if ref_my is None else f"{name} | L1={l1_y:.3e}"
                mx_std = None
                my_std = None

                if name in result.psd_repeat_samples:
                    samples = result.psd_repeat_samples[name][:, t_index, :, :]
                    mx_rep = np.sum(samples, axis=2)
                    my_rep = np.sum(samples, axis=1)
                    mx_std = np.std(mx_rep, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(mx)
                    my_std = np.std(my_rep, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(my)
                self.plotter.plot_line(
                    axes[0], x_axis_plot, mx, key=name, family=family, label=label_x,
                    error=mx_std, markevery=max(1, len(x_axis) // 8)
                )
                self.plotter.plot_line(
                    axes[1], y_axis_plot, my, key=name, family=family, label=label_y,
                    error=my_std, markevery=max(1, len(y_axis) // 8)
                )
                export_sheets[f"marginal_x_{name}"] = self._curve_sheet(
                    x_axis,
                    mx,
                    label=name,
                    x_label="component_1_volume",
                    y_label="marginal_count",
                    std=mx_std,
                    extra={"family": family, "axis": "x", "l1_to_ref": l1_x, "reference_name": ref_name},
                )
                export_sheets[f"marginal_y_{name}"] = self._curve_sheet(
                    y_axis,
                    my,
                    label=name,
                    x_label="component_2_volume",
                    y_label="marginal_count",
                    std=my_std,
                    extra={"family": family, "axis": "y", "l1_to_ref": l1_y, "reference_name": ref_name},
                )

            self.plotter.finalize_axes(
                axes[0],
                xlabel="Component 1 volume",
                ylabel="Marginal count",
                title=f"Marginal PSD of component 1 at t = {t_value:.3f} s",
                xscale="log",
                legend=True,
            )
            self.plotter.finalize_axes(
                axes[1],
                xlabel="Component 2 volume",
                ylabel="Marginal count",
                title=f"Marginal PSD of component 2 at t = {t_value:.3f} s",
                xscale="log",
                legend=True,
            )
            self.plotter.tighten(fig)
            self._save_figure(fig, export_stem, suffix="marginal")
            figures["marginal"] = fig

        if total:
            fig, ax = self.plotter.figure()
            total_axis, _ = self._build_total_support(result.reference_x_centers, result.reference_y_centers)
            total_axis_plot = self._axis_for_plot(total_axis, log_scale=True)
            ref_total = None if ref_2d is None else self._collapse_total_distribution(
                ref_2d,
                result.reference_x_centers,
                result.reference_y_centers,
            )
            for name in ordered:
                if name not in result.psd_counts:
                    continue
                family = result.base_result.methods[name].family if name in result.base_result.methods else "dpbe"
                total_mean = self._collapse_total_distribution(
                    result.psd_counts[name][t_index],
                    result.reference_x_centers,
                    result.reference_y_centers,
                )
                l1_total = np.nan if ref_total is None else self._relative_l1(total_mean, ref_total)
                label = name if ref_total is None else f"{name} | L1={l1_total:.3e}"
                total_std = None

                if name in result.psd_repeat_samples:
                    samples = result.psd_repeat_samples[name][:, t_index, :, :]
                    total_rep = np.asarray(
                        [
                            self._collapse_total_distribution(sample, result.reference_x_centers, result.reference_y_centers)
                            for sample in samples
                        ],
                        dtype=float,
                    )
                    total_std = np.std(total_rep, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(total_mean)
                self.plotter.plot_line(
                    ax, total_axis_plot, total_mean, key=name, family=family, label=label,
                    error=total_std, markevery=max(1, len(total_axis) // 8)
                )
                export_sheets[f"total_{name}"] = self._curve_sheet(
                    total_axis,
                    total_mean,
                    label=name,
                    x_label="total_particle_volume",
                    y_label="total_volume_count",
                    std=total_std,
                    extra={"family": family, "l1_to_ref": l1_total, "reference_name": ref_name},
                )

            self.plotter.finalize_axes(
                ax,
                xlabel="Total particle volume",
                ylabel="Total-volume count",
                title=f"Total-volume PSD at t = {t_value:.3f} s",
                xscale="log",
                legend=True,
            )
            self.plotter.tighten(fig)
            self._save_figure(fig, export_stem, suffix="total")
            figures["total"] = fig

        if q0 or q3:
            x_query = self._build_qx_query(result.reference_x_centers, result.reference_y_centers, n_points=200)
            bases = []
            if q0:
                bases.append(("Q0", "number"))
            if q3:
                bases.append(("Q3", "volume"))
            for q_label, basis in bases:
                fig, ax = self.plotter.figure()
                for name in ordered:
                    if name not in result.psd_counts:
                        continue
                    family = result.base_result.methods[name].family if name in result.base_result.methods else "dpbe"
                    q_mean = self._build_q_curve_from_counts(
                        result.psd_counts[name][t_index],
                        result.reference_x_centers,
                        result.reference_y_centers,
                        x_query=x_query,
                        basis=basis,
                    )
                    q_std = None
                    if name in result.psd_repeat_samples:
                        samples = result.psd_repeat_samples[name][:, t_index, :, :]
                        q_rep = np.asarray(
                            [
                                self._build_q_curve_from_counts(
                                    sample,
                                    result.reference_x_centers,
                                    result.reference_y_centers,
                                    x_query=x_query,
                                    basis=basis,
                                )
                                for sample in samples
                            ],
                            dtype=float,
                        )
                        q_std = np.std(q_rep, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(q_mean)
                    self.plotter.plot_line(
                        ax,
                        x_query,
                        q_mean,
                        key=f"{name}_{q_label}",
                        family=family,
                        label=name,
                        error=q_std,
                        markevery=max(1, len(x_query) // 8),
                    )
                    export_sheets[f"{q_label}_{name}"] = self._curve_sheet(
                        x_query,
                        q_mean,
                        label=name,
                        x_label="particle_diameter",
                        y_label=q_label,
                        std=q_std,
                        extra={"family": family, "basis": basis},
                    )
                self.plotter.finalize_axes(
                    ax,
                    xlabel="Particle diameter",
                    ylabel=q_label,
                    title=f"{q_label} cumulative distribution at t = {t_value:.3f} s",
                    xscale="log",
                    legend=True,
                )
                self.plotter.tighten(fig)
                self._save_figure(fig, export_stem, suffix=q_label)
                figures[q_label] = fig

        self._write_excel_book(
            method_name="plot_psd_snapshot",
            metadata=export_metadata,
            sheets=export_sheets if export_sheets else {"empty": pd.DataFrame([{"note": "No PSD sheets were generated."}])},
            export_stem=export_stem,
        )
        return figures

    def plot_psd_l1_error(self, result: AdvancedValidationResult) -> Optional[plt.Figure]:
        self._write_excel_book(
            method_name="plot_psd_l1_error",
            metadata={
                "content": "PSD L1 error plotting is currently disabled in this workflow.",
                "process": result.base_result.process,
                "kernel": result.base_result.kernel,
            },
            sheets={"note": pd.DataFrame([{"message": "This plot is currently not implemented."}])},
        )
        return None

    def plot_selected_moments(
        self,
        result: AdvancedValidationResult,
        relative: bool = True,
    ) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}
        export_sheets: Dict[str, pd.DataFrame] = {}
        export_stem = self._next_export_stem(
            "plot_selected_moments",
            suffix="relative" if relative else "absolute",
        )
        for key, (i, j) in self.MOMENT_KEYS.items():
            fig, ax = self.plotter.figure()
            sheet_data: Dict[str, object] = {"time_s": np.asarray(result.base_result.time, dtype=float)}
            for name, method in result.base_result.methods.items():
                raw_values = np.asarray(method.moments[i, j, :], dtype=float).copy()
                values = raw_values.copy()
                errors = None
                if method.std is not None:
                    errors = np.asarray(method.std[i, j, :], dtype=float).copy()
                raw_errors = None if errors is None else errors.copy()
                scale = 1.0

                if relative:
                    scale = raw_values[0] + MIN
                    values = values / scale
                    if errors is not None:
                        errors = errors / scale

                self.plotter.plot_line(
                    ax,
                    result.base_result.time,
                    values,
                    key=name,
                    family=method.family,
                    label=name,
                    error=errors,
                    markevery=max(1, len(result.base_result.time) // 8),
                )
                value_col = self._slugify(f"{name}_value")
                sheet_data[value_col] = values
                sheet_data[self._slugify(f"{name}_family")] = [method.family] * len(values)
                if relative:
                    sheet_data[self._slugify(f"{name}_raw_value")] = raw_values
                if errors is not None:
                    sheet_data[self._slugify(f"{name}_std")] = errors
                    if relative and raw_errors is not None:
                        sheet_data[self._slugify(f"{name}_raw_std")] = raw_errors
            self.plotter.finalize_axes(
                ax,
                xlabel="Time $t$ / s",
                ylabel=f"Relative {key} / $-$" if relative else key,
                title=f"{key} evolution over time (relative)" if relative else f"{key} evolution over time",
                legend=True,
            )
            self.plotter.tighten(fig)
            self._save_figure(fig, export_stem, suffix=key)
            figures[key] = fig
            export_sheets[key] = pd.DataFrame(sheet_data)
        self._write_excel_book(
            method_name="plot_selected_moments",
            metadata={
                "relative": relative,
                "reference_for_error_methods": "Analytical Solution",
                "time_points": len(result.base_result.time),
                "moments": ",".join(self.MOMENT_KEYS.keys()),
            },
            sheets=export_sheets,
            export_stem=export_stem,
        )
        return figures

    def plot_error_time_pareto(self, result: AdvancedValidationResult) -> plt.Figure:
        fig, ax = self.plotter.figure()
        export_stem = self._next_export_stem("plot_error_time_pareto")
        rows: List[Dict[str, object]] = []
        for name, score in result.aggregated_errors.items():
            if name == "Analytical Solution":
                continue
            family = result.base_result.methods[name].family
            cpu_time = result.cpu_times.get(name, np.nan)
            self.plotter.plot_scatter(ax, cpu_time, score, key=name, family=family, size=70)
            ax.annotate(name, (cpu_time, score), textcoords="offset points", xytext=(6, 4))
            rows.append(
                {
                    "method": name,
                    "family": family,
                    "cpu_time_s": cpu_time,
                    "aggregated_error": score,
                }
            )
        self.plotter.finalize_axes(
            ax,
            xlabel="CPU time / s",
            ylabel="Aggregated moment error / $-$",
            title="Error-time Pareto view",
        )
        self.plotter.tighten(fig)
        self._save_figure(fig, export_stem, suffix="pareto")
        self._write_excel_book(
            method_name="plot_error_time_pareto",
            metadata={
                "x_label": "CPU time / s",
                "y_label": "Aggregated moment error / -",
                "reference": "Analytical Solution",
            },
            sheets={"pareto": pd.DataFrame(rows)},
            export_stem=export_stem,
        )
        return fig

    def plot_moment_variances(self, result: AdvancedValidationResult) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}
        export_sheets: Dict[str, pd.DataFrame] = {}
        export_stem = self._next_export_stem("plot_moment_variances")
        for key, (i, j) in self.MOMENT_KEYS.items():
            fig, ax = self.plotter.figure()
            plotted = False
            sheet_data: Dict[str, object] = {"time_s": np.asarray(result.base_result.time, dtype=float)}
            for name, variance in result.variances.items():
                if name not in result.base_result.methods:
                    continue
                method = result.base_result.methods[name]
                if method.family != "wmcpbe":
                    continue
                mean_sq = np.asarray(method.moments[i, j, :], dtype=float) ** 2
                normalized_variance = np.asarray(variance[i, j, :], dtype=float) / (mean_sq + MIN)
                self.plotter.plot_line(
                    ax,
                    result.base_result.time,
                    normalized_variance,
                    key=name,
                    family=method.family,
                    label=name,
                    markevery=max(1, len(result.base_result.time) // 8),
                )
                plotted = True
                sheet_data[self._slugify(f"{name}_normalized_variance")] = normalized_variance
            self.plotter.finalize_axes(
                ax,
                xlabel="Time $t$ / s",
                ylabel=f"Normalized variance of {key} / $-$",
                title=f"{key} variance over time (WMCPBE only)",
                legend=plotted,
            )
            self.plotter.tighten(fig)
            self._save_figure(fig, export_stem, suffix=key)
            figures[key] = fig
            export_sheets[key] = pd.DataFrame(sheet_data)
        self._write_excel_book(
            method_name="plot_moment_variances",
            metadata={
                "normalization": "Var[M_k](t) / E[M_k](t)^2",
                "families_included": "wmcpbe",
                "moments": ",".join(self.MOMENT_KEYS.keys()),
            },
            sheets=export_sheets,
            export_stem=export_stem,
        )
        return figures

    def show(self) -> None:
        self.plotter.show()

    def _run_dpbe_variant(self, variant: DPBEVariantConfig) -> Tuple[MethodResult, np.ndarray]:
        solver = self.runner._build_dpbe_solver(variant)
        self.runner._initialize_dpbe_solver(solver, variant)
        self.runner._apply_case_params(solver)

        time_start = time.time()
        solver.core.calc_F_M()
        solver.core.calc_B_R()
        solver.core.calc_int_B_F()
        solver.core.solve_PBE()
        elapsed = time.time() - time_start

        moments = solver.post.calc_mom_t()
        psd_stack = self._build_dpbe_psd_stack(solver)
        result = MethodResult(
            name=variant.name,
            family="dpbe",
            moments=moments,
            meta={"elapsed_s": elapsed, "grid": variant.grid, "NS": variant.ns, "S": variant.s},
        )
        return result, psd_stack

    def _build_dpbe_psd_stack(self, solver) -> np.ndarray:
        """Extract the physical 2D PSD from dPBE.

        In 2D dPBE, N[0, 0] is the virtual zero cell, while the remaining first
        row/column represent pure-material states on the coordinate axes. Those
        axis states are physical and must be retained in PSD-based post-process.
        """
        stack = np.asarray([solver.N[:, :, tidx] for tidx in range(solver.t_num)], dtype=float)
        if stack.size > 0:
            stack[:, 0, 0] = 0.0
        return np.maximum(stack, 0.0)

    def _run_wmcpbe_variant(
        self,
        variant: WMCPBEVariantConfig,
        canonical,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        reference_moments: np.ndarray,
    ) -> WMCPBERunArtifacts:
        solver_template = MCPBESolver(
            dim=self.config.case.dim,
            t_vec=self.config.case.t_vec,
            verbose=True,
            load_attr=False,
            init=False,
        )
        solver_template.a0 = 100000
        solver_template.CDF_method = "disc"
        solver_template.G = self.config.case.g
        solver_template.process_type = self.config.case.process
        solver_template.alpha_prim = np.ones(self.config.case.dim ** 2)
        solver_template.break_dW_mode = "const"
        solver_template.break_dW_min = 1.0
        solver_template.break_dW_max = 50.0
        solver_template.agg_dW_min = 1.0
        solver_template.agg_dW_max = 20.0
        solver_template.recon_enable = True
        solver_template.V_eff_init = 1000
        solver_template.recon_N_max = 4000
        solver_template.recon_method = "4PMC"
        solver_template.recon_bins = 30
        solver_template.recon_RS_target = 1000
        self.runner._apply_case_params(solver_template)
        self.runner._apply_attrs(solver_template, variant.attrs)

        ref_solver = self.runner._build_reference_dpbe_initialized()
        weighted_init = int(getattr(solver_template, "V_eff_init", 0) or 0) > 0
        vc, v_flat, w_init = self.runner._build_mc_initial_particles(
            ref_solver,
            canonical.n0,
            int(solver_template.a0),
            weighted_init=weighted_init,
        )

        time_start = time.time()
        repeat_records, _ = solver_template._run_repeat_records(
            N=variant.repeats,
            base_seed=variant.base_seed,
            maxiter=variant.maxiter,
            init_Vc=False,
            Vc=vc,
            V_flat=v_flat,
            W_init=w_init,
            workers=variant.workers,
            collect_hist2d_edges=(x_edges, y_edges),
            collect_event_stats=True,
        )
        elapsed = time.time() - time_start

        if not repeat_records:
            raise RuntimeError("WMCPBE repeat execution produced no completed results.")

        repeat_moments = [
            np.asarray(record["result"]["moments"], dtype=float)
            for record in repeat_records
        ]
        repeat_moment_samples = self._select_repeat_moment_samples(repeat_moments)
        repeat_psd = [
            np.asarray(record["hist2d_stack"], dtype=float)
            for record in repeat_records
        ]
        repeat_seed_info: List[Dict[str, object]] = []
        for repeat_index, record in enumerate(repeat_records):
            seed_info = copy.deepcopy((record.get("result") or {}).get("seed_info", {}))
            if not isinstance(seed_info, dict):
                seed_info = {"seed_info": seed_info}
            seed_info.setdefault("repeat_index", int(record.get("idx", repeat_index)))
            repeat_seed_info.append(seed_info)
        repeat_sim_agg_events = [
            float((record.get("event_stats") or {}).get("sim_agg_events", np.nan))
            for record in repeat_records
        ]
        repeat_sim_break_events = [
            float((record.get("event_stats") or {}).get("sim_break_events", np.nan))
            for record in repeat_records
        ]
        repeat_real_agg_events = [
            float((record.get("event_stats") or {}).get("real_agg_events", np.nan))
            for record in repeat_records
        ]
        repeat_real_break_events = [
            float((record.get("event_stats") or {}).get("real_break_events", np.nan))
            for record in repeat_records
        ]
        elapsed_mean = elapsed / max(variant.repeats, 1)

        moments_mean = np.mean(repeat_moments, axis=0)
        if len(repeat_moments) > 1:
            moments_std = np.std(repeat_moments, axis=0, ddof=1)
            moments_var = moments_std ** 2
            psd_mean = np.mean(repeat_psd, axis=0)
        else:
            moments_std = None
            moments_var = np.zeros_like(moments_mean)
            psd_mean = repeat_psd[0]

        aggregated_error_override = np.nan
        aggregated_error_variance = np.nan
        aggregated_error_repeat_count = 0
        if variant.aggregate_error_per_repeat:
            repeat_aggregated_errors = np.asarray(
                [
                    self._compute_aggregated_error_for_moments(moments, reference_moments)
                    for moments in repeat_moments
                ],
                dtype=float,
            )
            aggregated_error_repeat_count = int(repeat_aggregated_errors.size)
            aggregated_error_override = float(np.mean(repeat_aggregated_errors))
            aggregated_error_variance = (
                float(np.var(repeat_aggregated_errors, ddof=1))
                if repeat_aggregated_errors.size > 1
                else 0.0
            )

        sim_agg_mean = float(np.nanmean(repeat_sim_agg_events)) if repeat_sim_agg_events else np.nan
        sim_break_mean = float(np.nanmean(repeat_sim_break_events)) if repeat_sim_break_events else np.nan
        sim_total_mean = (
            sim_agg_mean + sim_break_mean
            if np.isfinite(sim_agg_mean) and np.isfinite(sim_break_mean)
            else np.nan
        )
        real_agg_mean = float(np.nanmean(repeat_real_agg_events)) if repeat_real_agg_events else np.nan
        real_break_mean = float(np.nanmean(repeat_real_break_events)) if repeat_real_break_events else np.nan
        real_total_mean = (
            real_agg_mean + real_break_mean
            if np.isfinite(real_agg_mean) and np.isfinite(real_break_mean)
            else np.nan
        )

        result = MethodResult(
            name=variant.name,
            family="wmcpbe",
            moments=moments_mean,
            std=moments_std,
            meta={
                "elapsed_s": elapsed,
                "elapsed_mean_s": elapsed_mean,
                "repeats": variant.repeats,
                "base_seed": variant.base_seed,
                "workers": variant.workers,
                "aggregate_error_per_repeat": variant.aggregate_error_per_repeat,
                "aggregated_error_mode": "per_repeat_mean" if variant.aggregate_error_per_repeat else "mean_moments",
                "aggregated_error_override": aggregated_error_override,
                "aggregated_error_variance": aggregated_error_variance,
                "aggregated_error_repeat_count": aggregated_error_repeat_count,
                "sim_agg_events_mean": sim_agg_mean,
                "sim_break_events_mean": sim_break_mean,
                "sim_total_events_mean": sim_total_mean,
                "real_agg_events_mean": real_agg_mean,
                "real_break_events_mean": real_break_mean,
                "real_total_events_mean": real_total_mean,
                **copy.deepcopy(variant.attrs),
            },
        )
        return WMCPBERunArtifacts(
            result=result,
            psd_mean=psd_mean,
            moment_variance=moments_var,
            psd_repeat_samples=np.asarray(repeat_psd, dtype=float),
            moment_repeat_samples=repeat_moment_samples,
            repeat_seed_info=repeat_seed_info,
        )

    def _build_wmcpbe_psd_stack(self, solver, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
        t_count = min(len(solver.V_save), len(solver.W_save), len(solver.t_vec))
        stack = []
        for tidx in range(t_count):
            v_snap = np.asarray(solver.V_save[tidx], dtype=float)
            w_snap = np.asarray(solver.W_save[tidx], dtype=float)
            hist, _, _ = np.histogram2d(
                v_snap[0, :],
                v_snap[1, :],
                bins=[x_edges, y_edges],
                weights=w_snap,
            )
            stack.append(hist)
        return np.asarray(stack, dtype=float)

    def _axis_for_plot(self, values: np.ndarray, *, log_scale: bool = False) -> np.ndarray:
        axis = np.asarray(values, dtype=float).copy()
        if not log_scale:
            return axis
        positive = axis[np.isfinite(axis) & (axis > 0.0)]
        if positive.size == 0:
            return np.ones_like(axis)
        anchor = float(np.min(positive)) * 0.5
        axis[~np.isfinite(axis) | (axis <= 0.0)] = max(anchor, MIN)
        return axis

    def _build_total_support(
        self,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        total_grid = x_centers[:, None] + y_centers[None, :]
        rounded = np.round(total_grid.ravel(), decimals=12)
        unique_vals, inverse = np.unique(rounded, return_inverse=True)
        return unique_vals.astype(float), inverse

    def _collapse_total_distribution(
        self,
        counts_2d: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
    ) -> np.ndarray:
        total_axis, inverse = self._build_total_support(x_centers, y_centers)
        flat_counts = np.asarray(counts_2d, dtype=float).ravel()
        total_counts = np.bincount(inverse, weights=flat_counts, minlength=total_axis.size)
        return total_counts.astype(float)

    def _vol2diam(self, V: np.ndarray) -> np.ndarray:
        return np.power(6.0 * np.asarray(V, dtype=float) / math.pi, 1.0 / 3.0)

    def _build_qx_query(
        self,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        n_points: int = 200,
    ) -> np.ndarray:
        total_volume = np.asarray(x_centers, dtype=float)[:, None] + np.asarray(y_centers, dtype=float)[None, :]
        diam = self._vol2diam(total_volume.ravel())
        diam = diam[np.isfinite(diam) & (diam > 0.0)]
        if diam.size == 0:
            return np.linspace(1.0, 2.0, int(n_points))
        dmin = float(np.min(diam))
        dmax = float(np.max(diam))
        if dmax <= dmin:
            dmax = dmin * (1.0 + 1e-12)
        return np.geomspace(dmin, dmax, int(n_points))

    def _compute_psd_cdf_from_counts(
        self,
        counts_2d: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        basis: str = "volume",
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        total_volume = np.asarray(x_centers, dtype=float)[:, None] + np.asarray(y_centers, dtype=float)[None, :]
        diam = self._vol2diam(total_volume.ravel())
        counts = np.asarray(counts_2d, dtype=float).ravel()
        if basis == "number":
            weights = counts
        elif basis == "volume":
            weights = counts * total_volume.ravel()
        else:
            raise ValueError(f"Unsupported basis {basis!r}; expected 'number' or 'volume'.")
        mask = np.isfinite(diam) & np.isfinite(weights) & (diam > 0.0) & (weights > 0.0)
        if not np.any(mask):
            return None
        diam = diam[mask]
        weights = weights[mask]
        order = np.argsort(diam)
        diam = diam[order]
        weights = weights[order]
        weight_cum = np.cumsum(weights)
        total = float(weight_cum[-1])
        if total <= 0.0:
            return None
        return diam, weight_cum / total

    def _eval_Q_of_x(
        self,
        x_sorted: np.ndarray,
        Q_sorted: np.ndarray,
        x_query: np.ndarray,
    ) -> np.ndarray:
        xq = np.asarray(x_query, dtype=float)
        Qq = np.zeros_like(xq, dtype=float)
        idx = np.searchsorted(x_sorted, xq, side="right") - 1
        Qq[idx < 0] = 0.0
        valid = idx >= 0
        if np.any(valid):
            idx_clipped = np.clip(idx[valid], 0, len(Q_sorted) - 1)
            Qq[valid] = Q_sorted[idx_clipped]
        return Qq

    def _build_q_curve_from_counts(
        self,
        counts_2d: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        x_query: np.ndarray,
        basis: str = "volume",
    ) -> np.ndarray:
        cdf = self._compute_psd_cdf_from_counts(counts_2d, x_centers, y_centers, basis=basis)
        if cdf is None:
            return np.zeros_like(np.asarray(x_query, dtype=float))
        x_sorted, q_sorted = cdf
        return self._eval_Q_of_x(x_sorted, q_sorted, x_query)

    def _relative_l1(self, values: np.ndarray, reference: np.ndarray) -> float:
        val = np.asarray(values, dtype=float)
        ref = np.asarray(reference, dtype=float)
        return float(np.sum(np.abs(val - ref)) / (np.sum(np.abs(ref)) + MIN))

    def _compute_moment_error_summary(
        self,
        base_result: ValidationResult,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        reference = base_result.methods["Analytical Solution"].moments
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for name, method in base_result.methods.items():
            entries: Dict[str, Dict[str, float]] = {}
            for key, (i, j) in self.MOMENT_KEYS.items():
                rel = np.abs(method.moments[i, j, :] - reference[i, j, :]) / (np.abs(reference[i, j, :]) + MIN)
                entries[key] = {
                    "max_rel_err": float(np.max(rel)),
                    "final_rel_err": float(rel[-1]),
                }
            summary[name] = entries
        return summary

    def _compute_aggregated_error_for_moments(
        self,
        moments: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        terms = []
        for i, j in self.MOMENT_KEYS.values():
            rel = (moments[i, j, :] - reference[i, j, :]) / (reference[i, j, :] + MIN)
            terms.append(float(np.max(rel ** 2)))
        return float(np.sqrt(np.sum(terms)))

    def _compute_aggregated_errors(self, base_result: ValidationResult) -> Dict[str, float]:
        reference = base_result.methods["Analytical Solution"].moments
        aggregated: Dict[str, float] = {}
        for name, method in base_result.methods.items():
            if name == "Analytical Solution":
                aggregated[name] = 0.0
                continue
            override = method.meta.get("aggregated_error_override", np.nan)
            try:
                override_value = float(override)
            except (TypeError, ValueError):
                override_value = np.nan
            if np.isfinite(override_value):
                aggregated[name] = override_value
                continue
            aggregated[name] = self._compute_aggregated_error_for_moments(method.moments, reference)
        return aggregated

    def _collect_aggregated_error_variances(self, base_result: ValidationResult) -> Dict[str, float]:
        variances: Dict[str, float] = {}
        for name, method in base_result.methods.items():
            if name == "Analytical Solution":
                variances[name] = 0.0
            else:
                variances[name] = float(method.meta.get("aggregated_error_variance", np.nan))
        return variances
