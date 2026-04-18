"""Reconstruction-error monitoring workflow for wmcpbe_recon_debug."""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

from validation import MIN, DPBEVariantConfig, ValidationConfig, WMCPBEVariantConfig  # noqa: E402
from pbe_validation_advance import Dirichlet2DValidationRunner, DirichletInitialCondition  # noqa: E402
from pbe_core.plotter.plotter_new import PaperPlotter  # noqa: E402
from wmcpbe_recon_debug import MCPBESolver  # noqa: E402


@dataclass
class ReconstructionRunData:
    run_id: int
    time: np.ndarray
    iter_count: np.ndarray
    count_before: np.ndarray
    count_after: np.ndarray
    M00_rel_err: np.ndarray
    M01_rel_err: np.ndarray
    M11_rel_err: np.ndarray
    M02_rel_err: np.ndarray
    L1_err: np.ndarray


@dataclass
class ReconstructionMethodResult:
    name: str
    event_index: np.ndarray
    reference_time: np.ndarray
    mean_metrics: Dict[str, np.ndarray]
    std_metrics: Dict[str, Optional[np.ndarray]]
    raw_runs: List[ReconstructionRunData]
    cpu_time_s: float
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass
class ReconstructionMonitorResult:
    methods: Dict[str, ReconstructionMethodResult]


class ReconstructionMonitor:
    METRIC_LABELS = {
        "M00_rel_err": "Relative error of M00 / -",
        "M01_rel_err": "Relative error of M01 / -",
        "M11_rel_err": "Relative error of M11 / -",
        "M02_rel_err": "Relative error of M02 / -",
        "L1_err": "L1 error of N(x, y) / -",
    }

    def __init__(
        self,
        config: ValidationConfig,
        init_dist: DirichletInitialCondition,
        plotter: Optional[PaperPlotter] = None,
        export_dir: Optional[Path] = None,
    ) -> None:
        if config.case.dim != 2:
            raise ValueError("ReconstructionMonitor only supports dim=2.")
        enabled_dpbe = [variant for variant in config.dpbe_variants if variant.enabled]
        if len(enabled_dpbe) != 1:
            raise ValueError("ReconstructionMonitor expects exactly one enabled dPBE variant.")
        self.config = config
        self.init_dist = init_dist
        self.runner = Dirichlet2DValidationRunner(config, init_dist)
        self.plotter = plotter or PaperPlotter()
        self.export_dir = Path(export_dir) if export_dir is not None else Path(__file__).resolve().parent / "exports_reconstruction_monitor"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._export_counts: Dict[str, int] = {}

    def run(self) -> ReconstructionMonitorResult:
        canonical = self.runner.build_canonical_initial_state()
        ref_solver = self.runner._build_reference_dpbe_initialized()
        methods: Dict[str, ReconstructionMethodResult] = {}
        for variant in self.config.wmcpbe_variants:
            if not variant.enabled:
                continue
            weighted_init = int(variant.attrs.get("V_eff_init", 0) or 0) > 0
            vc, v_flat, w_init = self.runner._build_mc_initial_particles(
                ref_solver,
                canonical.n0,
                int(variant.attrs.get("a0", 100000)),
                weighted_init=weighted_init,
            )
            methods[variant.name] = self._run_variant(variant, vc, v_flat, w_init)
        return ReconstructionMonitorResult(methods=methods)

    def print_summary(self, result: ReconstructionMonitorResult) -> None:
        rows: List[Dict[str, object]] = []
        raw_rows: List[Dict[str, object]] = []
        print("\nReconstruction monitor summary")
        print("-" * 88)
        for name, method in result.methods.items():
            print(name)
            print(f"  CPU time         = {method.cpu_time_s:.3f} s")
            print(f"  recon samples    = {sum(run.time.size for run in method.raw_runs)}")
            row: Dict[str, object] = {
                "method": name,
                "cpu_time_s": method.cpu_time_s,
                "reconstruction_samples": int(sum(run.time.size for run in method.raw_runs)),
            }
            for metric in ("M00_rel_err", "M01_rel_err", "M11_rel_err", "M02_rel_err", "L1_err"):
                values = method.mean_metrics[metric]
                max_val = float(np.nanmax(values)) if values.size > 0 else float("nan")
                final_val = float(values[-1]) if values.size > 0 else float("nan")
                print(f"  max {metric:<12} = {max_val:.6e}")
                print(f"  final {metric:<10} = {final_val:.6e}")
                row[f"max_{metric}"] = max_val
                row[f"final_{metric}"] = final_val
            print("")
            rows.append(row)
            for run in method.raw_runs:
                for idx in range(run.time.size):
                    raw_rows.append(
                        {
                            "method": name,
                            "run_id": run.run_id,
                            "time_s": float(run.time[idx]),
                            "iter_count": int(run.iter_count[idx]),
                            "count_before": int(run.count_before[idx]),
                            "count_after": int(run.count_after[idx]),
                            "M00_rel_err": float(run.M00_rel_err[idx]),
                            "M01_rel_err": float(run.M01_rel_err[idx]),
                            "M11_rel_err": float(run.M11_rel_err[idx]),
                            "M02_rel_err": float(run.M02_rel_err[idx]),
                            "L1_err": float(run.L1_err[idx]),
                        }
                    )
        self._write_excel_book(
            method_name="print_summary",
            metadata={
                "workflow": "reconstruction_monitor",
                "kernel": self.config.case.kernel,
                "process": self.config.case.process,
                "x_label": "Reconstruction event index / -",
            },
            sheets={
                "summary": pd.DataFrame(rows),
                "raw_events": pd.DataFrame(raw_rows),
            },
        )

    def plot_moment_errors(self, result: ReconstructionMonitorResult) -> object:
        fig, axes = self.plotter.subplots(2, 2, width_scale=1.6, height_scale=1.2)
        axes = np.asarray(axes).reshape(2, 2)
        metrics = ["M00_rel_err", "M01_rel_err", "M11_rel_err", "M02_rel_err"]
        export_sheets: Dict[str, pd.DataFrame] = {}
        for ax, metric in zip(axes.ravel(), metrics):
            sheet_frames: List[pd.DataFrame] = []
            for name, method in result.methods.items():
                x = method.event_index
                y = method.mean_metrics[metric]
                std = method.std_metrics[metric]
                self.plotter.plot_line(
                    ax,
                    x,
                    y,
                    key=name,
                    family="wmcpbe_recon_debug",
                    label=name,
                    error=std,
                    markevery=max(1, len(x) // 8) if len(x) > 0 else None,
                )
                if x.size > 0:
                    extra = {"reference_time_s": method.reference_time}
                    sheet_frames.append(
                        self._curve_sheet(
                            x,
                            y,
                            label=name,
                            x_label="recon_index",
                            y_label=metric,
                            std=std,
                            extra=extra,
                        )
                    )
            self.plotter.finalize_axes(
                ax,
                xlabel="Reconstruction event index / -",
                ylabel=self.METRIC_LABELS[metric],
                title=metric,
                legend=True,
            )
            export_sheets[metric] = (
                pd.concat(sheet_frames, ignore_index=True)
                if sheet_frames
                else pd.DataFrame([{"note": "No data."}])
            )
        self.plotter.tighten(fig)
        self._write_excel_book(
            method_name="plot_moment_errors",
            metadata={
                "layout": "2x2",
                "metrics": ",".join(metrics),
                "x_label": "Reconstruction event index / -",
                "reference_time_note": "reference_time_s is taken from the first run for the same reconstruction index",
            },
            sheets=export_sheets,
        )
        return fig

    def plot_psd_l1_error(self, result: ReconstructionMonitorResult) -> object:
        fig, ax = self.plotter.figure()
        sheets: Dict[str, pd.DataFrame] = {}
        for name, method in result.methods.items():
            x = method.event_index
            y = method.mean_metrics["L1_err"]
            std = method.std_metrics["L1_err"]
            self.plotter.plot_line(
                ax,
                x,
                y,
                key=name,
                family="wmcpbe_recon_debug",
                label=name,
                error=std,
                markevery=max(1, len(x) // 8) if len(x) > 0 else None,
            )
            extra = {"reference_time_s": method.reference_time}
            sheets[self._sheet_name(name)] = self._curve_sheet(
                x,
                y,
                label=name,
                x_label="recon_index",
                y_label="L1_err",
                std=std,
                extra=extra,
            )
        self.plotter.finalize_axes(
            ax,
            xlabel="Reconstruction event index / -",
            ylabel=self.METRIC_LABELS["L1_err"],
            title="L1 error of reconstructed N(x, y)",
            legend=True,
        )
        self.plotter.tighten(fig)
        self._write_excel_book(
            method_name="plot_psd_l1_error",
            metadata={
                "x_label": "Reconstruction event index / -",
                "y_label": self.METRIC_LABELS["L1_err"],
                "reference_time_note": "reference_time_s is taken from the first run for the same reconstruction index",
            },
            sheets=sheets,
        )
        return fig

    def show(self) -> None:
        self.plotter.show()

    def _run_variant(
        self,
        variant: WMCPBEVariantConfig,
        vc: float,
        v_flat: np.ndarray,
        w_init: Optional[np.ndarray],
    ) -> ReconstructionMethodResult:
        solver_template = MCPBESolver(
            dim=self.config.case.dim,
            t_vec=self.config.case.t_vec,
            verbose=True,
            load_attr=False,
            init=False,
        )
        solver_template.a0 = int(variant.attrs.get("a0", 100000))
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
        solver_template.recon_monitor_enable = True
        solver_template.recon_monitor_psd_bins = 100
        self.runner._apply_case_params(solver_template)
        self.runner._apply_attrs(solver_template, variant.attrs)

        seed_sequence = np.random.SeedSequence(variant.base_seed)
        seeds = seed_sequence.spawn(variant.repeats)
        raw_runs: List[ReconstructionRunData] = []

        time_start = time.time()
        for run_id, seed in enumerate(seeds):
            solver = copy.deepcopy(solver_template)
            solver._rng = np.random.default_rng(seed)
            solver.V_flat = None
            solver.Vc = vc
            solver._initialize_particles(
                init_Vc=False,
                V_flat=v_flat.copy(),
                W_init=None if w_init is None else w_init.copy(),
                init_cdf=None,
            )
            solver._init_lmc()
            solver._initialize_samplers()
            solver.solve(maxiter=variant.maxiter)
            raw_runs.append(
                ReconstructionRunData(
                    run_id=run_id,
                    time=np.asarray(getattr(solver, "recon_monitor_time", []), dtype=float),
                    iter_count=np.asarray(getattr(solver, "recon_monitor_iter", []), dtype=int),
                    count_before=np.asarray(getattr(solver, "recon_monitor_count_before", []), dtype=int),
                    count_after=np.asarray(getattr(solver, "recon_monitor_count_after", []), dtype=int),
                    M00_rel_err=np.asarray(getattr(solver, "recon_monitor_M00_rel_err", []), dtype=float),
                    M01_rel_err=np.asarray(getattr(solver, "recon_monitor_M01_rel_err", []), dtype=float),
                    M11_rel_err=np.asarray(getattr(solver, "recon_monitor_M11_rel_err", []), dtype=float),
                    M02_rel_err=np.asarray(getattr(solver, "recon_monitor_M02_rel_err", []), dtype=float),
                    L1_err=np.asarray(getattr(solver, "recon_monitor_L1_err", []), dtype=float),
                )
            )
        elapsed = time.time() - time_start

        event_index, reference_time = self._build_event_grid(raw_runs)
        mean_metrics: Dict[str, np.ndarray] = {}
        std_metrics: Dict[str, Optional[np.ndarray]] = {}
        for metric in ("M00_rel_err", "M01_rel_err", "M11_rel_err", "M02_rel_err", "L1_err"):
            mean_metrics[metric], std_metrics[metric] = self._aggregate_metric_by_event(
                raw_runs,
                metric,
                event_index.size,
            )

        return ReconstructionMethodResult(
            name=variant.name,
            event_index=event_index,
            reference_time=reference_time,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            raw_runs=raw_runs,
            cpu_time_s=elapsed,
            attrs=copy.deepcopy(variant.attrs),
        )

    def _build_event_grid(self, raw_runs: List[ReconstructionRunData]) -> tuple[np.ndarray, np.ndarray]:
        if not raw_runs:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        ref_time = np.asarray(raw_runs[0].time, dtype=float)
        target_len = int(ref_time.size)
        if target_len <= 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        event_index = np.arange(1, target_len + 1, dtype=int)
        return event_index, ref_time[:target_len].copy()

    def _aggregate_metric_by_event(
        self,
        raw_runs: List[ReconstructionRunData],
        metric: str,
        target_len: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if target_len <= 0:
            return np.asarray([], dtype=float), None
        stacked = np.full((len(raw_runs), target_len), np.nan, dtype=float)
        for idx, run in enumerate(raw_runs):
            y = np.asarray(getattr(run, metric), dtype=float)
            if y.size == 0:
                continue
            n_use = min(target_len, y.size)
            stacked[idx, :n_use] = y[:n_use]
        mean = np.nanmean(stacked, axis=0)
        if len(raw_runs) > 1:
            counts = np.sum(np.isfinite(stacked), axis=0)
            std = np.nanstd(stacked, axis=0, ddof=1)
            std[counts < 2] = np.nan
        else:
            std = None
        return mean, std

    def _slugify(self, text: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "export"

    def _sheet_name(self, text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in text)
        cleaned = cleaned.strip() or "Sheet"
        return cleaned[:31]

    def _next_export_path(self, method_name: str, suffix: str = "") -> Path:
        base = self._slugify(method_name)
        if suffix:
            base = f"{base}_{self._slugify(suffix)}"
        index = self._export_counts.get(base, 0) + 1
        self._export_counts[base] = index
        return self.export_dir / f"{base}_{index:02d}.xlsx"

    def _curve_sheet(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str,
        x_label: str,
        y_label: str,
        std: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        data: Dict[str, object] = {
            x_label: np.asarray(x, dtype=float),
            y_label: np.asarray(y, dtype=float),
            "label": [label] * len(x),
        }
        if std is not None:
            data[f"{y_label}_std"] = np.asarray(std, dtype=float)
        if extra is not None:
            for key, values in extra.items():
                data[key] = np.asarray(values)
        return pd.DataFrame(data)

    def _write_excel_book(
        self,
        method_name: str,
        metadata: Dict[str, object],
        sheets: Dict[str, pd.DataFrame],
        suffix: str = "",
    ) -> Path:
        path = self._next_export_path(method_name, suffix=suffix)
        with pd.ExcelWriter(path) as writer:
            meta_df = pd.DataFrame(
                [{"key": key, "value": value if np.isscalar(value) or value is None else str(value)} for key, value in metadata.items()]
            )
            meta_df.to_excel(writer, sheet_name="metadata", index=False)
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=self._sheet_name(name), index=False)
        print(f"Saved Excel export: {path}")
        return path


if __name__ == "__main__":
    from validation import CaseConfig

    case = CaseConfig(
        dim=2,
        kernel="const",
        process="mix",
        t_vec=np.arange(0.0, 30.0 + 1e-12, 2.0),
        x=2e-3,
        beta0=1e-6,
        p1=1e-1,
        p2=1.0,
        use_psd=False,
    )

    config = ValidationConfig(
        case=case,
        dpbe_variants=[
            DPBEVariantConfig(name="dPBE", grid="geo", ns=50, s=1.5, enabled=False),
        ],
        wmcpbe_variants=[
            WMCPBEVariantConfig(
                name="4PM",
                repeats=20,
                attrs={
                    "a0": 1e5,
                    "V_eff_init": 1000,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "recon_monitor_psd_bins": 100,
                },
            ),
            WMCPBEVariantConfig(
                name="RS",
                repeats=20,
                attrs={
                    "a0": 1e5,
                    "V_eff_init": 1000,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "RS",
                    "recon_RS_target": 1500,
                    "recon_monitor_psd_bins": 100,
                },
            ),
            WMCPBEVariantConfig(
                name="CAM",
                repeats=20,
                attrs={
                    "a0": 1e5,
                    "V_eff_init": 1000,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "CAM",
                    "recon_monitor_psd_bins": 100,
                },
            ),
        ],
        qmom_variants=[],
        reference_dpbe_name="dPBE",
    )

    init_dist = DirichletInitialCondition(
        alpha_x=1.5,
        alpha_y=3.0,
        alpha_rest=3.0,
        x_min_scale=2.0,
        x_max_scale=0.1,
        y_min_scale=2.0,
        y_max_scale=0.1,
        total_number=1e5,
        volume_concentration=None,
    )

    monitor = ReconstructionMonitor(config=config, init_dist=init_dist)
    result = monitor.run()
    monitor.print_summary(result)
    monitor.plot_moment_errors(result)
    monitor.plot_psd_l1_error(result)
    monitor.show()
