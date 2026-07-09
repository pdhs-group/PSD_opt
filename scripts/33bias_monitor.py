"""Bias-monitor validation workflow for the legacy WMCPBE backup solver."""

from __future__ import annotations

import copy
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


from validation import MIN, CaseConfig, WMCPBEVariantConfig
from pbe_core.plotter.plotter_new import PaperPlotter
from wmcpbe_backup import MCPBESolver


@dataclass
class BetaInitialCondition1D:
    alpha: float
    beta: float
    x_min: float
    x_max: float
    n_init: int
    total_number: Optional[float] = None
    volume_concentration: Optional[float] = None


@dataclass
class BiasMethodResult:
    name: str
    moments_mean: np.ndarray
    moments_std: Optional[np.ndarray]
    bias_cum_m2_mean: np.ndarray
    bias_cum_m2_std: Optional[np.ndarray]
    bias_err_pred_m2_mean: np.ndarray
    bias_err_pred_m2_std: Optional[np.ndarray]
    bias_ratio_m2_mean: np.ndarray
    bias_ratio_m2_std: Optional[np.ndarray]
    cpu_time_s: float
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass
class BiasMonitorResult:
    time: np.ndarray
    volumes: np.ndarray
    weights: np.ndarray
    exact_moments: np.ndarray
    methods: Dict[str, BiasMethodResult]


class BiasMonitor1D:
    """Unified bias-monitor workflow for the legacy 1D breakage solver."""

    def __init__(
        self,
        case: CaseConfig,
        init_dist: BetaInitialCondition1D,
        wmcpbe_variants: List[WMCPBEVariantConfig],
        plotter: Optional[PaperPlotter] = None,
        export_dir: Optional[Path] = None,
    ) -> None:
        if case.dim != 1:
            raise ValueError("BiasMonitor1D only supports dim=1.")
        if str(case.process).lower() != "breakage":
            raise ValueError("BiasMonitor1D currently supports process='breakage' only.")
        self.case = case
        self.init_dist = init_dist
        self.wmcpbe_variants = [variant for variant in wmcpbe_variants if variant.enabled]
        if not self.wmcpbe_variants:
            raise ValueError("At least one enabled WMCPBE variant is required.")
        self.plotter = plotter or PaperPlotter()
        self.export_dir = Path(export_dir) if export_dir is not None else Path(__file__).resolve().parent / "exports_bias_monitor"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._export_counts: Dict[str, int] = {}

    def run(self) -> BiasMonitorResult:
        volumes, weights = self._build_beta_initial_condition()
        exact = self._exact_moments_1d_uniform_breakage(self.case.t_vec, volumes, weights, self.case.p1)
        methods: Dict[str, BiasMethodResult] = {}
        for variant in self.wmcpbe_variants:
            methods[variant.name] = self._run_variant(variant, volumes, weights)
        return BiasMonitorResult(
            time=np.asarray(self.case.t_vec, dtype=float),
            volumes=volumes,
            weights=weights,
            exact_moments=exact,
            methods=methods,
        )

    def print_summary(self, result: BiasMonitorResult) -> None:
        rows: List[Dict[str, object]] = []
        print("\nBias monitor summary (backup WMCPBE)")
        print("-" * 80)
        for name, method in result.methods.items():
            err_m0 = method.moments_mean[0, 0, :] - result.exact_moments[0, 0, :]
            err_m1 = method.moments_mean[1, 0, :] - result.exact_moments[1, 0, :]
            err_m2 = method.moments_mean[2, 0, :] - result.exact_moments[2, 0, :]
            max_rel_m2 = float(np.max(np.abs(err_m2) / (np.abs(result.exact_moments[2, 0, :]) + MIN)))
            print(name)
            print(f"  CPU time            = {method.cpu_time_s:.3f} s")
            print(f"  max |error M0|      = {np.max(np.abs(err_m0)):.6e}")
            print(f"  max |error M1|      = {np.max(np.abs(err_m1)):.6e}")
            print(f"  max |error M2|      = {np.max(np.abs(err_m2)):.6e}")
            print(f"  max rel error M2    = {max_rel_m2:.6e}")
            print(f"  max |raw bias M2|   = {np.max(np.abs(method.bias_cum_m2_mean)):.6e}")
            print(f"  max |pred bias M2|  = {np.max(np.abs(method.bias_err_pred_m2_mean)):.6e}")
            print(f"  max |r2|            = {np.nanmax(np.abs(method.bias_ratio_m2_mean)):.6e}")
            print("")
            rows.append(
                {
                    "method": name,
                    "cpu_time_s": method.cpu_time_s,
                    "max_abs_error_M0": float(np.max(np.abs(err_m0))),
                    "max_abs_error_M1": float(np.max(np.abs(err_m1))),
                    "max_abs_error_M2": float(np.max(np.abs(err_m2))),
                    "max_rel_error_M2": max_rel_m2,
                    "max_abs_raw_bias_M2": float(np.max(np.abs(method.bias_cum_m2_mean))),
                    "max_abs_pred_bias_M2": float(np.max(np.abs(method.bias_err_pred_m2_mean))),
                    "max_abs_r2": float(np.nanmax(np.abs(method.bias_ratio_m2_mean))),
                }
            )
        self._write_excel_book(
            method_name="print_summary",
            metadata={
                "workflow": "bias_monitor_1d_breakage",
                "kernel": self.case.kernel,
                "process": self.case.process,
                "lambda": self.case.p1,
                "time_points": len(result.time),
            },
            sheets={"summary": pd.DataFrame(rows)},
        )

    def plot_m2_evolution(self, result: BiasMonitorResult) -> object:
        fig, ax = self.plotter.figure()
        self.plotter.plot_line(
            ax,
            result.time,
            result.exact_moments[2, 0, :],
            key="Analytical Solution",
            family="analytical",
            label="Analytical Solution",
            markevery=max(1, len(result.time) // 8),
        )
        sheets: Dict[str, pd.DataFrame] = {
            "analytical_m2": self._curve_sheet(
                result.time,
                result.exact_moments[2, 0, :],
                label="Analytical Solution",
                x_label="time_s",
                y_label="M2",
            )
        }
        for name, method in result.methods.items():
            self.plotter.plot_line(
                ax,
                result.time,
                method.moments_mean[2, 0, :],
                key=name,
                family="wmcpbe_backup",
                label=name,
                error=None if method.moments_std is None else method.moments_std[2, 0, :],
                markevery=max(1, len(result.time) // 8),
            )
            sheets[self._sheet_name(f"m2_{name}")] = self._curve_sheet(
                result.time,
                method.moments_mean[2, 0, :],
                label=name,
                x_label="time_s",
                y_label="M2",
                std=None if method.moments_std is None else method.moments_std[2, 0, :],
                extra={"family": "wmcpbe_backup"},
            )
        self.plotter.finalize_axes(
            ax,
            xlabel="Time $t$ / s",
            ylabel="M2",
            title="Second moment evolution",
            legend=True,
        )
        self.plotter.tighten(fig)
        self._write_excel_book(
            method_name="plot_m2_evolution",
            metadata={
                "x_label": "Time t / s",
                "y_label": "M2",
                "reference": "Analytical Solution",
            },
            sheets=sheets,
        )
        return fig

    def plot_m2_error_prediction(self, result: BiasMonitorResult) -> object:
        fig, ax = self.plotter.figure(width_scale=1.15)
        sheets: Dict[str, pd.DataFrame] = {}
        for name, method in result.methods.items():
            actual_error = method.moments_mean[2, 0, :] - result.exact_moments[2, 0, :]
            actual_std = None if method.moments_std is None else method.moments_std[2, 0, :]
            self.plotter.plot_line(
                ax,
                result.time,
                actual_error,
                key=f"{name} actual",
                family="wmcpbe_backup",
                label=f"{name} actual error",
                error=actual_std,
                markevery=max(1, len(result.time) // 8),
            )
            self.plotter.plot_line(
                ax,
                result.time,
                method.bias_err_pred_m2_mean,
                key=f"{name} pred",
                family="wmcpbe_backup",
                label=f"{name} propagated prediction",
                error=method.bias_err_pred_m2_std,
                markevery=max(1, len(result.time) // 8),
            )
            self.plotter.plot_line(
                ax,
                result.time,
                method.bias_cum_m2_mean,
                key=f"{name} raw",
                family="wmcpbe_backup",
                label=f"{name} raw cumulative bias",
                error=method.bias_cum_m2_std,
                markevery=max(1, len(result.time) // 8),
            )
            sheets[self._sheet_name(f"actual_error_{name}")] = self._curve_sheet(
                result.time,
                actual_error,
                label=f"{name} actual error",
                x_label="time_s",
                y_label="actual_error_M2",
                std=actual_std,
            )
            sheets[self._sheet_name(f"pred_bias_{name}")] = self._curve_sheet(
                result.time,
                method.bias_err_pred_m2_mean,
                label=f"{name} propagated prediction",
                x_label="time_s",
                y_label="predicted_bias_M2",
                std=method.bias_err_pred_m2_std,
            )
            sheets[self._sheet_name(f"raw_bias_{name}")] = self._curve_sheet(
                result.time,
                method.bias_cum_m2_mean,
                label=f"{name} raw cumulative bias",
                x_label="time_s",
                y_label="raw_cumulative_bias_M2",
                std=method.bias_cum_m2_std,
            )
        self.plotter.finalize_axes(
            ax,
            xlabel="Time $t$ / s",
            ylabel="M2 error / prediction",
            title="Actual M2 error and bias-based predictions",
            legend=True,
        )
        self.plotter.tighten(fig)
        self._write_excel_book(
            method_name="plot_m2_error_prediction",
            metadata={
                "x_label": "Time t / s",
                "y_label": "M2 error / prediction",
                "reference": "Analytical Solution",
            },
            sheets=sheets,
        )
        return fig

    def plot_bias_ratio(self, result: BiasMonitorResult) -> object:
        fig, ax = self.plotter.figure()
        sheets: Dict[str, pd.DataFrame] = {}
        for name, method in result.methods.items():
            self.plotter.plot_line(
                ax,
                result.time,
                method.bias_ratio_m2_mean,
                key=name,
                family="wmcpbe_backup",
                label=name,
                error=method.bias_ratio_m2_std,
                markevery=max(1, len(result.time) // 8),
            )
            sheets[self._sheet_name(f"r2_{name}")] = self._curve_sheet(
                result.time,
                method.bias_ratio_m2_mean,
                label=name,
                x_label="time_s",
                y_label="r2",
                std=method.bias_ratio_m2_std,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        self.plotter.finalize_axes(
            ax,
            xlabel="Time $t$ / s",
            ylabel="$r_2$",
            title="Relative bias ratio over time",
            legend=True,
        )
        self.plotter.tighten(fig)
        self._write_excel_book(
            method_name="plot_bias_ratio",
            metadata={
                "x_label": "Time t / s",
                "y_label": "r2",
            },
            sheets=sheets,
        )
        return fig

    def show(self) -> None:
        self.plotter.show()

    def _build_beta_initial_condition(self) -> tuple[np.ndarray, np.ndarray]:
        x_min = float(self.init_dist.x_min)
        x_max = float(self.init_dist.x_max)
        if not (x_min > 0.0 and x_max > x_min):
            raise ValueError("Beta support bounds must satisfy 0 < x_min < x_max.")
        edges = np.linspace(x_min, x_max, int(self.init_dist.n_init) + 1, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dx = np.diff(edges)

        span = max(x_max - x_min, MIN)
        u = (centers - x_min) / span
        coef = math.gamma(self.init_dist.alpha + self.init_dist.beta)
        coef /= math.gamma(self.init_dist.alpha) * math.gamma(self.init_dist.beta)
        pdf = coef * np.power(np.clip(u, MIN, 1.0 - MIN), self.init_dist.alpha - 1.0)
        pdf *= np.power(np.clip(1.0 - u, MIN, 1.0), self.init_dist.beta - 1.0)
        pdf /= span
        cell_prob = np.maximum(pdf * dx, 0.0)
        cell_prob /= max(float(np.sum(cell_prob)), MIN)

        c_target = self.init_dist.volume_concentration
        n0_target = self.init_dist.total_number
        if c_target is not None and n0_target is not None:
            warnings.warn(
                "Both volume_concentration and total_number are provided; volume_concentration will be used.",
                RuntimeWarning,
            )
        if c_target is not None:
            scale = float(c_target) / max(float(np.sum(cell_prob * centers)), MIN)
        elif n0_target is not None:
            scale = float(n0_target)
        else:
            scale = float(self.case.c) / max(float(np.sum(cell_prob * centers)), MIN)
        weights = cell_prob * scale
        self.case.c = float(np.sum(weights * centers))
        return centers, weights

    def _exact_moments_1d_uniform_breakage(
        self,
        t_vec: np.ndarray,
        volumes: np.ndarray,
        weights: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        mu = np.zeros((3, 1, t_vec.size), dtype=float)
        m0_0 = float(np.sum(weights))
        m1_0 = float(np.sum(weights * volumes))
        m2_0 = float(np.sum(weights * volumes * volumes))
        mu[0, 0, :] = m0_0 * np.exp(lam * t_vec)
        mu[1, 0, :] = m1_0
        mu[2, 0, :] = m2_0 * np.exp(-(lam / 3.0) * t_vec)
        return mu

    def _run_variant(self, variant: WMCPBEVariantConfig, volumes: np.ndarray, weights: np.ndarray) -> BiasMethodResult:
        template = self._build_solver_template(variant)
        seed_sequence = np.random.SeedSequence(variant.base_seed)
        seeds = seed_sequence.spawn(variant.repeats)
        repeat_moments: List[np.ndarray] = []
        repeat_bias_cum: List[np.ndarray] = []
        repeat_bias_pred: List[np.ndarray] = []
        repeat_bias_ratio: List[np.ndarray] = []

        time_start = time.time()
        for seed in seeds:
            solver = self._instantiate_solver(template, volumes, weights, int(seed.generate_state(1)[0]))
            solver.solve(maxiter=variant.maxiter)
            moments, _ = solver.calc_moments_over_time(max_i=2, max_j=0, normalize=True)
            repeat_moments.append(np.asarray(moments, dtype=float))
            repeat_bias_cum.append(np.asarray(solver.bias_cum_M2[: moments.shape[2]], dtype=float))
            repeat_bias_pred.append(np.asarray(solver.bias_err_pred_M2[: moments.shape[2]], dtype=float))
            repeat_bias_ratio.append(np.asarray(solver.bias_ratio_M2[: moments.shape[2]], dtype=float))
        elapsed = time.time() - time_start

        moments_mean = np.mean(repeat_moments, axis=0)
        moments_std = np.std(repeat_moments, axis=0, ddof=1) if variant.repeats > 1 else None
        bias_cum_mean = np.mean(repeat_bias_cum, axis=0)
        bias_cum_std = np.std(repeat_bias_cum, axis=0, ddof=1) if variant.repeats > 1 else None
        bias_pred_mean = np.mean(repeat_bias_pred, axis=0)
        bias_pred_std = np.std(repeat_bias_pred, axis=0, ddof=1) if variant.repeats > 1 else None
        bias_ratio_mean = np.nanmean(repeat_bias_ratio, axis=0)
        bias_ratio_std = np.nanstd(repeat_bias_ratio, axis=0, ddof=1) if variant.repeats > 1 else None

        return BiasMethodResult(
            name=variant.name,
            moments_mean=moments_mean,
            moments_std=moments_std,
            bias_cum_m2_mean=bias_cum_mean,
            bias_cum_m2_std=bias_cum_std,
            bias_err_pred_m2_mean=bias_pred_mean,
            bias_err_pred_m2_std=bias_pred_std,
            bias_ratio_m2_mean=bias_ratio_mean,
            bias_ratio_m2_std=bias_ratio_std,
            cpu_time_s=elapsed,
            attrs=copy.deepcopy(variant.attrs),
        )

    def _build_solver_template(self, variant: WMCPBEVariantConfig) -> MCPBESolver:
        solver = MCPBESolver(
            dim=1,
            t_vec=self.case.t_vec,
            verbose=True,
            load_attr=False,
            init=False,
            seed=variant.base_seed,
        )
        solver.process_type = "breakage"
        solver.BREAKRVAL = 1
        solver.BREAKFVAL = 2
        solver.pl_P1 = float(self.case.p1)
        solver.pl_P2 = 1
        solver.pl_v = 1.0
        solver.pl_q = 1.0
        solver.bias_enable = True
        solver.Vc = 1.0
        solver.a0 = int(variant.attrs.get("a0", 5000))
        solver.recon_enable = bool(variant.attrs.get("recon_enable", True))
        solver.recon_N_max = int(variant.attrs.get("recon_N_max", 2000))
        solver.recon_method = str(variant.attrs.get("recon_method", "2PM"))
        solver.recon_bins = int(variant.attrs.get("recon_bins", 100))
        solver.recon_RS_target = int(variant.attrs.get("recon_RS_target", 1000))
        solver.break_dW_mode = "const"
        delta_w = float(variant.attrs.get("break_dW_max", 100.0))
        solver.break_dW_min = float(variant.attrs.get("break_dW_min", delta_w))
        solver.break_dW_max = delta_w
        for key, value in variant.attrs.items():
            setattr(solver, key, value)
        return solver

    def _instantiate_solver(
        self,
        template: MCPBESolver,
        volumes: np.ndarray,
        weights: np.ndarray,
        seed: int,
    ) -> MCPBESolver:
        solver = copy.deepcopy(template)
        solver.seed = seed
        solver._rng = np.random.default_rng(seed)
        n_init = volumes.size
        v_flat = np.zeros((2, n_init), dtype=float)
        v_flat[0, :] = volumes
        v_flat[1, :] = volumes
        solver._initialize_particles(init_Vc=False, V_flat=v_flat)
        a = int(solver.a_tot)
        solver.W[:a] = np.asarray(weights, dtype=float)
        solver.W0 = solver.W[:a].copy()
        solver.W0_save = [solver.W0.copy()]
        solver.W_save = [solver.W[:a].copy()]
        solver.W_save_left = [solver.W[:a].copy()]
        solver._init_lmc()
        solver._initialize_samplers()
        return solver

    def _next_export_path(self, method_name: str, suffix: str = "") -> Path:
        base = self._slugify(method_name)
        if suffix:
            base = f"{base}_{self._slugify(suffix)}"
        index = self._export_counts.get(base, 0) + 1
        self._export_counts[base] = index
        return self.export_dir / f"{base}_{index:02d}.xlsx"

    def _slugify(self, text: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "export"

    def _sheet_name(self, text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in text)
        cleaned = cleaned.strip() or "Sheet"
        return cleaned[:31]

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
    a0 = 100000
    V_eff_init = 1000
    case = CaseConfig(
        dim=1,
        kernel="const",
        process="breakage",
        t_vec=np.linspace(0.0, 10.0, 101),
        c=1.0,
        p1=0.1,
        p2=1.0,
        x=1e-3,
    )

    init_dist = BetaInitialCondition1D(
        alpha=1.5,
        beta=3.0,
        x_min=2e-3,
        x_max=2e-3*2**20,
        n_init=V_eff_init,
        total_number=a0,
        volume_concentration=None,
    )

    variants = [
        WMCPBEVariantConfig(
            name="WMCPBE (bias)",
            repeats=1,
            attrs={
                "a0": a0,
                "V_eff_init": V_eff_init,
                "break_dW_max": 100.0,
                "recon_enable": True,
                "recon_N_max": 2000,
                "recon_method": "2PM",
                "recon_bins": 400,
            },
        ),
    ]

    monitor = BiasMonitor1D(case=case, init_dist=init_dist, wmcpbe_variants=variants)
    result = monitor.run()
    monitor.print_summary(result)
    monitor.plot_m2_evolution(result)
    monitor.plot_m2_error_prediction(result)
    monitor.plot_bias_ratio(result)
    monitor.show()
