"""Granulation-only validation workflow for WMCPBE variants.

This module is intentionally smaller than scripts/pbe_validation/new/validation.py:
- it runs only wmcpbe_granulation.MCPBESolver variants;
- it initializes every solver from explicit particles + weights;
- it keeps analytical moments and plotting utilities for post-processing.

The example initial-particle builder is deliberately simple. Users can replace
build_example_initial_particles() with their own particle cloud construction.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from wmcpbe_granulation import MCPBESolver  

MIN = 1e-40

@dataclass
class CaseConfig:
    dim: int
    kernel: str
    process: str
    t_vec: np.ndarray
    x: float = 2e-6
    beta0: float = 1e-2
    g: float = 1.0
    p1: float = 1e-2
    p2: float = 1.0
    pl_v: float = 1.0
    pl_q: float = 1.0
    initial_number_density: float = 1.0
    initial_total_weight: float = 100000.0

    def __post_init__(self) -> None:
        self.t_vec = np.asarray(self.t_vec, dtype=float)
        if self.dim not in (1, 2):
            raise NotImplementedError("This granulation validator supports dim=1 or dim=2.")
        if self.process not in ("agglomeration", "breakage", "mix"):
            raise ValueError("process must be 'agglomeration', 'breakage', or 'mix'.")
        if self.kernel not in ("const", "sum"):
            raise ValueError("kernel must be 'const' or 'sum'.")


@dataclass
class WMCPBEVariantConfig:
    name: str
    repeats: int = 5
    base_seed: int = 42
    maxiter: int = int(1e9)
    enabled: bool = True
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GranulationValidationConfig:
    case: CaseConfig
    wmcpbe_variants: List[WMCPBEVariantConfig]
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.wmcpbe_variants:
            raise ValueError("At least one WMCPBE variant is required.")


@dataclass
class InitialParticleState:
    Vc: float
    V_flat: np.ndarray
    W_init: np.ndarray


@dataclass
class MethodResult:
    name: str
    family: str
    moments: np.ndarray
    std: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    time: np.ndarray
    dim: int
    kernel: str
    process: str
    initial_state: InitialParticleState
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def add_method(self, result: MethodResult) -> None:
        self.methods[result.name] = result

    def methods_by_family(self, family: str) -> List[MethodResult]:
        return [item for item in self.methods.values() if item.family == family]


class GranulationValidationRunner:
    """Run several WMCPBE parameter variants from one explicit weighted initial state."""

    def __init__(self, config: GranulationValidationConfig):
        self.config = config

    def run(self) -> ValidationResult:
        initial_state = self.build_example_initial_particles()
        result = ValidationResult(
            time=self.config.case.t_vec.copy(),
            dim=self.config.case.dim,
            kernel=self.config.case.kernel,
            process=self.config.case.process,
            initial_state=initial_state,
        )

        wmcpbe_results: List[MethodResult] = []
        for variant in self.config.wmcpbe_variants:
            if not variant.enabled:
                continue
            wmcpbe_results.append(self.run_wmcpbe_variant(variant, initial_state))

        if not wmcpbe_results:
            raise ValueError("No enabled WMCPBE variants were run.")

        analytic = self.compute_analytical_moments(wmcpbe_results[0].moments)
        result.add_method(
            MethodResult(
                name="Analytical Solution",
                family="analytical",
                moments=analytic,
                meta={"initial_moments_source": wmcpbe_results[0].name},
            )
        )
        for item in wmcpbe_results:
            result.add_method(item)
        return result

    def build_example_initial_particles(self) -> InitialParticleState:
        """Build a minimal weighted initial state from explicit particles.

        1D example:
            three representative particle volumes: v0, 2*v0, 4*v0.

        2D example:
            four representative particles on a tiny component-volume grid.

        The weights are real-particle counts represented by each compute particle.
        The control volume is chosen so that mu00(0) equals
        case.initial_number_density.
        """
        case = self.config.case
        v0 = self._reference_particle_volume()
        total_weight = float(case.initial_total_weight)
        if total_weight <= 0.0:
            raise ValueError("initial_total_weight must be positive.")
        if case.initial_number_density <= 0.0:
            raise ValueError("initial_number_density must be positive.")

        if case.dim == 1:
            component_volumes = np.array([[v0, 2.0 * v0, 4.0 * v0]], dtype=float)
            fractions = np.array([0.60, 0.30, 0.10], dtype=float)
        else:
            component_volumes = np.array(
                [
                    [v0, 2.0 * v0, v0, 2.0 * v0],
                    [v0, v0, 2.0 * v0, 2.0 * v0],
                ],
                dtype=float,
            )
            fractions = np.array([0.40, 0.20, 0.20, 0.20], dtype=float)

        fractions = fractions / float(np.sum(fractions))
        weights = total_weight * fractions
        V_flat = np.zeros((case.dim + 1, component_volumes.shape[1]), dtype=float)
        V_flat[: case.dim, :] = component_volumes
        V_flat[-1, :] = np.sum(component_volumes, axis=0)
        Vc = total_weight / float(case.initial_number_density)
        return InitialParticleState(Vc=Vc, V_flat=V_flat, W_init=weights)

    def run_wmcpbe_variant(
        self,
        variant: WMCPBEVariantConfig,
        initial_state: InitialParticleState,
    ) -> MethodResult:
        solver = self._build_wmcpbe_solver(variant)

        time_start = time.time()
        results, _ = solver.solve_repeats(
            N=variant.repeats,
            base_seed=variant.base_seed,
            maxiter=variant.maxiter,
            init_Vc=False,
            Vc=float(initial_state.Vc),
            V_flat=initial_state.V_flat,
            W_init=initial_state.W_init,
        )
        elapsed = time.time() - time_start

        trajectories: List[np.ndarray] = []
        for item in results:
            tv = np.asarray(item["t_vec"], dtype=float)
            if tv.shape != self.config.case.t_vec.shape or not np.allclose(tv, self.config.case.t_vec):
                raise ValueError(f"WMCPBE variant '{variant.name}' returned a different time grid.")
            trajectories.append(np.asarray(item["moments"], dtype=float))

        if not trajectories:
            raise ValueError(f"WMCPBE variant '{variant.name}' produced no repeat results.")

        stack = np.stack(trajectories, axis=0)
        moments = np.mean(stack, axis=0)
        std = np.std(stack, axis=0, ddof=1) if variant.repeats > 1 else None
        return MethodResult(
            name=variant.name,
            family="wmcpbe",
            moments=moments,
            std=std,
            meta={
                "elapsed_s": elapsed,
                "repeats": variant.repeats,
                "base_seed": variant.base_seed,
                **copy.deepcopy(variant.attrs),
            },
        )

    def compute_analytical_moments(self, wmcpbe_reference_moments: np.ndarray) -> np.ndarray:
        case = self.config.case
        t = case.t_vec
        mu0 = np.asarray(wmcpbe_reference_moments, dtype=float)
        mu = np.zeros((3, 3, len(t)), dtype=float)

        if case.kernel == "const":
            if case.dim == 1:
                mu00_0 = float(mu0[0, 0, 0])
                mu10_0 = float(mu0[1, 0, 0])
                mu20_0 = float(mu0[2, 0, 0])
                if case.process == "agglomeration":
                    mu[0, 0, :] = 2.0 * mu00_0 / (2.0 + case.beta0 * mu00_0 * t)
                    mu[1, 0, :] = mu10_0
                    mu[2, 0, :] = mu20_0 + case.beta0 * (mu10_0 ** 2) * t
                elif case.process == "breakage":
                    for k in range(3):
                        mu[k, 0, :] = mu0[k, 0, 0] * np.exp(case.p1 * (2.0 / (k + 1) - 1.0) * t)
                elif case.process == "mix":
                    kappa = float(case.beta0)
                    b0 = float(case.p1)
                    mu[1, 0, :] = mu10_0
                    if abs(b0) < MIN:
                        mu[0, 0, :] = mu00_0 / (1.0 + 0.5 * kappa * mu00_0 * t)
                        mu[2, 0, :] = mu20_0 + kappa * (mu10_0 ** 2) * t
                    else:
                        e = np.exp(b0 * t)
                        mu[0, 0, :] = (mu00_0 * e) / (1.0 + (0.5 * kappa * mu00_0 / b0) * (e - 1.0))
                        decay = np.exp(-(b0 / 3.0) * t)
                        mu[2, 0, :] = mu20_0 * decay + (3.0 * kappa * (mu10_0 ** 2) / b0) * (1.0 - decay)
                else:
                    raise NotImplementedError(f"Unsupported process '{case.process}'.")
            elif case.dim == 2:
                mu00_0 = float(mu0[0, 0, 0])
                mu10_0 = float(mu0[1, 0, 0])
                mu01_0 = float(mu0[0, 1, 0])
                mu11_0 = float(mu0[1, 1, 0])
                mu20_0 = float(mu0[2, 0, 0])
                mu02_0 = float(mu0[0, 2, 0])
                if case.process == "agglomeration":
                    kappa = float(case.beta0)
                    mu[0, 0, :] = mu00_0 / (1.0 + 0.5 * kappa * mu00_0 * t)
                    mu[1, 0, :] = mu10_0
                    mu[0, 1, :] = mu01_0
                    mu[1, 1, :] = mu11_0 + kappa * mu10_0 * mu01_0 * t
                    mu[2, 0, :] = mu20_0 + kappa * (mu10_0 ** 2) * t
                    mu[0, 2, :] = mu02_0 + kappa * (mu01_0 ** 2) * t
                elif case.process == "breakage":
                    for i in range(3):
                        for j in range(3):
                            mu[i, j, :] = mu0[i, j, 0] * np.exp(
                                case.p1 * (2.0 / ((i + 1) * (j + 1)) - 1.0) * t
                            )
                elif case.process == "mix":
                    kappa = float(case.beta0)
                    b0 = float(case.p1)
                    mu[1, 0, :] = mu10_0
                    mu[0, 1, :] = mu01_0
                    if abs(b0) < MIN:
                        mu[0, 0, :] = mu00_0 / (1.0 + 0.5 * kappa * mu00_0 * t)
                        mu[1, 1, :] = mu11_0 + kappa * mu10_0 * mu01_0 * t
                        mu[2, 0, :] = mu20_0 + kappa * (mu10_0 ** 2) * t
                        mu[0, 2, :] = mu02_0 + kappa * (mu01_0 ** 2) * t
                    else:
                        e = np.exp(b0 * t)
                        mu[0, 0, :] = (mu00_0 * e) / (1.0 + (0.5 * kappa * mu00_0 / b0) * (e - 1.0))
                        decay11 = np.exp(-(b0 / 2.0) * t)
                        mu[1, 1, :] = mu11_0 * decay11 + (2.0 * kappa * mu10_0 * mu01_0 / b0) * (1.0 - decay11)
                        decay2 = np.exp(-(b0 / 3.0) * t)
                        mu[2, 0, :] = mu20_0 * decay2 + (3.0 * kappa * (mu10_0 ** 2) / b0) * (1.0 - decay2)
                        mu[0, 2, :] = mu02_0 * decay2 + (3.0 * kappa * (mu01_0 ** 2) / b0) * (1.0 - decay2)
                else:
                    raise NotImplementedError(f"Unsupported process '{case.process}'.")
        elif case.kernel == "sum":
            if case.dim == 1 and case.process == "agglomeration":
                mu00_0 = float(mu0[0, 0, 0])
                mu10_0 = float(mu0[1, 0, 0])
                mu20_0 = float(mu0[2, 0, 0])
                phi = 1.0 - np.exp(-case.beta0 * mu00_0 * t)
                mu[0, 0, :] = mu00_0 * np.exp(-case.beta0 * mu00_0 * t)
                mu[1, 0, :] = mu10_0
                mu[2, 0, :] = mu10_0 * (
                    self._reference_particle_volume()
                    + mu10_0 * (2.0 - phi) * phi / (mu00_0 * (1.0 - phi) ** 2)
                )
                mu[2, 0, 0] = mu20_0
            else:
                raise NotImplementedError("Sum-kernel analytical support is currently limited to 1D agglomeration.")
        else:
            raise NotImplementedError(f"Unsupported kernel '{case.kernel}'.")

        mu[:, :, 0] = mu0[:, :, 0]
        return mu

    def _build_wmcpbe_solver(self, variant: WMCPBEVariantConfig) -> MCPBESolver:
        case = self.config.case
        solver = MCPBESolver(
            dim=case.dim,
            t_vec=case.t_vec,
            verbose=self.config.verbose,
            load_attr=False,
            init=False,
        )
        solver.process_type = case.process
        solver.G = case.g
        solver.alpha_prim = np.ones(case.dim ** 2, dtype=float)
        solver.pl_v = case.pl_v
        solver.pl_q = case.pl_q
        solver.recon_enable = False
        solver.recon_method = "4PMC"
        solver.recon_N_max = 4000
        solver.recon_bins = 30
        solver.recon_RS_target = 1000
        solver.break_dW_max = 50.0
        solver.agg_dW_min = 1.0
        solver.agg_dW_max = 20.0
        self._apply_case_params(solver)
        self._apply_attrs(solver, variant.attrs)
        return solver

    def _apply_case_params(self, solver: MCPBESolver) -> None:
        case = self.config.case
        if case.kernel == "const":
            solver.COLEVAL = 3
            solver.SIZEEVAL = 1
            solver.CORR_BETA = case.beta0
            solver.BREAKRVAL = 1
            solver.BREAKFVAL = 2
            solver.pl_P1 = case.p1
            solver.pl_P2 = case.p2
            solver.pl_P3 = case.p1
            solver.pl_P4 = case.p2
        elif case.kernel == "sum":
            solver.COLEVAL = 4
            solver.SIZEEVAL = 1
            solver.CORR_BETA = case.beta0 / max(self._reference_particle_volume(), MIN)
            solver.BREAKRVAL = 2
            solver.BREAKFVAL = 2
            solver.pl_P1 = case.p1
            solver.pl_P2 = case.p2
            solver.pl_P3 = case.p1
            solver.pl_P4 = case.p2
        else:
            raise NotImplementedError(f"Unsupported kernel '{case.kernel}'.")

    @staticmethod
    def _apply_attrs(solver: MCPBESolver, attrs: Dict[str, Any]) -> None:
        for key, value in attrs.items():
            setattr(solver, key, value)

    def _reference_particle_volume(self) -> float:
        radius = float(self.config.case.x) / 2.0
        return float((4.0 / 3.0) * math.pi * radius ** 3)


class ValidationPlotter:
    """Plot analytical and WMCPBE moments from a ValidationResult."""

    FAMILY_COLORS = {
        "analytical": "black",
        "wmcpbe": "#b22222",
    }
    MARKERS = ["o", "^", "s", "D", "v", "P", "X"]
    LINESTYLES = ["-", "--", "-.", ":"]

    def __init__(self, result: ValidationResult):
        self.result = result

    def plot_all_moments(self, relative: bool = True, include_total_volume: bool = True) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}
        figures["mu00"] = self.plot_moment(0, 0, relative=relative)
        figures["mu10"] = self.plot_moment(1, 0, relative=relative)
        figures["mu20"] = self.plot_moment(2, 0, relative=relative)
        if self.result.dim == 2:
            figures["mu11"] = self.plot_moment(1, 1, relative=relative, skip_initial=True)
            if include_total_volume:
                figures["total_volume"] = self.plot_total_volume(relative=relative)
        return figures

    def plot_moment(
        self,
        i: int,
        j: int,
        relative: bool = False,
        skip_initial: bool = False,
    ) -> plt.Figure:
        fig, ax = plt.subplots()
        time_vec = self.result.time[1:] if skip_initial else self.result.time
        ylabel = (
            f"Relative Moment $\\mu_{{{i}{j}}} / \\mu_{{{i}{j}}}(0)$ / $-$"
            if relative
            else f"Moment $\\mu_{{{i}{j}}}$"
        )

        family_counts: Dict[str, int] = {}
        for method in self.result.methods.values():
            moments = method.moments[:, :, 1:] if skip_initial else method.moments
            std = method.std[:, :, 1:] if (skip_initial and method.std is not None) else method.std
            family_counts.setdefault(method.family, 0)
            style_idx = family_counts[method.family]
            family_counts[method.family] += 1
            self._plot_series(
                ax=ax,
                time_vec=time_vec,
                values=moments[i, j, :],
                errors=None if std is None else std[i, j, :],
                label=method.name,
                family=method.family,
                style_index=style_idx,
                relative=relative,
            )

        ax.set_xlabel("Time $t$ / s")
        ax.set_ylabel(ylabel)
        if i + j >= 2 and not relative:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_total_volume(self, relative: bool = False) -> plt.Figure:
        if self.result.dim != 2:
            raise ValueError("Total volume plot is only available for dim=2.")

        fig, ax = plt.subplots()
        family_counts: Dict[str, int] = {}
        for method in self.result.methods.values():
            values = method.moments[1, 0, :] + method.moments[0, 1, :]
            errors = None
            if method.std is not None:
                errors = np.sqrt(method.std[1, 0, :] ** 2 + method.std[0, 1, :] ** 2)
            family_counts.setdefault(method.family, 0)
            style_idx = family_counts[method.family]
            family_counts[method.family] += 1
            self._plot_series(
                ax=ax,
                time_vec=self.result.time,
                values=values,
                errors=errors,
                label=method.name,
                family=method.family,
                style_index=style_idx,
                relative=relative,
            )

        ax.set_xlabel("Time $t$ / s")
        ylabel = (
            "Relative Total Particle Volume / $-$"
            if relative
            else "Total Particle Volume $(\\mu_{10} + \\mu_{01})$"
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig

    def show(self) -> None:
        plt.show()

    def _plot_series(
        self,
        ax: plt.Axes,
        time_vec: np.ndarray,
        values: np.ndarray,
        errors: Optional[np.ndarray],
        label: str,
        family: str,
        style_index: int,
        relative: bool,
    ) -> None:
        series = values.astype(float).copy()
        err = None if errors is None else errors.astype(float).copy()
        if relative:
            scale = series[0] + MIN
            series = series / scale
            if err is not None:
                err = err / scale

        color = self.FAMILY_COLORS.get(family, "0.25")
        linestyle = self.LINESTYLES[style_index % len(self.LINESTYLES)]
        marker = self.MARKERS[style_index % len(self.MARKERS)]
        ax.plot(
            time_vec,
            series,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )
        if err is not None:
            ax.fill_between(time_vec, series - err, series + err, color=color, alpha=0.15)


__all__ = [
    "CaseConfig",
    "WMCPBEVariantConfig",
    "GranulationValidationConfig",
    "GranulationValidationRunner",
    "ValidationResult",
    "ValidationPlotter",
]
