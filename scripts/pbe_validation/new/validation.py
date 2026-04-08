"""Modern validation workflow for dPBE, WMCPBE, QMOM, and analytical moments.

This module keeps four responsibilities separate:
1. configuration
2. canonical initial-state construction
3. solver orchestration
4. plotting

The canonical initial state is always built from a reference dPBE initialization.
WMCPBE and QMOM then consume equivalent initial data derived from that reference.
"""

from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _bootstrap_project_paths() -> None:
    """Allow running this script directly from the repository."""
    root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        root,
        root / "dpbe" / "src",
        root / "mcpbe" / "src",
        root / "qmom" / "src",
        root / "pbe-core" / "src",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_project_paths()

from dpbe import DPBESolver, ExtruderPBESolver  # noqa: E402
from qmom import PBMSolver  # noqa: E402
from wmcpbe import MCPBESolver  # noqa: E402


MIN = 1e-40


@dataclass
class CaseConfig:
    dim: int
    kernel: str
    process: str
    t_vec: np.ndarray
    c: float = 1.0
    x: float = 2e-6
    beta0: float = 1e-2
    nc: int = 3
    extruder: bool = False
    use_psd: bool = False
    dist_path: Optional[str] = None
    g: float = 1.0
    v_unit: float = 1.0
    p1: float = 1e-2
    p2: float = 1.0
    new_x: Optional[float] = None

    def __post_init__(self) -> None:
        self.t_vec = np.asarray(self.t_vec, dtype=float)
        if self.use_psd and not self.dist_path:
            raise ValueError("dist_path must be provided when use_psd=True.")


@dataclass
class DPBEVariantConfig:
    name: str
    grid: str
    ns: int
    s: int
    enabled: bool = True
    extra_attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WMCPBEVariantConfig:
    name: str
    repeats: int = 40
    base_seed: int = 42
    maxiter: int = int(1e9)
    enabled: bool = True
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QMOMVariantConfig:
    name: str
    n_order: int = 2
    n_add: int = 10
    enabled: bool = True
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    case: CaseConfig
    dpbe_variants: List[DPBEVariantConfig]
    wmcpbe_variants: List[WMCPBEVariantConfig] = field(default_factory=list)
    qmom_variants: List[QMOMVariantConfig] = field(default_factory=list)
    reference_dpbe_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.dpbe_variants:
            raise ValueError("At least one dPBE variant is required.")
        if self.reference_dpbe_name is None:
            self.reference_dpbe_name = self.dpbe_variants[0].name

    def get_reference_dpbe(self) -> DPBEVariantConfig:
        for variant in self.dpbe_variants:
            if variant.name == self.reference_dpbe_name:
                return variant
        raise ValueError(f"Reference dPBE variant '{self.reference_dpbe_name}' was not found.")


@dataclass
class CanonicalInitialState:
    dim: int
    time: np.ndarray
    n0: float
    v0: float
    reference_initial_moments: np.ndarray


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
    canonical_initial_state: CanonicalInitialState
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def add_method(self, result: MethodResult) -> None:
        self.methods[result.name] = result

    def methods_by_family(self, family: str) -> List[MethodResult]:
        return [item for item in self.methods.values() if item.family == family]


class ValidationRunner:
    """Run all configured solvers from one canonical initial state."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def run(self) -> ValidationResult:
        canonical = self.build_canonical_initial_state()
        result = ValidationResult(
            time=self.config.case.t_vec.copy(),
            dim=self.config.case.dim,
            kernel=self.config.case.kernel,
            process=self.config.case.process,
            canonical_initial_state=canonical,
        )

        analytic = self.compute_analytical_moments(canonical.reference_initial_moments)
        result.add_method(MethodResult(name="Analytical Solution", family="analytical", moments=analytic))

        for variant in self.config.dpbe_variants:
            if not variant.enabled:
                continue
            result.add_method(self.run_dpbe_variant(variant))

        for variant in self.config.wmcpbe_variants:
            if not variant.enabled:
                continue
            result.add_method(self.run_wmcpbe_variant(variant, canonical))

        for variant in self.config.qmom_variants:
            if not variant.enabled:
                continue
            result.add_method(self.run_qmom_variant(variant, canonical))

        return result

    def build_canonical_initial_state(self) -> CanonicalInitialState:
        ref_solver = self._build_reference_dpbe_initialized()

        initial_moments = self._extract_initial_moments(ref_solver)
        n0 = self._compute_initial_number_density(ref_solver)
        v0 = self._compute_reference_v0(ref_solver)

        return CanonicalInitialState(
            dim=self.config.case.dim,
            time=self.config.case.t_vec.copy(),
            n0=n0,
            v0=v0,
            reference_initial_moments=initial_moments,
        )

    def run_dpbe_variant(self, variant: DPBEVariantConfig) -> MethodResult:
        solver = self._build_dpbe_solver(variant)
        self._initialize_dpbe_solver(solver, variant)
        self._apply_case_params(solver)

        time_start = time.time()
        solver.core.calc_F_M()
        solver.core.calc_B_R()
        solver.core.calc_int_B_F()
        solver.core.solve_PBE()
        elapsed = time.time() - time_start

        moments = solver.post.calc_mom_t()
        return MethodResult(
            name=variant.name,
            family="dpbe",
            moments=moments,
            meta={"elapsed_s": elapsed, "grid": variant.grid, "NS": variant.ns, "S": variant.s},
        )

    def run_wmcpbe_variant(
        self,
        variant: WMCPBEVariantConfig,
        canonical: CanonicalInitialState,
    ) -> MethodResult:
        solver = MCPBESolver(
            dim=self.config.case.dim,
            t_vec=self.config.case.t_vec,
            verbose=True,
            load_attr=False,
            init=False,
        )
        solver.a0 = 100000
        solver.CDF_method = "disc"
        solver.G = self.config.case.g
        solver.process_type = self.config.case.process
        solver.alpha_prim = np.ones(self.config.case.dim ** 2)
        solver.break_dW_mode = "const"
        solver.break_dW_min = 1.0
        solver.break_dW_max = 50.0
        solver.agg_dW_min = 1.0
        solver.agg_dW_max = 20.0
        solver.recon_enable = True
        solver.V_eff_init = 1000
        solver.recon_N_max = 4000
        solver.recon_method = "4PMC"
        solver.recon_bins = 30
        solver.recon_RS_target = 1000
        self._apply_case_params(solver)
        self._apply_attrs(solver, variant.attrs)
        ref_solver = self._build_reference_dpbe_initialized()
        mc_vc, mc_v_flat = self._build_mc_initial_particles(
            ref_solver,
            canonical.n0,
            int(solver.a0),
        )

        time_start = time.time()
        results, _ = solver.solve_repeats(
            N=variant.repeats,
            base_seed=variant.base_seed,
            maxiter=variant.maxiter,
            init_Vc=False,
            Vc=mc_vc,
            V_flat=mc_v_flat,
        )
        elapsed = time.time() - time_start

        trajectories = [item["moments"] for item in results]
        moments = np.mean(trajectories, axis=0)
        std = np.std(trajectories, axis=0, ddof=1) if variant.repeats > 1 else None
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

    def run_qmom_variant(
        self,
        variant: QMOMVariantConfig,
        canonical: CanonicalInitialState,
    ) -> MethodResult:
        solver = PBMSolver(self.config.case.dim, t_vec=self.config.case.t_vec, load_attr=False)
        solver.n_order = variant.n_order
        solver.n_add = variant.n_add
        solver.GQMOM = False
        solver.GQMOM_method = "gaussian"
        solver.USE_PSD = self.config.case.use_psd
        solver.process_type = self.config.case.process
        solver.G = self.config.case.g
        solver.alpha_prim = np.ones(self.config.case.dim ** 2)
        solver.V_unit = self.config.case.v_unit
        solver.DIST1 = self.config.case.dist_path
        solver.DIST3 = self.config.case.dist_path
        self._apply_case_params(solver)
        self._apply_attrs(solver, variant.attrs)
        ref_solver = self._build_reference_dpbe_initialized()
        qmom_initial_moments = self._build_qmom_initial_moments(ref_solver, variant.n_order, variant.n_add)
        self._initialize_qmom_from_canonical(solver, qmom_initial_moments)

        time_start = time.time()
        solver.core.solve_PBM()
        elapsed = time.time() - time_start

        moments = np.zeros((3, 3, solver.t_num))
        if solver.dim == 1:
            moments[:, 0, :] = solver.moments[:3, :]
        else:
            for idx, (i, j) in enumerate(solver.indices):
                if i < 3 and j < 3:
                    moments[i, j, :] = solver.moments[idx, :]

        return MethodResult(
            name=variant.name,
            family="qmom",
            moments=moments,
            meta={
                "elapsed_s": elapsed,
                "n_order": variant.n_order,
                "n_add": variant.n_add,
                **copy.deepcopy(variant.attrs),
            },
        )

    def compute_analytical_moments(self, initial_moments: np.ndarray) -> np.ndarray:
        case = self.config.case
        t = case.t_vec
        mu = np.ones((3, 3, len(t)))
        mu0 = initial_moments

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
                    if abs(b0) < 1e-30:
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
                            mu[i, j, :] = mu0[i, j, 0] * np.exp(case.p1 * (2.0 / ((i + 1) * (j + 1)) - 1.0) * t)
                elif case.process == "mix":
                    kappa = float(case.beta0)
                    b0 = float(case.p1)
                    mu[1, 0, :] = mu10_0
                    mu[0, 1, :] = mu01_0
                    if abs(b0) < 1e-30:
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
            else:
                raise NotImplementedError("Only dim=1 and dim=2 are supported.")
        elif case.kernel == "sum":
            if case.dim == 1:
                if case.process == "agglomeration":
                    mu[0, 0, :] = canonical_n0 = float(mu0[0, 0, 0]) * np.exp(-case.beta0 * float(mu0[0, 0, 0]) * t)
                    mu[1, 0, :] = float(mu0[1, 0, 0])
                    phi = 1.0 - np.exp(-case.beta0 * float(mu0[0, 0, 0]) * t)
                    mu[2, 0, :] = mu[1, 0, :] * (
                        self.config.case.x ** 3 / 6.0 + mu[1, 0, :] * (2.0 - phi) * phi / (float(mu0[0, 0, 0]) * (1.0 - phi) ** 2)
                    )
                elif case.process == "breakage":
                    raise NotImplementedError("Analytical solution for 1D sum-kernel breakage is not implemented in the new validator.")
                else:
                    raise NotImplementedError("Analytical solution for this sum-kernel case is not implemented.")
            else:
                raise NotImplementedError("Sum-kernel analytical support is intentionally limited in the new validator.")
        else:
            raise NotImplementedError(f"Unsupported kernel '{case.kernel}'.")

        return mu

    def _build_dpbe_solver(self, variant: DPBEVariantConfig) -> Any:
        case = self.config.case
        if not case.extruder:
            return DPBESolver(dim=case.dim, t_vec=case.t_vec, load_attr=False, f=variant.grid)
        return ExtruderPBESolver(dim=case.dim, NC=case.nc, t_vec=case.t_vec, load_attr=False, disc=variant.grid)

    def _initialize_dpbe_solver(self, solver: Any, variant: DPBEVariantConfig) -> None:
        case = self.config.case
        solver.NS = variant.ns
        solver.S = variant.s
        solver.USE_PSD = case.use_psd
        init_x = case.new_x if case.new_x is not None else case.x
        solver.R01 = init_x / 2.0
        solver.R03 = init_x / 2.0
        solver.DIST1 = case.dist_path
        solver.DIST3 = case.dist_path
        solver.alpha_prim = np.ones(case.dim ** 2)
        solver.G = case.g
        solver.V_unit = case.v_unit
        solver.process_type = case.process
        self._apply_attrs(solver, variant.extra_attrs)
        solver.core.calc_R()
        n0 = 3.0 * case.c / (4.0 * math.pi * (case.x / 2.0) ** 3)
        solver.core.init_N(reset_N=True, N01=n0, N03=n0)

    def _apply_case_params(self, solver: Any) -> None:
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
            solver.CORR_BETA = case.beta0 / max(self._compute_reference_v0_from_case(), MIN)
            solver.BREAKRVAL = 2
            solver.BREAKFVAL = 2
            solver.pl_P1 = case.p1
            solver.pl_P2 = case.p2
            solver.pl_P3 = case.p1
            solver.pl_P4 = case.p2
        else:
            raise NotImplementedError(f"Unsupported kernel '{case.kernel}'.")

    def _apply_attrs(self, solver: Any, attrs: Dict[str, Any]) -> None:
        for key, value in attrs.items():
            setattr(solver, key, value)

    def _compute_initial_number_density(self, solver: Any) -> float:
        if self.config.case.dim == 1:
            return float(np.sum(solver.N[:, 0]) / max(self.config.case.v_unit, MIN))
        return float(np.sum(solver.N[:, :, 0]) / max(self.config.case.v_unit, MIN))

    def _compute_reference_v0(self, solver: Any) -> float:
        if self.config.case.dim == 1:
            return float(solver.V[1])
        return float((solver.V1[1] + solver.V3[1]) / 2.0)

    def _compute_reference_v0_from_case(self) -> float:
        radius = self.config.case.x / 2.0
        return float((4.0 / 3.0) * math.pi * radius ** 3)

    def _build_mc_initial_particles(self, solver: Any, n0: float, a0: int) -> tuple[float, np.ndarray]:
        case = self.config.case
        vc = a0 / max(n0, MIN)
        n_disc = solver.N / max(case.v_unit, MIN)
        a_array = np.round(n_disc[..., 0] * vc).astype(int)
        v_flat = np.zeros((case.dim + 1, int(np.sum(a_array))))

        cnt = 0
        if case.dim == 1:
            for i in range(1, len(solver.V)):
                if a_array[i] <= 0:
                    continue
                v_flat[0, cnt:cnt + a_array[i]] = solver.V[i]
                cnt += a_array[i]
        else:
            for i in range(solver.V.shape[0]):
                for j in range(solver.V.shape[1]):
                    if a_array[i, j] <= 0:
                        continue
                    v_flat[0, cnt:cnt + a_array[i, j]] = solver.V[i, 0]
                    v_flat[1, cnt:cnt + a_array[i, j]] = solver.V[0, j]
                    cnt += a_array[i, j]
        v_flat[-1, :] = np.sum(v_flat[:case.dim, :], axis=0)
        return vc, v_flat

    def _build_qmom_initial_moments(self, solver: Any, n_order: int, n_add: int) -> np.ndarray:
        case = self.config.case
        if case.dim == 1:
            moments = np.zeros((2 * n_order, len(case.t_vec)))
            moments[:, 0] = np.array(
                [np.sum((solver.V ** k) * solver.N[:, 0]) for k in range(2 * n_order)],
                dtype=float,
            )
            return moments

        qmom_probe = PBMSolver(case.dim, t_vec=case.t_vec, load_attr=False)
        qmom_probe.n_order = n_order
        qmom_probe.n_add = n_add
        qmom_probe.moment_2d_indices_c()
        mu_num = len(qmom_probe.indices)
        moments = np.zeros((mu_num, qmom_probe.t_num))
        n0 = solver.N[:, :, 0].copy()
        n0[0, 0] = 0.0
        for idx, (i, j) in enumerate(qmom_probe.indices):
            moments[idx, 0] = np.sum(
                (solver.X1_vol * solver.V) ** i
                * (solver.X3_vol * solver.V) ** j
                * n0
            ) * case.v_unit
        return moments

    def _build_reference_dpbe_initialized(self) -> Any:
        ref_variant = self.config.get_reference_dpbe()
        solver = self._build_dpbe_solver(ref_variant)
        self._initialize_dpbe_solver(solver, ref_variant)
        return solver

    def _initialize_qmom_from_canonical(self, solver: Any, moments: np.ndarray) -> None:
        if solver.dim == 1:
            solver.x_max = 1
            solver.moments = moments.copy()
            solver.normalize_mom()
        else:
            solver.moment_2d_indices_c()
            solver.moments = moments.copy()
        solver.set_tol(solver.moments[:, 0])

    def _extract_initial_moments(self, solver: Any) -> np.ndarray:
        mu = np.zeros((3, 3, len(self.config.case.t_vec)))
        if self.config.case.dim == 1:
            for i in range(3):
                mu[i, 0, 0] = np.sum((solver.V ** i) * solver.N[:, 0])
            return mu

        n0 = solver.N[:, :, 0].copy()
        n0[0, 0] = 0.0
        for i in range(3):
            for j in range(3):
                mu[i, j, 0] = np.sum(
                    (solver.X1_vol * solver.V) ** i
                    * (solver.X3_vol * solver.V) ** j
                    * n0
                ) * self.config.case.v_unit
        return mu


class ValidationPlotter:
    """Plot moments from a ValidationResult without mutating stored data."""

    FAMILY_COLORS = {
        "analytical": "black",
        "dpbe": "#2e8b57",
        "wmcpbe": "#b22222",
        "qmom": "#1f77b4",
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
        time = self.result.time[1:] if skip_initial else self.result.time
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
                time=time,
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
                time=self.result.time,
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
        time: np.ndarray,
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
            time,
            series,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )
        if err is not None:
            ax.fill_between(time, series - err, series + err, color=color, alpha=0.15)


__all__ = [
    "CaseConfig",
    "DPBEVariantConfig",
    "WMCPBEVariantConfig",
    "QMOMVariantConfig",
    "ValidationConfig",
    "ValidationRunner",
    "ValidationResult",
    "ValidationPlotter",
]
