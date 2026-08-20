# -*- coding: utf-8 -*-
"""Phase-1 LMC--MCPBE parameter study.

Run this file directly from Spyder.  Edit only the configuration block below.
The study scans MAS, initial X1, STR and gamma while keeping the complete
wmcpbe + aggregate-pool + energy-surrogate chain active.

Parallelism is intentionally at the *condition* level.  The independent
random repeats of one condition are serial inside one worker, so there is no
nested ProcessPoolExecutor and only the parent process writes the HDF5 file.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np


# def _bootstrap_project_paths() -> None:
#     """Make monorepo source packages importable for Spyder and spawned workers."""
#     project_root = Path(__file__).resolve().parents[2]
#     for source_path in (
#         project_root / "pbe-core" / "src",
#         project_root / "lmc" / "src",
#         project_root / "breakage-rate-model" / "src",
#         project_root / "mcpbe" / "src",
#     ):
#         if not source_path.is_dir():
#             raise FileNotFoundError(f"Required project source directory is missing: {source_path}")
#         source_text = str(source_path)
#         if source_text not in sys.path:
#             sys.path.insert(0, source_text)


# _bootstrap_project_paths()

from breakage_rate_model.ann_model import ANNEnergyModel  # noqa: E402
from breakage_rate_model.base import BaseEnergyModel  # noqa: E402
from breakage_rate_model.mlp_model import MLPEnergyModel  # noqa: E402
from breakage_rate_model.parametric_model import ParametricEnergyModel  # noqa: E402
from breakage_rate_model.powerlaw_separable import PowerLawSeparableModel  # noqa: E402
from lmc.agg_pool_npz_sqlite import AggPool  # noqa: E402
from wmcpbe import MCPBESolver  # noqa: E402


# =============================================================================
# Spyder configuration: edit this block for a real study.
# =============================================================================

DATA_ROOT = Path(r"D:\LMC")
AGGREGATE_POOL_ROOT = DATA_ROOT
BREAKAGE_MODEL_KIND = "mlp"  # "mlp", "ann", "powerlaw", or "parametric"
BREAKAGE_MODEL_PATH = DATA_ROOT / f"{BREAKAGE_MODEL_KIND}_model.pkl"

# Old model pickles do not persist this metadata.  Bounds use the model input
# log(V / LMC_A0_RUNTIME), not the raw PBE volume.
MODEL_LOGV_BOUNDS = (float(np.log(100.0)), float(np.log(50_000.0)))
WARN_MODEL_EXTRAPOLATION = True
WARN_POOL_OUT_OF_BOUNDS = True

# Study output.  RESUME=True only accepts a file with the identical numerical
# configuration fingerprint; completed conditions are then skipped.
OUTPUT_DIRECTORY = DATA_ROOT / "pbe_parameter_scan_phase1"
RESULT_FILENAME = "phase1_results.h5"
RESUME = True
N_WORKERS = 1  # Increase to the scheduler allocation; never nest worker pools.
BASE_SEED = 42
N_REPEATS = 3

# Scan dimensions: STR0 and STR2 are phase-swap symmetric, while STR1 is not.
MAS_VALUES = (0.1, 0.5, 0.9)
X1_VALUES = (0.1, 0.5, 0.9)
STR_VALUES = (1.0, 1.0e1, 1.0e2, 1.0e3)
GAMMA_VALUES = (1.0e-3, 1.0, 1.0e3)

# Fixed LMC inputs for phase 1.  Only edit these together with compatible
# aggregate-pool and energy-model assets.
DF = 1.8
NO_FRAG = 2
INT_BRE = 0.0
LMC_A0_RUNTIME = 1.0
SMALL_PARTICLE_POLICY = "disable"
DELTA_CELLS = 0.1

# Initial 2D PBE state.  ``build_initial_state`` supplies V_flat/W_init
# directly, so this is an explicit monodisperse representative-particle
# population rather than the solver's PGV-based initialization path.  Every
# condition's X1 controls the phase partition within each identical particle.
INITIAL_COMPUTE_PARTICLES = 4
INITIAL_WEIGHT_PER_COMPUTE_PARTICLE = 1.0
INITIAL_PARTICLE_VOLUME = 1_000.0
CONTROL_VOLUME = 1.0

# PBE time/process controls.  The default isolates breakage; agglomeration
# inputs remain explicit so process_type can later be changed to "mix".
PROCESS_TYPE = "breakage"  # "breakage", "agglomeration", or "mix"
END_TIME = 10.0
N_TIME_POINTS = 11
MAX_EVENTS = 100_000
VERBOSE_SOLVER = False

# Breakage controls.  The energy surrogate supplies the breakage rate; the
# remaining values stay explicit for compatibility with the solver interface.
BREAKRVAL = 1
BREAKFVAL = 2
PL_V = 2.0
PL_Q = 1.0
PL_P1 = 1.0
PL_P2 = 1.0
PL_P3 = 1.0
PL_P4 = 1.0
BREAK_DW_CONST = 1.0
LAMBDA_E = 1.0
ENERGY_EXPONENT = 1.0
RATE_MIN = 0.0
RATE_MAX: Optional[float] = None

# Agglomeration controls, inactive for PROCESS_TYPE="breakage".
COLEVAL = 3
CORR_BETA = 1.0
SHEAR_RATE = 1.0
ALPHA_PRIM = (1.0, 1.0, 1.0, 1.0)
AGG_DW_CONST = 1.0

# Reconstruction controls.  Keep disabled for the baseline scan.  When
# enabled, wmcpbe evaluates these controls after every simulated event and
# rebuilds its samplers as part of a completed reconstruction.
RECON_ENABLE = False
RECON_METHOD = "RS"  # "CAM", "RS", "2PM", "QMX", "4PM", or "4PMC"
RECON_N_MAX = 4_000
RECON_EVERY_EVENTS = 0
RECON_BINS = 500
RECON_GRID_LOG = True
RECON_TAIL_PROTECT = 100
RECON_COOLDOWN_EVENTS = 50
RECON_RS_TARGET = 2_000
RECON_RS_MIN_PER_CELL = 1
RECON_RS_MAX_PER_CELL = 200
RECON_QMX_Q_SMALL = 0.65
RECON_QMX_Q_TAIL = 0.95
RECON_QMX_SMALL_METHOD = "2PM"
RECON_QMX_MID_METHOD = "RS"
RECON_QMX_TAIL_METHOD = "CAM"  # also accepts "NONE"
RECON_4PM_EPS_W = 1.0e-14
RECON_4PM_COND_MAX = 1.0e12
RECON_4PMC_EPS_VAR = 1.0e-30

# PSD is sampled on this fixed diameter grid.  The runner fails if a completed
# realization has support outside it, rather than silently truncating a PSD.
PSD_X_GRID = np.logspace(-3.0, 4.0, 257)
PSD_BASIS = "volume"  # "volume" or "number"
MOMENT_MAX_ORDER = 2
MASS_CONSERVATION_RTOL = 1e-12
PRINT_EVERY_CONDITION = True
PLOT_CASE_IDS: tuple[str, ...] = ()

# Use None for all 1080 conditions.  This is useful for a manual pilot without
# changing the parameter-space definition or the deterministic case ordering.
CASE_INDICES: Optional[tuple[int, ...]] = [1]


_MODEL_TYPES = {
    "mlp": MLPEnergyModel,
    "ann": ANNEnergyModel,
    "powerlaw": PowerLawSeparableModel,
    "parametric": ParametricEnergyModel,
}
_EVENT_NAMES = ("sim_agg", "sim_break", "real_agg", "real_break")


@dataclass(frozen=True)
class StudyConfig:
    """All numerical, asset and output settings needed by a worker task."""

    data_root: Path
    aggregate_pool_root: Path
    model_kind: str
    model_path: Path
    model_logV_bounds: tuple[float, float]
    warn_model_extrapolation: bool
    warn_pool_out_of_bounds: bool
    output_directory: Path
    result_filename: str
    resume: bool
    n_workers: int
    base_seed: int
    n_repeats: int
    mas_values: tuple[float, ...]
    x1_values: tuple[float, ...]
    str_values: tuple[float, ...]
    gamma_values: tuple[float, ...]
    Df: float
    NO_FRAG: int
    int_bre: float
    lmc_A0_runtime: float
    small_particle_policy: str
    delta_cells: float
    initial_compute_particles: int
    initial_weight_per_compute_particle: float
    initial_particle_volume: float
    control_volume: float
    process_type: str
    end_time: float
    n_time_points: int
    max_events: int
    verbose_solver: bool
    BREAKRVAL: int
    BREAKFVAL: int
    pl_v: float
    pl_q: float
    pl_P1: float
    pl_P2: float
    pl_P3: float
    pl_P4: float
    break_dW_const: float
    lambda_E: float
    energy_exp: float
    rate_min: float
    rate_max: Optional[float]
    COLEVAL: int
    CORR_BETA: float
    G: float
    alpha_prim: tuple[float, ...]
    agg_dW_const: float
    recon_enable: bool
    recon_method: str
    recon_N_max: int
    recon_every_events: int
    recon_bins: int
    recon_grid_log: bool
    recon_tail_protect: int
    recon_cooldown_events: int
    recon_RS_target: int
    recon_RS_min_per_cell: int
    recon_RS_max_per_cell: int
    recon_QMX_q_small: float
    recon_QMX_q_tail: float
    recon_QMX_small_method: str
    recon_QMX_mid_method: str
    recon_QMX_tail_method: str
    recon_4pm_eps_w: float
    recon_4pm_cond_max: float
    recon_4pmc_eps_var: float
    psd_x_grid: tuple[float, ...]
    psd_basis: str
    moment_max_order: int
    mass_conservation_rtol: float
    print_every_condition: bool
    plot_case_ids: tuple[str, ...]
    case_indices: Optional[tuple[int, ...]]

    @property
    def result_path(self) -> Path:
        return self.output_directory / self.result_filename

    @property
    def t_vec(self) -> np.ndarray:
        return np.linspace(0.0, self.end_time, self.n_time_points, dtype=float)

    @property
    def initial_represented_particle_count(self) -> float:
        return float(self.initial_compute_particles) * float(self.initial_weight_per_compute_particle)

    @property
    def initial_represented_total_volume(self) -> float:
        return self.initial_represented_particle_count * float(self.initial_particle_volume)


@dataclass(frozen=True)
class ScanCase:
    """One deterministic point in the phase-1 parameter space."""

    index: int
    case_id: str
    MAS: float
    X1: float
    STR: tuple[float, float, float]
    gamma: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "case_id": self.case_id,
            "MAS": self.MAS,
            "X1": self.X1,
            "STR": list(self.STR),
            "gamma": self.gamma,
        }


@dataclass(frozen=True)
class RefreshRequirements:
    """Explicit dependency information for editing an initialized solver."""

    rebuild_particles: bool
    rebuild_lmc: bool
    rebuild_samplers: bool


def build_default_config() -> StudyConfig:
    """Collect the editable Spyder constants into one serializable object."""
    return StudyConfig(
        data_root=DATA_ROOT,
        aggregate_pool_root=AGGREGATE_POOL_ROOT,
        model_kind=BREAKAGE_MODEL_KIND,
        model_path=BREAKAGE_MODEL_PATH,
        model_logV_bounds=MODEL_LOGV_BOUNDS,
        warn_model_extrapolation=WARN_MODEL_EXTRAPOLATION,
        warn_pool_out_of_bounds=WARN_POOL_OUT_OF_BOUNDS,
        output_directory=OUTPUT_DIRECTORY,
        result_filename=RESULT_FILENAME,
        resume=RESUME,
        n_workers=N_WORKERS,
        base_seed=BASE_SEED,
        n_repeats=N_REPEATS,
        mas_values=MAS_VALUES,
        x1_values=X1_VALUES,
        str_values=STR_VALUES,
        gamma_values=GAMMA_VALUES,
        Df=DF,
        NO_FRAG=NO_FRAG,
        int_bre=INT_BRE,
        lmc_A0_runtime=LMC_A0_RUNTIME,
        small_particle_policy=SMALL_PARTICLE_POLICY,
        delta_cells=DELTA_CELLS,
        initial_compute_particles=INITIAL_COMPUTE_PARTICLES,
        initial_weight_per_compute_particle=INITIAL_WEIGHT_PER_COMPUTE_PARTICLE,
        initial_particle_volume=INITIAL_PARTICLE_VOLUME,
        control_volume=CONTROL_VOLUME,
        process_type=PROCESS_TYPE,
        end_time=END_TIME,
        n_time_points=N_TIME_POINTS,
        max_events=MAX_EVENTS,
        verbose_solver=VERBOSE_SOLVER,
        BREAKRVAL=BREAKRVAL,
        BREAKFVAL=BREAKFVAL,
        pl_v=PL_V,
        pl_q=PL_Q,
        pl_P1=PL_P1,
        pl_P2=PL_P2,
        pl_P3=PL_P3,
        pl_P4=PL_P4,
        break_dW_const=BREAK_DW_CONST,
        lambda_E=LAMBDA_E,
        energy_exp=ENERGY_EXPONENT,
        rate_min=RATE_MIN,
        rate_max=RATE_MAX,
        COLEVAL=COLEVAL,
        CORR_BETA=CORR_BETA,
        G=SHEAR_RATE,
        alpha_prim=ALPHA_PRIM,
        agg_dW_const=AGG_DW_CONST,
        recon_enable=RECON_ENABLE,
        recon_method=RECON_METHOD,
        recon_N_max=RECON_N_MAX,
        recon_every_events=RECON_EVERY_EVENTS,
        recon_bins=RECON_BINS,
        recon_grid_log=RECON_GRID_LOG,
        recon_tail_protect=RECON_TAIL_PROTECT,
        recon_cooldown_events=RECON_COOLDOWN_EVENTS,
        recon_RS_target=RECON_RS_TARGET,
        recon_RS_min_per_cell=RECON_RS_MIN_PER_CELL,
        recon_RS_max_per_cell=RECON_RS_MAX_PER_CELL,
        recon_QMX_q_small=RECON_QMX_Q_SMALL,
        recon_QMX_q_tail=RECON_QMX_Q_TAIL,
        recon_QMX_small_method=RECON_QMX_SMALL_METHOD,
        recon_QMX_mid_method=RECON_QMX_MID_METHOD,
        recon_QMX_tail_method=RECON_QMX_TAIL_METHOD,
        recon_4pm_eps_w=RECON_4PM_EPS_W,
        recon_4pm_cond_max=RECON_4PM_COND_MAX,
        recon_4pmc_eps_var=RECON_4PMC_EPS_VAR,
        psd_x_grid=tuple(float(value) for value in PSD_X_GRID),
        psd_basis=PSD_BASIS,
        moment_max_order=MOMENT_MAX_ORDER,
        mass_conservation_rtol=MASS_CONSERVATION_RTOL,
        print_every_condition=PRINT_EVERY_CONDITION,
        plot_case_ids=PLOT_CASE_IDS,
        case_indices=CASE_INDICES,
    )


DEFAULT_CONFIG = build_default_config()


# =============================================================================
# Parameter module
# =============================================================================


def _number_tag(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def build_scan_cases(config: StudyConfig) -> list[ScanCase]:
    """Return the stable 1080-case phase-1 Cartesian product.

    STR0 and STR2 are retained only in non-decreasing order.  STR1 is an
    independent mixed-bond strength, which matches the 40 STR combinations
    available in the current energy-pool assets.
    """
    cases: list[ScanCase] = []
    index = 0
    for MAS in config.mas_values:
        for X1 in config.x1_values:
            for STR0 in config.str_values:
                for STR1 in config.str_values:
                    for STR2 in config.str_values:
                        if STR0 > STR2:
                            continue
                        for gamma in config.gamma_values:
                            case_id = (
                                f"case_{index:04d}_mas_{_number_tag(MAS)}"
                                f"_x1_{_number_tag(X1)}"
                                f"_str_{_number_tag(STR0)}_{_number_tag(STR1)}_{_number_tag(STR2)}"
                                f"_gamma_{_number_tag(gamma)}"
                            )
                            cases.append(
                                ScanCase(
                                    index=index,
                                    case_id=case_id,
                                    MAS=float(MAS),
                                    X1=float(X1),
                                    STR=(float(STR0), float(STR1), float(STR2)),
                                    gamma=float(gamma),
                                )
                            )
                            index += 1
    return cases


def build_initial_state(config: StudyConfig, X1: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the explicit monodisperse 2D representative-particle state.

    Each column represents ``initial_weight_per_compute_particle`` physical
    particles of total volume ``initial_particle_volume``.  ``X1`` changes
    their shared two-phase composition for the current scan condition.
    """
    phase_1 = config.initial_particle_volume * float(X1)
    phase_2 = config.initial_particle_volume * (1.0 - float(X1))
    V_flat = np.empty((3, config.initial_compute_particles), dtype=float)
    V_flat[0, :] = phase_1
    V_flat[1, :] = phase_2
    V_flat[2, :] = config.initial_particle_volume
    W_init = np.full(
        config.initial_compute_particles,
        config.initial_weight_per_compute_particle,
        dtype=float,
    )
    return V_flat, W_init


def apply_case_to_solver(solver: MCPBESolver, config: StudyConfig, case: ScanCase) -> None:
    """Apply static physics and one scan point before solver initialization."""
    solver.process_type = config.process_type
    # ``init_Vc=False`` below means a0 is informational only; retain the
    # physical represented count for state diagnostics rather than the number
    # of computational columns.
    solver.a0 = config.initial_represented_particle_count
    solver.Vc = config.control_volume
    solver.G = config.G
    solver.COLEVAL = config.COLEVAL
    solver.CORR_BETA = config.CORR_BETA
    solver.alpha_prim = np.asarray(config.alpha_prim, dtype=float)
    solver.agg_dW_const = config.agg_dW_const

    solver.BREAKRVAL = config.BREAKRVAL
    solver.BREAKFVAL = config.BREAKFVAL
    solver.pl_v = config.pl_v
    solver.pl_q = config.pl_q
    solver.pl_P1 = config.pl_P1
    solver.pl_P2 = config.pl_P2
    solver.pl_P3 = config.pl_P3
    solver.pl_P4 = config.pl_P4
    solver.break_dW_const = config.break_dW_const

    solver.recon_enable = config.recon_enable
    solver.recon_method = config.recon_method
    solver.recon_N_max = config.recon_N_max
    solver.recon_every_events = config.recon_every_events
    solver.recon_bins = config.recon_bins
    solver.recon_grid_log = config.recon_grid_log
    solver.recon_tail_protect = config.recon_tail_protect
    solver.recon_cooldown_events = config.recon_cooldown_events
    solver.recon_RS_target = config.recon_RS_target
    solver.recon_RS_min_per_cell = config.recon_RS_min_per_cell
    solver.recon_RS_max_per_cell = config.recon_RS_max_per_cell
    solver.recon_QMX_q_small = config.recon_QMX_q_small
    solver.recon_QMX_q_tail = config.recon_QMX_q_tail
    solver.recon_QMX_small_method = config.recon_QMX_small_method
    solver.recon_QMX_mid_method = config.recon_QMX_mid_method
    solver.recon_QMX_tail_method = config.recon_QMX_tail_method
    solver.recon_4pm_eps_w = config.recon_4pm_eps_w
    solver.recon_4pm_cond_max = config.recon_4pm_cond_max
    solver.recon_4pmc_eps_var = config.recon_4pmc_eps_var

    solver.use_lmc_pre_model = False
    solver.use_lmc_live = True
    solver.lmc_pool_dir = str(config.aggregate_pool_root)
    solver.lmc_A0_runtime = config.lmc_A0_runtime
    solver.lmc_Df = config.Df
    solver.lmc_MAS = case.MAS
    solver.lmc_NO_FRAG = config.NO_FRAG
    solver.lmc_gamma = case.gamma
    solver.lmc_int_bre = config.int_bre
    solver.lmc_STR = np.asarray(case.STR, dtype=float)
    solver.lmc_allow_loops = True
    solver.lmc_accept_all_cracks = False
    solver.lmc_use_weighted_start = False
    solver.lmc_small_particle_policy = config.small_particle_policy
    solver.lmc_delta_cells = config.delta_cells
    solver.lmc_warn_pool_out_of_bounds = config.warn_pool_out_of_bounds

    solver.lmc_use_breakage_model = True
    solver.lmc_breakage_model_kind = config.model_kind
    solver.lmc_breakage_model_path = str(config.model_path)
    solver.lmc_breakage_model_logV_bounds = config.model_logV_bounds
    solver.lmc_warn_model_extrapolation = config.warn_model_extrapolation
    solver.lmc_lambda_E = config.lambda_E
    solver.lmc_energy_exp = config.energy_exp
    solver.lmc_rate_min = config.rate_min
    solver.lmc_rate_max = config.rate_max


def classify_parameter_changes(changed_names: Iterable[str]) -> RefreshRequirements:
    """State the required refresh sequence for a changed initialized solver."""
    changed = set(changed_names)
    particle_inputs = {
        "X1",
        "initial_compute_particles",
        "initial_weight_per_compute_particle",
        "initial_particle_volume",
        "initial_state",
    }
    lmc_inputs = {
        "MAS", "gamma", "STR", "Df", "NO_FRAG", "int_bre", "lmc_pool_dir",
        "lmc_A0_runtime", "model_kind", "model_path", "model_logV_bounds",
        "lambda_E", "energy_exp", "rate_min", "rate_max",
    }
    sampler_inputs = {
        "process_type", "G", "COLEVAL", "CORR_BETA", "alpha_prim",
        "agg_dW_const", "break_dW_const", "BREAKRVAL", "BREAKFVAL",
        "pl_v", "pl_q", "pl_P1", "pl_P2", "pl_P3", "pl_P4",
    }
    rebuild_particles = bool(changed & particle_inputs)
    rebuild_lmc = bool(changed & lmc_inputs)
    rebuild_samplers = rebuild_particles or rebuild_lmc or bool(changed & sampler_inputs)
    return RefreshRequirements(rebuild_particles, rebuild_lmc, rebuild_samplers)


def refresh_initialized_solver(
    solver: MCPBESolver,
    requirements: RefreshRequirements,
    *,
    V_flat: Optional[np.ndarray] = None,
    W_init: Optional[np.ndarray] = None,
) -> None:
    """Refresh only the dependencies declared by ``classify_parameter_changes``.

    The phase-1 runner creates fresh trajectories and therefore does not call
    this helper.  It is included for controlled later reuse in an interactive
    parameter study.
    """
    if requirements.rebuild_particles:
        if V_flat is None or W_init is None:
            raise ValueError("Particle refresh requires explicit V_flat and W_init.")
        solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
    if requirements.rebuild_lmc:
        if solver.lmc_live is not None and solver.lmc_live._sim is not None:
            solver.lmc_live._sim.close()
        solver._init_lmc()
    if requirements.rebuild_samplers:
        solver._initialize_samplers()


def derive_repeat_seeds(config: StudyConfig, case_index: int) -> tuple[int, ...]:
    """Derive order-independent reproducible seeds for one condition."""
    root = np.random.SeedSequence([int(config.base_seed), int(case_index)])
    children = root.spawn(config.n_repeats)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)


# =============================================================================
# Startup module
# =============================================================================


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _config_json(config: StudyConfig, *, include_runtime_controls: bool = True) -> str:
    payload = asdict(config)
    if not include_runtime_controls:
        for key in (
            "output_directory", "result_filename", "resume", "n_workers",
            "print_every_condition", "plot_case_ids", "case_indices",
        ):
            payload.pop(key)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def configuration_fingerprint(config: StudyConfig) -> str:
    """Hash numerical settings and scan definition, excluding runtime scheduling only."""
    return hashlib.sha256(_config_json(config, include_runtime_controls=False).encode("utf-8")).hexdigest()


def _validate_config_values(config: StudyConfig, cases: Sequence[ScanCase]) -> None:
    if config.model_kind not in _MODEL_TYPES:
        raise ValueError(f"model_kind must be one of {tuple(_MODEL_TYPES)}, got {config.model_kind!r}.")
    if config.process_type not in {"breakage", "agglomeration", "mix"}:
        raise ValueError("process_type must be 'breakage', 'agglomeration', or 'mix'.")
    if config.n_workers < 1 or config.n_repeats < 1:
        raise ValueError("n_workers and n_repeats must both be positive.")
    if (
        isinstance(config.initial_compute_particles, bool)
        or not isinstance(config.initial_compute_particles, (int, np.integer))
        or config.initial_compute_particles < 1
    ):
        raise ValueError("initial_compute_particles must be a positive integer.")
    if config.max_events < 1 or config.n_time_points < 2 or config.end_time <= 0.0:
        raise ValueError("max_events, n_time_points and end_time must be positive.")
    if config.moment_max_order < 0:
        raise ValueError("moment_max_order must be non-negative.")
    if config.NO_FRAG < 2 or config.lmc_A0_runtime <= 0.0:
        raise ValueError("NO_FRAG >= 2 and positive lmc_A0_runtime are required.")
    for name, value in (
        ("initial_weight_per_compute_particle", config.initial_weight_per_compute_particle),
        ("initial_particle_volume", config.initial_particle_volume),
        ("control_volume", config.control_volume),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(config.delta_cells) or config.delta_cells < 0.0:
        raise ValueError("delta_cells must be finite and non-negative.")
    alpha_prim = np.asarray(config.alpha_prim, dtype=float)
    if alpha_prim.shape != (4,) or not np.all(np.isfinite(alpha_prim)):
        raise ValueError("alpha_prim must be a finite length-4 array for the 2D solver.")
    if config.small_particle_policy not in {"fallback", "disable"}:
        raise ValueError("small_particle_policy must be 'fallback' or 'disable'.")
    if config.rate_min < 0.0 or (config.rate_max is not None and config.rate_max < config.rate_min):
        raise ValueError("Breakage-rate bounds are inconsistent.")
    if not np.isfinite(config.break_dW_const) or config.break_dW_const <= 0.0:
        raise ValueError("break_dW_const must be finite and positive.")
    if not np.isfinite(config.agg_dW_const) or config.agg_dW_const <= 0.0:
        raise ValueError("agg_dW_const must be finite and positive.")

    if not isinstance(config.recon_enable, (bool, np.bool_)):
        raise ValueError("recon_enable must be a boolean.")
    if not isinstance(config.recon_grid_log, (bool, np.bool_)):
        raise ValueError("recon_grid_log must be a boolean.")

    recon_methods = {"CAM", "RS", "2PM", "QMX", "4PM", "4PMC"}
    recon_kernel_methods = {"CAM", "RS", "2PM", "4PM", "4PMC"}
    if str(config.recon_method).upper() not in recon_methods:
        raise ValueError(f"recon_method must be one of {tuple(sorted(recon_methods))}.")
    if str(config.recon_QMX_small_method).upper() not in recon_kernel_methods:
        raise ValueError("recon_QMX_small_method must select a reconstruction kernel.")
    if str(config.recon_QMX_mid_method).upper() not in recon_kernel_methods:
        raise ValueError("recon_QMX_mid_method must select a reconstruction kernel.")
    if str(config.recon_QMX_tail_method).upper() not in (recon_kernel_methods | {"NONE"}):
        raise ValueError("recon_QMX_tail_method must select a reconstruction kernel or 'NONE'.")
    for name, value, minimum in (
        ("recon_N_max", config.recon_N_max, 0),
        ("recon_every_events", config.recon_every_events, 0),
        ("recon_bins", config.recon_bins, 1),
        ("recon_tail_protect", config.recon_tail_protect, 0),
        ("recon_cooldown_events", config.recon_cooldown_events, 0),
        ("recon_RS_target", config.recon_RS_target, 1),
        ("recon_RS_min_per_cell", config.recon_RS_min_per_cell, 0),
        ("recon_RS_max_per_cell", config.recon_RS_max_per_cell, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if config.recon_RS_max_per_cell < config.recon_RS_min_per_cell:
        raise ValueError("recon_RS_max_per_cell must be >= recon_RS_min_per_cell.")
    q_small = float(config.recon_QMX_q_small)
    q_tail = float(config.recon_QMX_q_tail)
    if not np.isfinite(q_small) or not np.isfinite(q_tail) or not (0.0 <= q_small <= q_tail <= 1.0):
        raise ValueError("recon_QMX quantiles must satisfy 0 <= q_small <= q_tail <= 1.")
    for name, value, allow_zero in (
        ("recon_4pm_eps_w", config.recon_4pm_eps_w, True),
        ("recon_4pm_cond_max", config.recon_4pm_cond_max, False),
        ("recon_4pmc_eps_var", config.recon_4pmc_eps_var, False),
    ):
        if not np.isfinite(value) or value < 0.0 or (not allow_zero and value <= 0.0):
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {relation}.")

    logV_bounds = np.asarray(config.model_logV_bounds, dtype=float)
    if logV_bounds.shape != (2,) or not np.all(np.isfinite(logV_bounds)) or logV_bounds[0] >= logV_bounds[1]:
        raise ValueError("model_logV_bounds must be two finite increasing log(V/A0) values.")
    initial_logV = math.log(config.initial_particle_volume / config.lmc_A0_runtime)
    if initial_logV < logV_bounds[0] or initial_logV > logV_bounds[1]:
        raise ValueError(
            "The initial normalized particle volume lies outside model_logV_bounds: "
            f"log(V/A0)={initial_logV:.6g}, bounds={tuple(logV_bounds)}."
        )

    psd_grid = np.asarray(config.psd_x_grid, dtype=float)
    if psd_grid.ndim != 1 or psd_grid.size < 2 or not np.all(np.isfinite(psd_grid)):
        raise ValueError("psd_x_grid must be a finite one-dimensional grid with at least two values.")
    if np.any(psd_grid <= 0.0) or np.any(np.diff(psd_grid) <= 0.0):
        raise ValueError("psd_x_grid must be strictly increasing and positive.")
    if config.psd_basis not in {"volume", "number"}:
        raise ValueError("psd_basis must be 'volume' or 'number'.")

    if len(cases) != 1080:
        raise RuntimeError(f"Phase-1 scan must contain 1080 cases, got {len(cases)}.")
    if any(case.STR[0] > case.STR[2] for case in cases):
        raise RuntimeError("STR symmetry enumeration contains a forbidden STR0 > STR2 case.")
    if any(not (0.0 < case.X1 < 1.0) for case in cases):
        raise ValueError("All scanned X1 values must lie strictly between zero and one.")
    if any(case.gamma <= 0.0 for case in cases):
        raise ValueError("All scanned gamma values must be positive.")
    if any(np.any(np.asarray(case.STR) <= 0.0) for case in cases):
        raise ValueError("All scanned STR values must be positive.")

    if config.case_indices is not None:
        invalid = [index for index in config.case_indices if index < 0 or index >= len(cases)]
        if invalid:
            raise IndexError(f"CASE_INDICES contains out-of-range values: {invalid}")


def _validate_assets(config: StudyConfig) -> None:
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Selected energy model does not exist: {config.model_path}")
    if not config.aggregate_pool_root.is_dir():
        raise FileNotFoundError(f"Aggregate-pool root does not exist: {config.aggregate_pool_root}")

    model = BaseEnergyModel.load(str(config.model_path), device="cpu")
    expected_type = _MODEL_TYPES[config.model_kind]
    if not isinstance(model, expected_type):
        raise TypeError(
            f"model_kind={config.model_kind!r} requires {expected_type.__name__}, "
            f"got {type(model).__name__}."
        )
    if not model.is_fitted:
        raise RuntimeError("The selected energy model is not fitted.")

    normalized_volume = config.initial_particle_volume / config.lmc_A0_runtime
    pool = AggPool(str(config.aggregate_pool_root), warn_out_of_bounds=False)
    try:
        for MAS in config.mas_values:
            cache = pool._get_cache(config.Df, float(MAS))
            A_min, A_max = float(cache["Np_vals"][0]), float(cache["Np_vals"][-1])
            X_min, X_max = float(cache["XA_vals"][0]), float(cache["XA_vals"][-1])
            if normalized_volume < A_min or normalized_volume > A_max:
                raise ValueError(
                    f"Initial A/A0={normalized_volume} is outside pool coverage "
                    f"[{A_min}, {A_max}] for Df={config.Df}, MAS={MAS}."
                )
            if min(config.x1_values) < X_min or max(config.x1_values) > X_max:
                raise ValueError(
                    f"Scanned X1 range [{min(config.x1_values)}, {max(config.x1_values)}] is outside "
                    f"pool coverage [{X_min}, {X_max}] for Df={config.Df}, MAS={MAS}."
                )
    finally:
        pool.close_pool_cache()


def validate_startup(config: StudyConfig, cases: Sequence[ScanCase]) -> None:
    """Perform all expensive, static validation exactly once in the parent."""
    _validate_config_values(config, cases)
    _validate_assets(config)


def _decode_h5_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def prepare_result_file(config: StudyConfig, cases: Sequence[ScanCase]) -> list[ScanCase]:
    """Create/validate the resumable result file and return incomplete cases."""
    config.output_directory.mkdir(parents=True, exist_ok=True)
    result_path = config.result_path
    fingerprint = configuration_fingerprint(config)

    if result_path.exists() and not config.resume:
        raise FileExistsError(
            f"Result file already exists and RESUME=False: {result_path}. "
            "Choose a new output directory or enable strict resume."
        )

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(result_path, "a") as h5_file:
        if "config_json" not in h5_file:
            h5_file.attrs["format_version"] = 2
            h5_file.attrs["config_fingerprint"] = fingerprint
            h5_file.attrs["initial_state_kind"] = "monodisperse_2d_explicit"
            h5_file.attrs["initial_compute_particles"] = config.initial_compute_particles
            h5_file.attrs["initial_weight_per_compute_particle"] = config.initial_weight_per_compute_particle
            h5_file.attrs["initial_represented_particle_count"] = config.initial_represented_particle_count
            h5_file.attrs["initial_particle_volume"] = config.initial_particle_volume
            h5_file.attrs["initial_represented_total_volume"] = config.initial_represented_total_volume
            h5_file.create_dataset(
                "config_json",
                data=_config_json(config, include_runtime_controls=False),
                dtype=string_dtype,
            )
            h5_file.create_dataset(
                "initial_runtime_config_json",
                data=_config_json(config, include_runtime_controls=True),
                dtype=string_dtype,
            )
            h5_file.create_dataset("t_vec", data=config.t_vec)
            h5_file.create_dataset("psd_x_grid", data=np.asarray(config.psd_x_grid, dtype=float))
            case_json = np.asarray(
                [json.dumps(case.as_dict(), sort_keys=True) for case in cases],
                dtype=object,
            )
            h5_file.create_dataset("case_table_json", data=case_json, dtype=string_dtype)
            h5_file.require_group("conditions")
        else:
            stored_fingerprint = _decode_h5_string(h5_file.attrs["config_fingerprint"])
            if stored_fingerprint != fingerprint:
                raise ValueError(
                    "Existing result file has a different configuration fingerprint. "
                    "Refusing to combine non-comparable scans."
                )
            stored_cases = [_decode_h5_string(value) for value in h5_file["case_table_json"][...]]
            expected_cases = [json.dumps(case.as_dict(), sort_keys=True) for case in cases]
            if stored_cases != expected_cases:
                raise ValueError("Existing result file has a different ordered case table.")

        selected_indices = set(config.case_indices) if config.case_indices is not None else None
        conditions = h5_file["conditions"]
        pending: list[ScanCase] = []
        for case in cases:
            if selected_indices is not None and case.index not in selected_indices:
                continue
            if case.case_id in conditions and bool(conditions[case.case_id].attrs.get("complete", False)):
                continue
            pending.append(case)
    return pending


# =============================================================================
# Solver module
# =============================================================================


def _close_live_lmc(solver: MCPBESolver) -> None:
    live_adapter = getattr(solver, "lmc_live", None)
    if live_adapter is not None and live_adapter._sim is not None:
        live_adapter._sim.close()


def _weighted_phase_volume(V_snap: np.ndarray, W_snap: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(V_snap[:2], dtype=float) * np.asarray(W_snap, dtype=float)[None, :], axis=1)


def _extract_repeat_result(solver: MCPBESolver, config: StudyConfig) -> dict[str, Any]:
    """Collect fixed-shape numerical outputs from one completed trajectory."""
    moments, t_moments = solver.calc_moments_over_time(
        max_i=config.moment_max_order,
        max_j=config.moment_max_order,
        normalize=True,
    )
    expected_t = config.t_vec
    if moments.shape[-1] != expected_t.size or not np.allclose(t_moments, expected_t, rtol=0.0, atol=0.0):
        raise RuntimeError("Solver did not produce snapshots aligned with the configured t_vec.")

    cdf_list, t_psd = solver.compute_psd_cdf_over_time(
        psd_basis=config.psd_basis,
        time_scheme="interp",
    )
    if len(cdf_list) != expected_t.size or not np.allclose(t_psd, expected_t, rtol=0.0, atol=0.0):
        raise RuntimeError("PSD snapshots are not aligned with the configured t_vec.")

    psd_grid = np.asarray(config.psd_x_grid, dtype=float)
    psd_Q = np.empty((expected_t.size, psd_grid.size), dtype=float)
    x50 = np.empty(expected_t.size, dtype=float)
    support = np.empty((expected_t.size, 2), dtype=float)
    for time_index, cdf in enumerate(cdf_list):
        if cdf is None:
            raise RuntimeError(f"No PSD CDF is available at time index {time_index}.")
        x_sorted, Q_sorted = cdf
        support[time_index] = (float(x_sorted[0]), float(x_sorted[-1]))
        if support[time_index, 0] < psd_grid[0] or support[time_index, 1] > psd_grid[-1]:
            raise ValueError(
                f"PSD support {tuple(support[time_index])} lies outside configured psd_x_grid "
                f"[{psd_grid[0]}, {psd_grid[-1]}] at time index {time_index}."
            )
        psd_Q[time_index] = solver._eval_Q_of_x(x_sorted, Q_sorted, psd_grid)
        x50[time_index] = solver._invert_cdf_monotone(x_sorted, Q_sorted, q=0.5)

    event_columns = (
        solver.sim_agg_events_save,
        solver.sim_break_events_save,
        solver.real_agg_events_save,
        solver.real_break_events_save,
    )
    events = np.column_stack([np.asarray(values, dtype=float) for values in event_columns])
    if events.shape != (expected_t.size, len(_EVENT_NAMES)):
        raise RuntimeError(f"Unexpected event-statistics shape {events.shape}.")

    initial_phase_volume = _weighted_phase_volume(solver.V0, solver.W0)
    final_phase_volume = _weighted_phase_volume(solver.V_save[-1], solver.W_save[-1])
    if not np.allclose(final_phase_volume, initial_phase_volume, rtol=config.mass_conservation_rtol, atol=0.0):
        raise RuntimeError(
            "Phase-wise represented volume is not conserved: "
            f"initial={initial_phase_volume}, final={final_phase_volume}."
        )
    relative_phase_error = np.max(np.abs(final_phase_volume - initial_phase_volume) / initial_phase_volume)

    return {
        "moments": np.asarray(moments, dtype=float),
        "psd_Q": psd_Q,
        "x50": x50,
        "psd_support": support,
        "events": events,
        "initial_phase_volume": initial_phase_volume,
        "final_phase_volume": final_phase_volume,
        "phase_volume_relative_error": float(relative_phase_error),
        "reconstruction_count": int(solver._recon_count),
        "machine_seconds": float(solver.MACHINE_TIME),
    }


def run_one_repeat(config: StudyConfig, case: ScanCase, seed: int) -> dict[str, Any]:
    """Run one fresh trajectory; reinitialization here is physically required."""
    V_flat, W_init = build_initial_state(config, case.X1)
    solver = MCPBESolver(
        dim=2,
        t_vec=config.t_vec,
        verbose=config.verbose_solver,
        load_attr=False,
        init=False,
        seed=int(seed),
    )
    try:
        apply_case_to_solver(solver, config, case)
        solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
        solver._init_lmc()
        solver._initialize_samplers()
        solver.solve(maxiter=config.max_events)
        result = _extract_repeat_result(solver, config)
        result["seed"] = int(seed)
        return result
    finally:
        _close_live_lmc(solver)


def run_condition_task(config: StudyConfig, case: ScanCase) -> dict[str, Any]:
    """Run all serial random repeats for one condition in one process worker."""
    records = [run_one_repeat(config, case, seed) for seed in derive_repeat_seeds(config, case.index)]
    stacked: dict[str, Any] = {"case": case.as_dict()}
    for name in (
        "moments", "psd_Q", "x50", "psd_support", "events",
        "initial_phase_volume", "final_phase_volume", "phase_volume_relative_error",
        "reconstruction_count", "machine_seconds", "seed",
    ):
        stacked[name] = np.asarray([record[name] for record in records])
    return stacked


# =============================================================================
# Post-processing module
# =============================================================================


def _sample_std(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1:], dtype=float)


def _replace_dataset(group: h5py.Group, name: str, value: np.ndarray) -> None:
    if name in group:
        del group[name]
    array = np.asarray(value)
    kwargs: dict[str, Any] = {}
    if array.ndim > 0 and array.size > 1:
        kwargs["compression"] = "gzip"
        kwargs["shuffle"] = True
    group.create_dataset(name, data=array, **kwargs)


def write_condition_result(h5_file: h5py.File, result: Mapping[str, Any], config: StudyConfig) -> None:
    """Write one fully completed condition from the parent process only."""
    case_data = result["case"]
    case_id = str(case_data["case_id"])
    conditions = h5_file["conditions"]
    if case_id in conditions:
        del conditions[case_id]
    group = conditions.create_group(case_id)
    group.attrs["complete"] = False
    group.attrs["index"] = int(case_data["index"])
    group.attrs["MAS"] = float(case_data["MAS"])
    group.attrs["X1"] = float(case_data["X1"])
    group.attrs["STR"] = np.asarray(case_data["STR"], dtype=float)
    group.attrs["gamma"] = float(case_data["gamma"])
    group.attrs["n_repeats"] = int(config.n_repeats)

    for name in (
        "seed", "moments", "psd_Q", "x50", "psd_support", "events",
        "initial_phase_volume", "final_phase_volume", "phase_volume_relative_error",
        "reconstruction_count", "machine_seconds",
    ):
        _replace_dataset(group, name, np.asarray(result[name]))

    for source_name, output_prefix in (
        ("moments", "moments"),
        ("psd_Q", "psd_Q"),
        ("x50", "x50"),
        ("events", "events"),
        ("reconstruction_count", "reconstruction_count"),
    ):
        values = np.asarray(result[source_name], dtype=float)
        _replace_dataset(group, f"{output_prefix}_mean", np.mean(values, axis=0))
        _replace_dataset(group, f"{output_prefix}_std", _sample_std(values))

    group.attrs["complete"] = True
    h5_file.flush()


def _case_attributes(group: h5py.Group) -> dict[str, Any]:
    return {
        "case_id": group.name.rsplit("/", 1)[-1],
        "case_index": int(group.attrs["index"]),
        "MAS": float(group.attrs["MAS"]),
        "X1": float(group.attrs["X1"]),
        "STR0": float(group.attrs["STR"][0]),
        "STR1": float(group.attrs["STR"][1]),
        "STR2": float(group.attrs["STR"][2]),
        "gamma": float(group.attrs["gamma"]),
    }


def export_csv_summaries(result_path: Path) -> tuple[Path, Path]:
    """Rebuild user-facing CSV summaries from completed HDF5 condition groups."""
    condition_csv = result_path.with_name("condition_summary.csv")
    time_csv = result_path.with_name("time_summary.csv")
    with h5py.File(result_path, "r") as h5_file:
        t_vec = np.asarray(h5_file["t_vec"], dtype=float)
        groups = [
            group for _, group in sorted(h5_file["conditions"].items())
            if bool(group.attrs.get("complete", False))
        ]

        moment_order = int(next(iter(groups))["moments_mean"].shape[0] - 1) if groups else 0
        moment_names = [f"mu_{i}_{j}" for i in range(moment_order + 1) for j in range(moment_order + 1)]
        common_fields = ["case_id", "case_index", "MAS", "X1", "STR0", "STR1", "STR2", "gamma"]
        event_fields = [f"{name}_{stat}" for name in _EVENT_NAMES for stat in ("mean", "std")]
        moment_fields = [f"{name}_{stat}" for name in moment_names for stat in ("mean", "std")]

        with condition_csv.open("w", newline="", encoding="utf-8") as handle:
            fields = common_fields + [
                "n_repeats", "machine_seconds_mean", "machine_seconds_std",
                "reconstruction_count_mean", "reconstruction_count_std",
                "phase_volume_relative_error_max", "x50_final_mean", "x50_final_std",
            ] + event_fields + moment_fields
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for group in groups:
                row = _case_attributes(group)
                row["n_repeats"] = int(group.attrs["n_repeats"])
                machine = np.asarray(group["machine_seconds"], dtype=float)
                phase_error = np.asarray(group["phase_volume_relative_error"], dtype=float)
                row["machine_seconds_mean"] = float(np.mean(machine))
                row["machine_seconds_std"] = float(_sample_std(machine[:, None])[0])
                reconstruction_count = np.asarray(group["reconstruction_count"], dtype=float)
                row["reconstruction_count_mean"] = float(np.mean(reconstruction_count))
                row["reconstruction_count_std"] = float(_sample_std(reconstruction_count[:, None])[0])
                row["phase_volume_relative_error_max"] = float(np.max(phase_error))
                x50_mean = np.asarray(group["x50_mean"], dtype=float)
                x50_std = np.asarray(group["x50_std"], dtype=float)
                row["x50_final_mean"] = float(x50_mean[-1])
                row["x50_final_std"] = float(x50_std[-1])
                events_mean = np.asarray(group["events_mean"], dtype=float)[-1]
                events_std = np.asarray(group["events_std"], dtype=float)[-1]
                for event_index, event_name in enumerate(_EVENT_NAMES):
                    row[f"{event_name}_mean"] = float(events_mean[event_index])
                    row[f"{event_name}_std"] = float(events_std[event_index])
                moments_mean = np.asarray(group["moments_mean"], dtype=float)
                moments_std = np.asarray(group["moments_std"], dtype=float)
                for i in range(moment_order + 1):
                    for j in range(moment_order + 1):
                        row[f"mu_{i}_{j}_mean"] = float(moments_mean[i, j, -1])
                        row[f"mu_{i}_{j}_std"] = float(moments_std[i, j, -1])
                writer.writerow(row)

        with time_csv.open("w", newline="", encoding="utf-8") as handle:
            fields = common_fields + ["time", "x50_mean", "x50_std"] + event_fields + moment_fields
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for group in groups:
                common = _case_attributes(group)
                x50_mean = np.asarray(group["x50_mean"], dtype=float)
                x50_std = np.asarray(group["x50_std"], dtype=float)
                events_mean = np.asarray(group["events_mean"], dtype=float)
                events_std = np.asarray(group["events_std"], dtype=float)
                moments_mean = np.asarray(group["moments_mean"], dtype=float)
                moments_std = np.asarray(group["moments_std"], dtype=float)
                for time_index, time_value in enumerate(t_vec):
                    row = dict(common)
                    row["time"] = float(time_value)
                    row["x50_mean"] = float(x50_mean[time_index])
                    row["x50_std"] = float(x50_std[time_index])
                    for event_index, event_name in enumerate(_EVENT_NAMES):
                        row[f"{event_name}_mean"] = float(events_mean[time_index, event_index])
                        row[f"{event_name}_std"] = float(events_std[time_index, event_index])
                    for i in range(moment_order + 1):
                        for j in range(moment_order + 1):
                            row[f"mu_{i}_{j}_mean"] = float(moments_mean[i, j, time_index])
                            row[f"mu_{i}_{j}_std"] = float(moments_std[i, j, time_index])
                    writer.writerow(row)
    return condition_csv, time_csv


def print_condition_summary(result: Mapping[str, Any], completed: int, total: int) -> None:
    """Print one concise, non-graphical research progress line."""
    case = result["case"]
    x50 = np.asarray(result["x50"], dtype=float)
    events = np.asarray(result["events"], dtype=float)
    moments = np.asarray(result["moments"], dtype=float)
    x50_final = float(np.mean(x50[:, -1]))
    break_final = float(np.mean(events[:, -1, 1]))
    mu00_final = float(np.mean(moments[:, 0, 0, -1]))
    reconstruction_mean = float(np.mean(np.asarray(result["reconstruction_count"], dtype=float)))
    reconstruction_std = float(_sample_std(np.asarray(result["reconstruction_count"], dtype=float)[:, None])[0])
    print(
        f"[{completed:4d}/{total:4d}] {case['case_id']} | "
        f"x50_final={x50_final:.6g}, mu00_final={mu00_final:.6g}, "
        f"sim_break_final={break_final:.6g}, recon={reconstruction_mean:.6g}\u00b1{reconstruction_std:.6g}"
    )


def plot_selected_cases(result_path: Path, case_ids: Sequence[str]) -> None:
    """Write PSD and selected moment plots only for explicitly named cases."""
    if not case_ids:
        return
    import matplotlib.pyplot as plt

    plot_dir = result_path.with_name("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(result_path, "r") as h5_file:
        t_vec = np.asarray(h5_file["t_vec"], dtype=float)
        psd_x_grid = np.asarray(h5_file["psd_x_grid"], dtype=float)
        for case_id in case_ids:
            if case_id not in h5_file["conditions"]:
                raise KeyError(f"Requested PLOT_CASE_IDS entry is not present: {case_id}")
            group = h5_file["conditions"][case_id]
            if not bool(group.attrs.get("complete", False)):
                raise RuntimeError(f"Requested plot case is not complete: {case_id}")
            Q_mean = np.asarray(group["psd_Q_mean"], dtype=float)
            moments_mean = np.asarray(group["moments_mean"], dtype=float)

            figure, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(psd_x_grid, Q_mean[-1], label=f"t={t_vec[-1]:.6g}")
            axes[0].set_xscale("log")
            axes[0].set_xlabel("diameter")
            axes[0].set_ylabel(f"Q ({PSD_BASIS})")
            axes[0].grid(True)
            axes[0].legend()
            for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
                if i < moments_mean.shape[0] and j < moments_mean.shape[1]:
                    axes[1].plot(t_vec, moments_mean[i, j], label=rf"$\mu_{{{i},{j}}}$")
            axes[1].set_xlabel("time")
            axes[1].set_ylabel("normalized moment")
            axes[1].grid(True)
            axes[1].legend()
            figure.suptitle(case_id)
            figure.tight_layout()
            figure.savefig(plot_dir / f"{case_id}.png", dpi=160)
            plt.close(figure)


# =============================================================================
# Orchestration
# =============================================================================


def _write_completed_result(result_path: Path, result: Mapping[str, Any], config: StudyConfig) -> None:
    with h5py.File(result_path, "a") as h5_file:
        write_condition_result(h5_file, result, config)


def run_parameter_scan(config: StudyConfig = DEFAULT_CONFIG) -> Path:
    """Validate once, execute pending cases, save HDF5/CSV, and return HDF5 path."""
    cases = build_scan_cases(config)
    validate_startup(config, cases)
    pending_cases = prepare_result_file(config, cases)
    print(
        f"Phase-1 parameter scan: {len(cases)} defined conditions, "
        f"{len(pending_cases)} pending, {config.n_repeats} repeats each, "
        f"workers={config.n_workers}."
    )
    print(
        "Initial state: monodisperse_2d_explicit | "
        f"compute_particles={config.initial_compute_particles}, "
        f"weight_per_compute_particle={config.initial_weight_per_compute_particle:.6g}, "
        f"represented_particles={config.initial_represented_particle_count:.6g}, "
        f"particle_volume={config.initial_particle_volume:.6g}, "
        f"represented_total_volume={config.initial_represented_total_volume:.6g}."
    )

    completed = 0
    total = len(pending_cases)
    if config.n_workers == 1:
        for case in pending_cases:
            result = run_condition_task(config, case)
            _write_completed_result(config.result_path, result, config)
            completed += 1
            if config.print_every_condition:
                print_condition_summary(result, completed, total)
    else:
        # The HDF5 file is not open while worker processes are spawned.  Workers
        # never access it; the parent opens it only after results are returned.
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = {
                executor.submit(run_condition_task, config, case): case
                for case in pending_cases
            }
            try:
                for future in as_completed(futures):
                    result = future.result()
                    _write_completed_result(config.result_path, result, config)
                    completed += 1
                    if config.print_every_condition:
                        print_condition_summary(result, completed, total)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    condition_csv, time_csv = export_csv_summaries(config.result_path)
    plot_selected_cases(config.result_path, config.plot_case_ids)
    print(f"Saved HDF5 results: {config.result_path}")
    print(f"Saved condition summary: {condition_csv}")
    print(f"Saved time summary: {time_csv}")
    return config.result_path


if __name__ == "__main__":
    RESULT_PATH = run_parameter_scan()
