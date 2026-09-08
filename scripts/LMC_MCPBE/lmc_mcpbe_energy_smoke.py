# -*- coding: utf-8 -*-
"""One-event 2D smoke test for wmcpbe + aggregate pool + energy model.

Run this file directly from Spyder.  It deliberately uses only breakage and
one event, but requires the real online LMC path to read one aggregate from the
configured pool and the selected energy model to provide the breakage rate.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _bootstrap_project_paths() -> None:
    """Make the monorepo source packages importable in a Spyder file run."""
    project_root = Path(__file__).resolve().parents[2]
    source_paths = (
        project_root / "pbe-core" / "src",
        project_root / "lmc" / "src",
        project_root / "breakage-rate-model" / "src",
        project_root / "mcpbe" / "src",
    )
    for path in source_paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Required project source directory is missing: {path}")
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


_bootstrap_project_paths()

from wmcpbe import MCPBESolver  # noqa: E402


# =============================================================================
# Spyder configuration: edit only this section for parameter studies.
# =============================================================================

DATA_ROOT = Path(r"D:\LMC")
BREAKAGE_MODEL_KIND = "mlp"  # "mlp", "ann", "powerlaw", or "parametric"
BREAKAGE_MODEL_PATH = DATA_ROOT / f"{BREAKAGE_MODEL_KIND}_model.pkl"
AGGREGATE_POOL_ROOT = DATA_ROOT

# This temporary MLP was trained with energy-pool A0=1.0.  Keep this identical
# to the energy model and aggregate-pool lattice-cell scale unless both assets
# are regenerated on a different scale.
LMC_A0_RUNTIME = 1.0

# The currently available temporary model files predate persisted training
# volume metadata.  These bounds are therefore explicit and refer to
# log(V / LMC_A0_RUNTIME), not to raw PBE volume.
MODEL_LOGV_BOUNDS = (float(np.log(100.0)), float(np.log(50_000.0)))
WARN_MODEL_EXTRAPOLATION = True
WARN_POOL_OUT_OF_BOUNDS = True

# A 2D parent is required to expose X1.  Its components are
# [PARENT_VOLUME * X1, PARENT_VOLUME * (1 - X1)].
PARENT_VOLUME = 1_000.0
X1 = 0.5
INITIAL_PARTICLES = 4

# Parameters supplied both to the LMC fracture calculation and, through the
# full 10-column feature vector, to the energy-model adapter.
NO_FRAG = 2
GAMMA = 1.0
INT_BRE = 0.0
DF = 1.8
MAS = 0.1
STR = np.array([1.0, 1.0, 1.0], dtype=float)

# The MLP adapter computes S(V) = E_in(V) / E_need(V).  These parameters are
# intentionally explicit because they set the event-time scale of the PBE.
LAMBDA_E = 1.0
ENERGY_EXPONENT = 1.0
RATE_MIN = 0.0
RATE_MAX = None

# Minimal PBE/event settings.  ``MAX_EVENTS=1`` is intentional: this script is
# an integration smoke test, not a statistically meaningful PBE simulation.
SEED = 42
END_TIME = 10.0
MAX_EVENTS = 1
BREAK_DW_CONST = 1.0
DELTA_CELLS = 0.1
MASS_CONSERVATION_RTOL = 1e-12  # double-precision summation check


def _pool_directory_name(df: float, mas: float) -> str:
    """Mirror the NPZ+SQLite aggregate-pool naming convention."""
    return f"aggregate_pool_Df{str(df).replace('.', 'p')}_MAS{mas:.2f}".replace(".", "p") + "_npz_single"


def _validate_configuration() -> None:
    """Fail before constructing a solver when external assets or controls are invalid."""
    if BREAKAGE_MODEL_KIND not in {"mlp", "ann", "powerlaw", "parametric"}:
        raise ValueError("BREAKAGE_MODEL_KIND must select one of the four trained models.")
    if not BREAKAGE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Selected energy model does not exist: {BREAKAGE_MODEL_PATH}"
        )
    if not AGGREGATE_POOL_ROOT.is_dir():
        raise FileNotFoundError(f"Aggregate-pool root does not exist: {AGGREGATE_POOL_ROOT}")
    expected_pool = AGGREGATE_POOL_ROOT / _pool_directory_name(DF, MAS)
    if not expected_pool.is_dir() or not (expected_pool / "pool_index.sqlite").is_file():
        raise FileNotFoundError(
            "No completed NPZ+SQLite aggregate pool matches the selected Df/MAS: "
            f"{expected_pool}"
        )
    if not np.isfinite(PARENT_VOLUME) or PARENT_VOLUME <= 0.0:
        raise ValueError("PARENT_VOLUME must be finite and positive.")
    if not np.isfinite(X1) or not 0.0 < X1 < 1.0:
        raise ValueError("X1 must be strictly between 0 and 1 for this 2D smoke test.")
    if INITIAL_PARTICLES < 1 or MAX_EVENTS != 1:
        raise ValueError("INITIAL_PARTICLES must be positive and MAX_EVENTS must remain exactly 1.")
    if NO_FRAG < 2:
        raise ValueError("NO_FRAG must be at least 2.")
    if GAMMA <= 0.0 or LMC_A0_RUNTIME <= 0.0:
        raise ValueError("GAMMA and LMC_A0_RUNTIME must both be positive.")
    if np.asarray(STR, dtype=float).shape != (3,) or not np.all(np.isfinite(STR)):
        raise ValueError("STR must be a finite array with shape (3,).")
    if RATE_MIN < 0.0 or (RATE_MAX is not None and RATE_MAX < RATE_MIN):
        raise ValueError("Breakage-rate bounds are inconsistent.")
    if not np.isfinite(BREAK_DW_CONST) or BREAK_DW_CONST <= 0.0:
        raise ValueError("BREAK_DW_CONST must be finite and positive.")
    logV_bounds = np.asarray(MODEL_LOGV_BOUNDS, dtype=float)
    if (
        logV_bounds.shape != (2,)
        or not np.all(np.isfinite(logV_bounds))
        or logV_bounds[0] >= logV_bounds[1]
    ):
        raise ValueError("MODEL_LOGV_BOUNDS must be two finite increasing log-volume values.")


def _initial_particle_state() -> tuple[np.ndarray, np.ndarray]:
    """Build identical two-phase particles with unit representative weights."""
    phase_1 = PARENT_VOLUME * X1
    phase_2 = PARENT_VOLUME * (1.0 - X1)
    volumes = np.empty((3, INITIAL_PARTICLES), dtype=float)
    volumes[0, :] = phase_1
    volumes[1, :] = phase_2
    volumes[2, :] = PARENT_VOLUME
    return volumes, np.ones(INITIAL_PARTICLES, dtype=float)


def _build_solver() -> MCPBESolver:
    """Create a minimal 2D breakage solver and initialize its live dependencies."""
    volumes, weights = _initial_particle_state()
    solver = MCPBESolver(
        dim=2,
        t_vec=np.array([0.0, END_TIME], dtype=float),
        verbose=True,
        load_attr=False,
        init=False,
        seed=SEED,
    )
    solver.process_type = "breakage"
    solver.a0 = INITIAL_PARTICLES
    solver.Vc = 1.0
    solver.G = 1.0

    # Required breakage controls, even though the live LMC provides fragments.
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2
    solver.pl_v = 2.0
    solver.pl_q = 1.0
    solver.pl_P1 = 1.0
    solver.pl_P2 = 1.0
    solver.pl_P3 = 1.0
    solver.pl_P4 = 1.0
    solver.break_dW_const = BREAK_DW_CONST

    # Live LMC + aggregate pool.
    solver.use_lmc_live = True
    solver.lmc_pool_dir = str(AGGREGATE_POOL_ROOT)
    solver.lmc_A0_runtime = LMC_A0_RUNTIME
    solver.lmc_Df = DF
    solver.lmc_MAS = MAS
    solver.lmc_NO_FRAG = NO_FRAG
    solver.lmc_gamma = GAMMA
    solver.lmc_int_bre = INT_BRE
    solver.lmc_STR = np.asarray(STR, dtype=float)
    solver.lmc_allow_loops = True
    solver.lmc_accept_all_cracks = False
    solver.lmc_use_weighted_start = False
    solver.lmc_delta_cells = DELTA_CELLS

    # Energy-rate adapter.  It keeps this full ten-feature runtime interface
    # regardless of which subset was active when the selected model was trained.
    solver.lmc_use_breakage_model = True
    solver.lmc_breakage_model_kind = BREAKAGE_MODEL_KIND
    solver.lmc_breakage_model_path = str(BREAKAGE_MODEL_PATH)
    solver.lmc_breakage_model_logV_bounds = MODEL_LOGV_BOUNDS
    solver.lmc_warn_model_extrapolation = WARN_MODEL_EXTRAPOLATION
    solver.lmc_warn_pool_out_of_bounds = WARN_POOL_OUT_OF_BOUNDS
    solver.lmc_lambda_E = LAMBDA_E
    solver.lmc_energy_exp = ENERGY_EXPONENT
    solver.lmc_rate_min = RATE_MIN
    solver.lmc_rate_max = RATE_MAX

    solver._initialize_particles(init_Vc=False, V_flat=volumes, W_init=weights)
    solver._init_lmc()
    solver._initialize_samplers()
    return solver


def run_smoke() -> dict[str, object]:
    """Run one real breakage event and verify the energy-model and pool pathways."""
    _validate_configuration()
    solver = _build_solver()
    if solver.lmc_live is None or solver.lmc_breakage_adapter is None:
        raise RuntimeError("Live LMC or energy breakage-rate adapter was not initialized.")

    initial_phase_volume = np.sum(solver.V_flat[:2, : solver.a_tot], axis=1)
    initial_propensity = float(solver._break_sampler.total())
    if not np.isfinite(initial_propensity) or initial_propensity <= 0.0:
        raise FloatingPointError(
            f"Initial total breakage propensity must be finite and positive, got {initial_propensity}."
        )

    adapter = solver.lmc_breakage_adapter
    features, _normalized_volume = adapter._build_features_batch(solver)
    predicted_log_energy = adapter.model.predict(features)
    if not np.all(np.isfinite(predicted_log_energy)):
        raise FloatingPointError("Energy model produced non-finite log-energy predictions.")

    try:
        solver.solve(maxiter=MAX_EVENTS)
    finally:
        # ``solve`` already clears the cache on normal completion; ``close``
        # also handles failures before that cleanup point.
        solver.lmc_live._sim.close()

    pool_stats = solver.lmc_live._sim.agg_pool.debug_stats()
    if pool_stats["pool_read_calls"] < 1 or pool_stats["pool_npz_file_opens"] < 1:
        raise RuntimeError("No aggregate-pool sample was read during the breakage event.")
    if solver.sim_break_events != 1.0:
        raise RuntimeError(
            f"Expected exactly one simulated breakage event, got {solver.sim_break_events}."
        )

    final_phase_volume = np.sum(solver.V_flat[:2, : solver.a_tot], axis=1)
    if not np.allclose(
        final_phase_volume,
        initial_phase_volume,
        rtol=MASS_CONSERVATION_RTOL,
        atol=0.0,
    ):
        raise RuntimeError(
            "Phase-wise volume was not conserved by the live-LMC breakage event: "
            f"initial={initial_phase_volume}, final={final_phase_volume}."
        )

    result = {
        "initial_particles": int(INITIAL_PARTICLES),
        "final_particles": int(solver.a_tot),
        "initial_total_breakage_propensity": initial_propensity,
        "simulated_breakage_events": float(solver.sim_break_events),
        "real_breakage_events": float(solver.real_break_events),
        "initial_phase_volume": initial_phase_volume,
        "final_phase_volume": final_phase_volume,
        "energy_model_active_features": tuple(adapter.model.active_feature_names),
        "log_energy_range": (
            float(np.min(predicted_log_energy)),
            float(np.max(predicted_log_energy)),
        ),
        "pool_stats": pool_stats,
    }
    print("Smoke test passed.")
    for name, value in result.items():
        print(f"  {name}: {value}")
    return result


if __name__ == "__main__":
    SMOKE_RESULT = run_smoke()
