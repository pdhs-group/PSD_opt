# -*- coding: utf-8 -*-
"""
Serial acceptance-rate probe for the MPTSA + MAS aggregate sampler.

This script mirrors the candidate-generation and acceptance logic from
aggregates_sampler_npz_sqlite.py, but intentionally does not write NPZ files,
SQLite indexes, restart checkpoints, or error dumps. It is meant for quick,
side-by-side acceptance-rate comparisons before changing the sampler logic.

For each (Np, frac_A) pair:
  - submit SAMPLES_PER_PARAM serial "sample slots";
  - each slot uses the same _task_seed scheme as the NPZ+SQLite builder;
  - each slot runs at most MAX_TRIES_FACTOR MPTSA grid attempts;
  - each MPTSA grid is reused for MIX_TRIES_PER_MPTSA material-mix attempts;
  - each material-mix attempt uses a fresh seed_mix and one lambda schedule;
  - the first accepted material-mix attempt completes that slot.

Reported rates:
  - mix_accept_rate = accepted slots / total material-mix attempts
  - mptsa_accept_rate = accepted slots / total MPTSA grid attempts
  - task_success_rate = accepted slots / submitted sample slots
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from agggenerator.mptsa2d import (
    MPTSALatticeParams2D,
    estimate_fractal_dimension_2d,
    generate_mptsa_lattice_2d,
)
from agggenerator.material_mix import (
    MASPhysicalParams,
    MaterialMixParams,
    assign_materials_with_target_mas,
    probe_low_mas_geometry,
)


# ============================================================
# User configuration (kept close to aggregates_sampler_npz_sqlite.py)
# ============================================================

TARGET_DF: float = 1.8
TARGET_MAS: float = 0.10

DF_TOL: float = 0.1  # kept for reporting only; not enforced in the source script
MAS_TOL: float = 0.05

NP_LIST: List[int] = [10000]

FRAC_A_LIST: List[float] = [0.1, 0.5, 0.9]

SAMPLES_PER_PARAM: int = 100
MAX_TRIES_FACTOR: int = 50
MIX_TRIES_PER_MPTSA: int = 5

GEOMETRY_PRESCREEN_ENABLED: bool = True
GEOMETRY_PRESCREEN_MAS_THRESHOLD: float = 0.3

# Tuple fields are:
#   (lambda_min, lambda_max, sweeps_per_eval, max_bisect, temperature)
MIX_LAMBDA_SCHEDULES: List[Tuple[float, float, int, int, float]] = [
    (-3.0, 3.0, 8, 20, 1.0),
    (0.0, 6.0, 12, 24, 1.0),
    (1.0, 10.0, 16, 24, 0.8),
    (2.0, 14.0, 20, 28, 0.7),
]

MASTER_SEED: int = 42

# Set to 0 to silence within-group progress lines.
PRINT_PROGRESS_EVERY: int = 1


# ============================================================
# Parameter helpers copied from the NPZ+SQLite builder
# ============================================================

def _task_seed(master_seed: int, Np: int, frac_A: float, attempt_index: int) -> int:
    frac_key = int(round(float(frac_A) * 10000.0))
    seq = np.random.SeedSequence(
        [int(master_seed), int(Np), frac_key, int(attempt_index)]
    )
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _make_mptsa_params(Np: int, Df: float, seed: int) -> MPTSALatticeParams2D:
    return MPTSALatticeParams2D(
        Np=int(Np),
        Df=float(Df),
        k=1.0,
        max_attempts=50000,
        seed=int(seed),
        fill_hole=True,
        hole_area_max=4,
        compensate_alpha=1.0,
        compensate_beta=0.25,
        verbose=False,
    )


def _compute_mix_window_stride(grid: np.ndarray) -> Tuple[int, int]:
    if grid.ndim < 2:
        raise ValueError(f"Expected a 2D grid, got shape={grid.shape!r}")

    shorter_side = int(min(grid.shape[0], grid.shape[1]))
    window = max(2, min(12, shorter_side // 5))
    stride = max(1, min(3, window // 4))
    return window, stride


def _make_mix_params(
    frac_A: float,
    target_MAS: float,
    seed: int,
    grid: np.ndarray,
    lambda_min: float = -3.0,
    lambda_max: float = 3.0,
    sweeps_per_eval: int = 8,
    max_bisect: int = 20,
    temperature: float = 1.0,
) -> MaterialMixParams:
    window, stride = _compute_mix_window_stride(grid)

    return MaterialMixParams(
        frac_A=float(frac_A),
        target_MAS=float(target_MAS),
        tol_MAS=0.05,
        window=window,
        stride=stride,
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        sweeps_per_eval=int(sweeps_per_eval),
        max_bisect=int(max_bisect),
        temperature=float(temperature),
        seed=int(seed),
    )


def _mix_schedule_for_attempt(mix_attempt_index: int) -> Tuple[float, float, int, int, float]:
    if not MIX_LAMBDA_SCHEDULES:
        return -3.0, 3.0, 8, 20, 1.0
    return MIX_LAMBDA_SCHEDULES[int(mix_attempt_index) % len(MIX_LAMBDA_SCHEDULES)]


def _accepted_by_original_mas_filter(
    frac_A: float,
    mas_actual: float,
    target_MAS: float,
    mas_tol: float,
) -> bool:
    if 0.0 < frac_A < 1.0:
        return bool(
            np.isfinite(mas_actual)
            and abs(float(mas_actual) - float(target_MAS)) <= float(mas_tol)
        )
    return True


# ============================================================
# Probe bookkeeping
# ============================================================

@dataclass
class TaskProbeResult:
    success: bool
    mptsa_tries: int
    mix_tries: int
    mas_rejects: int
    errors: int
    finite_mas_evals: int
    best_mas: float
    best_abs_mas_error: float
    accepted_mas: float = float("nan")
    accepted_df: float = float("nan")
    accepted_slope: float = float("nan")
    accepted_seed_mptsa: int = -1
    accepted_seed_mix: int = -1
    accepted_mix_try: int = -1
    accepted_lambda_min: float = float("nan")
    accepted_lambda_max: float = float("nan")
    accepted_lambda_used: float = float("nan")
    geometry_prescreens: int = 0
    geometry_rejects: int = 0
    best_probe_mas: float = float("nan")
    last_error: str = ""


@dataclass
class GroupProbeStats:
    Np: int
    frac_A: float
    tasks: int = 0
    accepted: int = 0
    mptsa_attempts: int = 0
    mix_attempts: int = 0
    mas_rejects: int = 0
    errors: int = 0
    finite_mas_evals: int = 0
    geometry_prescreens: int = 0
    geometry_rejects: int = 0
    success_mptsa_tries_sum: int = 0
    success_mix_tries_sum: int = 0
    accepted_mas_sum: float = 0.0
    accepted_mas_min: float = float("inf")
    accepted_mas_max: float = float("-inf")
    accepted_df_sum: float = 0.0
    best_mas: float = float("nan")
    best_abs_mas_error: float = float("inf")
    best_probe_mas: float = float("nan")
    elapsed_s: float = 0.0
    last_error: str = ""

    def add(self, result: TaskProbeResult) -> None:
        self.tasks += 1
        self.mptsa_attempts += int(result.mptsa_tries)
        self.mix_attempts += int(result.mix_tries)
        self.mas_rejects += int(result.mas_rejects)
        self.errors += int(result.errors)
        self.finite_mas_evals += int(result.finite_mas_evals)
        self.geometry_prescreens += int(result.geometry_prescreens)
        self.geometry_rejects += int(result.geometry_rejects)

        if result.last_error:
            self.last_error = result.last_error

        if np.isfinite(result.best_probe_mas):
            if (
                not np.isfinite(self.best_probe_mas)
                or result.best_probe_mas < self.best_probe_mas
            ):
                self.best_probe_mas = float(result.best_probe_mas)

        if result.best_abs_mas_error < self.best_abs_mas_error:
            self.best_abs_mas_error = float(result.best_abs_mas_error)
            self.best_mas = float(result.best_mas)

        if result.success:
            self.accepted += 1
            self.success_mptsa_tries_sum += int(result.mptsa_tries)
            self.success_mix_tries_sum += int(result.mix_tries)
            if np.isfinite(result.accepted_mas):
                mas = float(result.accepted_mas)
                self.accepted_mas_sum += mas
                self.accepted_mas_min = min(self.accepted_mas_min, mas)
                self.accepted_mas_max = max(self.accepted_mas_max, mas)
            if np.isfinite(result.accepted_df):
                self.accepted_df_sum += float(result.accepted_df)

    @property
    def failed_tasks(self) -> int:
        return self.tasks - self.accepted

    @property
    def mix_accept_rate(self) -> float:
        return self.accepted / self.mix_attempts if self.mix_attempts else float("nan")

    @property
    def mptsa_accept_rate(self) -> float:
        return (
            self.accepted / self.mptsa_attempts
            if self.mptsa_attempts
            else float("nan")
        )

    @property
    def task_success_rate(self) -> float:
        return self.accepted / self.tasks if self.tasks else float("nan")

    @property
    def geometry_reject_rate(self) -> float:
        return (
            self.geometry_rejects / self.geometry_prescreens
            if self.geometry_prescreens
            else float("nan")
        )

    @property
    def mix_attempts_per_accept(self) -> float:
        return self.mix_attempts / self.accepted if self.accepted else float("inf")

    @property
    def mptsa_attempts_per_accept(self) -> float:
        return self.mptsa_attempts / self.accepted if self.accepted else float("inf")

    @property
    def mean_success_mptsa_tries(self) -> float:
        return (
            self.success_mptsa_tries_sum / self.accepted
            if self.accepted
            else float("nan")
        )

    @property
    def mean_success_mix_tries(self) -> float:
        return (
            self.success_mix_tries_sum / self.accepted
            if self.accepted
            else float("nan")
        )

    @property
    def mean_accepted_mas(self) -> float:
        return self.accepted_mas_sum / self.accepted if self.accepted else float("nan")

    @property
    def mean_accepted_df(self) -> float:
        return self.accepted_df_sum / self.accepted if self.accepted else float("nan")


# ============================================================
# Probe runner
# ============================================================

def _probe_one_sampler_task(
    Np: int,
    frac_A: float,
    target_Df: float,
    target_MAS: float,
    mas_tol: float,
    max_tries_factor: int,
    base_seed: int,
) -> TaskProbeResult:
    rng = np.random.default_rng(base_seed)
    phys_params = MASPhysicalParams()
    max_mptsa_tries = int(max(1, max_tries_factor))
    mix_tries_per_mptsa = int(max(1, MIX_TRIES_PER_MPTSA))

    mptsa_tries = 0
    mix_tries = 0
    mas_rejects = 0
    errors = 0
    finite_mas_evals = 0
    geometry_prescreens = 0
    geometry_rejects = 0
    best_mas = float("nan")
    best_abs_mas_error = float("inf")
    best_probe_mas = float("nan")
    last_error = ""

    while mptsa_tries < max_mptsa_tries:
        mptsa_tries += 1
        seed_mptsa = int(rng.integers(0, 2**31 - 1))

        mptsa_params = _make_mptsa_params(Np=Np, Df=target_Df, seed=seed_mptsa)

        try:
            _positions, Ns, Rgs, grid, _origin = generate_mptsa_lattice_2d(
                mptsa_params
            )
            Df_est, slope = estimate_fractal_dimension_2d(Ns, Rgs)
        except Exception as exc:
            errors += 1
            last_error = str(exc).replace("\n", " | ")[:200]
            continue

        if (
            GEOMETRY_PRESCREEN_ENABLED
            and target_MAS < GEOMETRY_PRESCREEN_MAS_THRESHOLD
            and 0.0 < frac_A < 1.0
        ):
            geometry_prescreens += 1
            try:
                probe_params = _make_mix_params(
                    frac_A=frac_A,
                    target_MAS=target_MAS,
                    seed=seed_mptsa,
                    grid=grid,
                )
                _probe_labels, probe_stats = probe_low_mas_geometry(
                    grid,
                    probe_params,
                    phys_params,
                    seed=seed_mptsa,
                )
            except Exception as exc:
                errors += 1
                last_error = str(exc).replace("\n", " | ")[:200]
                continue

            probe_mas = float(probe_stats.get("MAS", np.nan))
            if np.isfinite(probe_mas):
                if (not np.isfinite(best_probe_mas)) or probe_mas < best_probe_mas:
                    best_probe_mas = float(probe_mas)

            if (not np.isfinite(probe_mas)) or probe_mas > target_MAS + mas_tol:
                geometry_rejects += 1
                continue

        for mix_try in range(mix_tries_per_mptsa):
            mix_tries += 1
            seed_mix = int(rng.integers(0, 2**31 - 1))
            (
                lambda_min,
                lambda_max,
                sweeps_per_eval,
                max_bisect,
                temperature,
            ) = _mix_schedule_for_attempt(mix_try)

            try:
                mix_params = _make_mix_params(
                    frac_A=frac_A,
                    target_MAS=target_MAS,
                    seed=seed_mix,
                    grid=grid,
                    lambda_min=lambda_min,
                    lambda_max=lambda_max,
                    sweeps_per_eval=sweeps_per_eval,
                    max_bisect=max_bisect,
                    temperature=temperature,
                )
                _labels, stats = assign_materials_with_target_mas(
                    grid, mix_params, phys_params
                )
            except Exception as exc:
                errors += 1
                last_error = str(exc).replace("\n", " | ")[:200]
                continue

            MAS_actual = float(stats.get("MAS", np.nan))
            if np.isfinite(MAS_actual):
                finite_mas_evals += 1
                err = abs(MAS_actual - float(target_MAS))
                if err < best_abs_mas_error:
                    best_abs_mas_error = float(err)
                    best_mas = float(MAS_actual)

            if _accepted_by_original_mas_filter(
                frac_A=frac_A,
                mas_actual=MAS_actual,
                target_MAS=target_MAS,
                mas_tol=mas_tol,
            ):
                return TaskProbeResult(
                    success=True,
                    mptsa_tries=mptsa_tries,
                    mix_tries=mix_tries,
                    mas_rejects=mas_rejects,
                    errors=errors,
                    finite_mas_evals=finite_mas_evals,
                    best_mas=best_mas,
                    best_abs_mas_error=best_abs_mas_error,
                    accepted_mas=MAS_actual,
                    accepted_df=float(Df_est),
                    accepted_slope=float(slope),
                    accepted_seed_mptsa=seed_mptsa,
                    accepted_seed_mix=seed_mix,
                    accepted_mix_try=int(mix_try + 1),
                    accepted_lambda_min=float(lambda_min),
                    accepted_lambda_max=float(lambda_max),
                    accepted_lambda_used=float(stats.get("lambda_used", np.nan)),
                    geometry_prescreens=geometry_prescreens,
                    geometry_rejects=geometry_rejects,
                    best_probe_mas=best_probe_mas,
                    last_error=last_error,
                )

            mas_rejects += 1

    return TaskProbeResult(
        success=False,
        mptsa_tries=mptsa_tries,
        mix_tries=mix_tries,
        mas_rejects=mas_rejects,
        errors=errors,
        finite_mas_evals=finite_mas_evals,
        best_mas=best_mas,
        best_abs_mas_error=best_abs_mas_error,
        geometry_prescreens=geometry_prescreens,
        geometry_rejects=geometry_rejects,
        best_probe_mas=best_probe_mas,
        last_error=last_error,
    )


def _format_rate(value: float) -> str:
    if not np.isfinite(value):
        return "   n/a "
    return f"{100.0 * value:6.2f}%"


def _format_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _print_group_progress(stats: GroupProbeStats) -> None:
    print(
        f"  progress {stats.tasks:4d}/{SAMPLES_PER_PARAM}: "
        f"accepted={stats.accepted:4d}, "
        f"mix_accept={_format_rate(stats.mix_accept_rate)}, "
        f"task_success={_format_rate(stats.task_success_rate)}, "
        f"mptsa_attempts={stats.mptsa_attempts}, "
        f"mix_attempts={stats.mix_attempts}, "
        f"geom_rejects={stats.geometry_rejects}",
        flush=True,
    )


def _print_group_summary(stats: GroupProbeStats) -> None:
    print(
        f"[RESULT] Np={stats.Np}, frac_A={stats.frac_A:.4f}: "
        f"accepted={stats.accepted}/{stats.tasks}, "
        f"mptsa_attempts={stats.mptsa_attempts}, "
        f"mix_attempts={stats.mix_attempts}, "
        f"mix_accept={_format_rate(stats.mix_accept_rate)}, "
        f"mptsa_accept={_format_rate(stats.mptsa_accept_rate)}, "
        f"task_success={_format_rate(stats.task_success_rate)}, "
        f"mix/accept={_format_float(stats.mix_attempts_per_accept, 2)}, "
        f"mptsa/accept={_format_float(stats.mptsa_attempts_per_accept, 2)}, "
        f"mean_success_mix={_format_float(stats.mean_success_mix_tries, 2)}, "
        f"mean_MAS={_format_float(stats.mean_accepted_mas, 4)}, "
        f"best_MAS_seen={_format_float(stats.best_mas, 4)}, "
        f"best_probe_MAS={_format_float(stats.best_probe_mas, 4)}, "
        f"geom_rejects={stats.geometry_rejects}/{stats.geometry_prescreens}, "
        f"errors={stats.errors}, elapsed={stats.elapsed_s:.1f}s",
        flush=True,
    )


def _print_final_table(all_stats: List[GroupProbeStats]) -> None:
    print("\n[SUMMARY]")
    header = (
        "Np      frac_A  tasks  ok   mptsa_att  mix_att  mix_accept  "
        "mptsa_accept  task_success  mix/ok  mean_MAS  best_MAS  "
        "probe_MAS  geom_skip  errors  seconds"
    )
    print(header)
    print("-" * len(header))
    for stats in all_stats:
        print(
            f"{stats.Np:<7d} "
            f"{stats.frac_A:<7.4f} "
            f"{stats.tasks:<5d} "
            f"{stats.accepted:<4d} "
            f"{stats.mptsa_attempts:<9d} "
            f"{stats.mix_attempts:<7d} "
            f"{_format_rate(stats.mix_accept_rate):>10s} "
            f"{_format_rate(stats.mptsa_accept_rate):>12s} "
            f"{_format_rate(stats.task_success_rate):>12s} "
            f"{_format_float(stats.mix_attempts_per_accept, 2):>6s} "
            f"{_format_float(stats.mean_accepted_mas, 4):>9s} "
            f"{_format_float(stats.best_mas, 4):>8s} "
            f"{_format_float(stats.best_probe_mas, 4):>9s} "
            f"{stats.geometry_rejects:>4d}/{stats.geometry_prescreens:<4d} "
            f"{stats.errors:<6d} "
            f"{stats.elapsed_s:>7.1f}",
            flush=True,
        )


def run_probe() -> None:
    param_pairs: List[Tuple[int, float]] = [
        (Np, frac_A) for Np in NP_LIST for frac_A in FRAC_A_LIST
    ]

    print(
        f"[PROBE] Target pairs: {len(param_pairs)} (Np, frac_A) combinations, "
        f"{SAMPLES_PER_PARAM} serial sample slots per pair."
    )
    print(
        f"[PROBE] Fixed targets: Df={TARGET_DF}, MAS={TARGET_MAS}, "
        f"DF_TOL={DF_TOL} (not enforced), MAS_TOL={MAS_TOL}"
    )
    print(
        f"[PROBE] Repeat budget: max {MAX_TRIES_FACTOR} MPTSA grids per slot, "
        f"{MIX_TRIES_PER_MPTSA} mix attempts per grid, "
        f"{len(MIX_LAMBDA_SCHEDULES)} lambda schedules."
    )
    print(
        f"[PROBE] Geometry prescreen: enabled={GEOMETRY_PRESCREEN_ENABLED}, "
        f"active for MAS<{GEOMETRY_PRESCREEN_MAS_THRESHOLD}, "
        "reject if probe_MAS > target + MAS_TOL.\n"
    )

    all_stats: List[GroupProbeStats] = []
    total_start = time.perf_counter()

    for Np, frac_A in param_pairs:
        stats = GroupProbeStats(Np=int(Np), frac_A=float(frac_A))
        start = time.perf_counter()
        print(
            f"[GROUP] Np={int(Np)}, frac_A={float(frac_A):.4f}",
            flush=True,
        )

        for task_index in range(int(SAMPLES_PER_PARAM)):
            task_seed = _task_seed(
                MASTER_SEED,
                int(Np),
                float(frac_A),
                int(task_index),
            )
            result = _probe_one_sampler_task(
                Np=int(Np),
                frac_A=float(frac_A),
                target_Df=TARGET_DF,
                target_MAS=TARGET_MAS,
                mas_tol=MAS_TOL,
                max_tries_factor=MAX_TRIES_FACTOR,
                base_seed=int(task_seed),
            )
            stats.add(result)

            if (
                PRINT_PROGRESS_EVERY > 0
                and (
                    stats.tasks % int(PRINT_PROGRESS_EVERY) == 0
                    or stats.tasks == int(SAMPLES_PER_PARAM)
                )
            ):
                _print_group_progress(stats)

        stats.elapsed_s = time.perf_counter() - start
        all_stats.append(stats)
        _print_group_summary(stats)
        print("")

    total_elapsed = time.perf_counter() - total_start
    _print_final_table(all_stats)
    print(f"\n[PROBE] Done in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    run_probe()
