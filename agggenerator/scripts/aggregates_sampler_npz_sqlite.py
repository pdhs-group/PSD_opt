# -*- coding: utf-8 -*-
"""
Offline builder for an MPTSA + MAS aggregate pool stored as NPZ files with a
SQLite index.

Key behavior
------------
1. The generation logic is kept aligned with `aggregates_sampler.py`: samples
   are created from the same MPTSA + MAS workflow and converted to the same LMC
   grid representation.
2. Before generating a parameter pair `(Np, frac_A)`, the script checks the
   existing SQLite index:
   - if enough samples already exist, the pair is skipped;
   - if the pair does not exist, it is generated from scratch;
   - if the pair exists but has too few samples, only the missing number is
     targeted.
3. Each worker generates at most one accepted sample, writes its NPZ file
   directly, and returns only lightweight metadata to the main process.
4. SQLite is treated as the authoritative checkpoint state. If a run is
   interrupted after an NPZ file is written but before SQLite is updated, that
   sample may be regenerated in a later run.
5. Each parameter group stores a persistent `next_attempt_index` in SQLite.
   The attempt cursor is advanced when a task is submitted, so restarted runs do
   not repeatedly replay the same rejected random attempts.
"""

from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from lmc import GridFactory

# ============================================================
# User configuration (edit here inside Spyder)
# ============================================================

TARGET_DF: float = 1.8
TARGET_MAS: float = 0.10

DF_TOL: float = 0.1
MAS_TOL: float = 0.05

NP_LIST: List[int] = [100, 200, 400, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 16000,
                      20000, 30000, 40000, 50000]

FRAC_A_LIST: List[float] = [0.1, 0.5, 0.9]

SAMPLES_PER_PARAM: int = 100
MAX_TRIES_FACTOR: int = 100
MIX_TRIES_PER_MPTSA: int = 4

GEOMETRY_PRESCREEN_ENABLED: bool = True
GEOMETRY_PRESCREEN_MAS_THRESHOLD: float = 0.3

# Each MPTSA geometry is reused across several material-mixing attempts.
# Tuple fields are:
#   (lambda_min, lambda_max, sweeps_per_eval, max_bisect, temperature)
# The first entry mirrors the original sampler settings. Later entries widen
# the positive-lambda side, which is usually the useful direction for low MAS.
MIX_LAMBDA_SCHEDULES: List[Tuple[float, float, int, int, float]] = [
    (-3.0, 3.0, 8, 20, 1.0),
    (0.0, 6.0, 12, 24, 1.0),
    (1.0, 10.0, 16, 24, 0.8),
    (2.0, 14.0, 20, 28, 0.7),
]

OUTPUT_POOL_DIR: str = "aggregate_pool_Df1p8_MAS0p10_npz_single"
# OUTPUT_POOL_DIR = os.path.join(os.environ.get('STORAGE_PATH'), OUTPUT_POOL_DIR)

WORKERS: int = 1
MASTER_SEED: int = 42

A0_CELL_AREA: float = 1.0
INT_BRE: float = 0.0

SAVE_COMPRESSED: bool = False
SQLITE_NAME: str = "pool_index.sqlite"
SAMPLES_SUBDIR: str = "samples"


# ============================================================
# Internal helpers
# ============================================================

def _sanitize_group_name(Np: int, frac_A: float) -> str:
    xa_int = int(round(frac_A * 10000))
    return f"Np{Np}_XA{xa_int:04d}"



def _sample_name(sample_index: int) -> str:
    return f"sample_{int(sample_index):04d}"



def _task_seed(master_seed: int, Np: int, frac_A: float, attempt_index: int) -> int:
    frac_key = int(round(float(frac_A) * 10000.0))
    seq = np.random.SeedSequence([int(master_seed), int(Np), frac_key, int(attempt_index)])
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



def _make_mix_params(
    frac_A: float,
    target_MAS: float,
    seed: int,
    lambda_min: float = -3.0,
    lambda_max: float = 3.0,
    sweeps_per_eval: int = 8,
    max_bisect: int = 20,
    temperature: float = 1.0,
) -> MaterialMixParams:
    return MaterialMixParams(
        frac_A=float(frac_A),
        target_MAS=float(target_MAS),
        tol_MAS=0.05,
        window_rel=0.05,
        window_min=10,
        window_max=2500,
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



def _compute_actual_frac_A(labels: np.ndarray) -> float:
    occ = labels >= 0
    total = int(occ.sum())
    if total == 0:
        return 0.0
    nA = int((labels == 0).sum())
    return nA / total



def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")



def _dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, default=_json_default)



def _ensure_groups_schema(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(groups)").fetchall()}
    if "next_attempt_index" not in columns:
        conn.execute("ALTER TABLE groups ADD COLUMN next_attempt_index INTEGER NOT NULL DEFAULT 0")



def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS groups (
            group_name TEXT PRIMARY KEY,
            np_target REAL,
            frac_a_target REAL,
            n_samples INTEGER NOT NULL,
            attrs_json TEXT NOT NULL,
            next_attempt_index INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT NOT NULL,
            sample_name TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            npz_relpath TEXT NOT NULL,
            array_prefix TEXT NOT NULL,
            attrs_json TEXT NOT NULL,
            FOREIGN KEY(group_name) REFERENCES groups(group_name),
            UNIQUE(group_name, sample_name)
        );

        CREATE INDEX IF NOT EXISTS idx_samples_group_name ON samples(group_name);
        CREATE INDEX IF NOT EXISTS idx_samples_group_index ON samples(group_name, sample_index);
        """
    )
    _ensure_groups_schema(conn)



def _get_meta_value(conn: sqlite3.Connection, key: str) -> Any | None:
    row = conn.execute("SELECT value_json FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return json.loads(str(row[0]))



def _set_meta_if_missing(conn: sqlite3.Connection, key: str, value: Any) -> None:
    existing = _get_meta_value(conn, key)
    if existing is None:
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            (key, _dumps_json(value)),
        )
    elif existing != value:
        print(
            f"[POOL][WARN] SQLite meta '{key}' = {existing!r} differs from current config ({value!r}). "
            f"Keeping existing value."
        )



def _get_group_state(conn: sqlite3.Connection, group_name: str) -> Tuple[int, int, int]:
    row_group = conn.execute(
        "SELECT next_attempt_index FROM groups WHERE group_name = ?",
        (group_name,),
    ).fetchone()
    next_attempt_index = int(row_group[0]) if row_group is not None else 0

    row_samples = conn.execute(
        "SELECT COUNT(*), MAX(sample_index) FROM samples WHERE group_name = ?",
        (group_name,),
    ).fetchone()
    n_existing = int(row_samples[0]) if row_samples is not None else 0
    max_index = int(row_samples[1]) if row_samples is not None and row_samples[1] is not None else -1

    next_attempt_index = max(next_attempt_index, max_index + 1)
    return n_existing, max_index, next_attempt_index



def _upsert_group(
    conn: sqlite3.Connection,
    group_name: str,
    Np: int,
    frac_A: float,
    n_samples: int,
    next_attempt_index: int,
) -> None:
    attrs_json = _dumps_json(
        {
            "Np_target": int(Np),
            "frac_A_target": float(frac_A),
            "n_samples": int(n_samples),
            "next_attempt_index": int(next_attempt_index),
        }
    )
    conn.execute(
        """
        INSERT INTO groups(group_name, np_target, frac_a_target, n_samples, attrs_json, next_attempt_index)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(group_name) DO UPDATE SET
            np_target = excluded.np_target,
            frac_a_target = excluded.frac_a_target,
            n_samples = excluded.n_samples,
            attrs_json = excluded.attrs_json,
            next_attempt_index = excluded.next_attempt_index
        """,
        (
            group_name,
            float(Np),
            float(frac_A),
            int(n_samples),
            attrs_json,
            int(next_attempt_index),
        ),
    )



def _register_sample(
    conn: sqlite3.Connection,
    group_name: str,
    sample_name: str,
    sample_index: int,
    npz_relpath: str,
    sample_attrs: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO samples(group_name, sample_name, sample_index, npz_relpath, array_prefix, attrs_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(group_name, sample_name) DO UPDATE SET
            sample_index = excluded.sample_index,
            npz_relpath = excluded.npz_relpath,
            array_prefix = excluded.array_prefix,
            attrs_json = excluded.attrs_json
        """,
        (
            group_name,
            sample_name,
            int(sample_index),
            npz_relpath,
            "",
            _dumps_json(sample_attrs),
        ),
    )



def _generate_one_sample(
    Np: int,
    frac_A: float,
    target_Df: float,
    df_tol: float,
    target_MAS: float,
    mas_tol: float,
    max_tries_factor: int,
    base_seed: int,
) -> Dict[str, Any] | None:
    rng = np.random.default_rng(base_seed)
    grid_factory = GridFactory()
    phys_params = MASPhysicalParams()
    max_mptsa_tries = int(max(1, max_tries_factor))
    mix_tries_per_mptsa = int(max(1, MIX_TRIES_PER_MPTSA))

    mptsa_tries = 0
    while mptsa_tries < max_mptsa_tries:
        mptsa_tries += 1
        seed_mptsa = int(rng.integers(0, 2**31 - 1))

        mptsa_params = _make_mptsa_params(Np=Np, Df=target_Df, seed=seed_mptsa)

        try:
            _positions, Ns, Rgs, grid, origin = generate_mptsa_lattice_2d(mptsa_params)
            Df_est, slope = estimate_fractal_dimension_2d(Ns, Rgs)
        except Exception as e:
            dump = {
                "error": str(e),
                "stage": "mptsa",
                "Np": Np,
                "frac_A": frac_A,
                "Df_target": target_Df,
                "seed_mptsa": seed_mptsa,
                "mptsa_try": mptsa_tries,
                "grid_shape": None,
            }
            np.save(
                f"error_dump_{Np}_{frac_A}_{seed_mptsa}_mptsa.npy",
                dump,
                allow_pickle=True,
            )
            continue

        if (not np.isfinite(Df_est)) or abs(float(Df_est) - float(target_Df)) > float(df_tol):
            continue

        geometry_probe_MAS = float("nan")
        geometry_probe_strategy = ""
        geometry_prescreened = False
        if (
            GEOMETRY_PRESCREEN_ENABLED
            and target_MAS < GEOMETRY_PRESCREEN_MAS_THRESHOLD
            and 0.0 < frac_A < 1.0
        ):
            geometry_prescreened = True
            try:
                probe_params = _make_mix_params(
                    frac_A=frac_A,
                    target_MAS=target_MAS,
                    seed=seed_mptsa,
                )
                _probe_labels, probe_stats = probe_low_mas_geometry(
                    grid,
                    probe_params,
                    phys_params,
                    seed=seed_mptsa,
                )
                geometry_probe_MAS = float(probe_stats.get("MAS", np.nan))
                geometry_probe_strategy = str(probe_stats.get("strategy", ""))
            except Exception as e:
                dump = {
                    "error": str(e),
                    "stage": "geometry_prescreen",
                    "Np": Np,
                    "frac_A": frac_A,
                    "Df_target": target_Df,
                    "seed_mptsa": seed_mptsa,
                    "mptsa_try": mptsa_tries,
                    "grid_shape": grid.shape,
                }
                np.save(
                    f"error_dump_{Np}_{frac_A}_{seed_mptsa}_prescreen.npy",
                    dump,
                    allow_pickle=True,
                )
                continue

            if (
                (not np.isfinite(geometry_probe_MAS))
                or geometry_probe_MAS > target_MAS + mas_tol
            ):
                continue

        for mix_try in range(mix_tries_per_mptsa):
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
                    lambda_min=lambda_min,
                    lambda_max=lambda_max,
                    sweeps_per_eval=sweeps_per_eval,
                    max_bisect=max_bisect,
                    temperature=temperature,
                )
                labels, stats = assign_materials_with_target_mas(
                    grid, mix_params, phys_params
                )
            except Exception as e:
                dump = {
                    "error": str(e),
                    "stage": "material_mix",
                    "Np": Np,
                    "frac_A": frac_A,
                    "Df_target": target_Df,
                    "seed_mptsa": seed_mptsa,
                    "seed_mix": seed_mix,
                    "mptsa_try": mptsa_tries,
                    "mix_try": mix_try + 1,
                    "grid_shape": grid.shape,
                    "lambda_min": lambda_min,
                    "lambda_max": lambda_max,
                    "sweeps_per_eval": sweeps_per_eval,
                    "max_bisect": max_bisect,
                    "temperature": temperature,
                }
                np.save(
                    f"error_dump_{Np}_{frac_A}_{seed_mptsa}_{seed_mix}.npy",
                    dump,
                    allow_pickle=True,
                )
                continue

            MAS_actual = float(stats.get("MAS", np.nan))
            if 0.0 < frac_A < 1.0:
                if (
                    (not np.isfinite(MAS_actual))
                    or (abs(MAS_actual - target_MAS) > mas_tol)
                ):
                    continue

            frac_A_actual = _compute_actual_frac_A(labels)

            M, Hbond, Vbond, meta, bond_counts = grid_factory.make_from_array(
                labels,
                a_code=0,
                b_code=1,
                empty_code=-1,
                A0=A0_CELL_AREA,
                int_bre=INT_BRE,
            )

            return {
                "Np_target": int(Np),
                "frac_A_target": float(frac_A),
                "Df_target": float(target_Df),
                "Df_est": float(Df_est),
                "MAS_target": float(target_MAS),
                "MAS_actual": float(MAS_actual),
                "frac_A_actual": float(frac_A_actual),
                "slope": float(slope),
                "labels": labels.astype(np.int8, copy=False),
                "M": M,
                "Hbond": Hbond,
                "Vbond": Vbond,
                "meta": asdict(meta),
                "origin": np.array(origin, dtype=np.int32),
                "bond_counts": bond_counts,
                "seed_mptsa": int(seed_mptsa),
                "seed_mix": int(seed_mix),
                "mptsa_try": int(mptsa_tries),
                "mix_try": int(mix_try + 1),
                "mix_tries_per_mptsa": int(mix_tries_per_mptsa),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "lambda_used": float(stats.get("lambda_used", np.nan)),
                "sweeps_per_eval": int(sweeps_per_eval),
                "max_bisect": int(max_bisect),
                "temperature": float(temperature),
                "geometry_prescreened": bool(geometry_prescreened),
                "geometry_probe_MAS": float(geometry_probe_MAS),
                "geometry_probe_strategy": str(geometry_probe_strategy),
            }

    return None



def _write_npz_atomic(npz_path: Path, sample: Dict[str, Any], compressed: bool) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = npz_path.with_name(f"{npz_path.stem}.tmp_{os.getpid()}.npz")
    save_npz = np.savez_compressed if compressed else np.savez
    try:
        save_npz(
            tmp_path,
            M=np.asarray(sample["M"]),
            Hbond=np.asarray(sample["Hbond"]),
            Vbond=np.asarray(sample["Vbond"]),
            labels=np.asarray(sample["labels"]),
            origin=np.asarray(sample["origin"]),
        )
        os.replace(str(tmp_path), str(npz_path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass



def _generate_and_store_one_sample(
    output_dir: str,
    group_name: str,
    sample_index: int,
    Np: int,
    frac_A: float,
    target_Df: float,
    df_tol: float,
    target_MAS: float,
    mas_tol: float,
    max_tries_factor: int,
    task_seed: int,
    compressed: bool,
) -> Dict[str, Any]:
    sample_name = _sample_name(sample_index)
    sample = _generate_one_sample(
        Np=Np,
        frac_A=frac_A,
        target_Df=target_Df,
        df_tol=df_tol,
        target_MAS=target_MAS,
        mas_tol=mas_tol,
        max_tries_factor=max_tries_factor,
        base_seed=task_seed,
    )
    if sample is None:
        return {
            "success": False,
            "group_name": group_name,
            "sample_name": sample_name,
            "sample_index": int(sample_index),
        }

    output_root = Path(output_dir)
    npz_path = output_root / SAMPLES_SUBDIR / group_name / f"{sample_name}.npz"
    _write_npz_atomic(npz_path, sample, compressed=compressed)

    sample_attrs = {
        "Np_target": sample["Np_target"],
        "frac_A_target": sample["frac_A_target"],
        "Df_target": sample["Df_target"],
        "Df_est": sample["Df_est"],
        "MAS_target": sample["MAS_target"],
        "MAS_actual": sample["MAS_actual"],
        "frac_A_actual": sample["frac_A_actual"],
        "slope": sample["slope"],
        "seed_mptsa": sample["seed_mptsa"],
        "seed_mix": sample["seed_mix"],
        "mptsa_try": sample["mptsa_try"],
        "mix_try": sample["mix_try"],
        "mix_tries_per_mptsa": sample["mix_tries_per_mptsa"],
        "lambda_min": sample["lambda_min"],
        "lambda_max": sample["lambda_max"],
        "lambda_used": sample["lambda_used"],
        "sweeps_per_eval": sample["sweeps_per_eval"],
        "max_bisect": sample["max_bisect"],
        "temperature": sample["temperature"],
        "geometry_prescreened": sample["geometry_prescreened"],
        "geometry_probe_MAS": sample["geometry_probe_MAS"],
        "geometry_probe_strategy": sample["geometry_probe_strategy"],
        "meta": sample["meta"],
        "bond_counts": {str(key): int(val) for key, val in sample["bond_counts"].items()},
    }

    return {
        "success": True,
        "group_name": group_name,
        "sample_name": sample_name,
        "sample_index": int(sample_index),
        "npz_relpath": npz_path.relative_to(output_root).as_posix(),
        "sample_attrs": sample_attrs,
    }



def _make_group_states(conn: sqlite3.Connection, param_pairs: List[Tuple[int, float]]) -> Dict[Tuple[int, float], Dict[str, Any]]:
    states: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for Np, frac_A in param_pairs:
        group_name = _sanitize_group_name(Np, frac_A)
        existing_count, max_index, next_attempt_index = _get_group_state(conn, group_name)
        states[(Np, frac_A)] = {
            "Np": int(Np),
            "frac_A": float(frac_A),
            "group_name": group_name,
            "done": int(existing_count),
            "target_total": int(SAMPLES_PER_PARAM),
            "missing": max(0, int(SAMPLES_PER_PARAM) - int(existing_count)),
            "launched": 0,
            "inflight": 0,
            "next_sample_index": int(max_index) + 1,
            "next_attempt_index": int(next_attempt_index),
        }
    return states



def _can_submit_group(state: Dict[str, Any]) -> bool:
    return bool(
        state["launched"] < state["missing"]
        and (state["done"] + state["inflight"]) < state["target_total"]
    )



def _start_round(states: Dict[Tuple[int, float], Dict[str, Any]]) -> List[Dict[str, Any]]:
    todo_states: List[Dict[str, Any]] = []
    for state in states.values():
        state["missing"] = max(0, int(state["target_total"]) - int(state["done"]))
        state["launched"] = 0
        state["inflight"] = 0
        if state["missing"] > 0:
            todo_states.append(state)
    return todo_states



def _reserve_one_task(conn: sqlite3.Connection, state: Dict[str, Any]) -> Dict[str, Any]:
    sample_index = int(state["next_sample_index"])
    attempt_index = int(state["next_attempt_index"])
    task_seed = _task_seed(MASTER_SEED, int(state["Np"]), float(state["frac_A"]), attempt_index)

    state["launched"] += 1
    state["inflight"] += 1
    state["next_sample_index"] += 1
    state["next_attempt_index"] += 1

    _upsert_group(
        conn,
        str(state["group_name"]),
        int(state["Np"]),
        float(state["frac_A"]),
        int(state["done"]),
        int(state["next_attempt_index"]),
    )
    conn.commit()

    return {
        "group_name": str(state["group_name"]),
        "sample_index": sample_index,
        "Np": int(state["Np"]),
        "frac_A": float(state["frac_A"]),
        "task_seed": task_seed,
    }



def _submit_one_task(
    ex: ProcessPoolExecutor,
    output_dir: Path,
    payload: Dict[str, Any],
) -> Any:
    return ex.submit(
        _generate_and_store_one_sample,
        str(output_dir),
        str(payload["group_name"]),
        int(payload["sample_index"]),
        int(payload["Np"]),
        float(payload["frac_A"]),
        TARGET_DF,
        DF_TOL,
        TARGET_MAS,
        MAS_TOL,
        MAX_TRIES_FACTOR,
        int(payload["task_seed"]),
        SAVE_COMPRESSED,
    )



def build_pool() -> None:
    output_dir = Path(OUTPUT_POOL_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SAMPLES_SUBDIR).mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / SQLITE_NAME

    param_pairs: List[Tuple[int, float]] = [
        (Np, frac_A) for Np in NP_LIST for frac_A in FRAC_A_LIST
    ]

    print(
        f"[POOL] Target pairs: {len(param_pairs)} (Np, frac_A) combinations, "
        f"{SAMPLES_PER_PARAM} accepted samples per pair."
    )
    print(
        f"[POOL] Fixed targets: Df={TARGET_DF}, MAS={TARGET_MAS}, "
        f"DF_TOL={DF_TOL}, MAS_TOL={MAS_TOL}"
    )
    print(
        f"[POOL] Mixing retries: {MIX_TRIES_PER_MPTSA} mix attempts per MPTSA grid, "
        f"{len(MIX_LAMBDA_SCHEDULES)} lambda schedules"
    )
    print(
        f"[POOL] Geometry prescreen: enabled={GEOMETRY_PRESCREEN_ENABLED}, "
        f"active for MAS<{GEOMETRY_PRESCREEN_MAS_THRESHOLD}, "
        "reject if probe_MAS > target + MAS_TOL"
    )
    print(f"[POOL] NPZ+SQLite output: {output_dir}")
    print(f"[POOL] workers={WORKERS}")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        _init_db(conn)
        _set_meta_if_missing(conn, "format", "npz_sqlite_single")
        _set_meta_if_missing(conn, "source_builder", "aggregates_sampler_npz_sqlite")
        _set_meta_if_missing(conn, "datasets", ["M", "Hbond", "Vbond", "labels", "origin"])
        _set_meta_if_missing(conn, "TARGET_DF", TARGET_DF)
        _set_meta_if_missing(conn, "TARGET_MAS", TARGET_MAS)
        _set_meta_if_missing(conn, "DF_TOL", DF_TOL)
        _set_meta_if_missing(conn, "MAS_TOL", MAS_TOL)
        _set_meta_if_missing(conn, "MIX_TRIES_PER_MPTSA", MIX_TRIES_PER_MPTSA)
        _set_meta_if_missing(conn, "MIX_LAMBDA_SCHEDULES", MIX_LAMBDA_SCHEDULES)
        _set_meta_if_missing(conn, "GEOMETRY_PRESCREEN_ENABLED", GEOMETRY_PRESCREEN_ENABLED)
        _set_meta_if_missing(conn, "GEOMETRY_PRESCREEN_MAS_THRESHOLD", GEOMETRY_PRESCREEN_MAS_THRESHOLD)
        _set_meta_if_missing(conn, "A0_CELL_AREA", A0_CELL_AREA)
        _set_meta_if_missing(conn, "INT_BRE", INT_BRE)
        _set_meta_if_missing(conn, "MASTER_SEED", MASTER_SEED)
        conn.commit()

        states = _make_group_states(conn, param_pairs)
        round_index = 0

        if WORKERS <= 1:
            while True:
                todo_states = _start_round(states)
                if not todo_states:
                    if round_index == 0:
                        print("[POOL] All (Np, frac_A) pairs already have enough samples. Nothing to do.")
                    else:
                        print(f"[POOL] All parameter pairs reached target after {round_index} round(s).")
                    break

                round_index += 1
                print(
                    f"[POOL] Round {round_index}: {len(todo_states)} parameter pairs need more samples "
                    f"(up to {sum(int(state['missing']) for state in todo_states)} single-sample tasks)."
                )

                for state in todo_states:
                    print(
                        f"[POOL] (serial, round {round_index}) Np={state['Np']}, frac_A={state['frac_A']:.4f}: "
                        f"{state['done']} existing, targeting up to {state['missing']} more..."
                    )
                    while _can_submit_group(state):
                        payload = _reserve_one_task(conn, state)
                        result = _generate_and_store_one_sample(
                            str(output_dir),
                            str(payload["group_name"]),
                            int(payload["sample_index"]),
                            int(payload["Np"]),
                            float(payload["frac_A"]),
                            TARGET_DF,
                            DF_TOL,
                            TARGET_MAS,
                            MAS_TOL,
                            MAX_TRIES_FACTOR,
                            int(payload["task_seed"]),
                            SAVE_COMPRESSED,
                        )
                        state["inflight"] -= 1
                        if bool(result["success"]):
                            _register_sample(
                                conn,
                                str(result["group_name"]),
                                str(result["sample_name"]),
                                int(result["sample_index"]),
                                str(result["npz_relpath"]),
                                dict(result["sample_attrs"]),
                            )
                            state["done"] += 1
                            _upsert_group(
                                conn,
                                str(state["group_name"]),
                                int(state["Np"]),
                                float(state["frac_A"]),
                                int(state["done"]),
                                int(state["next_attempt_index"]),
                            )
                            conn.commit()
                            print(
                                f"[POOL] Np={state['Np']}, frac_A={state['frac_A']:.4f}: "
                                f"accepted sample {result['sample_name']} ({state['done']}/{state['target_total']})"
                            )
                        else:
                            print(
                                f"[POOL][WARN] Np={state['Np']}, frac_A={state['frac_A']:.4f}: "
                                f"task for {result['sample_name']} found no accepted sample within budget."
                            )

                remaining = sum(1 for state in states.values() if int(state["done"]) < int(state["target_total"]))
                if remaining > 0:
                    print(f"[POOL] Round {round_index} completed; {remaining} parameter pairs still need samples.")

        else:
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                while True:
                    todo_states = _start_round(states)
                    if not todo_states:
                        if round_index == 0:
                            print("[POOL] All (Np, frac_A) pairs already have enough samples. Nothing to do.")
                        else:
                            print(f"[POOL] All parameter pairs reached target after {round_index} round(s).")
                        break

                    round_index += 1
                    print(
                        f"[POOL] Round {round_index}: {len(todo_states)} parameter pairs need more samples "
                        f"(up to {sum(int(state['missing']) for state in todo_states)} single-sample tasks)."
                    )

                    state_by_key = {(int(state["Np"]), float(state["frac_A"])): state for state in todo_states}
                    pending: Dict[Any, Tuple[int, float]] = {}
                    round_robin = list(state_by_key.keys())
                    rr_index = 0

                    def refill() -> None:
                        nonlocal rr_index
                        if not round_robin:
                            return
                        while len(pending) < int(WORKERS):
                            submitted = False
                            for _ in range(len(round_robin)):
                                key = round_robin[rr_index % len(round_robin)]
                                rr_index += 1
                                state = state_by_key[key]
                                if _can_submit_group(state):
                                    payload = _reserve_one_task(conn, state)
                                    future = _submit_one_task(ex, output_dir, payload)
                                    pending[future] = key
                                    submitted = True
                                    break
                            if not submitted:
                                break

                    refill()
                    while pending:
                        done_set, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                        for fut in done_set:
                            key = pending.pop(fut)
                            state = state_by_key[key]
                            state["inflight"] -= 1
                            try:
                                result = fut.result()
                            except Exception as e:
                                print(
                                    f"[POOL][ERROR] Np={state['Np']}, frac_A={state['frac_A']:.4f}: worker failed: {e}"
                                )
                                result = None

                            if result is not None and bool(result["success"]):
                                _register_sample(
                                    conn,
                                    str(result["group_name"]),
                                    str(result["sample_name"]),
                                    int(result["sample_index"]),
                                    str(result["npz_relpath"]),
                                    dict(result["sample_attrs"]),
                                )
                                state["done"] += 1
                                _upsert_group(
                                    conn,
                                    str(state["group_name"]),
                                    int(state["Np"]),
                                    float(state["frac_A"]),
                                    int(state["done"]),
                                    int(state["next_attempt_index"]),
                                )
                                conn.commit()
                                print(
                                    f"[POOL] Np={state['Np']}, frac_A={state['frac_A']:.4f}: "
                                    f"accepted sample {result['sample_name']} ({state['done']}/{state['target_total']})"
                                )
                            else:
                                sample_label = "unknown"
                                if result is not None:
                                    sample_label = str(result.get("sample_name", sample_label))
                                print(
                                    f"[POOL][WARN] Np={state['Np']}, frac_A={state['frac_A']:.4f}: "
                                    f"task for {sample_label} found no accepted sample within budget."
                                )

                        refill()

                    remaining = sum(1 for state in states.values() if int(state["done"]) < int(state["target_total"]))
                    if remaining > 0:
                        print(f"[POOL] Round {round_index} completed; {remaining} parameter pairs still need samples.")

    finally:
        conn.close()

    print("[POOL] Pool construction / extension completed.")


if __name__ == "__main__":
    build_pool()
