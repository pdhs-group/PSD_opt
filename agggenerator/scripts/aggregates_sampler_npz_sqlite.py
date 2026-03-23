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



def _task_seed(master_seed: int, Np: int, frac_A: float, sample_index: int) -> int:
    frac_key = int(round(float(frac_A) * 10000.0))
    seq = np.random.SeedSequence([int(master_seed), int(Np), frac_key, int(sample_index)])
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
) -> MaterialMixParams:
    window, stride = _compute_mix_window_stride(grid)

    return MaterialMixParams(
        frac_A=float(frac_A),
        target_MAS=float(target_MAS),
        tol_MAS=0.05,
        window=window,
        stride=stride,
        sweeps_per_eval=8,
        max_bisect=20,
        seed=int(seed),
    )



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
            attrs_json TEXT NOT NULL
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



def _get_group_state(conn: sqlite3.Connection, group_name: str) -> Tuple[int, int]:
    row = conn.execute(
        "SELECT n_samples FROM groups WHERE group_name = ?",
        (group_name,),
    ).fetchone()
    n_existing = int(row[0]) if row is not None else 0

    row_max = conn.execute(
        "SELECT MAX(sample_index) FROM samples WHERE group_name = ?",
        (group_name,),
    ).fetchone()
    max_index = int(row_max[0]) if row_max is not None and row_max[0] is not None else -1
    return n_existing, max_index



def _upsert_group(
    conn: sqlite3.Connection,
    group_name: str,
    Np: int,
    frac_A: float,
    n_samples: int,
) -> None:
    attrs_json = _dumps_json(
        {
            "Np_target": int(Np),
            "frac_A_target": float(frac_A),
            "n_samples": int(n_samples),
        }
    )
    conn.execute(
        """
        INSERT INTO groups(group_name, np_target, frac_a_target, n_samples, attrs_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(group_name) DO UPDATE SET
            np_target = excluded.np_target,
            frac_a_target = excluded.frac_a_target,
            n_samples = excluded.n_samples,
            attrs_json = excluded.attrs_json
        """,
        (group_name, float(Np), float(frac_A), int(n_samples), attrs_json),
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
    target_MAS: float,
    mas_tol: float,
    max_tries_factor: int,
    base_seed: int,
) -> Dict[str, Any] | None:
    rng = np.random.default_rng(base_seed)
    grid_factory = GridFactory()
    phys_params = MASPhysicalParams()
    max_tries = int(max(1, max_tries_factor))

    tries = 0
    while tries < max_tries:
        tries += 1
        seed_mptsa = int(rng.integers(0, 2**31 - 1))
        seed_mix = int(rng.integers(0, 2**31 - 1))

        mptsa_params = _make_mptsa_params(Np=Np, Df=target_Df, seed=seed_mptsa)

        try:
            _positions, Ns, Rgs, grid, origin = generate_mptsa_lattice_2d(mptsa_params)
            Df_est, slope = estimate_fractal_dimension_2d(Ns, Rgs)

            mix_params = _make_mix_params(
                frac_A=frac_A,
                target_MAS=target_MAS,
                seed=seed_mix,
                grid=grid,
            )
            labels, stats = assign_materials_with_target_mas(grid, mix_params, phys_params)
        except Exception as e:
            dump = {
                "error": str(e),
                "Np": Np,
                "frac_A": frac_A,
                "Df_target": target_Df,
                "seed_mptsa": seed_mptsa,
                "seed_mix": seed_mix,
                "grid_shape": None if "grid" not in locals() else grid.shape,
            }
            np.save(
                f"error_dump_{Np}_{frac_A}_{seed_mptsa}_{seed_mix}.npy",
                dump,
                allow_pickle=True,
            )
            continue

        MAS_actual = float(stats.get("MAS", np.nan))
        if 0.0 < frac_A < 1.0:
            if (not np.isfinite(MAS_actual)) or (abs(MAS_actual - target_MAS) > mas_tol):
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
        existing_count, max_index = _get_group_state(conn, group_name)
        states[(Np, frac_A)] = {
            "Np": int(Np),
            "frac_A": float(frac_A),
            "group_name": group_name,
            "done": int(existing_count),
            "target_total": int(SAMPLES_PER_PARAM),
            "missing": max(0, int(SAMPLES_PER_PARAM) - int(existing_count)),
            "launched": 0,
            "inflight": 0,
            "next_index": int(max_index) + 1,
        }
    return states



def _can_submit_group(state: Dict[str, Any]) -> bool:
    return bool(
        state["launched"] < state["missing"]
        and (state["done"] + state["inflight"]) < state["target_total"]
    )



def _submit_one_task(
    ex: ProcessPoolExecutor,
    output_dir: Path,
    state: Dict[str, Any],
) -> Any:
    sample_index = int(state["next_index"])
    task_seed = _task_seed(MASTER_SEED, int(state["Np"]), float(state["frac_A"]), sample_index)
    future = ex.submit(
        _generate_and_store_one_sample,
        str(output_dir),
        str(state["group_name"]),
        sample_index,
        int(state["Np"]),
        float(state["frac_A"]),
        TARGET_DF,
        TARGET_MAS,
        MAS_TOL,
        MAX_TRIES_FACTOR,
        task_seed,
        SAVE_COMPRESSED,
    )
    state["launched"] += 1
    state["inflight"] += 1
    state["next_index"] += 1
    return future



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
        _set_meta_if_missing(conn, "A0_CELL_AREA", A0_CELL_AREA)
        _set_meta_if_missing(conn, "INT_BRE", INT_BRE)
        _set_meta_if_missing(conn, "MASTER_SEED", MASTER_SEED)
        conn.commit()

        states = _make_group_states(conn, param_pairs)
        todo_states = [state for state in states.values() if state["missing"] > 0]

        if not todo_states:
            print("[POOL] All (Np, frac_A) pairs already have enough samples. Nothing to do.")
            return

        print(
            f"[POOL] {len(todo_states)} parameter pairs need more samples "
            f"(SAMPLES_PER_PARAM={SAMPLES_PER_PARAM})."
        )

        if WORKERS <= 1:
            for state in todo_states:
                print(
                    f"[POOL] (serial) Np={state['Np']}, frac_A={state['frac_A']:.4f}: "
                    f"{state['done']} existing, targeting {state['missing']} more..."
                )
                while _can_submit_group(state):
                    sample_index = int(state["next_index"])
                    task_seed = _task_seed(MASTER_SEED, int(state["Np"]), float(state["frac_A"]), sample_index)
                    state["launched"] += 1
                    state["next_index"] += 1
                    result = _generate_and_store_one_sample(
                        str(output_dir),
                        str(state["group_name"]),
                        sample_index,
                        int(state["Np"]),
                        float(state["frac_A"]),
                        TARGET_DF,
                        TARGET_MAS,
                        MAS_TOL,
                        MAX_TRIES_FACTOR,
                        task_seed,
                        SAVE_COMPRESSED,
                    )
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
                        _upsert_group(conn, str(state["group_name"]), int(state["Np"]), float(state["frac_A"]), int(state["done"]))
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

        else:
            state_by_key = {(int(state["Np"]), float(state["frac_A"])): state for state in todo_states}
            pending: Dict[Any, Tuple[int, float]] = {}
            round_robin = list(state_by_key.keys())
            rr_index = 0

            def refill(executor: ProcessPoolExecutor) -> None:
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
                            future = _submit_one_task(executor, output_dir, state)
                            pending[future] = key
                            submitted = True
                            break
                    if not submitted:
                        break

            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                refill(ex)
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
                            _upsert_group(conn, str(state["group_name"]), int(state['Np']), float(state['frac_A']), int(state['done']))
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

                    refill(ex)

        for state in todo_states:
            if state["done"] < state["target_total"]:
                print(
                    f"[POOL][WARN] Np={state['Np']}, frac_A={state['frac_A']:.4f}: "
                    f"finished with {state['done']}/{state['target_total']} accepted samples."
                )

    finally:
        conn.close()

    print("[POOL] Pool construction / extension completed.")


if __name__ == "__main__":
    build_pool()

