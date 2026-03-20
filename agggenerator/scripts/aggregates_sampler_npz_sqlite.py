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
     generated, and the max-tries budget is scaled to that missing count.
3. Each accepted sample is stored in its own NPZ file, while SQLite stores the
   pool metadata, parameter-group index, and sample lookup information.
"""

from __future__ import annotations

import json
# import math
import os
# import shutil
import sqlite3
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from agggenerator.mptsa2d import (
    MPTSALatticeParams2D,
    generate_mptsa_lattice_2d,
    estimate_fractal_dimension_2d,
)
from agggenerator.material_mix import (
    MaterialMixParams,
    MASPhysicalParams,
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

SAVE_COMPRESSED: bool = True
SQLITE_NAME: str = "pool_index.sqlite"
SAMPLES_SUBDIR: str = "samples"


# ============================================================
# Internal helpers
# ============================================================

def _sanitize_group_name(Np: int, frac_A: float) -> str:
    xa_int = int(round(frac_A * 10000))
    return f"Np{Np}_XA{xa_int:04d}"



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
    base_seed: int,
) -> None:
    attrs_json = _dumps_json(
        {
            "Np_target": int(Np),
            "frac_A_target": float(frac_A),
            "base_seed": int(base_seed),
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



def _write_samples_to_npz_sqlite(
    conn: sqlite3.Connection,
    output_dir: Path,
    group_name: str,
    samples: List[Dict[str, Any]],
    start_index: int,
    compressed: bool,
) -> None:
    save_npz = np.savez_compressed if compressed else np.savez
    group_dir = output_dir / SAMPLES_SUBDIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, s in enumerate(samples):
        idx = start_index + local_idx
        sample_name = f"sample_{idx:04d}"
        npz_path = group_dir / f"{sample_name}.npz"
        save_npz(
            npz_path,
            M=np.asarray(s["M"]),
            Hbond=np.asarray(s["Hbond"]),
            Vbond=np.asarray(s["Vbond"]),
            labels=np.asarray(s["labels"]),
            origin=np.asarray(s["origin"]),
        )

        relpath = npz_path.relative_to(output_dir).as_posix()
        sample_attrs = {
            "Np_target": s["Np_target"],
            "frac_A_target": s["frac_A_target"],
            "Df_target": s["Df_target"],
            "Df_est": s["Df_est"],
            "MAS_target": s["MAS_target"],
            "MAS_actual": s["MAS_actual"],
            "frac_A_actual": s["frac_A_actual"],
            "slope": s["slope"],
            "seed_mptsa": s["seed_mptsa"],
            "seed_mix": s["seed_mix"],
            "meta": s["meta"],
            "bond_counts": {str(key): int(val) for key, val in s["bond_counts"].items()},
        }

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
                int(idx),
                relpath,
                "",
                _dumps_json(sample_attrs),
            ),
        )



def _generate_samples_for_param(
    Np: int,
    frac_A: float,
    target_Df: float,
    target_MAS: float,
    df_tol: float,
    mas_tol: float,
    n_samples: int,
    max_tries_factor: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(base_seed)
    grid_factory = GridFactory()
    phys_params = MASPhysicalParams()

    accepted: List[Dict[str, Any]] = []

    max_tries = int(max(1, n_samples * max_tries_factor))
    tries = 0

    while len(accepted) < n_samples and tries < max_tries:
        tries += 1
        seed_mptsa = int(rng.integers(0, 2**31 - 1))
        seed_mix = int(rng.integers(0, 2**31 - 1))

        mptsa_params = _make_mptsa_params(Np=Np, Df=target_Df, seed=seed_mptsa)

        try:
            positions, Ns, Rgs, grid, origin = generate_mptsa_lattice_2d(mptsa_params)
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
        else:
            MAS_actual = float(MAS_actual)

        frac_A_actual = _compute_actual_frac_A(labels)

        M, Hbond, Vbond, meta, bond_counts = grid_factory.make_from_array(
            labels,
            a_code=0,
            b_code=1,
            empty_code=-1,
            A0=A0_CELL_AREA,
            int_bre=INT_BRE,
        )

        meta_dict = asdict(meta)

        sample = dict(
            Np_target=int(Np),
            frac_A_target=float(frac_A),
            Df_target=float(target_Df),
            Df_est=float(Df_est),
            MAS_target=float(target_MAS),
            MAS_actual=float(MAS_actual),
            frac_A_actual=float(frac_A_actual),
            slope=float(slope),
            labels=labels.astype(np.int8, copy=False),
            M=M,
            Hbond=Hbond,
            Vbond=Vbond,
            meta=meta_dict,
            origin=np.array(origin, dtype=np.int32),
            bond_counts=bond_counts,
            seed_mptsa=int(seed_mptsa),
            seed_mix=int(seed_mix),
        )
        accepted.append(sample)

    return accepted


# ============================================================
# Top-level driver
# ============================================================

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

    rng_master = np.random.default_rng(MASTER_SEED)
    param_seeds = {
        (Np, frac_A): int(rng_master.integers(0, 2**31 - 1))
        for (Np, frac_A) in param_pairs
    }

    conn = sqlite3.connect(str(sqlite_path))
    try:
        _init_db(conn)
        _set_meta_if_missing(conn, "format", "npz_sqlite_single")
        _set_meta_if_missing(conn, "source_builder", "aggregates_sampler_npz_sqlite_single")
        _set_meta_if_missing(conn, "datasets", ["M", "Hbond", "Vbond", "labels", "origin"])
        _set_meta_if_missing(conn, "TARGET_DF", TARGET_DF)
        _set_meta_if_missing(conn, "TARGET_MAS", TARGET_MAS)
        _set_meta_if_missing(conn, "DF_TOL", DF_TOL)
        _set_meta_if_missing(conn, "MAS_TOL", MAS_TOL)
        _set_meta_if_missing(conn, "A0_CELL_AREA", A0_CELL_AREA)
        _set_meta_if_missing(conn, "INT_BRE", INT_BRE)
        _set_meta_if_missing(conn, "MASTER_SEED", MASTER_SEED)
        conn.commit()

        existing_counts: Dict[Tuple[int, float], int] = {}
        existing_max_index: Dict[Tuple[int, float], int] = {}
        for (Np, frac_A) in param_pairs:
            gname = _sanitize_group_name(Np, frac_A)
            n_existing, max_index = _get_group_state(conn, gname)
            existing_counts[(Np, frac_A)] = n_existing
            existing_max_index[(Np, frac_A)] = max_index

        todo_pairs: List[Tuple[int, float, int]] = []
        for (Np, frac_A) in param_pairs:
            already = existing_counts[(Np, frac_A)]
            need = max(0, SAMPLES_PER_PARAM - already)
            if need > 0:
                todo_pairs.append((Np, frac_A, need))

        if not todo_pairs:
            print("[POOL] All (Np, frac_A) pairs already have enough samples. Nothing to do.")
            return

        print(
            f"[POOL] {len(todo_pairs)} parameter pairs need more samples "
            f"(SAMPLES_PER_PARAM={SAMPLES_PER_PARAM})."
        )

        if WORKERS <= 1:
            for (Np, frac_A, need) in todo_pairs:
                already = existing_counts[(Np, frac_A)]
                print(
                    f"[POOL] (serial) Np={Np}, frac_A={frac_A:.4f}: "
                    f"{already} existing, generating {need} more..."
                )
                samples = _generate_samples_for_param(
                    Np=Np,
                    frac_A=frac_A,
                    target_Df=TARGET_DF,
                    target_MAS=TARGET_MAS,
                    df_tol=DF_TOL,
                    mas_tol=MAS_TOL,
                    n_samples=need,
                    max_tries_factor=MAX_TRIES_FACTOR,
                    base_seed=param_seeds[(Np, frac_A)],
                )
                print(
                    f"[POOL] (Np={Np}, frac_A={frac_A:.4f}) "
                    f"newly accepted samples: {len(samples)}"
                )

                gname = _sanitize_group_name(Np, frac_A)
                start_idx = existing_max_index[(Np, frac_A)] + 1
                _write_samples_to_npz_sqlite(
                    conn,
                    output_dir,
                    gname,
                    samples,
                    start_index=start_idx,
                    compressed=SAVE_COMPRESSED,
                )

                total_count = already + len(samples)
                _upsert_group(conn, gname, Np, frac_A, total_count, param_seeds[(Np, frac_A)])
                conn.commit()

                existing_counts[(Np, frac_A)] = total_count
                existing_max_index[(Np, frac_A)] = start_idx + len(samples) - 1 if samples else existing_max_index[(Np, frac_A)]

        else:
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                future_to_param: Dict[Any, Tuple[int, float, int]] = {}
                for (Np, frac_A, need) in todo_pairs:
                    seed = param_seeds[(Np, frac_A)]
                    fut = ex.submit(
                        _generate_samples_for_param,
                        Np,
                        frac_A,
                        TARGET_DF,
                        TARGET_MAS,
                        DF_TOL,
                        MAS_TOL,
                        need,
                        MAX_TRIES_FACTOR,
                        seed,
                    )
                    future_to_param[fut] = (Np, frac_A, need)

                for fut in as_completed(future_to_param):
                    Np, frac_A, need = future_to_param[fut]
                    already = existing_counts[(Np, frac_A)]
                    try:
                        samples = fut.result()
                    except Exception as e:
                        print(
                            f"[POOL][ERROR] Np={Np}, frac_A={frac_A:.4f} generation failed: {e}"
                        )
                        samples = []

                    print(
                        f"[POOL] (Np={Np}, frac_A={frac_A:.4f}) "
                        f"existing={already}, newly accepted={len(samples)}"
                    )

                    gname = _sanitize_group_name(Np, frac_A)
                    start_idx = existing_max_index[(Np, frac_A)] + 1
                    _write_samples_to_npz_sqlite(
                        conn,
                        output_dir,
                        gname,
                        samples,
                        start_index=start_idx,
                        compressed=SAVE_COMPRESSED,
                    )

                    total_count = already + len(samples)
                    _upsert_group(conn, gname, Np, frac_A, total_count, param_seeds[(Np, frac_A)])
                    conn.commit()

                    existing_counts[(Np, frac_A)] = total_count
                    existing_max_index[(Np, frac_A)] = start_idx + len(samples) - 1 if samples else existing_max_index[(Np, frac_A)]

    finally:
        conn.close()

    print("[POOL] Pool construction / extension completed.")


if __name__ == "__main__":
    build_pool()
