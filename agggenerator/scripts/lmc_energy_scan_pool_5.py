# -*- coding: utf-8 -*-
"""
Energy-volume scan using LMC + aggregate pool.

This version supports parameter-grid scans over:
- NO_FRAG
- int_bre
- gamma
- MAS
- X1
- STR

Each run is stored in HDF5 with a unique key and full parameter metadata.
The scan also maintains a SQLite checkpoint database so subtask results can be
written incrementally and resumed after interruptions.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np

from lmc import LMCSimulator
from aggregates_sampler_npz_sqlite import NP_LIST


def _ensure_sequence(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _coerce_str_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"STR must be a 1D array-like, got shape={arr.shape}")
    return arr.copy()


def _format_float_for_key(value: float) -> str:
    return np.format_float_positional(
        float(value), precision=6, unique=False, fractional=False, trim="-"
    ).replace(".", "p").replace("-", "m")


def _format_str_for_key(str_values: np.ndarray) -> str:
    return "x".join(_format_float_for_key(v) for v in _coerce_str_array(str_values))


def _status_db_path(h5_path: str) -> str:
    path = Path(h5_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_status.sqlite"))
    return str(path.with_name(f"{path.name}_status.sqlite"))


def _normalized_pool_dir(value: Any) -> str:
    return os.path.abspath(str(value))


def _make_scan_signature(params: Dict[str, Any], np_arr: np.ndarray) -> Tuple[Dict[str, Any], str, int]:
    payload = {
        "schema_version": 1,
        "NO_FRAG": int(params["NO_FRAG"]),
        "int_bre": float(params["int_bre"]),
        "gamma": float(params["gamma"]),
        "Df": float(params["Df"]),
        "MAS": float(params["MAS"]),
        "X1": float(params["X1"]),
        "STR": [float(v) for v in _coerce_str_array(params["STR"])],
        "A0": float(params["A0"]),
        "N_GRIDS": int(params["N_GRIDS"]),
        "N_FRACS": int(params["N_FRACS"]),
        "base_seed": int(params["base_seed"]),
        "np_list": [int(v) for v in np_arr.astype(int)],
    }
    signature_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature_crc32 = zlib.crc32(signature_json.encode("utf-8")) & 0xFFFFFFFF
    return payload, signature_json, int(signature_crc32)


def _read_attr_as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _read_attr_as_str(value.item())
    return str(value)


def _checkpoint_mismatch(grp_path: str, run_key: str, reason: str) -> None:
    raise ValueError(
        "[CHECKPOINT] Existing energy-scan checkpoint is incompatible with "
        "the current parameters.\n"
        f"  group   : {grp_path}\n"
        f"  run_key : {run_key}\n"
        f"  reason  : {reason}\n"
        "Use a new h5_path, delete the old HDF5 group/SQLite rows, or restore "
        "the original scan parameters."
    )


def _require_attr_int(grp: h5py.Group, name: str, expected: int, grp_path: str, run_key: str) -> None:
    if name not in grp.attrs:
        _checkpoint_mismatch(grp_path, run_key, f"missing HDF5 attr {name!r}")
    actual = int(grp.attrs[name])
    if actual != int(expected):
        _checkpoint_mismatch(grp_path, run_key, f"attr {name!r}: existing {actual!r}, current {int(expected)!r}")


def _require_attr_float(grp: h5py.Group, name: str, expected: float, grp_path: str, run_key: str) -> None:
    if name not in grp.attrs:
        _checkpoint_mismatch(grp_path, run_key, f"missing HDF5 attr {name!r}")
    actual = float(grp.attrs[name])
    if not np.isclose(actual, float(expected), rtol=1e-12, atol=1e-12):
        _checkpoint_mismatch(
            grp_path,
            run_key,
            f"attr {name!r}: existing {actual!r}, current {float(expected)!r}",
        )


def _require_attr_array(
    grp: h5py.Group,
    name: str,
    expected: np.ndarray,
    grp_path: str,
    run_key: str,
) -> None:
    if name not in grp.attrs:
        _checkpoint_mismatch(grp_path, run_key, f"missing HDF5 attr {name!r}")
    actual = np.asarray(grp.attrs[name], dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    if actual.shape != expected_arr.shape or not np.allclose(actual, expected_arr, rtol=1e-12, atol=1e-12):
        _checkpoint_mismatch(
            grp_path,
            run_key,
            f"attr {name!r}: existing {actual.tolist()!r}, current {expected_arr.tolist()!r}",
        )


def _validate_signature_or_legacy_attrs(
    grp: h5py.Group,
    *,
    grp_path: str,
    run_key: str,
    params: Dict[str, Any],
    signature_json: str,
    signature_crc32: int,
) -> None:
    has_signature_json = "scan_signature_json" in grp.attrs
    has_signature_crc = "scan_signature_crc32" in grp.attrs
    if has_signature_json or has_signature_crc:
        if has_signature_json:
            existing_json = _read_attr_as_str(grp.attrs["scan_signature_json"])
            if existing_json != signature_json:
                _checkpoint_mismatch(grp_path, run_key, "scan_signature_json differs")
        if has_signature_crc:
            existing_crc = int(grp.attrs["scan_signature_crc32"])
            if existing_crc != int(signature_crc32):
                _checkpoint_mismatch(
                    grp_path,
                    run_key,
                    f"scan_signature_crc32 differs: existing {existing_crc}, current {int(signature_crc32)}",
                )
        return

    _require_attr_int(grp, "NO_FRAG", int(params["NO_FRAG"]), grp_path, run_key)
    _require_attr_float(grp, "int_bre", float(params["int_bre"]), grp_path, run_key)
    _require_attr_float(grp, "gamma", float(params["gamma"]), grp_path, run_key)
    _require_attr_float(grp, "Df", float(params["Df"]), grp_path, run_key)
    _require_attr_float(grp, "MAS", float(params["MAS"]), grp_path, run_key)
    _require_attr_float(grp, "X1", float(params["X1"]), grp_path, run_key)
    _require_attr_float(grp, "A0", float(params["A0"]), grp_path, run_key)
    _require_attr_int(grp, "N_GRIDS", int(params["N_GRIDS"]), grp_path, run_key)
    _require_attr_int(grp, "N_FRACS", int(params["N_FRACS"]), grp_path, run_key)
    _require_attr_int(grp, "base_seed", int(params["base_seed"]), grp_path, run_key)
    _require_attr_array(grp, "STR", _coerce_str_array(params["STR"]), grp_path, run_key)


def _validate_dataset_shape(
    grp: h5py.Group,
    name: str,
    expected_shape: Tuple[int, ...],
    grp_path: str,
    run_key: str,
) -> None:
    if name in grp and tuple(grp[name].shape) != tuple(expected_shape):
        _checkpoint_mismatch(
            grp_path,
            run_key,
            f"dataset {name!r}: existing shape {tuple(grp[name].shape)}, current {tuple(expected_shape)}",
        )


def _validate_np_and_v(
    grp: h5py.Group,
    *,
    np_arr: np.ndarray,
    v_arr: np.ndarray,
    grp_path: str,
    run_key: str,
) -> None:
    if "Np" in grp:
        existing_np = np.asarray(grp["Np"][...], dtype=int)
        if existing_np.shape != np_arr.shape or not np.array_equal(existing_np, np_arr.astype(int)):
            _checkpoint_mismatch(
                grp_path,
                run_key,
                f"dataset 'Np' differs: existing {existing_np.tolist()!r}, current {np_arr.astype(int).tolist()!r}",
            )
    if "V" in grp:
        existing_v = np.asarray(grp["V"][...], dtype=float)
        if existing_v.shape != v_arr.shape or not np.allclose(existing_v, v_arr.astype(float), rtol=1e-12, atol=1e-12):
            _checkpoint_mismatch(grp_path, run_key, "dataset 'V' differs from np_list * A0")


def _validate_existing_checkpoint_group(
    grp: h5py.Group,
    *,
    grp_path: str,
    run_key: str,
    np_arr: np.ndarray,
    v_arr: np.ndarray,
    params: Dict[str, Any],
    signature_json: str,
    signature_crc32: int,
) -> None:
    n_np = int(np_arr.size)
    n_grids = int(params["N_GRIDS"])
    n_fracs = int(params["N_FRACS"])
    _validate_signature_or_legacy_attrs(
        grp,
        grp_path=grp_path,
        run_key=run_key,
        params=params,
        signature_json=signature_json,
        signature_crc32=signature_crc32,
    )
    _validate_np_and_v(grp, np_arr=np_arr, v_arr=v_arr, grp_path=grp_path, run_key=run_key)
    _validate_dataset_shape(grp, "E_samples", (n_np, n_grids, n_fracs), grp_path, run_key)
    _validate_dataset_shape(grp, "completed_mask", (n_np, n_grids), grp_path, run_key)
    _validate_dataset_shape(grp, "E_mean", (n_np,), grp_path, run_key)
    _validate_dataset_shape(grp, "E_std", (n_np,), grp_path, run_key)


def _validate_legacy_complete_group(
    grp: h5py.Group,
    *,
    grp_path: str,
    run_key: str,
    np_arr: np.ndarray,
    v_arr: np.ndarray,
    params: Dict[str, Any],
    n_runs_per_np: int,
    signature_json: str,
    signature_crc32: int,
) -> None:
    _validate_signature_or_legacy_attrs(
        grp,
        grp_path=grp_path,
        run_key=run_key,
        params=params,
        signature_json=signature_json,
        signature_crc32=signature_crc32,
    )
    _validate_np_and_v(grp, np_arr=np_arr, v_arr=v_arr, grp_path=grp_path, run_key=run_key)
    _validate_dataset_shape(grp, "E_samples", (int(np_arr.size), int(n_runs_per_np)), grp_path, run_key)


def _write_scan_signature_attrs(
    grp: h5py.Group,
    *,
    params: Dict[str, Any],
    signature_json: str,
    signature_crc32: int,
    mp_start_method: str,
) -> None:
    grp.attrs["pool_dir"] = _normalized_pool_dir(params["pool_dir"])
    grp.attrs["scan_signature_json"] = signature_json
    grp.attrs["scan_signature_crc32"] = int(signature_crc32)
    grp.attrs["mp_start_method"] = str(mp_start_method)


def _mp_start_method_for_workers(workers: int) -> str:
    if int(workers) <= 1:
        return "serial"
    return mp.get_context("spawn").get_start_method()


def _make_process_pool(workers: int) -> ProcessPoolExecutor:
    spawn_context = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=int(workers), mp_context=spawn_context)


def _grid_seed_for_task(base_seed: int, run_key: str, idx_np: int, grid_idx: int) -> int:
    run_crc = zlib.crc32(run_key.encode("utf-8")) & 0xFFFFFFFF
    seq = np.random.SeedSequence([int(base_seed), int(run_crc), int(idx_np), int(grid_idx)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


# =============================
# Worker: one grid + N_FRACS repeats
# =============================
def _energy_scan_worker_one_grid(args: Tuple[Any, ...]) -> Tuple[int, int, int, np.ndarray]:
    (
        idx_np,
        grid_idx,
        Np,
        A,
        seed_grid,
        pool_dir,
        Df,
        MAS,
        X1,
        A0,
        int_bre,
        STR,
        NO_FRAG,
        gamma,
        N_FRACS,
    ) = args

    sim = None
    try:
        sim = LMCSimulator(
            STR=STR,
            NO_FRAG=NO_FRAG,
            gamma=gamma,
            allow_loops=False,
            accept_all_cracks=False,
            use_weighted_start=True,
            plotter=None,
            pool_dir=pool_dir,
        )

        F = sim.mc_breakage_from_pool(
            pool_dir=pool_dir,
            Df=Df,
            MAS=MAS,
            A=A,
            X1=X1,
            N_GRIDS=1,
            N_FRACS=N_FRACS,
            A0=A0,
            int_bre=int_bre,
            seed=seed_grid,
            plot_each=False,
            interp="knn",
            KNN=1,
            sigma=0.35,
        )

        n_runs_local = N_FRACS
        try:
            F_run = F.reshape(n_runs_local, NO_FRAG, 4)
        except ValueError as exc:
            raise RuntimeError(
                f"[worker] Unexpected F shape for Np={Np}: "
                f"F.shape={F.shape}, expected {n_runs_local * NO_FRAG} rows"
            ) from exc

        energies = F_run[:, 0, 3].copy()
        return idx_np, grid_idx, Np, energies
    finally:
        if sim is not None and getattr(sim, "agg_pool", None) is not None:
            try:
                sim.agg_pool.close_pool_cache()
            except Exception as exc:
                print(
                    f"[WARN] Failed to close aggregate pool cache for "
                    f"Np={Np}, grid={grid_idx}: {exc!r}"
                )

def _fit_sigma_from_arrays(V: np.ndarray, E: np.ndarray) -> Tuple[float, float]:
    mask = (V > 0.0) & np.isfinite(V) & (E > 0.0) & np.isfinite(E)
    V_fit = V[mask]
    E_fit = E[mask]
    if V_fit.size < 2:
        return float("nan"), float("nan")

    logV = np.log(V_fit)
    logE = np.log(E_fit)
    r = np.corrcoef(logV, logE)[0, 1]

    A_mat = np.vstack([np.ones_like(logV), logV]).T
    coef, *_ = np.linalg.lstsq(A_mat, logE, rcond=None)
    _a, sigma = coef
    return float(sigma), float(r)

# =============================
# HDF5 / SQLite checkpoint helpers
# =============================
def _init_status_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS task_status (
            run_key TEXT NOT NULL,
            idx_np INTEGER NOT NULL,
            grid_idx INTEGER NOT NULL,
            np_value INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_key, idx_np, grid_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_task_status_run_status
        ON task_status(run_key, status);

        CREATE TABLE IF NOT EXISTS run_metadata (
            run_key TEXT PRIMARY KEY,
            signature_crc32 INTEGER NOT NULL,
            signature_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _task_row_count(conn: sqlite3.Connection, run_key: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM task_status WHERE run_key = ?",
        (run_key,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _ensure_run_metadata(
    conn: sqlite3.Connection,
    *,
    run_key: str,
    signature_json: str,
    signature_crc32: int,
    allow_legacy_migration: bool,
) -> None:
    row = conn.execute(
        """
        SELECT signature_crc32, signature_json
        FROM run_metadata
        WHERE run_key = ?
        """,
        (run_key,),
    ).fetchone()

    if row is not None:
        existing_crc = int(row[0])
        existing_json = str(row[1])
        if existing_crc != int(signature_crc32) or existing_json != signature_json:
            raise ValueError(
                "[CHECKPOINT] SQLite run_metadata is incompatible with the "
                "current scan parameters.\n"
                f"  run_key : {run_key}\n"
                f"  existing crc32 : {existing_crc}\n"
                f"  current crc32  : {int(signature_crc32)}\n"
                "Use a new h5_path, delete the old SQLite rows, or restore "
                "the original scan parameters."
            )
        conn.execute(
            "UPDATE run_metadata SET updated_at = CURRENT_TIMESTAMP WHERE run_key = ?",
            (run_key,),
        )
        conn.commit()
        return

    if _task_row_count(conn, run_key) > 0 and not allow_legacy_migration:
        raise ValueError(
            "[CHECKPOINT] SQLite task rows exist for this run_key, but no "
            "run_metadata row or compatible HDF5 group was found.\n"
            f"  run_key : {run_key}\n"
            "Use a new h5_path, delete the old SQLite rows, or restore the "
            "matching HDF5 checkpoint."
        )

    conn.execute(
        """
        INSERT INTO run_metadata(run_key, signature_crc32, signature_json)
        VALUES (?, ?, ?)
        """,
        (run_key, int(signature_crc32), signature_json),
    )
    conn.commit()


def _validate_existing_task_rows(
    conn: sqlite3.Connection,
    *,
    run_key: str,
    np_arr: np.ndarray,
    n_grids: int,
) -> None:
    rows = conn.execute(
        """
        SELECT idx_np, grid_idx, np_value
        FROM task_status
        WHERE run_key = ?
        """,
        (run_key,),
    ).fetchall()
    for idx_np, grid_idx, np_value in rows:
        idx_np = int(idx_np)
        grid_idx = int(grid_idx)
        np_value = int(np_value)
        if idx_np < 0 or idx_np >= int(np_arr.size):
            raise ValueError(
                f"[CHECKPOINT] SQLite task row has out-of-range idx_np={idx_np} "
                f"for run_key={run_key!r}."
            )
        if grid_idx < 0 or grid_idx >= int(n_grids):
            raise ValueError(
                f"[CHECKPOINT] SQLite task row has out-of-range grid_idx={grid_idx} "
                f"for run_key={run_key!r}."
            )
        expected_np = int(np_arr[idx_np])
        if np_value != expected_np:
            raise ValueError(
                "[CHECKPOINT] SQLite task row np_value is incompatible with "
                "the current np_list.\n"
                f"  run_key : {run_key}\n"
                f"  idx_np  : {idx_np}\n"
                f"  existing np_value : {np_value}\n"
                f"  current np_value  : {expected_np}"
            )


def _ensure_task_rows(
    conn: sqlite3.Connection,
    run_key: str,
    np_arr: np.ndarray,
    n_grids: int,
) -> None:
    rows = []
    for idx_np, np_value in enumerate(np_arr.astype(int)):
        for grid_idx in range(int(n_grids)):
            rows.append((run_key, int(idx_np), int(grid_idx), int(np_value)))
    conn.executemany(
        """
        INSERT OR IGNORE INTO task_status(run_key, idx_np, grid_idx, np_value)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def _mark_task_done(conn: sqlite3.Connection, run_key: str, idx_np: int, grid_idx: int) -> None:
    conn.execute(
        """
        UPDATE task_status
        SET status = 'done',
            updated_at = CURRENT_TIMESTAMP
        WHERE run_key = ? AND idx_np = ? AND grid_idx = ?
        """,
        (run_key, int(idx_np), int(grid_idx)),
    )
    conn.commit()


def _pending_tasks(conn: sqlite3.Connection, run_key: str) -> List[Tuple[int, int]]:
    rows = conn.execute(
        """
        SELECT idx_np, grid_idx
        FROM task_status
        WHERE run_key = ? AND status != 'done'
        ORDER BY idx_np, grid_idx
        """,
        (run_key,),
    ).fetchall()
    return [(int(row[0]), int(row[1])) for row in rows]


def _done_task_count(conn: sqlite3.Connection, run_key: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM task_status WHERE run_key = ? AND status = 'done'",
        (run_key,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _is_legacy_complete_group(grp: h5py.Group, n_np: int, n_runs_per_np: int) -> bool:
    if "completed_mask" in grp:
        return False
    if "E_samples" not in grp:
        return False
    e_samples = grp["E_samples"]
    return tuple(e_samples.shape) == (int(n_np), int(n_runs_per_np))


def _ensure_checkpoint_group(
    f: h5py.File,
    grp_path: str,
    *,
    np_arr: np.ndarray,
    v_arr: np.ndarray,
    params: Dict[str, Any],
    signature_json: str,
    signature_crc32: int,
    mp_start_method: str,
) -> h5py.Group:
    n_np = int(np_arr.size)
    n_grids = int(params["N_GRIDS"])
    n_fracs = int(params["N_FRACS"])

    if grp_path in f:
        grp = f[grp_path]
    else:
        grp = f.create_group(grp_path)

    grp.attrs["NO_FRAG"] = int(params["NO_FRAG"])
    grp.attrs["int_bre"] = float(params["int_bre"])
    grp.attrs["gamma"] = float(params["gamma"])
    grp.attrs["Df"] = float(params["Df"])
    grp.attrs["MAS"] = float(params["MAS"])
    grp.attrs["X1"] = float(params["X1"])
    grp.attrs["A0"] = float(params["A0"])
    grp.attrs["N_GRIDS"] = int(params["N_GRIDS"])
    grp.attrs["N_FRACS"] = int(params["N_FRACS"])
    grp.attrs["base_seed"] = int(params["base_seed"])
    grp.attrs["workers"] = int(params["workers"])
    grp.attrs["STR"] = _coerce_str_array(params["STR"])
    grp.attrs["checkpoint_version"] = 2
    _write_scan_signature_attrs(
        grp,
        params=params,
        signature_json=signature_json,
        signature_crc32=signature_crc32,
        mp_start_method=mp_start_method,
    )

    if "Np" not in grp:
        grp.create_dataset("Np", data=np_arr.astype(int), compression="gzip")
    if "V" not in grp:
        grp.create_dataset("V", data=v_arr.astype(float), compression="gzip")
    if "E_samples" not in grp:
        grp.create_dataset(
            "E_samples",
            shape=(n_np, n_grids, n_fracs),
            dtype=float,
            compression="gzip",
            fillvalue=np.nan,
        )
    if "completed_mask" not in grp:
        grp.create_dataset(
            "completed_mask",
            shape=(n_np, n_grids),
            dtype=np.bool_,
            compression="gzip",
            fillvalue=False,
        )
    if "E_mean" not in grp:
        grp.create_dataset("E_mean", shape=(n_np,), dtype=float, compression="gzip", fillvalue=np.nan)
    if "E_std" not in grp:
        grp.create_dataset("E_std", shape=(n_np,), dtype=float, compression="gzip", fillvalue=np.nan)

    return grp


def _sync_task_status_from_h5(
    conn: sqlite3.Connection,
    run_key: str,
    grp: h5py.Group,
) -> None:
    if "completed_mask" not in grp:
        return

    completed_mask = np.asarray(grp["completed_mask"][...], dtype=bool)
    rows = [(run_key, int(idx_np), int(grid_idx)) for idx_np, grid_idx in np.argwhere(completed_mask)]
    if not rows:
        return

    conn.executemany(
        """
        UPDATE task_status
        SET status = 'done',
            updated_at = CURRENT_TIMESTAMP
        WHERE run_key = ? AND idx_np = ? AND grid_idx = ?
        """,
        rows,
    )
    conn.commit()


def _update_np_summary(grp: h5py.Group, idx_np: int) -> None:
    completed_row = np.asarray(grp["completed_mask"][idx_np, :], dtype=bool)
    if not np.any(completed_row):
        grp["E_mean"][idx_np] = np.nan
        grp["E_std"][idx_np] = np.nan
        return

    row_samples = np.asarray(grp["E_samples"][idx_np, :, :], dtype=float)
    done_samples = row_samples[completed_row, :].reshape(-1)
    if done_samples.size == 0:
        grp["E_mean"][idx_np] = np.nan
        grp["E_std"][idx_np] = np.nan
        return

    grp["E_mean"][idx_np] = float(done_samples.mean())
    grp["E_std"][idx_np] = float(done_samples.std(ddof=1)) if done_samples.size > 1 else np.nan


def _update_all_summaries(grp: h5py.Group) -> None:
    n_np = int(grp["Np"].shape[0])
    for idx_np in range(n_np):
        _update_np_summary(grp, idx_np)


def _update_sigma_attrs(grp: h5py.Group) -> Tuple[float, float]:
    V = np.asarray(grp["V"][...], dtype=float)
    E_mean = np.asarray(grp["E_mean"][...], dtype=float)
    sigma, r = _fit_sigma_from_arrays(V, E_mean)
    grp.attrs["sigma"] = float(sigma)
    grp.attrs["pearson_r"] = float(r)
    return float(sigma), float(r)


def _subtask_payload(
    *,
    run_key: str,
    idx_np: int,
    grid_idx: int,
    np_arr: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[Any, ...]:
    np_value = int(np_arr[int(idx_np)])
    A = float(np_value) * float(params["A0"])
    seed_grid = _grid_seed_for_task(int(params["base_seed"]), run_key, int(idx_np), int(grid_idx))
    return (
        int(idx_np),
        int(grid_idx),
        int(np_value),
        A,
        seed_grid,
        str(params["pool_dir"]),
        float(params["Df"]),
        float(params["MAS"]),
        float(params["X1"]),
        float(params["A0"]),
        float(params["int_bre"]),
        _coerce_str_array(params["STR"]),
        int(params["NO_FRAG"]),
        float(params["gamma"]),
        int(params["N_FRACS"]),
    )


def _run_checkpointed_energy_scan(
    *,
    h5_path: str,
    sqlite_path: str,
    grp_path: str,
    run_key: str,
    params: Dict[str, Any],
    np_list: Sequence[int],
) -> Tuple[float, float]:
    np_arr = np.asarray(list(np_list), dtype=int)
    v_arr = np_arr.astype(float) * float(params["A0"])
    expected_tasks = int(np_arr.size) * int(params["N_GRIDS"])
    n_runs_per_np = int(params["N_GRIDS"]) * int(params["N_FRACS"])
    _signature_payload, signature_json, signature_crc32 = _make_scan_signature(params, np_arr)
    mp_start_method = _mp_start_method_for_workers(int(params["workers"]))

    with h5py.File(h5_path, "a") as h5f, sqlite3.connect(sqlite_path) as conn:
        _init_status_db(conn)

        group_exists = grp_path in h5f
        if not group_exists and _task_row_count(conn, run_key) > 0:
            raise ValueError(
                "[CHECKPOINT] SQLite task rows exist for this run_key, but "
                "the matching HDF5 group does not exist.\n"
                f"  group   : {grp_path}\n"
                f"  run_key : {run_key}\n"
                "Use a new h5_path, delete the old SQLite rows, or restore "
                "the matching HDF5 checkpoint."
            )

        if group_exists and _is_legacy_complete_group(h5f[grp_path], int(np_arr.size), n_runs_per_np):
            grp = h5f[grp_path]
            _validate_legacy_complete_group(
                grp,
                grp_path=grp_path,
                run_key=run_key,
                np_arr=np_arr,
                v_arr=v_arr,
                params=params,
                n_runs_per_np=n_runs_per_np,
                signature_json=signature_json,
                signature_crc32=signature_crc32,
            )
            _ensure_run_metadata(
                conn,
                run_key=run_key,
                signature_json=signature_json,
                signature_crc32=signature_crc32,
                allow_legacy_migration=True,
            )
            _write_scan_signature_attrs(
                grp,
                params=params,
                signature_json=signature_json,
                signature_crc32=signature_crc32,
                mp_start_method=mp_start_method,
            )
            h5f.flush()
            sigma = float(grp.attrs.get("sigma", np.nan))
            r = float(grp.attrs.get("pearson_r", np.nan))
            print(f"[SKIP] {grp_path} already exists in legacy-complete format, skip running LMC.")
            return sigma, r

        if group_exists:
            _validate_existing_checkpoint_group(
                h5f[grp_path],
                grp_path=grp_path,
                run_key=run_key,
                np_arr=np_arr,
                v_arr=v_arr,
                params=params,
                signature_json=signature_json,
                signature_crc32=signature_crc32,
            )

        _ensure_run_metadata(
            conn,
            run_key=run_key,
            signature_json=signature_json,
            signature_crc32=signature_crc32,
            allow_legacy_migration=group_exists,
        )
        _validate_existing_task_rows(
            conn,
            run_key=run_key,
            np_arr=np_arr,
            n_grids=int(params["N_GRIDS"]),
        )

        grp = _ensure_checkpoint_group(
            h5f,
            grp_path,
            np_arr=np_arr,
            v_arr=v_arr,
            params=params,
            signature_json=signature_json,
            signature_crc32=signature_crc32,
            mp_start_method=mp_start_method,
        )
        _ensure_task_rows(conn, run_key, np_arr, int(params["N_GRIDS"]))
        _sync_task_status_from_h5(conn, run_key, grp)

        pending = _pending_tasks(conn, run_key)
        done_count = _done_task_count(conn, run_key)
        grp.attrs["completed_tasks"] = int(done_count)
        grp.attrs["expected_tasks"] = int(expected_tasks)
        h5f.flush()

        if not pending:
            _update_all_summaries(grp)
            sigma, r = _update_sigma_attrs(grp)
            h5f.flush()
            print(f"[SKIP] {grp_path} already complete ({done_count}/{expected_tasks} subtasks).")
            return sigma, r

        print(
            f"[RESUME] {grp_path}: {done_count}/{expected_tasks} subtasks complete, "
            f"{len(pending)} remaining."
        )

        workers = int(params["workers"])
        if workers <= 1:
            for idx_np, grid_idx in pending:
                payload = _subtask_payload(
                    run_key=run_key,
                    idx_np=idx_np,
                    grid_idx=grid_idx,
                    np_arr=np_arr,
                    params=params,
                )
                idx_np_ret, grid_idx_ret, np_value, energies = _energy_scan_worker_one_grid(payload)
                grp["E_samples"][idx_np_ret, grid_idx_ret, :] = energies
                grp["completed_mask"][idx_np_ret, grid_idx_ret] = True
                _update_np_summary(grp, idx_np_ret)
                h5f.flush()
                _mark_task_done(conn, run_key, idx_np_ret, grid_idx_ret)
                done_count += 1
                grp.attrs["completed_tasks"] = int(done_count)
                h5f.flush()
                print(
                    f"[TASK-SEQ] Np={np_value:6d}, grid={grid_idx_ret:4d}, "
                    f"done {done_count}/{expected_tasks}"
                )
        else:
            pending_futures: Dict[Any, Tuple[int, int]] = {}
            pending_iter = iter(pending)

            def refill(executor: ProcessPoolExecutor) -> None:
                while len(pending_futures) < workers:
                    try:
                        idx_np, grid_idx = next(pending_iter)
                    except StopIteration:
                        break
                    payload = _subtask_payload(
                        run_key=run_key,
                        idx_np=idx_np,
                        grid_idx=grid_idx,
                        np_arr=np_arr,
                        params=params,
                    )
                    future = executor.submit(_energy_scan_worker_one_grid, payload)
                    pending_futures[future] = (idx_np, grid_idx)

            with _make_process_pool(workers) as ex:
                refill(ex)
                while pending_futures:
                    done_set, _ = wait(set(pending_futures.keys()), return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        idx_np_sub, grid_idx_sub = pending_futures.pop(fut)
                        idx_np_ret, grid_idx_ret, np_value, energies = fut.result()
                        if idx_np_ret != idx_np_sub or grid_idx_ret != grid_idx_sub:
                            raise RuntimeError(
                                f"Worker returned mismatched task identity: "
                                f"expected ({idx_np_sub}, {grid_idx_sub}), got ({idx_np_ret}, {grid_idx_ret})"
                            )

                        grp["E_samples"][idx_np_ret, grid_idx_ret, :] = energies
                        grp["completed_mask"][idx_np_ret, grid_idx_ret] = True
                        _update_np_summary(grp, idx_np_ret)
                        h5f.flush()
                        _mark_task_done(conn, run_key, idx_np_ret, grid_idx_ret)
                        done_count += 1
                        grp.attrs["completed_tasks"] = int(done_count)
                        h5f.flush()
                        print(
                            f"[TASK-PAR] Np={np_value:6d}, grid={grid_idx_ret:4d}, "
                            f"done {done_count}/{expected_tasks}"
                        )
                    refill(ex)

        _update_all_summaries(grp)
        sigma, r = _update_sigma_attrs(grp)
        grp.attrs["completed_tasks"] = int(done_count)
        grp.attrs["expected_tasks"] = int(expected_tasks)
        h5f.flush()
        return sigma, r

# =============================
# Parameter scan
# =============================
def _make_param_key(
    NO_FRAG: int,
    int_bre: float,
    gamma: float,
    Df: float,
    MAS: float,
    X1: float,
    STR: np.ndarray,
) -> str:
    return (
        f"NOF_{int(NO_FRAG)}"
        f"_GB_{_format_float_for_key(gamma)}"
        f"_BRE_{_format_float_for_key(int_bre)}"
        f"_Df_{_format_float_for_key(Df)}"
        f"_MAS_{_format_float_for_key(MAS)}"
        f"_X1_{_format_float_for_key(X1)}"
        f"_STR_{_format_str_for_key(STR)}"
    )


def run_full_parameter_scan(
    h5_path: str,
    pool_dir: str,
    Df: float,
    MAS: float | Sequence[float],
    STR: np.ndarray | Sequence[np.ndarray | Sequence[float]],
    A0: float,
    X1: float | Sequence[float],
    np_list: Sequence[int],
    no_frag_list: Sequence[int],
    int_bre_list: np.ndarray,
    gamma_list: np.ndarray,
    N_GRIDS: int,
    N_FRACS: int,
    base_seed: int,
    workers: int,
):
    sqlite_path = _status_db_path(h5_path)
    mas_list = [float(v) for v in _ensure_sequence(MAS)]
    x1_list = [float(v) for v in _ensure_sequence(X1)]
    str_list = [_coerce_str_array(v) for v in _ensure_sequence(STR)]

    for NO_FRAG in no_frag_list:
        for int_bre in int_bre_list:
            for gamma in gamma_list:
                for MAS_value in mas_list:
                    for X1_value in x1_list:
                        for STR_value in str_list:
                            key = _make_param_key(
                                int(NO_FRAG),
                                float(int_bre),
                                float(gamma),
                                float(Df),
                                float(MAS_value),
                                float(X1_value),
                                STR_value,
                            )
                            grp_path = f"/runs/{key}"

                            print("\n=====================================================")
                            print(
                                "SCAN: "
                                f"NO_FRAG={NO_FRAG}, int_bre={float(int_bre):.3f}, "
                                f"gamma={float(gamma):.3f}, MAS={MAS_value:.6g}, "
                                f"X1={X1_value:.6g}, STR={STR_value.tolist()}"
                            )
                            print("=====================================================\n")

                            params = dict(
                                pool_dir=os.path.abspath(pool_dir),
                                Df=float(Df),
                                MAS=float(MAS_value),
                                X1=float(X1_value),
                                A0=float(A0),
                                int_bre=float(int_bre),
                                STR=_coerce_str_array(STR_value),
                                NO_FRAG=int(NO_FRAG),
                                gamma=float(gamma),
                                N_GRIDS=int(N_GRIDS),
                                N_FRACS=int(N_FRACS),
                                base_seed=int(base_seed),
                                workers=int(workers),
                            )

                            sigma, r = _run_checkpointed_energy_scan(
                                h5_path=h5_path,
                                sqlite_path=sqlite_path,
                                grp_path=grp_path,
                                run_key=key,
                                params=params,
                                np_list=np_list,
                            )

                            print(f"[SCAN] sigma={sigma:.4f}, r={r:.4f}")

    print("\nAll scans finished!")
    print(f"Results saved to {h5_path}")
    print(f"Checkpoint DB saved to {sqlite_path}")


def print_h5_structure(h5_path):
    def _print(name, obj):
        indent = "  " * (name.count("/") - 1)

        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group ] {name}")
            for k, v in obj.attrs.items():
                print(f"{indent}    (attr) {k}: {v}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name} shape={obj.shape} dtype={obj.dtype}")
            for k, v in obj.attrs.items():
                print(f"{indent}    (attr) {k}: {v}")

    with h5py.File(h5_path, "r") as f:
        print(f"--- HDF5 structure of {h5_path} ---")
        f.visititems(_print)


if __name__ == "__main__":
    # pool_dir = r""
    # store_path = r""
    pool_dir = os.environ.get("TMP_PATH")
    store_path = os.path.join(os.environ.get("STORAGE_PATH"), "energy_pool")
    Df = 1.8
    MAS_list = [0.5]

    A0 = 1.0
    X1_list = [0.5]

    values = np.array([1.0, 1e1, 1e2, 1e3])
    a1, a2, a3 = np.meshgrid(values, values, values, indexing="ij")
    var_STR = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    var_STR = var_STR[~np.all(var_STR == 0, axis=1)]
    unique_STR = []
    for comp in var_STR:
        comp_reversed = comp[::-1]
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_STR):
            unique_STR.append(comp)
    STR_list = np.array(unique_STR)

    N_GRIDS = 100
    N_FRACS = 200
    base_seed = 42
    workers = 1

    np_list = NP_LIST
    no_frag_list = [2]
    int_bre_list = [0.0]
    gamma_list = np.logspace(-3, 3, 6)

    output_h5 = os.path.join(store_path, "psd_data5.h5")

    run_full_parameter_scan(
        h5_path=output_h5,
        pool_dir=pool_dir,
        Df=Df,
        MAS=MAS_list,
        STR=STR_list,
        A0=A0,
        X1=X1_list,
        np_list=np_list,
        no_frag_list=no_frag_list,
        int_bre_list=int_bre_list,
        gamma_list=gamma_list,
        N_GRIDS=N_GRIDS,
        N_FRACS=N_FRACS,
        base_seed=base_seed,
        workers=workers,
    )
    # print_h5_structure(output_h5)
