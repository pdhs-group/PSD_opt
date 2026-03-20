"""
Convert an aggregate-pool HDF5 file into a directory containing:
- SQLite metadata / index
- one NPZ file per sample

Output layout:
    <output_dir>/
        pool_index.sqlite
        samples/<group_name>/<sample_name>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np


DEFAULT_DATASETS = ("M", "Hbond", "Vbond", "labels", "origin")
SQLITE_NAME = "pool_index.sqlite"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _attrs_to_dict(attrs: h5py.AttributeManager) -> Dict[str, Any]:
    return {str(key): _json_default(value) for key, value in attrs.items()}


def _iter_sample_names(grp: h5py.Group) -> List[str]:
    return sorted(name for name in grp.keys() if str(name).startswith("sample_"))


def _sample_index(sample_name: str) -> int:
    try:
        return int(str(sample_name).split("_")[-1])
    except Exception:
        return -1


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE groups (
            group_name TEXT PRIMARY KEY,
            np_target REAL,
            frac_a_target REAL,
            n_samples INTEGER NOT NULL,
            attrs_json TEXT NOT NULL
        );

        CREATE TABLE samples (
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

        CREATE INDEX idx_samples_group_name ON samples(group_name);
        CREATE INDEX idx_samples_group_index ON samples(group_name, sample_index);
        """
    )


def convert_h5_pool_to_npz_sqlite_single(
    input_h5_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    overwrite: bool = False,
    compressed: bool = True,
) -> None:
    input_path = Path(input_h5_path).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_path}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    samples_root = output_path / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_path / SQLITE_NAME

    datasets = tuple(str(name) for name in datasets)
    save_npz = np.savez_compressed if compressed else np.savez

    print(f"[NPZ-SINGLE] Source HDF5: {input_path}")
    print(f"[NPZ-SINGLE] Target dir: {output_path}")
    print(f"[NPZ-SINGLE] SQLite: {sqlite_path}")
    print(f"[NPZ-SINGLE] datasets={datasets}")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        _init_db(conn)

        with h5py.File(input_path, "r") as h5:
            file_attrs = _attrs_to_dict(h5.attrs)
            group_names = sorted(str(name) for name in h5.keys())

            conn.execute(
                "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                ("format", json.dumps("npz_sqlite_single")),
            )
            conn.execute(
                "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                ("source_h5_path", json.dumps(str(input_path))),
            )
            conn.execute(
                "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                ("datasets", json.dumps(list(datasets), ensure_ascii=True, sort_keys=True)),
            )
            conn.execute(
                "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                ("file_attrs", json.dumps(file_attrs, ensure_ascii=True, sort_keys=True)),
            )

            total_samples = 0
            for group_name in group_names:
                grp = h5[group_name]
                if not isinstance(grp, h5py.Group):
                    continue

                group_attrs = _attrs_to_dict(grp.attrs)
                sample_names = _iter_sample_names(grp)
                total_samples += len(sample_names)

                conn.execute(
                    "INSERT INTO groups(group_name, np_target, frac_a_target, n_samples, attrs_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        group_name,
                        float(group_attrs.get("Np_target", np.nan)),
                        float(group_attrs.get("frac_A_target", np.nan)),
                        len(sample_names),
                        json.dumps(group_attrs, ensure_ascii=True, sort_keys=True),
                    ),
                )

                group_dir = samples_root / group_name
                group_dir.mkdir(parents=True, exist_ok=True)
                print(f"[NPZ-SINGLE] group={group_name} samples={len(sample_names)}")

                for sample_name in sample_names:
                    sample = grp[sample_name]
                    sample_attrs = _attrs_to_dict(sample.attrs)
                    sample_data = {}
                    for dataset_name in datasets:
                        if dataset_name in sample:
                            sample_data[dataset_name] = np.asarray(sample[dataset_name][()])

                    npz_path = group_dir / f"{sample_name}.npz"
                    save_npz(npz_path, **sample_data)
                    relpath = npz_path.relative_to(output_path).as_posix()

                    conn.execute(
                        "INSERT INTO samples(group_name, sample_name, sample_index, npz_relpath, array_prefix, attrs_json) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            group_name,
                            sample_name,
                            _sample_index(sample_name),
                            relpath,
                            "",
                            json.dumps(sample_attrs, ensure_ascii=True, sort_keys=True),
                        ),
                    )

                conn.commit()

            print(f"[NPZ-SINGLE] Conversion complete. groups={len(group_names)}, samples={total_samples}")
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an HDF5 aggregate pool into one-NPZ-per-sample plus SQLite index."
    )
    parser.add_argument("input_h5", help="Path to the source HDF5 pool file.")
    parser.add_argument("output_dir", help="Path to the new output directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to migrate for each sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use numpy.savez instead of numpy.savez_compressed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_h5_pool_to_npz_sqlite_single(
        args.input_h5,
        args.output_dir,
        datasets=args.datasets,
        overwrite=args.overwrite,
        compressed=not args.no_compress,
    )


if __name__ == "__main__":
    main()
