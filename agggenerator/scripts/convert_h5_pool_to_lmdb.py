"""
Convert an aggregate pool stored in HDF5 into an LMDB database.

The source HDF5 layout is expected to match the pool written by
`aggregates_sampler.py`:

    /<group_name>/
        attrs...
        /sample_0000/
            datasets: M, Hbond, Vbond, labels, origin
            attrs...

The target LMDB keeps a similarly navigable structure via string keys:

    __meta__/format_version
    __meta__/source_h5_path
    __meta__/file_attrs
    __meta__/group_names
    __meta__/group/<group_name>/attrs
    __meta__/group/<group_name>/sample_names
    __meta__/group/<group_name>/sample/<sample_name>/attrs
    data/<group_name>/<sample_name>/<dataset_name>

Arrays are serialized with `numpy.save(..., allow_pickle=False)` so they can be
loaded later with `numpy.load(BytesIO(value), allow_pickle=False)`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import lmdb
import numpy as np


FORMAT_VERSION = 1
META_PREFIX = "__meta__/"
DATA_PREFIX = "data/"
DEFAULT_DATASETS = ("M", "Hbond", "Vbond", "labels", "origin")


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


def _dumps_json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _dumps_array(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def _iter_sample_names(grp: h5py.Group) -> List[str]:
    return sorted(name for name in grp.keys() if str(name).startswith("sample_"))


def _key(*parts: str) -> bytes:
    return "/".join(parts).encode("utf-8")


def _default_map_size(h5_path: Path) -> int:
    source_size = max(h5_path.stat().st_size, 1)
    padded = int(math.ceil(source_size * 4.0 + 256 * 1024 * 1024))
    return max(padded, 1 << 30)


def _open_lmdb_env(output_path: Path, map_size: int) -> lmdb.Environment:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(output_path),
        map_size=int(map_size),
        subdir=True,
        create=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=False,
        writemap=False,
        max_dbs=1,
    )


def convert_h5_pool_to_lmdb(
    input_h5_path: str | os.PathLike[str],
    output_lmdb_path: str | os.PathLike[str],
    *,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    map_size: int | None = None,
    overwrite: bool = False,
) -> None:
    input_path = Path(input_h5_path).resolve()
    output_path = Path(output_lmdb_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_path}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output LMDB path already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    datasets = tuple(str(name) for name in datasets)
    if len(datasets) == 0:
        raise ValueError("At least one dataset name must be provided.")

    if map_size is None:
        map_size = _default_map_size(input_path)

    print(f"[LMDB] Source HDF5: {input_path}")
    print(f"[LMDB] Target LMDB: {output_path}")
    print(f"[LMDB] map_size={int(map_size)} bytes")
    print(f"[LMDB] datasets={datasets}")

    env = _open_lmdb_env(output_path, map_size=int(map_size))
    try:
        with h5py.File(input_path, "r") as h5:
            group_names = sorted(str(name) for name in h5.keys())

            with env.begin(write=True) as txn:
                txn.put(_key(META_PREFIX.rstrip("/"), "format_version"), str(FORMAT_VERSION).encode("utf-8"))
                txn.put(_key(META_PREFIX.rstrip("/"), "source_h5_path"), str(input_path).encode("utf-8"))
                txn.put(_key(META_PREFIX.rstrip("/"), "datasets"), _dumps_json(list(datasets)))
                txn.put(_key(META_PREFIX.rstrip("/"), "file_attrs"), _dumps_json(_attrs_to_dict(h5.attrs)))
                txn.put(_key(META_PREFIX.rstrip("/"), "group_names"), _dumps_json(group_names))

            total_samples = 0
            for group_name in group_names:
                grp = h5[group_name]
                if not isinstance(grp, h5py.Group):
                    continue

                sample_names = _iter_sample_names(grp)
                total_samples += len(sample_names)

                with env.begin(write=True) as txn:
                    txn.put(
                        _key(META_PREFIX.rstrip("/"), "group", group_name, "attrs"),
                        _dumps_json(_attrs_to_dict(grp.attrs)),
                    )
                    txn.put(
                        _key(META_PREFIX.rstrip("/"), "group", group_name, "sample_names"),
                        _dumps_json(sample_names),
                    )

                print(f"[LMDB] group={group_name} samples={len(sample_names)}")

                for sample_name in sample_names:
                    sample = grp[sample_name]
                    sample_attrs = _attrs_to_dict(sample.attrs)

                    with env.begin(write=True) as txn:
                        txn.put(
                            _key(META_PREFIX.rstrip("/"), "group", group_name, "sample", sample_name, "attrs"),
                            _dumps_json(sample_attrs),
                        )

                        for dataset_name in datasets:
                            if dataset_name not in sample:
                                continue
                            arr = sample[dataset_name][()]
                            txn.put(
                                _key(DATA_PREFIX.rstrip("/"), group_name, sample_name, dataset_name),
                                _dumps_array(np.asarray(arr)),
                            )

            env.sync()
            print(f"[LMDB] Conversion complete. groups={len(group_names)}, samples={total_samples}")
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an aggregate-pool HDF5 file into an LMDB database."
    )
    parser.add_argument("input_h5", help="Path to the source HDF5 pool file.")
    parser.add_argument("output_lmdb", help="Path to the target LMDB directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to migrate for each sample.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=None,
        help="LMDB map size in bytes. Defaults to a padded multiple of the input HDF5 size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output path before writing the LMDB database.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_h5_pool_to_lmdb(
        args.input_h5,
        args.output_lmdb,
        datasets=args.datasets,
        map_size=args.map_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
