from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path(r"C:\Users\px2030\Code\PSD_opt\lmc\tests")
MAX_VARS_PER_FIG = 5
EPS = 1e-12


def _find_latest_csv(directory: Path) -> Path:
    candidates = sorted(directory.glob("lmc_pool_memory_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No memory CSV found in {directory}")
    return candidates[-1]


def _load_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _numeric_columns(rows: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("CSV file is empty.")

    data: Dict[str, np.ndarray] = {}
    fieldnames = list(rows[0].keys())
    for field in fieldnames:
        if field in ("tag", "call"):
            continue

        values: List[float] = []
        ok = True
        for row in rows:
            raw = row.get(field, "")
            if raw == "":
                ok = False
                break
            try:
                values.append(float(raw))
            except ValueError:
                ok = False
                break

        if ok:
            data[field] = np.asarray(values, dtype=float)

    return data


def _relative_change(series: np.ndarray) -> np.ndarray | None:
    nz = np.flatnonzero(np.abs(series) > EPS)
    if nz.size == 0:
        return None

    base = float(series[int(nz[0])])
    if abs(base) <= EPS:
        return None

    rel = (series - base) / base
    if np.all(np.abs(rel) <= EPS):
        return None
    return rel


def _chunk(names: List[str], size: int) -> List[List[str]]:
    return [names[i:i + size] for i in range(0, len(names), size)]


def plot_memory_csv(csv_path: Path, output_dir: Path | None = None) -> List[Path]:
    rows = _load_csv(csv_path)
    calls = np.asarray([int(float(row["call"])) for row in rows], dtype=float)
    numeric = _numeric_columns(rows)

    rel_series: Dict[str, np.ndarray] = {}
    for name, values in numeric.items():
        rel = _relative_change(values)
        if rel is not None:
            rel_series[name] = rel

    if not rel_series:
        print(f"No changing numeric variables found in {csv_path}")
        return []

    if output_dir is None:
        output_dir = csv_path.parent / f"{csv_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    names = sorted(rel_series.keys())
    for fig_idx, group in enumerate(_chunk(names, MAX_VARS_PER_FIG), start=1):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in group:
            ax.plot(calls, rel_series[name], linewidth=1.8, label=name)

        ax.set_xlabel("call")
        ax.set_ylabel("relative change vs first non-zero value")
        ax.set_title(f"LMC Memory Relative Change ({fig_idx})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = output_dir / f"{csv_path.stem}_part{fig_idx:02d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LMC pool memory CSV diagnostics.")
    parser.add_argument("csv_path", nargs="?", help="Path to a memory CSV file. Default: latest CSV in lmc/tests.")
    parser.add_argument("--out-dir", dest="out_dir", help="Directory for generated figures.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else _find_latest_csv(DEFAULT_DIR)
    out_dir = Path(args.out_dir) if args.out_dir else None

    written = plot_memory_csv(csv_path, out_dir)
    if written:
        print(f"Processed CSV: {csv_path}")
        print(f"Generated {len(written)} figure(s) in {written[0].parent}")


if __name__ == "__main__":
    main()
