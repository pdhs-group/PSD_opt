# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:23:21 2025

@author: px2030
"""

import argparse
import os
import re
from pathlib import Path
from typing import List

import pandas as pd


def list_yappi_files(directory: Path, base: str) -> List[Path]:
    """
    List all files matching base*.csv in directory, sorted in natural order:
      timings_yappi.csv, timings_yappi (2).csv, timings_yappi (3).csv ...
    """
    # Collect candidates
    files = [p for p in directory.iterdir() if p.is_file() and p.name.endswith(".csv") and p.name.startswith(base)]
    if not files:
        return []

    # Parse sequence number: timings_yappi.csv -> 1, timings_yappi (N).csv -> N
    pat = re.compile(rf"^{re.escape(base)}(?:\s*\((\d+)\))?\.csv$", re.IGNORECASE)

    def sort_key(p: Path):
        m = pat.match(p.name)
        if not m:
            # Non-standard naming, put later, sort by name
            return (1, p.name.lower())
        num = m.group(1)
        # Base file (no parentheses) has priority, treated as sequence 1; parentheses files sorted by number ascending
        return (0, 1 if num is None else int(num))

    return sorted(files, key=sort_key)


def merge_yappi_csvs(directory: Path, base: str, out_path: Path, add_source: bool = True) -> int:
    """
    Merge base*.csv files in directory (natural order), write to out_path.
    Return number of successfully merged files.
    """
    files = list_yappi_files(directory, base)
    if not files:
        print(f"[WARN] No files found in {directory} matching {base}*.csv")
        return 0

    frames = []
    for p in files:
        try:
            # Try to read; empty files will raise EmptyDataError
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            print(f"[SKIP] Empty file: {p.name}")
            continue
        except Exception as e:
            print(f"[SKIP] Failed to read {p.name}: {e}")
            continue

        if add_source:
            df.insert(0, "source_file", p.name)
        frames.append(df)

    if not frames:
        print("[WARN] No non-empty CSVs to merge.")
        return 0

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[OK] Merged {len(frames)} file(s) -> {out_path}")
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description="Merge timings_yappi*.csv into one CSV.")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing CSV files (default: current dir).")
    parser.add_argument("--base", type=str, default="timings_yappi", help="Base filename prefix (default: timings_yappi).")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path. Default: <dir>/<base>_merged.csv")
    parser.add_argument("--no-source", action="store_true", help="Do NOT add source_file column.")
    args = parser.parse_args()

    directory = Path(args.dir).resolve()
    base = args.base
    out_path = Path(args.out).resolve() if args.out else (directory / f"{base}_merged.csv")
    add_source = not args.no_source

    merge_yappi_csvs(directory, base, out_path, add_source)


if __name__ == "__main__":
    directory = Path(r"C:\Users\px2030\Code\Ergebnisse\opt_para_study")
    merge_yappi_csvs(directory, "timings_yappi", directory / "timings_yappi_merged.csv", add_source=True)
    main()