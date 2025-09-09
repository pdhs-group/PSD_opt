# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:53:17 2025
Merge NPZ results from multiple run folders into a single folder per prefix.

Folder layout example:
  opt_results_MSE-0/
    multi_[('qx','MSE')]_Cmaes_wight_1_iter_50.npz
    50.sqlite
    50.sqlite.lock
  opt_results_MSE-1/
    multi_[('qx','MSE')]_Cmaes_wight_1_iter_50.npz
    ...
  ...

This script:
- Cleans temp files (*.sqlite, *.sqlite.lock, *.sqlite*) in each run folder.
- For files with the same name across run folders, merges their `results` dicts
  into a single aggregated `results`, and saves to `<prefix>/same_name.npz`.
  
@author: Haoran Ji (px2030@kit.edu)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Iterable
import numpy as np
import shutil

def find_run_dirs(base: Path, prefix: str) -> List[Path]:
    """Find subdirectories matching pattern <prefix>-* (single level, not recursive)."""
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix + "-")]
    runs.sort()
    return runs


def clean_temp_files(folder: Path) -> int:
    """Remove single-level *.sqlite and derived files (including .lock, -journal, etc.) from folder."""
    removed = 0
    for pat in ("*.sqlite", "*.sqlite.lock", "*.sqlite*"):
        for f in folder.glob(pat):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def index_npz_by_name(folders: Iterable[Path]) -> Dict[str, List[Path]]:
    """
    Collect path lists for each filename across multiple directories:
    Returns mapping: filename -> [path_in_run0, path_in_run1, ...]
    (missing directories won't appear)
    """
    table: Dict[str, List[Path]] = {}
    for d in folders:
        for p in sorted(d.glob("*.npz")):
            table.setdefault(p.name, []).append(p)
    return table


def load_results(npz_path: Path) -> list[dict]:
    """
    Read results from .npz and uniformly convert to List[dict]:
      - If single dict -> [dict]
      - If object array/list/tuple -> flatten to [dict, dict, ...]
    Throws error for non-dict elements (to ensure consistent structure).
    """
    def _to_list_of_dicts(obj) -> list[dict]:
        # Single dict
        if isinstance(obj, dict):
            return [obj]
        # numpy array
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                out = []
                for x in obj.ravel():
                    if isinstance(x, dict):
                        out.append(x)
                    elif isinstance(x, (list, tuple)):
                        for y in x:
                            if isinstance(y, dict):
                                out.append(y)
                            else:
                                raise TypeError(f"results contains non-dict element of type {type(y)}")
                    else:
                        raise TypeError(f"results contains non-dict element of type {type(x)}")
                return out
            else:
                # Non-object dtype is basically impossible for the expected structure
                raise TypeError(f"unexpected results array dtype: {obj.dtype}")
        # Python list/tuple
        if isinstance(obj, (list, tuple)):
            out = []
            for x in obj:
                if isinstance(x, dict):
                    out.append(x)
                else:
                    raise TypeError(f"results contains non-dict element of type {type(x)}")
            return out
        # Other types not supported
        raise TypeError(f"unsupported results type: {type(obj)}")

    with np.load(npz_path, allow_pickle=True) as data:
        res_obj = data["results"]
        # Some save methods wrap single dict in 0-d object array
        if isinstance(res_obj, np.ndarray) and res_obj.ndim == 0 and res_obj.dtype == object:
            res_obj = res_obj.item()
        return _to_list_of_dicts(res_obj)



def merge_results_lists(results_lists: list[list[dict]], source_dirs: list[str]) -> list[dict]:
    """
    Merge multiple List[dict] into a single List[dict], adding 'source_dir' field to each dict.
    results_lists[i] corresponds one-to-one with source_dirs[i].
    """
    merged: list[dict] = []
    for lst, src in zip(results_lists, source_dirs):
        for d in lst:
            # Shallow copy to avoid modifying original object
            nd = dict(d)
            nd["source_dir"] = src
            merged.append(nd)
    return merged



def save_npz(out_path: Path, results_obj: Any) -> None:
    """Save as .npz with structure consistent with original: np.savez(out_path, results=results_obj)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, results=results_obj)


def merge_for_prefix(base: Path, prefix: str) -> None:
    """Execute cleanup + merge for a specific prefix."""
    run_dirs = find_run_dirs(base, prefix)
    if VERBOSE:
        print(f"\n==> Prefix '{prefix}': found {len(run_dirs)} run folder(s)")
    if not run_dirs:
        return

    # Target output directory (without -number suffix)
    out_dir = base / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean temporary files
    if CLEAN_TEMP:
        total_removed = 0
        for d in run_dirs:
            removed = clean_temp_files(d)
            total_removed += removed
            if VERBOSE and removed:
                print(f"  cleaned {removed:3d} temp files in {d.name}")
        if VERBOSE:
            print(f"  total cleaned temp files: {total_removed}")

    table = index_npz_by_name(run_dirs)
    if VERBOSE:
        print(f"  found {len(table)} unique .npz filenames across runs")

    merged_count = 0
    for fname, paths in sorted(table.items()):
        results_lists = []
        src_dirs = []
        for p in paths:
            try:
                lst = load_results(p)              # -> List[dict]
                results_lists.append(lst)
                src_dirs.append(p.parent.name)     # e.g. "opt_results_MSE-0"
            except Exception as e:
                if VERBOSE:
                    print(f"  [skip] failed to load '{p}': {e}")
        
        if not results_lists:
            continue
        
        merged_results_list = merge_results_lists(results_lists, src_dirs)  # -> List[dict]
        out_path = out_dir / fname
        save_npz(out_path, merged_results_list)
        merged_count += 1
        if VERBOSE:
            print(f"  merged {len(paths):2d} -> {out_path.relative_to(base)}")

    if VERBOSE:
        print(f"==> Done: {merged_count} file(s) merged into '{out_dir.name}/'.")


def main():
    for prefix in PREFIXES:
        merge_for_prefix(BASE_DIR, prefix)


if __name__ == "__main__":
    
    # ====== Configuration Section (Edit directly here in Spyder) ======
    BASE_DIR = Path(r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_results\New_CAMES_results")      # Top-level directory (containing opt_results_MSE-0 / -1 / ...)
    PREFIXES: List[str] = ["opt_results_xMSE"]   # Prefixes to process, can have multiple
    CLEAN_TEMP = True                   # Whether to delete *.sqlite / *.sqlite.lock / *.sqlite* temp files
    VERBOSE = True                      # Print progress
    # ==========================================
    main()