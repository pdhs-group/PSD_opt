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
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import numpy as np
import shutil

def _to_list_of_dicts(obj) -> list[dict]:
    """Convert npz['results'] uniformly to List[dict], otherwise throw error."""
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0 and obj.dtype == object:
            return _to_list_of_dicts(obj.item())
        if obj.dtype != object:
            raise TypeError(f"unexpected results array dtype: {obj.dtype}")
        out: list[dict] = []
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
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            if isinstance(x, dict):
                out.append(x)
            else:
                raise TypeError(f"results contains non-dict element of type {type(x)}")
        return out
    raise TypeError(f"unsupported results type: {type(obj)}")

def load_results(npz_path: Path) -> list[dict]:
    """Read results from single npz file -> List[dict]."""
    with np.load(npz_path, allow_pickle=True) as data:
        if "results" not in data:
            raise KeyError(f"{npz_path} missing 'results'")
        return _to_list_of_dicts(data["results"])

def merge_results_lists(results_lists: list[list[dict]], src_tags: list[str], src_files: list[str]) -> list[dict]:
    """
    Merge multiple List[dict] into one List[dict], adding source information to each record:
    - source_tag: e.g. '0', '1' (extracted from filename -X_)
    - source_file: source filename
    """
    merged: list[dict] = []
    for lst, tag, file in zip(results_lists, src_tags, src_files):
        for d in lst:
            nd = dict(d)
            nd["source_tag"] = tag
            nd["source_file"] = file
            merged.append(nd)
    return merged

def save_npz(out_path: Path, results_obj: Any) -> None:
    """Save as npz with fixed key name 'results'."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, results=results_obj)

def find_group_files_single_dir(base: Path, prefix: str, iter_label: str | int) -> List[Path]:
    """
    Find files matching {prefix}-*_{iter}.npz in base directory (non-recursive).
    Example: prefix='opt_results_kva', iter_label=50
        -> matches 'opt_results_kva-0_50.npz', 'opt_results_kva-1_50.npz', ...
    """
    iter_str = str(iter_label)
    pattern = f"{prefix}-*_{iter_str}.npz"
    files = sorted(base.glob(pattern))
    return files

def extract_source_tag_from_name(fname: str, prefix: str, iter_label: str | int) -> str:
    """
    Extract -<tag>_ tag from filename, example:
        'opt_results_kva-12_50.npz' -> '12'
    If no match found, return empty string.
    """
    base = os.path.basename(fname)
    iter_str = str(iter_label)
    # Construct strict match: ^prefix-(tag)_(iter)\.npz$
    # prefix may contain underscores and alphanumeric chars, escape first
    pre_escaped = re.escape(prefix)
    m = re.match(rf"^{pre_escaped}-(?P<tag>[^_]+)_{re.escape(iter_str)}\.npz$", base, flags=re.IGNORECASE)
    return m.group("tag") if m else ""

def merge_single_dir(prefixes: Iterable[str],
                         iters: Iterable[int | str],
                         base_dir: Path,
                         out_dir: Path | None = None,
                         verbose: bool = True) -> None:
    """
    In single directory base_dir, for each (prefix, iter) combination:
      - Collect prefix-*_{iter}.npz
      - Load results -> List[dict] one by one
      - Merge and write out as out_dir / f"{prefix}_{iter}.npz"
    """
    if out_dir is None:
        out_dir = base_dir / "merged"

    total_groups = 0
    total_outputs = 0

    for prefix in prefixes:
        for it in iters:
            total_groups += 1
            files = find_group_files_single_dir(base_dir, prefix, it)
            if verbose:
                print(f"\n==> Group '{prefix}_{it}': found {len(files)} file(s)")

            if not files:
                continue

            results_lists: list[list[dict]] = []
            src_tags: list[str] = []
            src_files: list[str] = []

            for p in files:
                try:
                    lst = load_results(p)
                except Exception as e:
                    if verbose:
                        print(f"  [skip] load failed: {p.name} -> {e}")
                    continue
                tag = extract_source_tag_from_name(p.name, prefix, it)
                results_lists.append(lst)
                src_tags.append(tag)
                src_files.append(p.name)

            if not results_lists:
                if verbose:
                    print("  [skip] nothing loaded successfully.")
                continue

            def _tag_key(idx: int):
                t = src_tags[idx]
                try:
                    return (0, int(t))
                except Exception:
                    return (1, t)

            order = sorted(range(len(results_lists)), key=_tag_key)
            results_lists = [results_lists[i] for i in order]
            src_tags      = [src_tags[i] for i in order]
            src_files     = [src_files[i] for i in order]

            merged = merge_results_lists(results_lists, src_tags, src_files)
            out_path = out_dir / f"{prefix}_{it}.npz"
            save_npz(out_path, merged)
            total_outputs += 1
            if verbose:
                print(f"  merged {len(results_lists):2d} -> {out_path.relative_to(base_dir)}")

    if verbose:
        print(f"\n==> Done. planned groups: {total_groups}, written outputs: {total_outputs}")

if __name__ == "__main__":
    
    # ====== Configuration Section (Edit directly here in Spyder) ======
    BASE_DIR = Path(r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_results\New_CAMES_results\summaries_array")      # Top-level directory (containing opt_results_MSE-0 / -1 / ...)
    PREFIXES: List[str] = ["kva", "MSEa", "nna"]   # Prefixes to process, can have multiple
    ITERS: List[int] = [5,10,15,20,25,30,35,40,45,50,\
                55,60,65,70,75,80,85,90,95,100,\
                110,120,130,140,150,160,170,180,190,200,\
                220,240,260,280,300,320,340,360,380,400,\
                440,480,520,560,600,640,680,720,760,800,\
                880,960,1040,1120,1200,1280,1360,1440,1520,1600,\
                1680,1760,1840,1920,2000,2080,2160,2240,2320,2400,\
                2480,2560,2640,2720,2800,2880,2960,3040,3120,3200,\
                3360,3520,3680,3840,4000,4160,4320,4480,4640,4800,\
                4960,5120,5280,5440,5600,5760,5920,6080,6240,6400]
    VERBOSE = True                      # Print progress
    OUT_DIR = BASE_DIR / "merged"
    merge_single_dir(PREFIXES, ITERS, BASE_DIR, OUT_DIR, VERBOSE)