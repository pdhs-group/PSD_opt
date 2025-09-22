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

def find_run_dirs(base: Path, prefix: str) -> List[Path]:
    """Find subdirectories matching pattern <prefix>-* (single level, not recursive)."""
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix + "-")]
    runs.sort()
    return runs

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

def find_group_files_single_dir(base: Path, prefix: str, iter_label: str | int) -> List[Path]:
    """
    在 base 目录下查找匹配 {prefix}-*_{iter}.npz 的文件（不递归）。
    例如：prefix='opt_results_kva', iter_label=50
        -> 匹配 'opt_results_kva-0_50.npz', 'opt_results_kva-1_50.npz', ...
    """
    iter_str = str(iter_label)
    pattern = f"{prefix}-*_{iter_str}.npz"
    files = sorted(base.glob(pattern))
    return files

def extract_source_tag_from_name(fname: str, prefix: str, iter_label: str | int) -> str:
    """
    从文件名中提取 -<tag>_ 的 tag，例：
        'opt_results_kva-12_50.npz' -> '12'
    若未匹配到，返回空字符串。
    """
    base = os.path.basename(fname)
    iter_str = str(iter_label)
    # 构造严格匹配：^prefix-(tag)_(iter)\.npz$
    # prefix 可能含下划线和字母数字，先转义
    pre_escaped = re.escape(prefix)
    m = re.match(rf"^{pre_escaped}-(?P<tag>[^_]+)_{re.escape(iter_str)}\.npz$", base, flags=re.IGNORECASE)
    return m.group("tag") if m else ""

def merge_single_dir(prefixes: Iterable[str],
                         iters: Iterable[int | str],
                         base_dir: Path,
                         out_dir: Path | None = None,
                         verbose: bool = True) -> None:
    """
    在单个目录 base_dir 中，对每个 (prefix, iter) 组合：
      - 收集 prefix-*_{iter}.npz
      - 逐个加载 results -> List[dict]
      - 合并并写出为 out_dir / f"{prefix}_{iter}.npz"
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

            # 为了输出稳定，把 (tag, file, list) 按 tag 的“数字优先”排序
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

def merge_multi_dir():
    for prefix in PREFIXES:
        merge_for_prefix(BASE_DIR, prefix)


if __name__ == "__main__":
    
    # ====== Configuration Section (Edit directly here in Spyder) ======
    BASE_DIR = Path(r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_results\New_CAMES_results\summaries_array")      # Top-level directory (containing opt_results_MSE-0 / -1 / ...)
    PREFIXES: List[str] = ["kva", "MSEa", "nna"]   # Prefixes to process, can have multiple
    ITERS: List[int] = [50, 100, 200, 400, 800, 1600, 2400, 3200, 4800, 6400]
    VERBOSE = True                      # Print progress
    # ==========================================
    # merge_multi_dir()
    
    OUT_DIR = BASE_DIR / "merged"
    merge_single_dir(PREFIXES, ITERS, BASE_DIR, OUT_DIR, VERBOSE)