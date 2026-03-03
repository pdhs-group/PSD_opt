# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 10:48:23 2025

@author: Haoran Ji (px2030@kit.edu)

Offline summarization of the warm_params SQLite database written during Ray Tune.
For each data_name (i.e., your filename), compute the best result within the
first N steps and write a list of dictionaries that matches the original
result_dict structure into N.npz.

Usage:
1) CLI:
    python summarize_params_from_sqlite.py \
        --db /path/to/1600.sqlite \
        --steps 50 100 200 400 800 1600 \
        --outdir ./summaries \
        --filter nameA nameB

2) Debug/interactive mode (auto when no CLI args; convenient for Spyder/Jupyter):
    Run the script and follow the prompts.
"""

import argparse
import json
import os
import sqlite3
import sys
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# Core functionality
# -----------------------------
def load_all_records(db_path: str) -> Dict[str, List[Tuple[dict, float]]]:
    """Read all rows from SQLite, group by data_name, and keep insertion order."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite file does not exist: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT data_name, param_json, score, id
        FROM warm_params
        ORDER BY data_name ASC, id ASC
    """)
    rows = cursor.fetchall()
    conn.close()

    grouped: Dict[str, List[Tuple[dict, float]]] = defaultdict(list)
    for data_name, param_json, score, _id in rows:
        try:
            params = json.loads(param_json) if param_json is not None else {}
        except json.JSONDecodeError:
            params = {}
        grouped[data_name].append((params, float(score) if score is not None else float("inf")))
    return dict(grouped)


def best_so_far_prefix(records: List[Tuple[dict, float]], n: int) -> Tuple[dict, float]:
    """Within the first n records, find the minimal-score (params, score)."""
    if not records:
        return {}, float("inf")

    end = min(n, len(records))
    best_idx = 0
    best_score = records[0][1]
    for i in range(1, end):
        sc = records[i][1]
        if sc < best_score:
            best_idx, best_score = i, sc
    return records[best_idx][0], best_score


def build_result_dict(opt_params: dict, opt_score: float, data_name: str) -> dict:
    """Build a result_dict consistent with the framework; use data_name as file_path."""
    return {
        "opt_score": opt_score,
        "opt_params": opt_params,
        "file_path": data_name,
    }


def run_summarize(db_path: str,
                  steps: List[int],
                  outdir: str,
                  filters: Optional[List[str]],
                  prefix: str = "") -> None:
    """
    Compute summaries and save to N.npz.
    If prefix is provided, output file name is {prefix}_{N}.npz; otherwise {N}.npz.
    """
    steps = sorted(set(int(s) for s in steps if int(s) > 0))
    os.makedirs(outdir, exist_ok=True)

    grouped = load_all_records(db_path)
    if not grouped:
        print(f"[WARN] No records found in database: {db_path}")
        return

    data_names = sorted(grouped.keys())
    if filters:
        filt = set(filters)
        missing = [name for name in filt if name not in grouped]
        if missing:
            print(f"[WARN] The following data_name values do not exist in the DB and will be ignored: {missing}")
        data_names = [name for name in data_names if name in filt]

    if not data_names:
        print("[WARN] No data_name matched the given filter.")
        return

    print(f"[INFO] Loaded {len(data_names)} data_name groups from {db_path}.")
    print(f"[INFO] Steps to summarize: {steps}")
    if prefix:
        print(f"[INFO] Output filename prefix: {prefix}_")

    for n in steps:
        results_for_n: List[dict] = []

        for name in data_names:
            recs = grouped[name]
            if not recs:
                rd = build_result_dict({}, float("inf"), name)
            else:
                params, score = best_so_far_prefix(recs, n)
                rd = build_result_dict(params, score, name)

            results_for_n.append(rd)

        fname = f"{prefix}_{n}.npz" if prefix else f"{n}.npz"
        save_path = os.path.join(outdir, fname)
        np.savez_compressed(save_path, results=np.array(results_for_n, dtype=object))
        print(f"[OK] Saved: {save_path} (contains {len(results_for_n)} data_name best-of-first-{n} results)")

    print("[DONE] All steps summarized.")
    print("Read example:")
    print("  import numpy as np")
    print("  npz = np.load('N.npz', allow_pickle=True'); results = npz['results'].tolist()")
    print("  # results is a list[dict]; each dict matches the original result_dict structure.")


def run_summarize_for_root(results_root: str,
                           steps: List[int],
                           outdir: str,
                           db_name: str = "1600.sqlite",
                           filters: Optional[List[str]] = None) -> None:
    """
    Traverse the first-level subdirectories under results_root and look for
    SQLite files at {subdir}/{db_name}. For each subdirectory, run the
    summarization and use the subdirectory name as the output prefix:
        {subdir}_{N}.npz
    """
    if not os.path.isdir(results_root):
        raise NotADirectoryError(f"Not a valid directory: {results_root}")

    os.makedirs(outdir, exist_ok=True)

    subdirs = [d for d in os.listdir(results_root)
               if os.path.isdir(os.path.join(results_root, d))]
    # subdirs = [os.path.join(results_root, "KL")]

    if not subdirs:
        print(f"[WARN] No subdirectories found under root: {results_root}")
        return

    found_any = False
    for sub in sorted(subdirs):
        db_path = os.path.join(results_root, sub, db_name)
        if not os.path.exists(db_path):
            print(f"[SKIP] Subdirectory {sub} does not contain {db_name}")
            continue

        found_any = True
        print(f"\n====== Processing subdirectory: {sub} | DB: {db_path} ======")
        # Output to the shared outdir; filenames get the prefix
        run_summarize(db_path=db_path,
                      steps=steps,
                      outdir=outdir,
                      filters=filters,
                      prefix=sub)

    if not found_any:
        print(f"[WARN] Did not find {db_name} in any subdirectory")


# -----------------------------
# CLI entry (single DB or traverse root)
# -----------------------------
def cli_main():
    parser = argparse.ArgumentParser(
        description="Summarize best-of-first-N results from warm_params SQLite; support single DB or traversing a root directory (output {prefix}_{N}.npz)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--db", help="Single SQLite file path, e.g., /path/to/1600.sqlite")
    mode.add_argument("--root", help="Root directory; traverse its first-level subdirectories and look for {sub}/{db_name}")

    parser.add_argument("--db-name", default="1600.sqlite",
                        help="In --root mode, the sqlite filename to search for (default: 1600.sqlite)")
    parser.add_argument("--steps", nargs="+", type=int, required=True,
                        help="List of steps to summarize, e.g.: 50 100 200 400 800 1600")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (default: current directory)")
    parser.add_argument("--filter", nargs="*", default=None,
                        help="Only summarize the specified data_name values; empty means all")

    args = parser.parse_args()

    if args.root:
        run_summarize_for_root(results_root=args.root,
                               steps=args.steps,
                               outdir=args.outdir,
                               db_name=args.db_name,
                               filters=args.filter)
    else:
    # Single DB mode uses no prefix
        run_summarize(db_path=args.db,
                      steps=args.steps,
                      outdir=args.outdir,
                      filters=args.filter,
                      prefix="")


# -----------------------------
# Debug/interactive main (good for Spyder/Jupyter)
# -----------------------------
def debug_main():
    print("=== Debug interactive mode ===")
    print("Traverse first-level subdirectories under the specified root, find 1600.sqlite, and output {subdir}_{N}.npz")

    # Your default paths
    results_root = results_path  # Root directory containing subdirs like MSE, MAE, etc.
    db_name = "3200.sqlite"
    steps_str = "50,100,200,400,800,1600,2400,3200"
    # steps_str = "5,10,15,20,25,30,35,40,45,50,\
    #             55,60,65,70,75,80,85,90,95,100,\
    #             110,120,130,140,150,160,170,180,190,200,\
    #             220,240,260,280,300,320,340,360,380,400,\
    #             440,480,520,560,600,640,680,720,760,800,\
    #             880,960,1040,1120,1200,1280,1360,1440,1520,1600,\
    #             1680,1760,1840,1920,2000,2080,2160,2240,2320,2400,\
    #             2480,2560,2640,2720,2800,2880,2960,3040,3120,3200,\
    #             3360,3520,3680,3840,4000,4160,4320,4480,4640,4800,\
    #             4960,5120,5280,5440,5600,5760,5920,6080,6240,6400"
    outdir = os.path.join(results_root, "summaries_array")
    filters_str = ""  # Comma-separated data_name list; empty means all

    # Parse steps
    steps = []
    for tok in steps_str.split(","):
        tok = tok.strip()
        if tok:
            try:
                v = int(tok)
                if v > 0:
                    steps.append(v)
            except ValueError:
                print(f"[WARN] Failed to parse step value: {tok}; ignored.")

    # Parse data_name filters
    filters = [s.strip() for s in filters_str.split(",") if s.strip()] if filters_str else None

    print("\n[DEBUG] Parameter confirmation:")
    print(f"  root    : {results_root}")
    print(f"  db_name : {db_name}")
    print(f"  steps   : {steps}")
    print(f"  outdir  : {outdir}")
    print(f"  filters : {filters}")
    print()

    run_summarize_for_root(results_root=results_root,
                           steps=steps,
                           outdir=outdir,
                           db_name=db_name,
                           filters=filters)

def compare_npz(script_npz_path: str, framework_npz_path: str):
    """
    Compare the npz summarized by this script with the npz produced directly by the framework.

    Parameters:
        script_npz_path : str  Path to the script-summarized npz (e.g., "1600.npz")
        framework_npz_path : str  Path to the framework-produced npz (e.g., "1600_opt.npz")

    Returns:
        pandas.DataFrame : columns: data_name, score_script, score_framework, equal
    """
    # Read npz
    script_npz_path = os.path.join(results_path, script_npz_path)
    framework_npz_path = os.path.join(results_path, framework_npz_path)
    script_npz = np.load(script_npz_path, allow_pickle=True)["results"].tolist()
    framework_npz = np.load(framework_npz_path, allow_pickle=True)["results"].tolist()

    # Build a dict for the script data: {data_name -> opt_score}
    script_dict = {
        item["file_path"]: item["opt_score"] for item in script_npz
    }
    # script_dict = {}
    # for idx, item in enumerate(script_npz):
    #     data_name = item["file_path"]  # In the script output, file_path equals data_name
    #     if data_name in script_dict:
    #         print(f"[WARN] Duplicate in script npz: {data_name} "
    #               f"(old score={script_dict[data_name]}, new score={item['opt_score']}, index={idx})")
    #     script_dict[data_name] = item["opt_score"]

    # Framework data -> extract data_name (from file_path[0], take middle of Sim_...xlsx)
    framework_dict = {}
    i = 0
    for idx, item in enumerate(framework_npz):
        file_paths = item["file_path"]
        if not isinstance(file_paths, list):
            continue
        first_path = os.path.basename(file_paths[0])  # get filename
        if first_path.startswith("Sim_") and first_path.endswith(".xlsx"):
            data_name = first_path[len("Sim_"):-len(".xlsx")]
        else:
            data_name = first_path  # fallback
        if data_name in framework_dict:
            i += 1
            print(f"[WARN] Duplicate in framework npz: {data_name} "
                  f"(old score={framework_dict[data_name]}, new score={item['opt_score']}, index={idx})")
        print(f"Total duplicates encountered so far: {i}")
        framework_dict[data_name] = item["opt_score"]

    # Align and merge
    rows = []
    for data_name, score_s in script_dict.items():
        score_f = framework_dict.get(data_name, None)
        equal = (score_s == score_f) if score_f is not None else False
        rows.append({
            "data_name": data_name,
            "score_script": score_s,
            "score_framework": score_f,
            "equal": equal
        })

    df = pd.DataFrame(rows)
    return df

# -----------------------------
# Entry point: choose mode based on presence of CLI args
# -----------------------------
if __name__ == "__main__":
    # Set to True to force debug interactive mode (even if CLI args are present)
    FORCE_DEBUG = True
    results_path = r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_results\New_CAMES_results\results210925"
    
    if FORCE_DEBUG or len(sys.argv) == 1:
        # No CLI args -> enter interactive mode (Spyder/Jupyter friendly)
        debug_main()
    else:
        cli_main()
        
    
    # df = compare_npz("MSE_3200.npz", "MSE_3200_opt.npz")
    # print(df.head())
    # csv_save_path = os.path.join(results_path, "compare_3200.csv")
    # df.to_csv(csv_save_path, index=False)

