# -*- coding: utf-8 -*-
"""
Parameter sweep study for MCPBE (parallelized):
- scan parameter ranges
- compute R = x_50(end)/x_50(start)
- if x_50(t) is NOT monotonically decreasing -> mark R = -1 (still keep the record)
- save both model results and experimental ratio
"""

import os
from pathlib import Path
import numpy as np
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

from optframework import OptBase


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def is_monotone_decreasing(arr, tol=0.0):
    """Check monotone non-increasing."""
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2:
        return False
    return np.all(np.diff(arr) <= tol)


def safe_ratio(x50: np.ndarray) -> float:
    """Return x50[-1]/x50[0] if valid else nan."""
    x50 = np.asarray(x50, dtype=float)
    if x50.size < 2:
        return float("nan")
    if not np.isfinite(x50[0]) or x50[0] == 0:
        return float("nan")
    if not np.isfinite(x50[-1]):
        return float("nan")
    return float(x50[-1] / x50[0])


# ------------------------------------------------------------
# Parallel worker globals
# ------------------------------------------------------------
_G_OPT = None
_G_X_UNI_EXP = None
_G_DATA_EXP = None


def _init_worker(config_path: str, data_path: str, exp_data_path: str):
    """
    Initializer: runs once per process.
    Each process keeps its own OptBase + experimental data in globals.
    """
    global _G_OPT, _G_X_UNI_EXP, _G_DATA_EXP

    # Optional: avoid BLAS/OpenMP oversubscription inside each process
    # (uncomment if you see CPU oversubscription)
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    _G_OPT = OptBase(config_path=str(config_path), data_path=str(data_path))
    _G_X_UNI_EXP, _G_DATA_EXP = _G_OPT.core.p.get_all_data(str(exp_data_path))


def _run_one_case(args):
    """
    Run one parameter set in a worker process.
    Return one record row:
    [lmc_gamma, lmc_int_bre, lmc_energy_exp, lmc_lambda_E, lmc_NO_FRAG, CORR_BETA, ratio]
    """
    global _G_OPT, _G_X_UNI_EXP, _G_DATA_EXP

    (
        lmc_gamma, lmc_int_bre, lmc_energy_exp,
        lmc_lambda_E, lmc_NO_FRAG, CORR_BETA,
        pl_v, pl_P1, pl_P2
    ) = args

    pop_params = {
        "lmc_gamma": float(lmc_gamma),
        "lmc_int_bre": float(lmc_int_bre),
        "lmc_energy_exp": float(lmc_energy_exp),
        "lmc_lambda_E": float(lmc_lambda_E),
        "lmc_NO_FRAG": int(lmc_NO_FRAG),
        "CORR_BETA": float(CORR_BETA),
        "pl_v": float(pl_v),
        "pl_P1": float(pl_P1),
        "pl_P2": float(pl_P2),
    }

    ratio = -1.0  # default: invalid (non-monotone or failed)

    try:
        _ = _G_OPT.core.calc_delta(pop_params, _G_X_UNI_EXP, _G_DATA_EXP)

        p = _G_OPT.core.p
        x_50_mod = getattr(p, "x_50_mod", None)

        if x_50_mod is not None and len(x_50_mod) >= 2:
            x_50_mod = np.asarray(x_50_mod, dtype=float)
            if is_monotone_decreasing(x_50_mod):
                ratio = safe_ratio(x_50_mod)
            else:
                ratio = -1.0
        else:
            ratio = -1.0

    except Exception:
        ratio = -1.0

    return [
        float(lmc_gamma),
        float(lmc_int_bre),
        float(lmc_energy_exp),
        float(lmc_lambda_E),
        int(lmc_NO_FRAG),
        float(CORR_BETA),
        float(pl_v),
        float(pl_P1),
        float(pl_P2),
        float(ratio),
    ]


# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------
def run_param_sweep():
    # --------------------------------------------------------
    # Compute experimental ratio ONCE (serial, cheap)
    # --------------------------------------------------------
    opt0 = OptBase(config_path=str(config_path), data_path=str(data_path))
    _x_uni_exp, _data_exp = opt0.core.p.get_all_data(str(exp_data_path))

    p0 = opt0.core.p
    x50_exp = getattr(p0, "x_50_exp", None)
    exp_ratio = safe_ratio(x50_exp)

    # --------------------------------------------------------
    # Parameter ranges (your current choices)
    # --------------------------------------------------------
    lmc_gamma_list = np.array([5.0])
    lmc_int_bre_list = np.array([0.5])
    lmc_energy_exp_list = np.array([3.0])
    lmc_NO_FRAG_list = np.array([4])
    # lmc_lambda_E_list = np.array([1e-5, 1e-8, 1e-12])
    lmc_lambda_E_list = np.array([1])
    CORR_BETA_list = np.array([1])
    # CORR_BETA_list = np.array([1e-10, 1e-8, 1e-6])*1e3
    pl_v_list = np.array([0.5, 2.0])
    pl_P1_list = np.array([1e10, 1e15, 1e20])
    pl_P2_list = np.array([1.0,2.0,3.0])

    # --------------------------------------------------------
    # Prepare combinations
    # --------------------------------------------------------
    combos = list(itertools.product(
        lmc_gamma_list,
        lmc_int_bre_list,
        lmc_energy_exp_list,
        lmc_lambda_E_list,
        lmc_NO_FRAG_list,
        CORR_BETA_list,
        pl_v_list,
        pl_P1_list,
        pl_P2_list,
    ))

    total_cases = len(combos)
    print(f"Total parameter combinations: {total_cases}")
    print(f"Experimental x50 ratio (end/start): {exp_ratio}")

    # --------------------------------------------------------
    # Parallel sweep
    # --------------------------------------------------------
    records = []
    t_start = time.time()

    # Choose workers
    # - default: use SWEEP_WORKERS if set; else os.cpu_count()
    max_workers = 4
    chunksize = 2

    print(f"Using max_workers={max_workers}, chunksize={chunksize}")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(config_path), str(data_path), str(exp_data_path)),
    ) as ex:
        for i, row in enumerate(ex.map(_run_one_case, combos, chunksize=chunksize), start=1):
            records.append(row)
            if i % 50 == 0 or i == 1 or i == total_cases:
                print(f"[{i}/{total_cases}] done")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    records = np.asarray(records, dtype=float)

    np.savez(
        out_file,
        records=records,
        exp_ratio=np.asarray(exp_ratio, dtype=float),
        columns=np.asarray(
            [
                "lmc_gamma", "lmc_int_bre", "lmc_energy_exp",
                "lmc_lambda_E", "lmc_NO_FRAG", "CORR_BETA",
                "pl_v", "pl_P1", "pl_P2",
                "ratio",
            ],
            dtype=object,
        ),
        meta=np.asarray(
            {
                "exp_data_path": str(exp_data_path),
                "data_name": str(data_name),
                "note": "ratio=-1 indicates non-monotone x50(t) or failed run",
                "max_workers": int(max_workers),
                "chunksize": int(chunksize),
            },
            dtype=object,
        )
    )

    elapsed = time.time() - t_start
    print(f"Finished sweep in {elapsed:.1f} s")
    print(f"Total cases saved: {len(records)}")
    print(f"Results saved to: {out_file}")


def load_and_summarize(out_file: str | Path):
    out_file = Path(out_file)
    d = np.load(out_file, allow_pickle=True)
    records = d["records"]
    exp_ratio = float(d["exp_ratio"])
    cols = list(d["columns"].tolist())

    ratio_col = cols.index("ratio")
    valid_mask = records[:, ratio_col] >= 0.0

    n_total = records.shape[0]
    n_valid = int(np.sum(valid_mask))

    print("\n=== Loaded sweep results ===")
    print(f"File: {out_file}")
    print(f"Experimental x50 ratio (end/start): {exp_ratio}")
    print(f"Total cases: {n_total}")
    print(f"Valid (monotone) cases: {n_valid}")
    if n_valid > 0:
        r_valid = records[valid_mask, ratio_col]
        print(
            f"Model ratio (valid) min/mean/max: "
            f"{np.min(r_valid):.6g} / {np.mean(r_valid):.6g} / {np.max(r_valid):.6g}"
        )
        if np.isfinite(exp_ratio):
            err = np.abs(r_valid - exp_ratio)
            best_idx = np.argsort(err)[:5]
            print("\nTop-5 closest to experimental ratio:")
            valid_rows = records[valid_mask]
            for j in best_idx:
                row = valid_rows[j]
                print({cols[i]: row[i] for i in range(len(cols))})
    else:
        print("No valid cases found (all non-monotone or failed).")

    return records
# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    base_path = Path(os.getcwd()).resolve()
    # --- paths ---
    config_path = base_path / "config" / "opt_config_mcpbe.py"
    data_path = base_path / "data_mcpbe_CB"
    data_name = "CB_pur_N2000.h5"
    exp_data_path = data_path / data_name
    result_dir = base_path / "param_sweep_results"
    result_dir.mkdir(exist_ok=True)
    out_file = result_dir / "x50_ratio_scan_N2000_Q3.npz"
    
    run_param_sweep()
    records = load_and_summarize(out_file)
