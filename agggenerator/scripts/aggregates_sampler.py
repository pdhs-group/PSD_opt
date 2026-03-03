# -*- coding: utf-8 -*-
"""
Offline builder for an MPTSA + MAS aggregate pool (pre-generating grids for LMC).

Recommended usage (e.g. in Spyder):
    1. Open this file in the IDE
    2. Adjust the parameters in the “User configuration” section
    3. In the console, call: build_pool()
"""

from __future__ import annotations

import os
import math
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

import numpy as np
import h5py

from agggenerator.mptsa2d import (
    MPTSALatticeParams2D,
    generate_mptsa_lattice_2d,
    estimate_fractal_dimension_2d,
)
from agggenerator.material_mix import (
    MaterialMixParams,
    MASPhysicalParams,
    assign_materials_with_target_mas,
)
# GridFactory lives lmc.core.grid; adjust the import path if needed.
from lmc.core import GridFactory  # :contentReference[oaicite:1]{index=1}

# ============================================================
# User configuration (edit here inside Spyder)
# ============================================================

# Shape parameters: fixed for one full simulation campaign
TARGET_DF: float = 1.8          # target fractal dimension
TARGET_MAS: float = 0.40        # target Mischgüte (MAS)

# Acceptance tolerances (absolute)
DF_TOL: float = 0.1             # |Df_est - TARGET_DF| <= DF_TOL  (currently not enforced)
MAS_TOL: float = 0.05           # |MAS_actual - TARGET_MAS| <= MAS_TOL

# Parameter grid: we only sample in Np and frac_A
NP_LIST: List[int] = [100, 200, 400, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 16000,
                      20000, 30000, 40000, 50000]   # target occupied cell count
# NP_LIST: List[int] = [100, 200, 400, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000]
# FRAC_A_LIST: List[float] = [0.0, 0.3, 0.5, 0.8, 1.0]   # target fraction of material A
FRAC_A_LIST: List[float] = [1.0]

SAMPLES_PER_PARAM: int = 100    # accepted samples per (Np, frac_A) pair
MAX_TRIES_FACTOR: int = 100     # max attempts = SAMPLES_PER_PARAM * MAX_TRIES_FACTOR

# Output path for a single HDF5 container
OUTPUT_H5_PATH: str = "aggregate_pool_Df1p8_MAS0p40.h5"

# Parallelism: 1 = serial; >1 = use multiprocessing
WORKERS: int = 6

# Top-level RNG seed
MASTER_SEED: int = 42

# Parameters for GridFactory.make_from_array
A0_CELL_AREA: float = 1.0       # physical area per occupied cell
INT_BRE: float = 0.0            # initial breakage depth ratio (consistent with LMC config)


# ============================================================
# Internal helpers
# ============================================================

def _sanitize_group_name(Np: int, frac_A: float) -> str:
    """
    Convert a (Np, frac_A) pair into an HDF5 group name.

    Example
    -------
    Np = 3000, frac_A = 0.35  ->  'Np3000_XA0350'
    """
    xa_int = int(round(frac_A * 10000))
    return f"Np{Np}_XA{xa_int:04d}"


def _make_mptsa_params(Np: int, Df: float, seed: int) -> MPTSALatticeParams2D:
    """
    Construct MPTSA parameters.

    This mirrors the defaults used in generator.py, overriding only
    Np, Df and seed. Other settings (k, max_attempts, hole filling,
    etc.) are chosen to produce robust aggregates for the pool.
    :contentReference[oaicite:2]{index=2}
    """
    return MPTSALatticeParams2D(
        Np=int(Np),
        Df=float(Df),
        k=1.0,
        max_attempts=50000,
        seed=int(seed),
        fill_hole=True,
        hole_area_max=4,
        compensate_alpha=1.0,
        compensate_beta=0.25,
        verbose=False,   # keep silent when generating many samples
    )


def _make_mix_params(frac_A: float, target_MAS: float, seed: int) -> MaterialMixParams:
    """
    Construct material mixing parameters.

    This follows the defaults in generator.py, overriding only
    frac_A, target_MAS and seed. Here tol_MAS is kept relatively
    loose; the final acceptance is controlled by MAS_TOL at pool level.
    :contentReference[oaicite:3]{index=3}
    """
    return MaterialMixParams(
        frac_A=float(frac_A),
        target_MAS=float(target_MAS),
        tol_MAS=0.05,       # internal convergence tolerance; global filter uses MAS_TOL
        window=12,
        stride=3,
        sweeps_per_eval=8,
        max_bisect=10,
        seed=int(seed),
        # lambda_min / lambda_max can be adjusted here if needed
    )


def _compute_actual_frac_A(labels: np.ndarray) -> float:
    """
    Compute the actual fraction of material A from the label field.

    Labels convention:
      -1 : empty
       0 : material A
       1 : material B
    """
    occ = labels >= 0
    total = int(occ.sum())
    if total == 0:
        return 0.0
    nA = int((labels == 0).sum())
    return nA / total


def _generate_samples_for_param(
    Np: int,
    frac_A: float,
    target_Df: float,
    target_MAS: float,
    df_tol: float,
    mas_tol: float,
    n_samples: int,
    max_tries_factor: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    """
    Generate a collection of accepted samples for a given (Np, frac_A).

    This function runs in a worker process and returns a list of dicts,
    each representing one accepted aggregate and its metadata:

        {
          "Np_target": ...,
          "frac_A_target": ...,
          "Df_target": ...,
          "Df_est": ...,
          "MAS_target": ...,
          "MAS_actual": ...,
          "frac_A_actual": ...,
          "labels": labels (int8, -1/0/1),
          "M": M,
          "Hbond": Hbond,
          "Vbond": Vbond,
          "meta": meta_dict,
          "origin": origin,
          "bond_counts": bond_counts,
          "seed_mptsa": ...,
          "seed_mix": ...,
        }

    Notes
    -----
    - Any HDF5 writing is done in the main process; workers only return
      numpy arrays and small Python dicts.
    - Currently the Df filter is commented out; only MAS acceptance is
      enforced (except when frac_A is 0 or 1, where MAS is ignored).
    """
    rng = np.random.default_rng(base_seed)
    grid_factory = GridFactory()
    phys_params = MASPhysicalParams()  # default physical parameters for MAS

    accepted: List[Dict[str, Any]] = []

    max_tries = int(max(1, n_samples * max_tries_factor))
    tries = 0

    while len(accepted) < n_samples and tries < max_tries:
        tries += 1
        # Independent seeds for shape (MPTSA) and mixing (MCMC)
        seed_mptsa = int(rng.integers(0, 2**31 - 1))
        seed_mix = int(rng.integers(0, 2**31 - 1))

        # ---- 1) Generate aggregate geometry via MPTSA ----
        mptsa_params = _make_mptsa_params(Np=Np, Df=target_Df, seed=seed_mptsa)

        try:
            positions, Ns, Rgs, grid, origin = generate_mptsa_lattice_2d(mptsa_params)
            Df_est, slope = estimate_fractal_dimension_2d(Ns, Rgs)

            # ---- 2) Assign materials via MCMC + MAS target ----
            mix_params = _make_mix_params(frac_A=frac_A, target_MAS=target_MAS, seed=seed_mix)
            labels, stats = assign_materials_with_target_mas(grid, mix_params, phys_params)
        except Exception as e:
            # Critical: dump failing samples to help debugging
            dump = {
                "error": str(e),
                "Np": Np,
                "frac_A": frac_A,
                "Df_target": target_Df,
                "seed_mptsa": seed_mptsa,
                "seed_mix": seed_mix,
                "grid_shape": None if "grid" not in locals() else grid.shape,
            }
            np.save(
                f"error_dump_{Np}_{frac_A}_{seed_mptsa}_{seed_mix}.npy",
                dump,
                allow_pickle=True,
            )
            continue

        # Original Df filter (currently disabled for throughput debugging):
        # if abs(Df_est - target_Df) > df_tol:
        #     continue

        # MAS acceptance (skipped for pure A/B cases)
        MAS_actual = float(stats.get("MAS", np.nan))
        if 0.0 < frac_A < 1.0:
            if (not np.isfinite(MAS_actual)) or (abs(MAS_actual - target_MAS) > mas_tol):
                continue
        else:
            # For pure A or pure B, we keep the numeric value but do not filter on MAS.
            MAS_actual = float(MAS_actual)

        # Actual fraction of A in the final label field
        frac_A_actual = _compute_actual_frac_A(labels)

        # ---- 3) Convert to LMC internal grid (M, Hbond, Vbond, meta) ----
        # Convention: labels with -1=empty, 0=A, 1=B
        M, Hbond, Vbond, meta, bond_counts = grid_factory.make_from_array(
            labels,
            a_code=0,
            b_code=1,
            empty_code=-1,
            A0=A0_CELL_AREA,
            int_bre=INT_BRE,
        )

        meta_dict = asdict(meta)  # GridMeta -> dict for serialization

        sample = dict(
            Np_target=int(Np),
            frac_A_target=float(frac_A),
            Df_target=float(target_Df),
            Df_est=float(Df_est),
            MAS_target=float(target_MAS),
            MAS_actual=float(MAS_actual),
            frac_A_actual=float(frac_A_actual),
            slope=float(slope),
            labels=labels.astype(np.int8, copy=False),
            M=M,
            Hbond=Hbond,
            Vbond=Vbond,
            meta=meta_dict,
            origin=np.array(origin, dtype=np.int32),
            bond_counts=bond_counts,
            seed_mptsa=int(seed_mptsa),
            seed_mix=int(seed_mix),
        )
        accepted.append(sample)

    return accepted


def _write_samples_to_h5(
    h5: h5py.File,
    group_name: str,
    samples: List[Dict[str, Any]],
    start_index: int = 0,
) -> None:
    """
    Append a batch of samples into an open HDF5 file under /group_name/.

    This function does NOT clear or overwrite existing samples. New samples
    are written starting from index = start_index:

        /group_name/
            sample_0000/   (possibly from a previous run)
            ...
            sample_00NN/   (existing samples)
            sample_0NNN/   (newly appended samples)

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    group_name : str
        Name of the group for this (Np, frac_A) combination.
    samples : list of dict
        Samples to be written. Each dict must have keys "M", "Hbond",
        "Vbond", "labels", "origin", "meta", "bond_counts", etc.
    start_index : int, default 0
        Index offset for naming new samples. The first sample in this
        batch will be named f"sample_{start_index:04d}".
    """
    grp = h5.require_group(group_name)

    for local_idx, s in enumerate(samples):
        idx = start_index + local_idx
        sub = grp.create_group(f"sample_{idx:04d}")
        # main datasets
        sub.create_dataset("M", data=s["M"], compression="gzip")
        sub.create_dataset("Hbond", data=s["Hbond"], compression="gzip")
        sub.create_dataset("Vbond", data=s["Vbond"], compression="gzip")
        sub.create_dataset("labels", data=s["labels"], compression="gzip")
        sub.create_dataset("origin", data=s["origin"])

        # meta fields as attributes
        meta_dict: Dict[str, Any] = s["meta"]
        for k, v in meta_dict.items():
            sub.attrs[f"meta_{k}"] = v

        # scalar metadata
        sub.attrs["Np_target"] = s["Np_target"]
        sub.attrs["frac_A_target"] = s["frac_A_target"]
        sub.attrs["Df_target"] = s["Df_target"]
        sub.attrs["Df_est"] = s["Df_est"]
        sub.attrs["MAS_target"] = s["MAS_target"]
        sub.attrs["MAS_actual"] = s["MAS_actual"]
        sub.attrs["frac_A_actual"] = s["frac_A_actual"]
        sub.attrs["slope"] = s["slope"]
        sub.attrs["seed_mptsa"] = s["seed_mptsa"]
        sub.attrs["seed_mix"] = s["seed_mix"]

        # bond_counts (keys like 11,12,22 for bond types)
        bc: Dict[int, int] = s["bond_counts"]
        for key, val in bc.items():
            sub.attrs[f"bond_count_{key}"] = int(val)



# ============================================================
# Top-level driver
# ============================================================
def build_pool() -> None:
    """
    Build or extend the aggregate pool and write it to OUTPUT_H5_PATH.

    Incremental / safe behavior
    ---------------------------
    - The HDF5 file is opened in append mode ("a"):
        * If the file does not exist, it is created.
        * If it exists, existing groups / samples are preserved.
    - For each (Np, frac_A) pair:
        * If the group already has >= SAMPLES_PER_PARAM samples, it is skipped.
        * If it has fewer samples, only the missing samples are generated
          and appended (starting from the next sample index).
    - Top-level attributes (TARGET_DF, TARGET_MAS, etc.) are written only
      if they are not already present. If they exist and differ from the
      current configuration, a warning is printed but the values are not
      overwritten.

    This allows you to:
      - add new (Np, frac_A) points to an existing pool,
      - or increase SAMPLES_PER_PARAM and re-run to top up missing samples,
      without losing the data already stored in the HDF5 file.
    """
    os.makedirs(os.path.dirname(OUTPUT_H5_PATH) or ".", exist_ok=True)

    # List of parameter pairs [(Np, frac_A), ...]
    param_pairs: List[Tuple[int, float]] = [
        (Np, frac_A) for Np in NP_LIST for frac_A in FRAC_A_LIST
    ]

    print(
        f"[POOL] Target pairs: {len(param_pairs)} (Np, frac_A) combinations, "
        f"{SAMPLES_PER_PARAM} accepted samples per pair."
    )
    print(
        f"[POOL] Fixed targets: Df={TARGET_DF}, MAS={TARGET_MAS}, "
        f"DF_TOL={DF_TOL}, MAS_TOL={MAS_TOL}"
    )
    print(f"[POOL] HDF5 output: {OUTPUT_H5_PATH}")
    print(f"[POOL] workers={WORKERS}")

    rng_master = np.random.default_rng(MASTER_SEED)

    # Pre-generate a base_seed for each parameter pair to ensure reproducibility
    param_seeds = {
        (Np, frac_A): int(rng_master.integers(0, 2**31 - 1))
        for (Np, frac_A) in param_pairs
    }

    # Open the HDF5 file in append mode and write incrementally
    with h5py.File(OUTPUT_H5_PATH, "a") as h5:
        # --- Top-level attributes: write only if missing; warn if inconsistent ---
        global_attrs = dict(
            TARGET_DF=TARGET_DF,
            TARGET_MAS=TARGET_MAS,
            DF_TOL=DF_TOL,
            MAS_TOL=MAS_TOL,
            A0_CELL_AREA=A0_CELL_AREA,
            INT_BRE=INT_BRE,
            MASTER_SEED=MASTER_SEED,
        )
        for k, v in global_attrs.items():
            if k in h5.attrs:
                if h5.attrs[k] != v:
                    print(
                        f"[POOL][WARN] HDF5 attr '{k}' = {h5.attrs[k]} "
                        f"differs from current config ({v}). Keeping existing value."
                    )
            else:
                h5.attrs[k] = v

        # --- Inspect existing groups to know how many samples are already stored ---
        existing_counts: Dict[Tuple[int, float], int] = {}
        for (Np, frac_A) in param_pairs:
            gname = _sanitize_group_name(Np, frac_A)
            if gname in h5:
                grp = h5[gname]
                # Prefer the stored attribute; fall back to counting subgroups
                n_existing = int(grp.attrs.get("n_samples", 0))
                if n_existing <= 0:
                    # Count groups whose name starts with 'sample_'
                    n_existing = sum(
                        1 for name in grp.keys() if str(name).startswith("sample_")
                    )
                existing_counts[(Np, frac_A)] = n_existing
            else:
                existing_counts[(Np, frac_A)] = 0

        # --- Build a list of pairs that still need more samples ---
        todo_pairs: List[Tuple[int, float, int]] = []
        for (Np, frac_A) in param_pairs:
            already = existing_counts[(Np, frac_A)]
            need = max(0, SAMPLES_PER_PARAM - already)
            if need > 0:
                todo_pairs.append((Np, frac_A, need))

        if not todo_pairs:
            print("[POOL] All (Np, frac_A) pairs already have enough samples. Nothing to do.")
            return

        print(
            f"[POOL] {len(todo_pairs)} parameter pairs need more samples "
            f"(SAMPLES_PER_PARAM={SAMPLES_PER_PARAM})."
        )

        # --- Serial mode ---
        if WORKERS <= 1:
            for (Np, frac_A, need) in todo_pairs:
                already = existing_counts[(Np, frac_A)]
                print(
                    f"[POOL] (serial) Np={Np}, frac_A={frac_A:.4f}: "
                    f"{already} existing, generating {need} more..."
                )
                samples = _generate_samples_for_param(
                    Np=Np,
                    frac_A=frac_A,
                    target_Df=TARGET_DF,
                    target_MAS=TARGET_MAS,
                    df_tol=DF_TOL,
                    mas_tol=MAS_TOL,
                    n_samples=need,
                    max_tries_factor=MAX_TRIES_FACTOR,
                    base_seed=param_seeds[(Np, frac_A)],
                )
                print(
                    f"[POOL] (Np={Np}, frac_A={frac_A:.4f}) "
                    f"newly accepted samples: {len(samples)}"
                )

                gname = _sanitize_group_name(Np, frac_A)
                grp = h5.require_group(gname)
                # Set group-level attributes if they do not exist
                if "Np_target" not in grp.attrs:
                    grp.attrs["Np_target"] = int(Np)
                if "frac_A_target" not in grp.attrs:
                    grp.attrs["frac_A_target"] = float(frac_A)
                if "base_seed" not in grp.attrs:
                    grp.attrs["base_seed"] = int(param_seeds[(Np, frac_A)])

                start_idx = existing_counts[(Np, frac_A)]
                _write_samples_to_h5(h5, gname, samples, start_index=start_idx)

                # Update total sample count for this group
                grp.attrs["n_samples"] = int(start_idx + len(samples))
                existing_counts[(Np, frac_A)] = start_idx + len(samples)

                h5.flush()  # flush after each parameter pair

        # --- Multiprocessing mode ---
        else:
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                future_to_param: Dict[Any, Tuple[int, float, int]] = {}
                for (Np, frac_A, need) in todo_pairs:
                    seed = param_seeds[(Np, frac_A)]
                    fut = ex.submit(
                        _generate_samples_for_param,
                        Np,
                        frac_A,
                        TARGET_DF,
                        TARGET_MAS,
                        DF_TOL,
                        MAS_TOL,
                        need,
                        MAX_TRIES_FACTOR,
                        seed,
                    )
                    future_to_param[fut] = (Np, frac_A, need)

                for fut in as_completed(future_to_param):
                    Np, frac_A, need = future_to_param[fut]
                    already = existing_counts[(Np, frac_A)]
                    try:
                        samples = fut.result()
                    except Exception as e:
                        print(
                            f"[POOL][ERROR] Np={Np}, frac_A={frac_A:.4f} generation failed: {e}"
                        )
                        samples = []

                    print(
                        f"[POOL] (Np={Np}, frac_A={frac_A:.4f}) "
                        f"existing={already}, newly accepted={len(samples)}"
                    )

                    gname = _sanitize_group_name(Np, frac_A)
                    grp = h5.require_group(gname)
                    # Set group-level attributes if missing
                    if "Np_target" not in grp.attrs:
                        grp.attrs["Np_target"] = int(Np)
                    if "frac_A_target" not in grp.attrs:
                        grp.attrs["frac_A_target"] = float(frac_A)
                    if "base_seed" not in grp.attrs:
                        grp.attrs["base_seed"] = int(param_seeds[(Np, frac_A)])

                    start_idx = existing_counts[(Np, frac_A)]
                    _write_samples_to_h5(h5, gname, samples, start_index=start_idx)

                    grp.attrs["n_samples"] = int(start_idx + len(samples))
                    existing_counts[(Np, frac_A)] = start_idx + len(samples)

                    h5.flush()  # flush after each parameter pair

    print("[POOL] Pool construction / extension completed.")


if __name__ == "__main__":
    build_pool()
    # dump = np.load("error_dump_100_0.3_2062596029_244375622.npy", allow_pickle=True).item()
