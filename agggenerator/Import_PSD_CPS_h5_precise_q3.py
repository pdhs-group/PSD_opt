# -*- coding: utf-8 -*-
"""
PSD preprocessing and packaging into an HDF5 (h5) file.

This module reads experimental PSD data exported from CPS as a cumulative
volume-based distribution Q3(x) on a linear x-grid, converts it to a
differential q3(x), then to a number-based distribution q0(x), builds a
log-spaced grid, and finally computes the cumulative number-based
distribution Q0(x) on that grid together with a set of aggregate
diameters d_agg for Monte Carlo initialization.

Key stored items per experiment (per HDF5 group_label):
    Datasets:
        - d_agg       : aggregate diameters used for MC initialization
        - x_dis       : log-spaced diameter grid for q0 / Q0
        - q0_sum_log  : cumulative Q0(x) on x_dis (NOT the differential q0)
        - d3_cent     : original linear diameter grid from the .dat file
        - q3_sum_agg  : original cumulative Q3(x) distribution from CPS

    Attributes:
        - Vc       : control volume (m^3)
        - N_bins   : number of bins for log-spaced q0-discretization
        - N_aggs_target : target number of aggregates requested
        - N_aggs_eff    : effective number of aggregates actually generated
        - cell_size : cell size used in original pixel-based analysis
        - exp_t    : experimental sampling time (user-specified, in seconds)
        - phi_s    : solids volume fraction
        - V_mean   : mean aggregate volume used for Vc calculation
        - SiO2_particle_size : silica primary particle size (nm)
        - Carbon_Concentration : carbon concentration during synthesis

The HDF5 file is always opened in append mode ("a"), so it is never
overwritten as a whole. If a group with the same label already exists
and override=False, an error is raised to avoid silent overwrites.
"""

import re
import os
from glob import glob
from typing import Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def read_origin_dat(dat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read original cumulative PSD data (Q3) exported from CPS as .dat."""
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"No .dat file found at path: {dat_path!r}")

    # CPS export is typically tab-separated; adjust sep if needed.
    df = pd.read_csv(dat_path, sep="\t", encoding="latin1")

    d3_cent = df.iloc[:, 0].values.astype(float) * 1e-6
    q3_sum_agg = df.iloc[:, 1].values.astype(float) / 100.0

    if d3_cent.ndim != 1 or q3_sum_agg.ndim != 1 or d3_cent.size != q3_sum_agg.size:
        raise ValueError("d3_cent and q3_sum_agg must be 1D arrays of the same length.")

    return d3_cent, q3_sum_agg


def extract_minutes_from_name(path: str) -> int:
    """Extract the integer 'X' from patterns like 'CPS X min ...'. """
    name = os.path.basename(path)
    m = re.search(r"(\d+)\s*min", name)
    if not m:
        raise ValueError(f"Cannot extract minutes from filename {name!r}")
    return int(m.group(1))


# ----------------------------------------------------------------------
# PSD transformation
# ----------------------------------------------------------------------
def transform_PSD(
    d3_cent: np.ndarray,
    q3_sum_agg: np.ndarray,
    N_bins: int,
    N_aggs: int,
    force_N_aggs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Transform linear Q3-PSD to log-spaced Q0-PSD and MC diameters.

    Steps:
        0) Convert cumulative Q3(x) to differential q3(x) on the linear grid.
        1) Compute q0(x) from q3(x) using the -3/3 moment.
        2) Interpolate q0(x) onto a log-spaced grid x_dis with N_bins bins.
        3) Integrate q0(x) over each log-space interval to obtain per-bin
           fractions q0_i and a cumulative number-based distribution Q0(x)
           (stored as q0_sum_log on x_dis).
        4) Use q0_i to allocate a number of aggregates per bin and fill each
           interval with uniformly spaced diameters to build d_agg.
    """
    if N_bins <= 0:
        raise ValueError(f"N_bins must be positive, got {N_bins}.")
    if N_aggs <= 0:
        raise ValueError(f"N_aggs must be positive, got {N_aggs}.")

    d3_cent = np.asarray(d3_cent, dtype=float)
    q3_sum_agg = np.asarray(q3_sum_agg, dtype=float)

    if d3_cent.size < 3:
        raise ValueError("At least 3 points are required in d3_cent to compute the spacing.")
    if not np.all(np.diff(d3_cent) > 0):
        raise ValueError("d3_cent must be strictly increasing.")

    # ------------------------------------------------------------------
    # 0) linear Q3 → linear q3
    #     Replace finite differences with:
    #       - monotone repair of Q3
    #       - np.gradient(Q3, d3_cent)
    #       - non-negative clipping
    #       - normalization via trapz
    #     Keep original scaling convention by multiplying pdf by N_aggs,
    #     so ∫ q3_agg dx ≈ N_aggs.
    # ------------------------------------------------------------------
    Q3 = q3_sum_agg.copy()

    # basic sanitization
    mask_f = np.isfinite(Q3)
    if not np.any(mask_f):
        raise ValueError("q3_sum_agg has no finite values.")
    if np.any(~mask_f):
        # fill NaNs/Infs by linear interpolation over index
        idx = np.arange(Q3.size)
        Q3[~mask_f] = np.interp(idx[~mask_f], idx[mask_f], Q3[mask_f])

    # keep within [0,1] and enforce non-decreasing
    Q3 = np.clip(Q3, 0.0, 1.0)
    Q3 = np.maximum.accumulate(Q3)

    # derivative on (possibly) non-uniform grid
    q3_pdf = np.gradient(Q3, d3_cent)

    # non-negative truncation
    q3_pdf = np.clip(q3_pdf, 0.0, None)

    # normalize to a pdf: ∫ q3_pdf dx = 1
    area = float(np.trapz(q3_pdf, d3_cent))
    if area <= 0.0 or not np.isfinite(area):
        raise ValueError("Computed q3_pdf has non-positive integral; check Q3 and grid.")
    q3_pdf = q3_pdf / area

    # match your original scaling convention (integral ~ N_aggs)
    q3_agg = q3_pdf * float(N_aggs)

    # for consistency with later V_mean formula that used delta_x,
    # still define delta_x as a "representative" spacing; but do NOT
    # require constant spacing anymore.
    delta_x = float(np.median(np.diff(d3_cent)))

    # ------------------------------------------------------------------
    # 1) linear q3 → q0  (same as before)
    # ------------------------------------------------------------------
    # Original: M_neg3_3 = np.sum(d3_cent**(-3) * q3_agg * delta_x)
    # To preserve your existing "Riemann sum" style on the original grid,
    # keep the same expression with delta_x. (q3_agg already scaled by N_aggs)
    M_neg3_3 = np.sum(d3_cent**(-3) * q3_agg * delta_x)
    if M_neg3_3 <= 0.0:
        raise ValueError("Computed M_neg3_3 <= 0; check q3_agg and diameter grid.")

    q0_agg = d3_cent**(-3) * q3_agg / M_neg3_3

    # ------------------------------------------------------------------
    # 2) interpolation onto log-spaced grid (same as before)
    # ------------------------------------------------------------------
    x_dis = np.geomspace(d3_cent[0], d3_cent[-1], N_bins)
    q0_agg_dis = np.interp(x_dis, d3_cent, q0_agg)

    # ------------------------------------------------------------------
    # 2b) interval bounds in log space (same as before)
    # ------------------------------------------------------------------
    x_Int = np.zeros((len(x_dis), 1), dtype=float)
    for i in range(len(x_dis) - 1):
        x_Int[i] = (x_dis[i + 1] - x_dis[i]) / np.log(x_dis[i + 1] / x_dis[i])
    x_Int[-1] = x_dis[-1] + (x_dis[-1] - x_Int[-2])
    x_0 = x_dis[0] - (x_Int[0] - x_dis[0])

    # ------------------------------------------------------------------
    # 2c) per-interval amounts and cumulative Q0(x) (same as before)
    # ------------------------------------------------------------------
    q0_sum_log = np.zeros((len(x_dis), 1), dtype=float)
    q0_i = np.zeros((len(x_dis), 1), dtype=float)  # amount in each interval

    # first interval [x_0, x_Int[0]]
    q0_i[0] = (x_Int[0] - x_0) * q0_agg_dis[0]
    q0_sum_log[0] = q0_i[0]

    # remaining intervals [x_Int[i-1], x_Int[i]]
    for i in range(1, len(q0_agg_dis)):
        q0_i[i] = (x_Int[i] - x_Int[i - 1]) * q0_agg_dis[i]
        q0_sum_log[i] = np.sum(q0_i[: i + 1])  # (keep your bugfix)

    # ------------------------------------------------------------------
    # 3) allocation to bins using q0_i as weights (same as before)
    # ------------------------------------------------------------------
    frac_float = q0_i * float(N_aggs)
    frac_int = np.rint(frac_float).astype(int)

    # ------------------------------------------------------------------
    # optional correction to enforce total == N_aggs (same as before)
    # ------------------------------------------------------------------
    if force_N_aggs:
        total_now = int(frac_int.sum())
        delta = N_aggs - total_now  # number of particles to add/remove

        if delta > 0:
            frac_part = frac_float - frac_int
            idx = np.argsort(-frac_part.ravel())[:delta]
            for i in idx:
                frac_int[i] += 1
        elif delta < 0:
            need = -delta
            frac_part = frac_float - frac_int
            idx = np.argsort(frac_part.ravel())
            for i in idx:
                if need == 0:
                    break
                if frac_int[i] > 0:
                    frac_int[i] -= 1
                    need -= 1

    # ------------------------------------------------------------------
    # 4) build d_agg (same as before)
    # ------------------------------------------------------------------
    parts = []
    if frac_int[0] > 0:
        parts.append(np.linspace(x_0, x_Int[0], int(frac_int[0]), endpoint=False))

    for i in range(1, len(x_dis)):
        if frac_int[i] > 0:
            parts.append(
                np.linspace(x_Int[i - 1], x_Int[i], int(frac_int[i, 0]), endpoint=False)
            )

    if not parts:
        raise RuntimeError("No aggregates were generated; check q0_agg_dis and N_aggs.")

    d_agg = np.concatenate(parts)

    # guarantee exact count if force_N_aggs=True (same as before)
    if force_N_aggs and d_agg.size != N_aggs:
        if d_agg.size > N_aggs:
            keep_idx = np.random.choice(d_agg.size, N_aggs, replace=False)
            d_agg = d_agg[keep_idx]
        else:
            add = N_aggs - d_agg.size
            extra = np.random.choice(d_agg, add, replace=True)
            d_agg = np.concatenate([d_agg, extra])

    # q0_sum_log is (N,1); make it 1D
    q0_sum_log = q0_sum_log.ravel()
    q0_sum_log /= q0_sum_log[-1]

    # V_mean calculation (keep your original formula style)
    V_d = (np.pi / 6.0) * d3_cent**3
    V_mean = float(np.sum(V_d * q0_agg * delta_x))

    return d_agg, x_dis, q0_sum_log, V_mean


def invert_cdf_x50(x: np.ndarray, Q: np.ndarray, q: float = 0.5) -> float:
    """Return x such that CDF Q(x)=q using monotone fix + linear interpolation."""
    x = np.asarray(x, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    if x.size == 0 or Q.size == 0 or x.size != Q.size:
        return float("nan")

    m = np.isfinite(x) & np.isfinite(Q)
    if not np.any(m):
        return float("nan")

    x = x[m]
    Q = Q[m]

    order = np.argsort(x)
    x = x[order]
    Q = Q[order]

    Q = np.maximum.accumulate(Q)

    if Q[0] > q or Q[-1] < q:
        return float("nan")

    k = int(np.searchsorted(Q, q, side="left"))
    if k <= 0:
        return float(x[0])
    if k >= Q.size:
        return float(x[-1])

    q0, q1 = float(Q[k - 1]), float(Q[k])
    x0, x1 = float(x[k - 1]), float(x[k])
    if q1 <= q0 + 1e-15:
        return float(x1)
    return float(x0 + (q - q0) * (x1 - x0) / (q1 - q0))


def _compute_psd_cdf_from_d_agg(
    d_agg: np.ndarray,
    psd_basis: str = "volume",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build an empirical CDF from aggregate diameters d_agg."""
    d = np.asarray(d_agg, dtype=float).ravel()
    if d.size == 0:
        return None

    if psd_basis == "number":
        w = np.ones_like(d, dtype=float)
    elif psd_basis == "volume":
        w = (np.pi / 6.0) * d**3
    else:
        raise ValueError(f"psd_basis must be 'number' or 'volume', got {psd_basis!r}")

    mask = (d > 0.0) & (w > 0.0) & np.isfinite(d) & np.isfinite(w)
    if not np.any(mask):
        return None

    d = d[mask]
    w = w[mask]

    idx = np.argsort(d)
    x_sorted = d[idx]
    w_sorted = w[idx]
    w_cum = np.cumsum(w_sorted)
    total = float(w_cum[-1])
    if total <= 0.0:
        return None
    Q_sorted = w_cum / total
    return x_sorted, Q_sorted


def _eval_Q_of_x_stepwise(
    x_sorted: np.ndarray,
    Q_sorted: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """Evaluate step-wise empirical CDF Q(x) on a query grid."""
    xq = np.asarray(x_query, dtype=float)
    Qq = np.zeros_like(xq, dtype=float)
    idx = np.searchsorted(x_sorted, xq, side="right") - 1
    Qq[idx < 0] = 0.0
    valid = idx >= 0
    if np.any(valid):
        idx_clipped = np.clip(idx[valid], 0, len(Q_sorted) - 1)
        Qq[valid] = Q_sorted[idx_clipped]
    return Qq


# ----------------------------------------------------------------------
# HDF5 writer for a single .dat file
# ----------------------------------------------------------------------
def save_psd_to_h5(
    dat_path: str,
    h5_filename: str,
    group_label: str,
    override: bool,
    N_bins: int,
    N_aggs: int,
    cell_size: float,
    exp_t: float,
    phi_s: float,
    sio2_size: float,
    carbon_concentration: float,
) -> None:
    """Process a single PSD .dat file and store results into an HDF5 file."""
    if phi_s <= 0.0:
        raise ValueError(f"phi_s must be positive, got {phi_s}.")

    d3_cent, q3_sum_agg = read_origin_dat(dat_path)

    d_agg, x_dis, q0_sum_log, V_mean = transform_PSD(d3_cent, q3_sum_agg, N_bins, N_aggs)
    N_eff = d_agg.size

    Vc = float(N_eff) * float(V_mean) / float(phi_s)

    x50_Q3 = invert_cdf_x50(d3_cent, q3_sum_agg, q=0.5)
    x50_Q0 = invert_cdf_x50(x_dis, q0_sum_log, q=0.5)

    # reconstruction check at t=0 only
    if float(exp_t) == 0.0:
        cdf_num = _compute_psd_cdf_from_d_agg(d_agg, psd_basis="number")
        if cdf_num is None:
            Q0_recon_on_xdis = np.full_like(x_dis, np.nan, dtype=float)
        else:
            x_sorted, Q_sorted = cdf_num
            Q0_recon_on_xdis = _eval_Q_of_x_stepwise(x_sorted, Q_sorted, x_dis)

        cdf_vol = _compute_psd_cdf_from_d_agg(d_agg, psd_basis="volume")
        if cdf_vol is None:
            Q3_recon_on_d3 = np.full_like(d3_cent, np.nan, dtype=float)
        else:
            x_sorted, Q_sorted = cdf_vol
            Q3_recon_on_d3 = _eval_Q_of_x_stepwise(x_sorted, Q_sorted, d3_cent)

        plt.figure()
        plt.semilogx(x_dis, q0_sum_log, label="Original (stored) Q0 on x_dis")
        plt.semilogx(x_dis, Q0_recon_on_xdis, "--", label="Reconstructed Q0 from d_agg")
        plt.xlabel("Diameter x (m)")
        plt.ylabel("Q0(x)")
        plt.title(f"t=0: Q0 reconstruction check ({group_label})")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, which="both", linestyle=":")
        plt.legend()

        plt.figure()
        plt.semilogx(d3_cent, q3_sum_agg, label="Original (stored) Q3 on d3_cent")
        plt.semilogx(d3_cent, Q3_recon_on_d3, "--", label="Reconstructed Q3 from d_agg")
        plt.xlabel("Diameter x (m)")
        plt.ylabel("Q3(x)")
        plt.title(f"t=0: Q3 reconstruction check ({group_label})")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, which="both", linestyle=":")
        plt.legend()

        plt.show()

    os.makedirs(os.path.dirname(h5_filename) or ".", exist_ok=True)
    with h5py.File(h5_filename, mode="a") as h5f:
        if group_label in h5f:
            if not override:
                raise ValueError(
                    f"Group {group_label!r} already exists in {h5_filename!r}. "
                    "Choose a different label or set override=True."
                )
            del h5f[group_label]

        grp = h5f.create_group(group_label)

        grp.create_dataset("d_agg", data=d_agg, compression="gzip")
        grp.create_dataset("x_dis", data=x_dis, compression="gzip")
        grp.create_dataset("q0_sum_log", data=q0_sum_log, compression="gzip")
        grp.create_dataset("d3_cent", data=d3_cent, compression="gzip")
        grp.create_dataset("q3_sum_agg", data=q3_sum_agg, compression="gzip")
        grp.create_dataset("x50_Q3", data=np.array(x50_Q3, dtype=float))
        grp.create_dataset("x50_Q0", data=np.array(x50_Q0, dtype=float))

        grp.attrs["Vc"] = Vc
        grp.attrs["N_bins"] = int(N_bins)
        grp.attrs["N_aggs_target"] = int(N_aggs)
        grp.attrs["N_aggs_eff"] = int(N_eff)
        grp.attrs["cell_size"] = float(cell_size)
        grp.attrs["exp_t"] = float(exp_t)
        grp.attrs["phi_s"] = float(phi_s)
        grp.attrs["V_mean"] = float(V_mean)
        grp.attrs["SiO2_particle_size"] = float(sio2_size)
        grp.attrs["Carbon_Concentration"] = float(carbon_concentration)


# ----------------------------------------------------------------------
# Batch processing: folder of .dat → single HDF5 with multiple groups
# ----------------------------------------------------------------------
def batch_save_folder_to_h5(
    dat_folder: str,
    h5_filename: str,
    exp_t_list: Sequence[float],
    N_bins: int,
    N_aggs: int,
    cell_size: float,
    phi_s: float,
    sio2_size: float,
    carbon_concentration: float,
    override: bool = False,
    group_labels: Optional[Sequence[str]] = None,
) -> None:
    """Process all .dat files in a folder and store them into one HDF5 file."""
    dat_files = sorted(
        glob(os.path.join(dat_folder, "*.dat")),
        key=extract_minutes_from_name,
    )
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in folder {dat_folder!r}.")

    if len(dat_files) != len(exp_t_list):
        raise ValueError(
            f"Number of .dat files ({len(dat_files)}) does not match "
            f"length of exp_t_list ({len(exp_t_list)})."
        )

    if group_labels is None:
        group_labels = [
            os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
            for path in dat_files
        ]
    if len(group_labels) != len(dat_files):
        raise ValueError(
            "group_labels must have the same length as exp_t_list and the number of .dat files."
        )

    for dat_path, t_exp, label in zip(dat_files, exp_t_list, group_labels):
        print(f"Processing {dat_path!r} -> group {label!r}, exp_t={t_exp:.3g} s")
        save_psd_to_h5(
            dat_path=dat_path,
            h5_filename=h5_filename,
            group_label=label,
            override=override,
            N_bins=N_bins,
            N_aggs=N_aggs,
            cell_size=cell_size,
            exp_t=float(t_exp),
            phi_s=phi_s,
            sio2_size=sio2_size,
            carbon_concentration=carbon_concentration,
        )


# ----------------------------------------------------------------------
# Example main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dat_folder = os.path.join("input", "PSD_agg")
    h5_file = "CB_pur_series.h5"

    exp_t_min_list = [1.0, 3.0, 5.0, 10.0, 15.0, 30.0]
    exp_t_sec_list = [(t_min - 1) * 60.0 for t_min in exp_t_min_list]
    group_labels = [f"t_{int(t_min)}min" for t_min in exp_t_min_list]

    N_bins = 100
    N_aggs = 100000
    cell_size = 0.01
    phi_s = 0.00005
    sio2_size = 0.0
    carbon_concentration = 1.0

    batch_save_folder_to_h5(
        dat_folder=dat_folder,
        h5_filename=h5_file,
        exp_t_list=exp_t_sec_list,
        N_bins=N_bins,
        N_aggs=N_aggs,
        cell_size=cell_size,
        phi_s=phi_s,
        sio2_size=sio2_size,
        carbon_concentration=carbon_concentration,
        override=True,
        group_labels=group_labels,
    )

    print(f"Saved PSD data series into {h5_file!r}.")
