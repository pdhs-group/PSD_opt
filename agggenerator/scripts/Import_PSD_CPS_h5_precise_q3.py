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
    """Read original cumulative PSD data (Q3) exported from CPS as .dat.

    Parameters
    ----------
    dat_path : str
        Path to a single *.dat file.

    Returns
    -------
    d3_cent : ndarray, shape (N,)
        Linear diameter grid.
    q3_sum_agg : ndarray, shape (N,)
        Cumulative volume-based PSD Q3(x) on d3_cent as exported from CPS.
        In other words, q3_sum_agg[i] ≈ ∫_0^{d3_cent[i]} q3(x) dx.
    """
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

def invert_cdf(x: np.ndarray, Q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse CDF by monotone repair + linear interpolation.
    x must be strictly increasing; Q is cumulative in [0,1] (may have tiny noise).
    u are quantiles in (0,1).
    """
    x = np.asarray(x, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()

    if x.size < 2 or x.size != Q.size:
        raise ValueError("invert_cdf: x and Q must have same length >= 2.")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("invert_cdf: x must be strictly increasing.")

    # sanitize & monotone repair
    Q = np.clip(Q, 0.0, 1.0)
    Q = np.maximum.accumulate(Q)

    # avoid pathological flat all-zeros/all-ones
    if Q[-1] <= 0.0:
        return np.full_like(u, x[0], dtype=float)
    if Q[0] >= 1.0:
        return np.full_like(u, x[0], dtype=float)

    # keep u in [Q[0], Q[-1]] (typically [0,1])
    u = np.clip(u, Q[0], Q[-1])

    # use numpy interpolation: requires Q nondecreasing (we enforced)
    # If Q has duplicates (plateaus), np.interp still works but maps to the first occurrence effectively.
    # That is fine for quantile sampling.
    return np.interp(u, Q, x)


def sample_d_agg_from_Q0_quantiles_derived_from_Q3(
    d3_cent: np.ndarray,
    q3_sum_agg: np.ndarray,
    N_aggs: int,
    *,
    use_midpoints: bool = True,
) -> np.ndarray:
    """Derive q0 from experimental Q3, then sample d_agg by Q0 quantiles.

    This keeps consistency: volume-weighted CDF from sampled d_agg matches the input Q3 much better
    than sampling directly from Q3.
    """
    d3_cent = np.asarray(d3_cent, dtype=float).ravel()
    q3_sum_agg = np.asarray(q3_sum_agg, dtype=float).ravel()

    if d3_cent.size < 3:
        raise ValueError("Need >=3 points in d3_cent.")
    if d3_cent.size != q3_sum_agg.size:
        raise ValueError("d3_cent and q3_sum_agg must have same length.")
    if not np.all(np.diff(d3_cent) > 0.0):
        raise ValueError("d3_cent must be strictly increasing.")

    # ---- Step A: Q3 -> q3 on linear grid ----
    dx = d3_cent[2] - d3_cent[1]
    if dx <= 0:
        raise ValueError("Non-positive spacing in d3_cent.")
    Q3 = np.clip(q3_sum_agg, 0.0, 1.0)
    Q3 = np.maximum.accumulate(Q3)

    q3 = np.empty_like(Q3)
    q3[0] = Q3[0] / dx
    q3[1:] = (Q3[1:] - Q3[:-1]) / dx

    # ---- Step B: q3 -> q0 via d^{-3} and normalize ----
    # q0(d) ∝ d^{-3} q3(d)
    q0_unnorm = (d3_cent ** (-3.0)) * q3
    # Normalize as a PDF on the linear grid
    area = float(np.sum(q0_unnorm) * dx)
    if area <= 0.0:
        raise ValueError("Derived q0 has non-positive integral.")
    q0 = q0_unnorm / area

    # ---- Step C: build Q0 CDF on the same grid ----
    Q0 = np.cumsum(q0) * dx
    Q0 = np.clip(Q0, 0.0, 1.0)
    Q0 = np.maximum.accumulate(Q0)
    Q0[-1] = 1.0

    # ---- Step D: quantile sampling from Q0 ----
    if use_midpoints:
        u = (np.arange(N_aggs, dtype=float) + 0.5) / float(N_aggs)
    else:
        u = np.linspace(0.0, 1.0, N_aggs)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)    
        
    # alpha = 1.1    
    # t = (np.arange(N_aggs) + 0.5) / N_aggs
    # u = 1.0 - (1.0 - t)**alpha   # alpha > 1 -> tail dense

    d_agg = invert_cdf(d3_cent, Q0, u)
    return d_agg.astype(float)

def truncate_Q3_by_min_volume(
    d3_cent: np.ndarray,
    q3_sum_agg: np.ndarray,
    V_cut: float,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Truncate small particles by a minimum particle volume V_cut.

    Interpretation:
      - particles with volume < V_cut (diameter < d_cut) are removed from the system
      - remaining cumulative volume-based CDF Q3 is re-normalized to end at 1
      - returns the kept fraction f_keep = 1 - Q3(d_cut)

    Returns
    -------
    d_new : ndarray
        New diameter grid (starts at d_cut).
    Q3_new : ndarray
        Renormalized cumulative Q3 on d_new.
    d_cut : float
        Cutoff diameter corresponding to V_cut.
    Q3_cut : float
        Original cumulative Q3(d_cut) before truncation.
    f_keep : float
        Remaining volume fraction after truncation, i.e. 1 - Q3_cut.
    """
    d = np.asarray(d3_cent, dtype=float).ravel()
    Q3 = np.asarray(q3_sum_agg, dtype=float).ravel()

    if d.size < 3 or d.size != Q3.size:
        raise ValueError("truncate_Q3_by_min_volume: d3_cent and q3_sum_agg must be same length >= 3.")
    if not np.all(np.diff(d) > 0.0):
        raise ValueError("truncate_Q3_by_min_volume: d3_cent must be strictly increasing.")
    if V_cut is None:
        # no truncation
        return d, Q3, float("nan"), float("nan"), 1.0
    if V_cut <= 0.0:
        raise ValueError(f"V_cut must be positive, got {V_cut}.")

    # sphere: V = (pi/6) d^3  -> d = (6V/pi)^(1/3)
    d_cut = float((6.0 * float(V_cut) / np.pi) ** (1.0 / 3.0))

    # sanitize & monotone repair the CDF
    Q3 = np.clip(Q3, 0.0, 1.0)
    Q3 = np.maximum.accumulate(Q3)

    # handle edge cases
    if d_cut <= float(d[0]):
        # cutoff below measured range: effectively no truncation
        if verbose:
            print(f"[truncate] d_cut={d_cut:.6g} <= d_min={d[0]:.6g}: no truncation applied (f_keep=1.0).")
        return d, Q3, d_cut, 0.0, 1.0

    if d_cut >= float(d[-1]):
        raise ValueError(
            f"truncate_Q3_by_min_volume: d_cut={d_cut:.6g} >= d_max={d[-1]:.6g}. "
            "Cutoff too large; nothing would remain."
        )

    # get Q3(d_cut) by interpolation on original grid
    Q3_cut = float(np.interp(d_cut, d, Q3))
    Q3_cut = float(np.clip(Q3_cut, 0.0, 1.0))
    f_keep = 1.0 - Q3_cut

    if f_keep <= 0.0:
        raise ValueError(
            f"truncate_Q3_by_min_volume: remaining fraction f_keep={f_keep:.6g} <= 0. "
            "Cutoff removes (almost) everything."
        )

    # build new grid: start with (d_cut, Q3_cut) then all points >= d_cut
    idx0 = int(np.searchsorted(d, d_cut, side="left"))
    d_tail = d[idx0:]
    Q3_tail = Q3[idx0:]

    # ensure cutoff point included as first
    d_new = np.concatenate(([d_cut], d_tail))
    Q3_new_raw = np.concatenate(([Q3_cut], Q3_tail))

    # shift & renormalize:
    #   for d >= d_cut: Q3_new = (Q3 - Q3_cut) / (1 - Q3_cut)
    Q3_new = (Q3_new_raw - Q3_cut) / f_keep
    Q3_new = np.clip(Q3_new, 0.0, 1.0)
    Q3_new = np.maximum.accumulate(Q3_new)
    Q3_new[0] = 0.0
    Q3_new[-1] = 1.0

    if d_new.size < 3:
        raise ValueError(
            "truncate_Q3_by_min_volume: too few points remain after truncation. "
            "Lower V_cut or provide PSD with more points."
        )

    if verbose:
        print(
            f"[truncate] V_cut={V_cut:.6g} m^3 -> d_cut={d_cut:.6g} m | "
            f"kept volume fraction f_keep={f_keep:.6g} "
            f"({f_keep*100:.3f}% of original)."
        )

    return d_new, Q3_new, d_cut, Q3_cut, f_keep

# ----------------------------------------------------------------------
# PSD transformation
# ----------------------------------------------------------------------
def transform_PSD(
    d3_cent: np.ndarray,
    q3_sum_agg: np.ndarray,
    N_bins: int,
    N_aggs: int,
    force_N_aggs: bool = False,
    init_mode: str = "q0_bins",   # NEW: "q0_bins" (old) or "q3_quantile"
):
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

    Parameters
    ----------
    d3_cent : ndarray
        Original linear diameter grid.
    q3_sum_agg : ndarray
        Cumulative volume-based PSD Q3(x) on d3_cent.
    N_bins : int
        Number of log-spaced bins.
    N_aggs : int
        Target number of aggregates.
    force_N_aggs : bool, default False
        If True, after rounding the per-bin counts, apply a redistribution
        correction so that the final number of aggregates equals exactly
        N_aggs. This is done by adjusting a few bins by ±1 based on the
        fractional parts of their allocations.

    Returns
    -------
    d_agg : ndarray
        Generated aggregate diameters (MC initialization).
    x_dis : ndarray
        Log-spaced diameter grid.
    q0_sum_log : ndarray
        Cumulative number-based PSD Q0(x) on x_dis (same shape as x_dis).
    """
    if N_bins <= 0:
        raise ValueError(f"N_bins must be positive, got {N_bins}.")
    if N_aggs <= 0:
        raise ValueError(f"N_aggs must be positive, got {N_aggs}.")

    d3_cent = np.asarray(d3_cent, dtype=float)
    q3_sum_agg = np.asarray(q3_sum_agg, dtype=float)

    if d3_cent.size < 3:
        raise ValueError("At least 3 points are required in d3_cent to compute the spacing.")

    # ==============================================================
    # NEW init mode: generate d_agg directly from experimental Q3 quantiles
    # ==============================================================
    if init_mode == "q0_quantile_from_q3":
        d_agg = sample_d_agg_from_Q0_quantiles_derived_from_Q3(d3_cent, q3_sum_agg, N_aggs)
    
        # x_dis just defines where you store Q0 for plotting/metadata
        x_dis = np.geomspace(float(d3_cent[0]), float(d3_cent[-1]), int(N_bins))
    
        # Store Q0 on x_dis (number-based) derived from the sampled d_agg
        d_sorted = np.sort(d_agg)
        n = d_sorted.size
        Q_sorted = (np.arange(1, n + 1, dtype=float) / float(n))
        idx = np.searchsorted(d_sorted, x_dis, side="right") - 1
        q0_sum_log = np.zeros_like(x_dis, dtype=float)
        m = idx >= 0
        if np.any(m):
            q0_sum_log[m] = Q_sorted[np.clip(idx[m], 0, n - 1)]
    
        V_mean = float(np.mean((np.pi / 6.0) * d_agg**3))
        return d_agg, x_dis, q0_sum_log, V_mean


    # ------------------------------------------------------------------
    # 0) linear Q3 → linear q3 (finite differences)
    # ------------------------------------------------------------------
    delta_x = d3_cent[2] - d3_cent[1]
    if delta_x <= 0.0:
        raise ValueError("d3_cent must be strictly increasing with constant spacing.")

    # q3_agg[i] ≈ (Q3(x_i) - Q3(x_{i-1})) / (N_aggs * Δx)
    # Note: division by N_aggs is a scaling convention; it assumes that the
    # total "amount" represented by Q3_sum_agg corresponds to N_aggs units.
    q3_agg = np.zeros_like(q3_sum_agg, dtype=float)
    q3_agg[0] = q3_sum_agg[0] / (N_aggs * delta_x)
    for i in range(1, len(q3_sum_agg)):
        q3_agg[i] = (q3_sum_agg[i] - q3_sum_agg[i - 1]) / (N_aggs * delta_x)

    # ------------------------------------------------------------------
    # 1) linear q3 → q0
    # ------------------------------------------------------------------
    M_neg3_3 = np.sum(d3_cent**(-3) * q3_agg * delta_x)
    if M_neg3_3 <= 0.0:
        raise ValueError("Computed M_neg3_3 <= 0; check q3_agg and diameter grid.")

    q0_agg = d3_cent**(-3) * q3_agg / M_neg3_3

    # ------------------------------------------------------------------
    # 2) interpolation onto log-spaced grid
    # ------------------------------------------------------------------
    x_dis = np.geomspace(d3_cent[0], d3_cent[-1], N_bins)
    q0_agg_dis = np.interp(x_dis, d3_cent, q0_agg)

    # ------------------------------------------------------------------
    # 2b) interval bounds in log space
    # ------------------------------------------------------------------
    x_Int = np.zeros((len(x_dis), 1), dtype=float)
    for i in range(len(x_dis) - 1):
        x_Int[i] = (x_dis[i + 1] - x_dis[i]) / np.log(x_dis[i + 1] / x_dis[i])
    x_Int[-1] = x_dis[-1] + (x_dis[-1] - x_Int[-2])
    x_0 = x_dis[0] - (x_Int[0] - x_dis[0])

    # ------------------------------------------------------------------
    # 2c) per-interval amounts and cumulative Q0(x)
    # ------------------------------------------------------------------
    q0_sum_log = np.zeros((len(x_dis), 1), dtype=float)
    q0_i = np.zeros((len(x_dis), 1), dtype=float)  # amount in each interval

    # first interval [x_0, x_Int[0]]
    q0_i[0] = (x_Int[0] - x_0) * q0_agg_dis[0]
    q0_sum_log[0] = q0_i[0]

    # remaining intervals [x_Int[i-1], x_Int[i]]
    for i in range(1, len(q0_agg_dis)):
        q0_i[i] = (x_Int[i] - x_Int[i - 1]) * q0_agg_dis[i]
        # q0_sum_log[i] = np.sum(q0_i[:i])
        q0_sum_log[i] = np.sum(q0_i[:i+1])

    # ------------------------------------------------------------------
    # 3) allocation to bins using q0_i as weights
    # ------------------------------------------------------------------
    frac_float = q0_i * float(N_aggs)
    frac_int = np.rint(frac_float).astype(int)

    # ------------------------------------------------------------------
    # optional correction to enforce total == N_aggs
    # ------------------------------------------------------------------
    if force_N_aggs:
        total_now = int(frac_int.sum())
        delta = N_aggs - total_now  # number of particles to add/remove

        if delta > 0:
            # Need to add delta particles: choose bins with largest fractional part
            frac_part = frac_float - frac_int
            idx = np.argsort(-frac_part.ravel())[:delta]
            for i in idx:
                frac_int[i] += 1
        elif delta < 0:
            # Need to remove -delta particles: choose bins with smallest fractional part
            need = -delta
            frac_part = frac_float - frac_int
            idx = np.argsort(frac_part.ravel())  # smallest first
            for i in idx:
                if need == 0:
                    break
                if frac_int[i] > 0:
                    frac_int[i] -= 1
                    need -= 1

    # ------------------------------------------------------------------
    # 4) build d_agg
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

    # ------------------------------------------------------------------
    # guarantee exact count if force_N_aggs=True
    # (safety double-check, should already be exact)
    # ------------------------------------------------------------------
    if force_N_aggs and d_agg.size != N_aggs:
        if d_agg.size > N_aggs:
            keep_idx = np.random.choice(d_agg.size, N_aggs, replace=False)
            d_agg = d_agg[keep_idx]
        else:
            add = N_aggs - d_agg.size
            extra = np.random.choice(d_agg, add, replace=True)
            d_agg = np.concatenate([d_agg, extra])

    # q0_sum_log is (N,1); make it 1D to be more convenient downstream
    q0_sum_log = q0_sum_log.ravel()
    q0_sum_log /= q0_sum_log[-1]

    V_d = (np.pi / 6.0) * d3_cent**3
    V_mean = float(np.sum(V_d * q0_agg * delta_x))
    # print(f"{V_mean}")
    # delta_x = d3_cent[2] - d3_cent[1]
    # I0 = np.sum(q3_agg) * delta_x
    # M_neg3 = np.sum(d3_cent**(-3) * q3_agg * delta_x)
    # V_mean_q3 = (np.pi / 6.0) * I0 / M_neg3

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

    # enforce non-decreasing CDF (remove tiny numerical non-monotonicity)
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
    """Build an empirical CDF from aggregate diameters d_agg.

    This mirrors MCPBEPost._compute_psd_cdf_from_snapshot() logic:
    sort x, compute cumulative weight, normalize to (0,1].
    """
    d = np.asarray(d_agg, dtype=float).ravel()
    if d.size == 0:
        return None

    # weights
    if psd_basis == "number":
        w = np.ones_like(d, dtype=float)
    elif psd_basis == "volume":
        # volume weight proportional to particle volume ~ d^3
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
    """Evaluate step-wise empirical CDF Q(x) on a query grid.
    Mirrors MCPBEPost._eval_Q_of_x().
    """
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
    # V_mean: float,
    sio2_size: float,
    carbon_concentration: float,
    V_cut: float = 0.0,
) -> None:

    """Process a single PSD .dat file and store results into an HDF5 file.

    This creates (or overwrites) one group under `group_label` inside the
    given HDF5 file.

    Parameters
    ----------
    dat_path : str
        Path to the *.dat file containing the original cumulative Q3-PSD.
    h5_filename : str
        Target HDF5 file path. The file is opened in append mode ("a"),
        so it is never overwritten as a whole.
    group_label : str
        Name of the HDF5 group under which this experiment's data
        will be stored.
    override : bool
        If True and the group already exists in the HDF5 file, the group
        is deleted and recreated. If False, an existing group with the same
        label will cause a ValueError.
    N_bins : int
        Number of log-spaced bins for q0/Q0 discretization.
    N_aggs : int
        Number of aggregates to generate for MC initialization.
    cell_size : float
        Cell size used in the original pixel-based analysis (µm, etc.).
        Stored as metadata; not used in the PSD transformation itself.
    exp_t : float
        Experimental sampling time (in seconds).
    phi_s : float
        Solids volume fraction (dimensionless, 0 < phi_s <= 1).
    V_mean : float
        Mean aggregate volume (in m^3). Used together with phi_s and
        N_aggs_eff to compute the control volume Vc.
    sio2_size : float
        Silica primary particle size (nm).
    carbon_concentration : float
        Carbon concentration during synthesis (user-defined units).
    V_cut: float
        minimum particle volume cutoff (m^3); 0.0 disables

    Notes
    -----
    The control volume is computed as:

        Vc = N_aggs_eff * V_mean / phi_s

    where N_aggs_eff is the actual number of aggregates generated by
    transform_PSD (may differ slightly from N_aggs).
    """
    if phi_s <= 0.0:
        raise ValueError(f"phi_s must be positive, got {phi_s}.")
    # if V_mean <= 0.0:
    #     raise ValueError(f"V_mean must be positive, got {V_mean}.")

    # 1) Read original cumulative Q3-PSD
    d3_cent, q3_sum_agg = read_origin_dat(dat_path)
    
    # --- NEW: truncate tiny particles by minimum volume V_cut (if enabled) ---
    phi_s_original = float(phi_s)
    d_cut = float("nan")
    Q3_cut = float("nan")
    f_keep = 1.0
    
    if V_cut is not None and float(V_cut) > 0.0:
        d3_cent, q3_sum_agg, d_cut, Q3_cut, f_keep = truncate_Q3_by_min_volume(
            d3_cent, q3_sum_agg, float(V_cut), verbose=True
        )
    
    # NEW: solids volume fraction reduced because removed particles are considered "not in system"
    phi_s_eff = phi_s_original * float(f_keep)
    if phi_s_eff <= 0.0:
        raise ValueError(f"phi_s_eff <= 0 after truncation: {phi_s_eff}. Check V_cut / PSD.")
    
    # 2) Transform to log-space Q0-PSD and generate MC diameters
    d_agg, x_dis, q0_sum_log, V_mean = transform_PSD(
        d3_cent, q3_sum_agg, N_bins, N_aggs, init_mode="q0_quantile_from_q3"
    )
    N_eff = d_agg.size
    
    # 3) Compute control volume Vc (use phi_s_eff!)
    Vc = float(N_eff) * float(V_mean) / float(phi_s_eff)


    x50_Q3 = invert_cdf_x50(d3_cent, q3_sum_agg, q=0.5)
    x50_Q0 = invert_cdf_x50(x_dis, q0_sum_log, q=0.5)
    # --------------------------------------------------------------
    # Reconstruction test at t=0 only: d_agg -> empirical Q0 / Q3
    # (same logic as MCPBEPost: empirical CDF + stepwise evaluation)
    # --------------------------------------------------------------
    if float(exp_t) == 0.0:
        # Q0 from d_agg (number-weighted)
        cdf_num = _compute_psd_cdf_from_d_agg(d_agg, psd_basis="number")
        if cdf_num is None:
            Q0_recon_on_xdis = np.full_like(x_dis, np.nan, dtype=float)
        else:
            x_sorted, Q_sorted = cdf_num
            Q0_recon_on_xdis = _eval_Q_of_x_stepwise(x_sorted, Q_sorted, x_dis)
    
        # Q3 from d_agg (volume-weighted)
        cdf_vol = _compute_psd_cdf_from_d_agg(d_agg, psd_basis="volume")
        if cdf_vol is None:
            Q3_recon_on_d3 = np.full_like(d3_cent, np.nan, dtype=float)
        else:
            x_sorted, Q_sorted = cdf_vol
            Q3_recon_on_d3 = _eval_Q_of_x_stepwise(x_sorted, Q_sorted, d3_cent)
    
        # Plot comparison (t=0 only)
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
        
    # 4) Write to HDF5 (append mode, do not overwrite the file)
    os.makedirs(os.path.dirname(h5_filename) or ".", exist_ok=True)
    with h5py.File(h5_filename, mode="a") as h5f:
        if group_label in h5f:
            if not override:
                raise ValueError(
                    f"Group {group_label!r} already exists in {h5_filename!r}. "
                    "Choose a different label or set override=True."
                )
            # Overwrite old group: delete then recreate
            del h5f[group_label]

        grp = h5f.create_group(group_label)

        # datasets: processed + raw data
        grp.create_dataset("d_agg", data=d_agg, compression="gzip")
        grp.create_dataset("x_dis", data=x_dis, compression="gzip")
        grp.create_dataset("q0_sum_log", data=q0_sum_log, compression="gzip")
        grp.create_dataset("d3_cent", data=d3_cent, compression="gzip")
        grp.create_dataset("q3_sum_agg", data=q3_sum_agg, compression="gzip")

        # attributes: control volume and preprocessing parameters
        grp.attrs["Vc"] = Vc
        grp.attrs["N_bins"] = int(N_bins)
        grp.attrs["N_aggs_target"] = int(N_aggs)
        grp.attrs["N_aggs_eff"] = int(N_eff)
        grp.attrs["cell_size"] = float(cell_size)
        grp.attrs["exp_t"] = float(exp_t)
        grp.attrs["phi_s"] = float(phi_s_eff)                # CHANGED: effective phi_s after truncation
        grp.attrs["phi_s_original"] = float(phi_s_original)  # NEW
        grp.attrs["V_cut"] = float(V_cut)                    # NEW (0.0 means disabled)
        grp.attrs["d_cut"] = float(d_cut)                    # NEW (nan if disabled)
        grp.attrs["Q3_cut"] = float(Q3_cut)                  # NEW (nan if disabled)
        grp.attrs["f_keep_vol"] = float(f_keep)              # NEW remaining volume fraction
        grp.attrs["V_mean"] = float(V_mean)
        grp.attrs["x50_Q0"] = float(x50_Q0)
        grp.attrs["x50_Q3"] = float(x50_Q3)
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
    # V_mean: float,
    sio2_size: float,
    carbon_concentration: float,
    V_cut: float = 0.0,  # NEW
    override: bool = False,
    group_labels: Optional[Sequence[str]] = None,
) -> None:

    """Process all .dat files in a folder and store them into one HDF5 file.

    Parameters
    ----------
    dat_folder : str
        Folder containing CPS-exported .dat files.
    h5_filename : str
        Target HDF5 file to hold all experiments (one group per file/time).
    exp_t_list : sequence of float
        Experimental sampling times (in seconds) for each .dat file.
        The order must correspond to the sorted list of .dat files in
        `dat_folder` (or to group_labels if you provide them explicitly).
    N_bins, N_aggs, cell_size, phi_s, V_mean, sio2_size, carbon_concentration :
        Same meaning as in save_psd_to_h5.
    override : bool, default False
        Passed to save_psd_to_h5 for each group.
    group_labels : sequence of str, optional
        If provided, must have the same length as exp_t_list and the number
        of .dat files. Each label is used as the group name. If None, the
        group labels are derived from the .dat file basenames.
    """
    dat_files = sorted(
        glob(os.path.join(dat_folder, "*.dat")),
        key=extract_minutes_from_name,
    )
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in folder {dat_folder!r}.")

    if len(dat_files) != len(exp_t_list):
        raise ValueError(
            f"Number of .dat files ({len(dat_files)}) does not match "
            f"length of exp_t_list ({len(exp_t_list)}). Please ensure the "
            "time list corresponds to the files in sorted order."
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
            V_cut=V_cut,   # NEW
        )



# ----------------------------------------------------------------------
# Example main: process all CPS *.dat of one series into a single HDF5
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Folder containing CPS exports like:
    # "CPS 1 min CB_pur Q3.dat", "CPS 3 min CB_pur Q3.dat", ...
    dat_folder = os.path.join("input", "PSD_agg")

    # One HDF5 file for the whole series
    h5_file = "CB_pur_N100000_cut.h5"

    # Experimental times (in minutes) corresponding to the sorted .dat files
    # e.g. ["CPS 1 min ...", "CPS 3 min ...", "CPS 5 min ...", ...]
    exp_t_min_list = [1.0, 3.0, 5.0, 10.0, 15.0, 30.0]
    exp_t_sec_list = [(t_min-1) * 60.0 for t_min in exp_t_min_list]

    # Group labels; you can also derive them from filenames or define them manually
    group_labels = [f"t_{int(t_min)}min" for t_min in exp_t_min_list]

    N_bins = 200
    N_aggs = 100000
    cell_size = 0.01       # micrometers (for bookkeeping only)
    phi_s = 0.00005        # solids volume fraction (e.g. 0.005 mass % CB in water)
    # V_mean = 1e-18         # mean aggregate volume [m^3]; can be computed from PSD
    sio2_size = 0.0        # silica primary particle size in nm
    carbon_concentration = 1.0  # carbon concentration during synthesis
    V_cut = 1.76e-21       # d_cut = 150 nm

    batch_save_folder_to_h5(
        dat_folder=dat_folder,
        h5_filename=h5_file,
        exp_t_list=exp_t_sec_list,
        N_bins=N_bins,
        N_aggs=N_aggs,
        cell_size=cell_size,
        phi_s=phi_s,
        # V_mean=V_mean,
        sio2_size=sio2_size,
        carbon_concentration=carbon_concentration,
        V_cut=V_cut,
        override=True,
        group_labels=group_labels,
        
    )

    print(f"Saved PSD data series into {h5_file!r}.")
