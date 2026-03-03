# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:47:04 2025

@author: px2030
"""

# ===============================================
# Materials stage on a fixed lattice aggregate
# - exact composition (A/B counts)
# - target MAS (Ashton & Schmahl) at a given window size
# - geometry (grid) is NOT changed
# ===============================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# -----------------------------
# Public API
# -----------------------------
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np


@dataclass
class MaterialMixParams:
    """
    Parameters controlling the binary (A/B) material mixing on a fixed lattice.

    Attributes
    ----------
    frac_A : float
        Target fraction of material A among the occupied cells. The exact
        integer counts of A/B are enforced by rounding.
    target_MAS : float
        Target Mischgüte (mixing quality) in [0, 1], where 0 corresponds
        to complete segregation and 1 to an ideally randomized state.
    tol_MAS : float
        Acceptable tolerance for the achieved MAS relative to the target.
    window : int
        Sliding window size (in grid cells) used for local variance/MAS
        computation.
    stride : int
        Step of the sliding window. stride=1 means dense evaluation.
    min_occupancy_ratio : float
        A window is considered valid if (occupied cells / window^2)
        is at least this value.

    lambda_min, lambda_max : float
        Lower and upper bounds for the interaction strength λ in the
        Ising/Potts-like energy E = -λ * (# same-material neighbor pairs).
        The algorithm assumes that MAS is monotonic with respect to λ
        on this interval.

    sweeps_per_eval : int
        Number of MCMC sweeps (≈ #occupied proposals) used between
        MAS evaluations during the λ bisection.
    max_bisect : int
        Maximum number of bisection iterations used to tune λ.
    temperature : float
        Metropolis temperature parameter for the MCMC acceptance rule.
    seed : int | None
        Random seed used for the initial labeling and MCMC.
    """
    # composition of A/B (exact counts are enforced)
    frac_A: float = 0.7
    target_MAS: float = 0.5
    tol_MAS: float = 0.02

    # sliding-window MAS settings
    window: int = 16
    stride: int = 4
    min_occupancy_ratio: float = 0.5

    # lower/upper bounds for lambda (interaction strength)
    lambda_min: float = -3.0
    lambda_max: float = 3.0

    # MCMC / bisection controls
    sweeps_per_eval: int = 10
    max_bisect: int = 12
    temperature: float = 1.0
    seed: int | None = None


@dataclass
class MASPhysicalParams:
    """
    Optional physical parameters for the sigma_z^2 (random homogeneous lower
    bound) and sigma_0^2 (complete segregation upper bound) used in the MAS
    definition.

    If you do not have a specific physical model, the defaults correspond
    to a simple pixel-based model:
      - dc = dSi = dcSi = 1
      - Cc = CSi = 0
      - If transmission_weights is None, sigma_0^2 falls back to the
        standard binomial upper bound for a two-phase system.
    """
    # For sigma_z^2 (random homogeneous lower bound)
    dc: float = 1.0
    dSi: float = 1.0
    dcSi: float = 1.0      # effective "thickness" or denominator; dc*dSi/dcSi is used
    Cc: float = 0.0
    CSi: float = 0.0

    # For sigma_0^2 (complete segregation upper bound):
    # If you want to use a transmission-based upper bound, provide a
    # triple of weights (X_[10], X_[01], X_[11]) that sums to 1.
    transmission_weights: Tuple[float, float, float] | None = None


def assign_materials_with_target_mas(
    grid: np.ndarray,
    params: MaterialMixParams,
    phys: MASPhysicalParams = MASPhysicalParams(),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Assign binary materials A/B on a fixed aggregate geometry to achieve
    a target Mischgüte (MAS) via MCMC and λ-bisection.

    The geometry (occupied cells) is fixed by the input grid; only the
    assignment of A/B on those cells is randomized and tuned.

    Parameters
    ----------
    grid : (H, W) array of {0,1}
        Binary aggregate occupancy produced by the lattice generator
        (MPTSA or similar). 1 = occupied; 0 = empty.
    params : MaterialMixParams
        Target composition, MAS, and MCMC control parameters.
    phys : MASPhysicalParams, optional
        Physical parameters used in sigma_z^2 and sigma_0^2, which enter
        the MAS definition.

    Returns
    -------
    labels : (H, W) array of int8
        Label image with {-1 = empty, 0 = material A, 1 = material B}.
    stats : dict
        Summary statistics including:
          - N_total, nA, nB, frac_A
          - lambda_used
          - MAS, sigma2, sigma0_sq, sigmaz_sq
          - window, stride
    """
    rng = np.random.default_rng(params.seed)
    occ_y, occ_x = np.nonzero(grid)
    M = len(occ_y)
    if M == 0:
        raise ValueError("Empty aggregate grid.")

    # ---- exact composition counts ----
    nA = int(round(params.frac_A * M))
    nA = max(0, min(M, nA))
    nB = M - nA  # retained for clarity; not used directly further

    # Create a flat label vector over occupied indices.
    # 'order' is a random permutation of [0, M).
    # 'lbl_flat' assigns materials strictly according to the target counts.
    order = rng.permutation(M)
    lbl_flat = np.empty(M, dtype=np.int8)
    lbl_flat[order[:nA]] = 0  # A
    lbl_flat[order[nA:]] = 1  # B

    # Mapping between (y, x) <-> flat index on occupied cells.
    H, W = grid.shape
    to_id = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(occ_y, occ_x))}
    neighbors = _build_neighbors_4(occ_y, occ_x, to_id, H, W)

    # ---- tune lambda by bisection to match target MAS ----
    lam_lo, lam_hi = float(params.lambda_min), float(params.lambda_max)

    # Evaluate at bounds to establish monotonic order.
    # (In this Ising-like model MAS is assumed to be monotonic in λ
    # over the chosen range.)
    lbl_work = lbl_flat.copy()
    mas_lo, _, _, _ = _evaluate_mas_on_labels(grid, occ_y, occ_x, lbl_work, params, phys)
    _mcmc_exchange(lbl_work, neighbors, lam_lo,
                   sweeps=params.sweeps_per_eval,
                   T=params.temperature, rng=rng)  # slight thermalization

    lbl_work2 = lbl_flat.copy()
    _mcmc_exchange(lbl_work2, neighbors, lam_hi,
                   sweeps=params.sweeps_per_eval,
                   T=params.temperature, rng=rng)
    mas_hi, _, _, _ = _evaluate_mas_on_labels(grid, occ_y, occ_x, lbl_work2, params, phys)

    # Ensure mas_lo <= mas_hi by swapping bounds if needed
    if mas_lo > mas_hi:
        lam_lo, lam_hi = lam_hi, lam_lo
        lbl_work, lbl_work2 = lbl_work2, lbl_work
        mas_lo, mas_hi = mas_hi, mas_lo

    target = float(np.clip(params.target_MAS, 0.0, 1.0))
    best_lbl = lbl_flat.copy()
    best_mas, best_lam = None, None

    for _ in range(params.max_bisect):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        # Start from the configuration closer to the target side to help convergence.
        lbl_mid = lbl_work.copy() if (target <= mas_hi) else lbl_work2.copy()

        _mcmc_exchange(
            lbl_mid,
            neighbors,
            lam_mid,
            sweeps=max(1, params.sweeps_per_eval),
            T=params.temperature,
            rng=rng,
        )
        mas_mid, sigma2, sig0, sigz = _evaluate_mas_on_labels(
            grid, occ_y, occ_x, lbl_mid, params, phys
        )

        # Record best (closest to target) so far
        best_lbl, best_mas, best_lam = lbl_mid, mas_mid, lam_mid

        if abs(mas_mid - target) <= params.tol_MAS:
            lbl_flat = lbl_mid
            break

        # Bisection update based on measured MAS
        if mas_mid < target:
            lam_lo, lbl_work = lam_mid, lbl_mid
            mas_lo = mas_mid
        else:
            lam_hi, lbl_work2 = lam_mid, lbl_mid
            mas_hi = mas_mid
    else:
        # If convergence not reached within max_bisect, fall back to
        # the best solution encountered.
        lbl_flat = best_lbl
        mas_mid, sigma2, sig0, sigz = _evaluate_mas_on_labels(
            grid, occ_y, occ_x, lbl_flat, params, phys
        )

    # Build full label image
    labels = np.full((H, W), fill_value=-1, dtype=np.int8)
    labels[occ_y, occ_x] = lbl_flat

    # Final MAS evaluation on the chosen configuration
    MAS, sigma2, sigma0, sigmaz = _evaluate_mas_on_labels(
        grid, occ_y, occ_x, lbl_flat, params, phys
    )

    stats = dict(
        N_total=int(M),
        nA=int((labels == 0).sum()),
        nB=int((labels == 1).sum()),
        frac_A=(labels == 0).sum() / M,
        lambda_used=float(best_lam if best_lam is not None else lam_mid),
        MAS=float(MAS),
        sigma2=float(sigma2),
        sigma0_sq=float(sigma0),
        sigmaz_sq=float(sigmaz),
        window=int(params.window),
        stride=int(params.stride),
    )
    return labels, stats

# -----------------------------
# Visualization (optional)
# -----------------------------
def plot_materials_grid(grid: np.ndarray, labels: np.ndarray, origin: Tuple[int, int]) -> None:
    """
    Visualize a 2D aggregate with overlaid material labels.

    Parameters
    ----------
    grid : (H, W) array of {0,1}
        Binary occupancy of the aggregate (1 = occupied).
    labels : (H, W) array of int
        Material labels where {-1 = empty, 0 = A, 1 = B}.
        Only occupied positions are shown.
    origin : (x_min, y_min)
        The original lattice offset before cropping.

    Notes
    -----
    - The occupancy grid is drawn first with low alpha.
    - Material labels are drawn as an overlay with imshow colormap.
    - Gridlines are drawn for clarity.
    """
    H, W = grid.shape
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # Base layer: occupancy
    ax.imshow(grid, origin='lower', interpolation='nearest', alpha=0.35)

    # Overlay: labels at occupied cells
    show = np.ma.masked_where(labels < 0, labels)
    ax.imshow(show, origin='lower', interpolation='nearest')

    x_min, y_min = origin
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)

    ax.set_title("2D Lattice Aggregate with Materials")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=0.3)

    plt.tight_layout()
    plt.show()


# -----------------------------
# MAS Evaluation Internals
# -----------------------------

def _evaluate_mas_on_labels(grid, occ_y, occ_x, lbl_flat, params, phys):
    """
    Compute MAS (Mischgüte) and related statistics for a particular
    A/B labeling on a fixed aggregate geometry.

    Parameters
    ----------
    grid : (H, W) array of {0,1}
        Occupancy of the aggregate.
    occ_y, occ_x : arrays
        Coordinates of occupied cells.
    lbl_flat : length-M array of {0,1}
        Material labels for each occupied cell in flat ordering.
    params : MaterialMixParams
        Controls MAS evaluation (window, stride, etc.).
    phys : MASPhysicalParams
        Physical constants for sigma_z^2 and sigma_0^2 models.

    Returns
    -------
    MAS : float
        Final Mischgüte value in [0,1].
    sigma2 : float
        Observed local variance (σ²).
    sigma0_sq : float
        Upper variance bound (σ₀²).
    sigmaz_sq : float
        Lower variance bound (σ_z²).

    Notes
    -----
    - A sliding-window variance σ² is computed via integral images.
    - The grid must be large enough to accommodate the window size.
    """

    H, W = grid.shape
    labels = np.full((H, W), -1, dtype=np.int8)
    labels[occ_y, occ_x] = lbl_flat

    M = len(occ_y)
    nA = int((lbl_flat == 0).sum())
    X_A = nA / M

    # Check that the MAS window fits inside the grid
    if H < params.window or W < params.window:
        msg = (
            "[MAS] Grid too small for window-based MAS evaluation.\n"
            f"       grid shape = ({H}, {W}), "
            f"window = {params.window}, stride = {params.stride}.\n"
            "       Increase aggregate size or reduce window/stride."
        )
        print(msg)
        raise ValueError(msg)

    sigma2 = _sigma2_window_variance(
        labels, grid, params.window, params.stride, params.min_occupancy_ratio
    )

    # Upper bound σ₀²
    sigma0_sq = _sigma0_upper_bound(X_A, phys)

    # Lower bound σ_z²
    sigmaz_sq = _sigmaz_lower_bound(
        X_A, labels, grid, params.window, params.stride,
        phys.dc, phys.dSi, phys.dcSi, phys.Cc, phys.CSi
    )

    # Compute MAS = log(σ₀² / σ²) / log(σ₀² / σ_z²)
    eps = 1e-12
    num = np.log((sigma0_sq + eps) / (sigma2 + eps))
    den = np.log((sigma0_sq + eps) / (sigmaz_sq + eps))
    MAS = np.clip(num / (den + eps), 0.0, 1.0)

    return float(MAS), float(sigma2), float(sigma0_sq), float(sigmaz_sq)


@njit(cache=True)
def _sigma2_window_variance(labels: np.ndarray, grid: np.ndarray,
                            window: int, stride: int, min_occ_ratio: float) -> float:
    """
    Compute the local composition variance σ² using sliding
    windows with integral-image acceleration.

    Parameters
    ----------
    labels : (H, W) int array
        Material labels where {0,1} indicate materials and -1 = empty.
    grid : (H, W) array of {0,1}
        Aggregate occupancy map.
    window : int
        Window size (w × w).
    stride : int
        Sliding-window stride.
    min_occ_ratio : float
        Minimum fraction of occupied cells inside a window for it
        to be considered valid.

    Returns
    -------
    float
        The population variance of A-fraction across all valid windows.

    Notes
    -----
    - This uses integral images for O(1) window sums.
    - Only windows with sufficient occupancy contribute.
    - If no valid windows exist, σ² = 0 by definition.
    """

    H, W = grid.shape
    w = int(max(1, window))
    s = int(max(1, stride))

    # Occupancy mask
    occ = (grid.astype(np.uint8) == 1).astype(np.int32)

    # A-phase mask (only on occupied cells)
    A_mask = ((labels == 0) & (occ == 1)).astype(np.int32)

    occ_ii = _integral_image(occ)
    A_ii = _integral_image(A_mask)

    # Upper bound for all windows assuming all valid
    max_n = ((H - w) // s + 1) * ((W - w) // s + 1)
    vals = np.empty(max_n, dtype=np.float64)
    k = 0

    y = 0
    while y <= H - w:
        x = 0
        while x <= W - w:
            occ_cnt = _rect_sum(occ_ii, x, y, w, w)
            if occ_cnt >= min_occ_ratio * (w*w) and occ_cnt > 0:
                A_cnt = _rect_sum(A_ii, x, y, w, w)
                vals[k] = A_cnt / occ_cnt
                k += 1
            x += s
        y += s

    if k == 0:
        return 0.0

    # population variance
    mean = 0.0
    for i in range(k):
        mean += vals[i]
    mean /= k

    var = 0.0
    for i in range(k):
        d = vals[i] - mean
        var += d * d
    var /= k

    return var

def _sigma0_upper_bound(X_A: float, phys: MASPhysicalParams) -> float:
    """
    Compute the upper variance bound σ₀² corresponding to complete
    segregation between A and B.

    Two options
    -----------
    1. Default (no transmission model):
         σ₀² = X_A (1 - X_A)
    2. Transmission-weighted model:
         σ₀² = w10 * (1 - X_A)² + w01 * (0 - X_A)² + w11 * (0.5 - X_A)²

       where (w10, w01, w11) sum to 1.
    """
    if phys.transmission_weights is None:
        return float(X_A * (1.0 - X_A))

    w10, w01, w11 = phys.transmission_weights

    term = (
        w10 * (1.0 - X_A) ** 2 +
        w01 * (0.0 - X_A) ** 2 +
        w11 * (0.5 - X_A) ** 2
    )
    return float(term)


@njit(cache=True)
def _sigmaz_lower_bound(X_A: float, labels: np.ndarray, grid: np.ndarray,
                        window: int, stride: int,
                        phys_dc: float, phys_dSi: float, phys_dcSi: float,
                        phys_Cc: float, phys_CSi: float) -> float:
    """
    Compute the lower variance bound σ_z² for a random homogeneous
    mixture of A and B, using physical parameters.

    The bound is of the form:
        σ_z² = [ X_A (1 - X_A) / N̄ ] * geom_factor * size_factor

    where
        N̄          = average number of occupied pixels per valid window,
        geom_factor = (dc * dSi / dcSi)²,
        size_factor = 1 + (1 - X_A) * Cc² + X_A * CSi².

    Parameters
    ----------
    X_A : float
        Global fraction of phase A.
    labels : (H, W) array
        Label map; not used directly here but kept for consistency.
    grid : (H, W) array of {0,1}
        Occupancy mask of the aggregate.
    window : int
        Window size for counting occupancy.
    stride : int
        Step size between windows.
    phys_dc, phys_dSi, phys_dcSi : float
        Physical thickness / scaling parameters.
    phys_Cc, phys_CSi : float
        Coefficients encoding size contrast of the two phases.

    Returns
    -------
    float
        Lower variance bound σ_z².
    """
    H, W = grid.shape
    w = int(max(1, window))
    s = int(max(1, stride))

    occ = (grid.astype(np.uint8) == 1).astype(np.int32)
    occ_ii = _integral_image(occ)

    # Accumulate occupied counts over all windows and take the mean
    total = 0.0
    count = 0
    y = 0
    while y <= H - w:
        x = 0
        while x <= W - w:
            cnt = _rect_sum(occ_ii, x, y, w, w)
            if cnt > 0:
                total += cnt
                count += 1
            x += s
        y += s

    if count == 0:
        return 0.0

    N_bar = total / count

    geom_factor = (phys_dc * phys_dSi / max(phys_dcSi, 1e-12)) ** 2
    size_factor = 1.0 + (1.0 - X_A) * (phys_Cc ** 2) + X_A * (phys_CSi ** 2)

    return (X_A * (1.0 - X_A) / max(N_bar, 1e-12)) * geom_factor * size_factor


@njit(cache=True)
def _integral_image(a: np.ndarray) -> np.ndarray:
    """
    Compute the (0,0)-anchored integral image of a 2D array.

    ii[i+1, j+1] contains the sum of a[0..i, 0..j], i.e.
        ii[y+1, x+1] = sum_{0<=i<=y, 0<=j<=x} a[i,j]

    Parameters
    ----------
    a : (H, W) array
        Input array (typically int32).

    Returns
    -------
    ii : (H+1, W+1) int64 array
        Integral image with one extra row and column of zeros.
    """
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
    # Row-wise prefix sums + column-wise accumulation
    for i in range(a.shape[0]):
        row_sum = 0
        for j in range(a.shape[1]):
            row_sum += int(a[i, j])
            ii[i + 1, j + 1] = ii[i, j + 1] + row_sum
    return ii


@njit(cache=True)
def _rect_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """
    Compute the sum over a rectangle of size (w, h) in the original
    array using its integral image.

    The rectangle is [y, y+h) × [x, x+w).

    Parameters
    ----------
    ii : (H+1, W+1) array
        Integral image computed by _integral_image.
    x, y : int
        Top-left corner of the rectangle in the original array.
    w, h : int
        Width and height of the rectangle.

    Returns
    -------
    int
        Sum of a[y:y+h, x:x+w].
    """
    x2, y2 = x + w, y + h
    return int(ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x])


# -----------------------------
# Internals: MCMC with pairwise exchanges (exact composition)
# -----------------------------

def _build_neighbors_4(ys: np.ndarray, xs: np.ndarray,
                       to_id: Dict[Tuple[int, int], int],
                       H: int, W: int) -> List[List[int]]:
    """
    Build a 4-neighbor adjacency list for occupied sites.

    Parameters
    ----------
    ys, xs : arrays
        Coordinates of occupied cells.
    to_id : dict[(y, x) -> int]
        Mapping from lattice coordinates to flat indices.
    H, W : int
        Grid height and width.

    Returns
    -------
    neighbors : list of list of int
        neighbors[u] contains flat indices of 4-neighbor occupied sites of u.
    """
    M = len(ys)
    neigh = [[] for _ in range(M)]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = int(y + dy), int(x + dx)
            if 0 <= ny < H and 0 <= nx < W:
                j = to_id.get((ny, nx), None)
                if j is not None:
                    neigh[i].append(j)
    return neigh


def _neighbors_to_csr(neighbors: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list-of-lists adjacency structure to a CSR-like format.

    Parameters
    ----------
    neighbors : list of list of int
        neighbors[u] is the list of neighbors for site u.

    Returns
    -------
    indices : (E,) int32 array
        Flattened neighbor indices.
    indptr : (M+1,) int32 array
        Row-pointer array where neighbors of u are stored in
        indices[indptr[u]:indptr[u+1]].
    """
    indptr = [0]
    flat = []
    for lst in neighbors:
        flat.extend(lst)
        indptr.append(len(flat))
    return np.asarray(flat, dtype=np.int32), np.asarray(indptr, dtype=np.int32)


@njit(cache=True)
def _count_same_material_edges(labels: np.ndarray,
                               indices: np.ndarray,
                               indptr: np.ndarray,
                               u: int,
                               mat: int) -> int:
    """
    Count how many neighbors of node u have the same material label.

    Parameters
    ----------
    labels : (M,) int array
        Material label for each occupied site.
    indices, indptr : arrays
        CSR adjacency representation: neighbors of u are in
        indices[indptr[u] : indptr[u+1]].
    u : int
        Node index to inspect.
    mat : int
        Material label to compare against.

    Returns
    -------
    int
        Number of neighbors v of u with labels[v] == mat.
    """
    s = 0
    start = indptr[u]
    end = indptr[u + 1]
    for k in range(start, end):
        n = indices[k]
        if labels[n] == mat:
            s += 1
    return s


def _mcmc_exchange(labels: np.ndarray,
                   neighbors: List[List[int]],
                   lam: float,
                   sweeps: int,
                   T: float,
                   rng: np.random.Generator) -> None:
    """
    Run an MCMC chain that exchanges A/B labels between occupied sites
    while keeping the global composition fixed.

    Energy model
    ------------
      E = - λ * S,    where
      S = number of same-material neighbor pairs (over the 4-neighbor graph).

    Each MCMC step proposes swapping labels of one A-site and one B-site,
    then accepts or rejects via a Metropolis rule.

    Parameters
    ----------
    labels : (M,) int array
        Flat label array over occupied sites, with {0 = A, 1 = B}.
        Modified in-place.
    neighbors : list of list of int
        4-neighbor adjacency list for occupied sites.
    lam : float
        Interaction strength λ in the Ising-like energy.
    sweeps : int
        Number of sweeps; each sweep ≈ M proposals.
    T : float
        Metropolis temperature.
    rng : np.random.Generator
        Random number generator.
    """
    M = labels.shape[0]
    steps = int(max(1, sweeps) * M)

    # Precompute CSR adjacency (one-time conversion)
    indices, indptr = _neighbors_to_csr(neighbors)

    idxA = np.where(labels == 0)[0]
    idxB = np.where(labels == 1)[0]

    for _ in range(steps):
        if len(idxA) == 0 or len(idxB) == 0:
            break

        i = int(rng.integers(0, len(idxA)))
        j = int(rng.integers(0, len(idxB)))
        u = int(idxA[i])  # label 0 (A)
        v = int(idxB[j])  # label 1 (B)

        # ΔS: same-material edges before vs after the swap
        Su_before = _count_same_material_edges(labels, indices, indptr, u, 0)
        Sv_before = _count_same_material_edges(labels, indices, indptr, v, 1)
        Su_after  = _count_same_material_edges(labels, indices, indptr, u, 1)
        Sv_after  = _count_same_material_edges(labels, indices, indptr, v, 0)
        dS = (Su_after + Sv_after) - (Su_before + Sv_before)

        dE = -lam * dS
        accept = (dE <= 0.0) or (rng.random() < np.exp(-dE / max(T, 1e-9)))

        if accept:
            labels[u], labels[v] = 1, 0
            idxA[i], idxB[j] = v, u
