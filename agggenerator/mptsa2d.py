# -*- coding: utf-8 -*-
"""
2D MPTSA on-lattice（四邻、对齐的网格版）
- 新增：可选的小洞填补 + 等量外缘删点补偿 (fill_hole=True)
- 不依赖 SciPy，仅 numpy

依赖：numpy、matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Set, List
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# -----------------------------
# Utilities (shared)
# -----------------------------

@njit(cache=True, fastmath=True)
def radius_of_gyration_2d(positions: np.ndarray) -> float:
    """
    Compute the 2D radius of gyration of a set of points.

    Parameters
    ----------
    positions : (N, 2) array
        Coordinates of occupied sites or particles.

    Returns
    -------
    float
        Radius of gyration. Returns 0.0 if no positions are given.
    """
    if positions.size == 0:
        return 0.0

    n = positions.shape[0]

    # Compute center of mass
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += positions[i, 0]
        my += positions[i, 1]
    mx /= n
    my /= n

    # Accumulate squared distances from the center of mass
    acc = 0.0
    for i in range(n):
        dx = positions[i, 0] - mx
        dy = positions[i, 1] - my
        acc += dx * dx + dy * dy

    return (acc / n) ** 0.5

def estimate_fractal_dimension_2d(Ns: np.ndarray, Rgs: np.ndarray) -> Tuple[float, float]:
    """
    Estimate the fractal dimension from radius-of-gyration data in 2D.

    Uses the scaling relation Rg ~ N^(1/Df), so that
    log(Rg) = (1/Df) * log(N) + const.

    Parameters
    ----------
    Ns : array
        Number of occupied sites / particles for each measurement.
    Rgs : array
        Corresponding radius of gyration values.

    Returns
    -------
    Df_est : float
        Estimated fractal dimension (NaN if it cannot be estimated).
    slope : float
        Slope of log(Rg) vs. log(N). Should be 1/Df when the scaling holds.
    """
    mask = (Ns >= 2) & (Rgs > 0)
    if np.count_nonzero(mask) < 2:
        # Not enough valid data points for a reliable fit
        return float("nan"), float("nan")

    x = np.log(Ns[mask])
    y = np.log(Rgs[mask])
    slope, intercept = np.polyfit(x, y, 1)
    Df_est = float("nan") if slope <= 0 else float(1.0 / slope)
    return Df_est, float(slope)

# -----------------------------
# Lattice MPTSA (2D, 4-neighbor)
# -----------------------------

@dataclass
class MPTSALatticeParams2D:
    """
    Parameters controlling the 2D lattice MPTSA growth.

    Attributes
    ----------
    Np : int
        Target number of occupied lattice sites (particle size).
    Df : float
        Target fractal dimension.
    k : float
        Prefactor for the Rg–N scaling relation.
    max_attempts : int
        Maximum number of growth attempts before giving up.
    seed : int | None
        Random seed for the generator.

    fill_hole : bool
        Whether to run the post-processing step that fills small internal
        holes and compensates by removing the same number of sites elsewhere.
    hole_area_max : int
        Maximum area (in cells) that is still considered a "small hole".
    compensate_alpha : float
        Weight on r^2 when scoring candidate sites to remove during
        compensation (smaller → biased towards sites closer to the centroid).
    compensate_beta : float
        Weight on "exposedness" when scoring removal candidates
        (larger → biased towards sites on the outer rim).
    verbose : bool
        If True, print diagnostic information during growth/post-processing.
    """
    Np: int = 300
    Df: float = 1.8
    k: float = 1.0
    max_attempts: int = 20000
    seed: int | None = None

    # Post-processing to fill small holes and remove the same number of sites
    fill_hole: bool = False
    hole_area_max: int = 4
    compensate_alpha: float = 1.0
    compensate_beta: float = 0.25
    verbose: bool = False


@njit(cache=True, fastmath=True)
def _compute_radial_weights(
    positions: np.ndarray,
    Df: float,
    k: float,
    adaptive_bias: float,
) -> np.ndarray:
    """
    Compute radial weights used to bias growth or deletion decisions.

    The weight of each site scales roughly like (r + eps)^beta, where
    r is the distance from the current center of mass. The exponent beta
    is adapted based on the ratio between the current Rg and the target Rg
    implied by (Df, k).

    Parameters
    ----------
    positions : (N, 2) array
        Coordinates of the currently occupied sites.
    Df : float
        Target fractal dimension.
    k : float
        Prefactor in the Rg–N scaling relation.
    adaptive_bias : float
        Sensitivity of beta to deviations between current and target Rg.
        Larger values make the bias stronger.

    Returns
    -------
    w : (N,) array
        Radial weights for each site.
    """
    n = positions.shape[0]

    # Center of mass
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += positions[i, 0]
        my += positions[i, 1]
    mx /= n
    my /= n

    # Current radius of gyration
    acc = 0.0
    for i in range(n):
        dx = positions[i, 0] - mx
        dy = positions[i, 1] - my
        acc += dx * dx + dy * dy
    Rg = (acc / n) ** 0.5

    # Target Rg from the scaling law Rg ~ a * N^(1/Df)
    a = 0.6 * (k ** (1.0 / max(Df, 1e-6)))
    Rg_target = a * (n ** (1.0 / max(Df, 1e-6)))

    ratio = Rg_target / max(Rg, 1e-12)
    if ratio < 0.1:
        ratio = 0.1

    # Adapt beta depending on whether we are too compact (ratio > 1)
    # or too extended (ratio < 1). adaptive_bias controls the strength.
    if ratio > 1.0:
        beta = 1.0 + adaptive_bias * (ratio - 1.0)
    else:
        beta = 1.0 + 0.5 * (ratio - 1.0)

    # Clamp beta to roughly match the original parameter range
    if beta < 0.2:
        beta = 0.2
    if beta > 6.0:
        beta = 6.0

    w = np.empty(n, dtype=np.float64)
    eps = 1e-6
    for i in range(n):
        dx = positions[i, 0] - mx
        dy = positions[i, 1] - my
        r = (dx * dx + dy * dy) ** 0.5
        w[i] = (r + eps) ** beta
    return w

def _choose_base_index_lattice(
    positions: np.ndarray,
    rng: np.random.Generator,
    Df: float,
    k: float,
    adaptive_bias: float = 2.0,
) -> int:
    """
    Choose a base index (existing occupied site) from which a new site
    will be grown in the MPTSA lattice algorithm.

    The radial weight computation is delegated to `_compute_radial_weights`
    (Numba-accelerated), and the final sampling is done in Python.

    Parameters
    ----------
    positions : (N, 2) array
        Coordinates of currently occupied sites.
    rng : np.random.Generator
        Random number generator.
    Df : float
        Target fractal dimension.
    k : float
        Scaling prefactor for the Rg–N relation.
    adaptive_bias : float
        Controls how strongly deviations from the target Rg influence
        the sampling exponent beta.

    Returns
    -------
    int
        Index of the chosen base site.
    """
    w = _compute_radial_weights(positions, float(Df), float(k), float(adaptive_bias))
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        # Fallback: uniform choice
        return int(rng.integers(low=0, high=positions.shape[0]))

    p = w / w_sum
    return int(rng.choice(np.arange(positions.shape[0]), p=p))


def _empty_orth_neighbors(
    p: Tuple[int, int],
    occ: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Return the 4-neighborhood empty lattice sites around point p.

    Parameters
    ----------
    p : (x, y)
        The lattice coordinate being inspected.
    occ : set of (x, y)
        Occupied lattice coordinates.

    Returns
    -------
    list of (x, y)
        List of orthogonally-adjacent empty lattice sites.
    """
    x, y = p
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [q for q in candidates if q not in occ]


def _to_min_bounding_grid(
    occ: Set[Tuple[int, int]]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Convert an unordered set of occupied (x, y) coordinates into a compact,
    tightly bounding binary grid.

    Parameters
    ----------
    occ : set of (x, y)
        Occupied lattice coordinates.

    Returns
    -------
    grid : (H, W) uint8 array
        A binary grid containing the aggregate shape.
    origin : (x_min, y_min)
        The original coordinate offset such that:
        grid[y - y_min, x - x_min] corresponds to the original point (x, y).
    """
    xs = [x for x, _ in occ]
    ys = [y for _, y in occ]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    W = x_max - x_min + 1
    H = y_max - y_min + 1

    grid = np.zeros((H, W), dtype=np.uint8)
    for x, y in occ:
        grid[y - y_min, x - x_min] = 1

    return grid, (x_min, y_min)


def generate_mptsa_lattice_2d(
    params: MPTSALatticeParams2D
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Generate a 2D lattice aggregate using the MPTSA algorithm (4-neighbor growth).

    Optionally apply a post-processing step that fills small internal holes
    and compensates by deleting an equal number of exposed boundary sites.

    Parameters
    ----------
    params : MPTSALatticeParams2D
        Growth and post-processing parameters.

    Returns
    -------
    positions : (N, 2) float array
        Final occupied site coordinates in the global lattice coordinate system.
    Ns : array
        Sequence of occupancy counts during growth (used for statistics).
    Rgs : array
        Sequence of radius-of-gyration values corresponding to Ns.
    grid : (H, W) uint8 array
        Bounding binary grid of the final aggregate (after post-processing).
    origin : (x_min, y_min)
        Offset mapping grid coordinates back to global coordinates.
    """
    rng = np.random.default_rng(params.seed)
    Df = float(params.Df)
    k = float(params.k)

    # Initialize with two adjacent sites
    occ: Set[Tuple[int, int]] = set()
    occ.add((0, 0))
    occ.add((1, 0))

    positions = np.array([(0, 0), (1, 0)], dtype=float)
    Ns = [2]
    Rgs = [radius_of_gyration_2d(positions)]

    # --------------------
    # Main MPTSA growth
    # --------------------
    while positions.shape[0] < params.Np:
        placed = False

        for _ in range(params.max_attempts):
            base_idx = _choose_base_index_lattice(positions, rng, Df=Df, k=k)
            bx, by = map(int, positions[base_idx])

            empties = _empty_orth_neighbors((bx, by), occ)
            if not empties:
                continue

            cx, cy = empties[rng.integers(0, len(empties))]
            occ.add((cx, cy))
            positions = np.vstack([positions, [cx, cy]])
            placed = True
            break

        if not placed:
            raise RuntimeError(
                "Placement failed: no valid empty 4-neighbor site found after "
                "many attempts. Consider increasing max_attempts or adjusting Df."
            )

        Ns.append(positions.shape[0])
        Rgs.append(radius_of_gyration_2d(positions))

    # Convert to minimal bounding grid
    grid, origin = _to_min_bounding_grid(occ)

    # -------------------------------
    # Optional post-processing: fill small holes + delete boundary sites
    # -------------------------------
    if params.fill_hole:
        grid, n_fill, n_del = fill_small_holes_and_compensate(
            grid,
            area_max=params.hole_area_max,
            alpha=params.compensate_alpha,
            beta=params.compensate_beta,
            verbose=params.verbose,
        )

        if params.verbose:
            print(f"[Post] holes filled: {n_fill}, boundary deletions: {n_del}")

        # Reconstruct positions from the cleaned grid
        ys, xs = np.nonzero(grid)
        positions = np.column_stack([xs + origin[0], ys + origin[1]]).astype(float)

        # Recompute Ns and Rgs for consistency with the final shape
        N_final = positions.shape[0]
        Ns = np.arange(2, N_final + 1, dtype=float)
        Rgs = np.empty_like(Ns)
        for i, n in enumerate(Ns):
            Rgs[i] = radius_of_gyration_2d(positions[:int(n)])

    return (
        positions,
        np.asarray(Ns, dtype=float),
        np.asarray(Rgs, dtype=float),
        grid,
        origin,
    )

# -----------------------------
# Post-process: Fill holes + equal deletions
# -----------------------------
def fill_small_holes_and_compensate(
    grid: np.ndarray,
    area_max: int = 4,
    alpha: float = 1.0,
    beta: float = 0.25,
    verbose: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """
    Fill small internal holes in a binary aggregate and compensate by
    deleting the same number of boundary pixels.

    Steps
    -----
    1) Fill all interior 4-connected background components (holes) whose
       area is <= area_max. The total number of filled pixels is N_fill.
    2) Starting from the outer boundary, delete N_fill foreground pixels
       using a scoring strategy that prefers leaf-like or exposed pixels,
       while preserving global connectivity:
           - After removal, the foreground must remain a single component.
           - Score for a candidate pixel is:
                 score = alpha * r^2 - beta * exposure
             where smaller score is deleted first.

    Parameters
    ----------
    grid : (H, W) array of uint8
        Binary foreground mask (1 = occupied, 0 = empty).
    area_max : int
        Maximum area of a 4-connected 0-component to be considered a
        "small hole" and filled.
    alpha : float
        Weight on r^2 (distance to centroid) in the deletion score.
        Smaller alpha favors deleting pixels closer to the centroid.
    beta : float
        Weight on the exposure term (# of empty/edge neighbors).
        Larger beta favors deleting pixels on the outer rim.
    verbose : bool
        If True, print basic diagnostic information.

    Returns
    -------
    g : (H, W) array of uint8
        Processed binary mask after hole filling and boundary deletions.
    n_fill : int
        Total number of pixels filled (sum of all small-hole areas).
    n_del : int
        Total number of boundary pixels removed (should equal n_fill
        when n_fill > 0).
    """
    g = grid.copy().astype(np.uint8)
    n_fill = _fill_holes_inplace(g, area_max=area_max)
    if verbose:
        print(f"[Fill] filled holes (area <= {area_max}): {n_fill}")

    n_del = 0
    if n_fill > 0:
        n_del = _delete_boundary_pixels(
            g, n_delete=n_fill, alpha=alpha, beta=beta, verbose=verbose
        )

    return g, n_fill, n_del


# ---- hole filling helpers ---- #

def _fill_holes_inplace(g: np.ndarray, area_max: int) -> int:
    """
    In-place filling of small interior holes in a binary mask.

    Uses a 4-neighbor flood-fill from the padded outer background
    to mark the "outside". Remaining 0 pixels that are not connected
    to the outside are interior holes. Only those holes whose area
    is <= area_max are filled.

    Parameters
    ----------
    g : (H, W) uint8 array
        Binary mask, modified in-place (1 = foreground, 0 = background).
    area_max : int
        Maximum hole area (in pixels) to be filled.

    Returns
    -------
    filled : int
        Total number of pixels that were turned from 0 to 1.
    """
    H, W = g.shape
    # Pad by 1 cell to avoid special-casing borders
    gp = np.zeros((H + 2, W + 2), dtype=np.uint8)
    gp[1:-1, 1:-1] = g

    outside = np.zeros_like(gp, dtype=np.uint8)
    dq = deque()

    # Seeds: background pixels on the padded outer frame
    for x in range(W + 2):
        if gp[0, x] == 0:
            outside[0, x] = 1
            dq.append((0, x))
        if gp[H + 1, x] == 0:
            outside[H + 1, x] = 1
            dq.append((H + 1, x))

    for y in range(H + 2):
        if gp[y, 0] == 0:
            outside[y, 0] = 1
            dq.append((y, 0))
        if gp[y, W + 1] == 0:
            outside[y, W + 1] = 1
            dq.append((y, W + 1))

    # Flood-fill the outer background (4-neighbor)
    while dq:
        y, x = dq.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < H + 2 and 0 <= nx < W + 2:
                if gp[ny, nx] == 0 and outside[ny, nx] == 0:
                    outside[ny, nx] = 1
                    dq.append((ny, nx))

    # Remaining 0-pixels that are not marked as outside are candidate holes.
    # We traverse each connected component and fill only if its area <= area_max.
    visited = np.zeros_like(gp, dtype=np.uint8)
    filled = 0

    for y in range(1, H + 1):
        for x in range(1, W + 1):
            if gp[y, x] == 0 and outside[y, x] == 0 and visited[y, x] == 0:
                # Found one 4-connected hole component
                comp = []
                dq.clear()
                visited[y, x] = 1
                dq.append((y, x))

                while dq:
                    cy, cx = dq.popleft()
                    comp.append((cy, cx))
                    for ny, nx in ((cy + 1, cx), (cy - 1, cx),
                                   (cy, cx + 1), (cy, cx - 1)):
                        if 1 <= ny <= H and 1 <= nx <= W:
                            if (gp[ny, nx] == 0 and
                                outside[ny, nx] == 0 and
                                visited[ny, nx] == 0):
                                visited[ny, nx] = 1
                                dq.append((ny, nx))

                # Fill only if the hole is small enough
                if len(comp) <= area_max:
                    for (cy, cx) in comp:
                        gp[cy, cx] = 1
                    filled += len(comp)

    g[:, :] = gp[1:-1, 1:-1]
    return filled


# ---- boundary deletion helpers ---- #

@njit(cache=True)
def _boundary_points_njit(g: np.ndarray) -> np.ndarray:
    """
    Collect all boundary foreground pixels (4-neighbor) in a binary mask.

    A pixel is considered boundary if it is foreground (1) and at least one
    of its 4 neighbors is background (0) or out of bounds.

    Parameters
    ----------
    g : (H, W) uint8 array
        Binary mask.

    Returns
    -------
    out : (K, 2) int32 array
        List of (y, x) coordinates for boundary pixels.
    """
    H, W = g.shape

    # First pass: count boundary pixels
    cnt = 0
    for y in range(H):
        for x in range(W):
            if g[y, x] == 1:
                if ((y == 0   or g[y - 1, x] == 0) or
                    (y == H-1 or g[y + 1, x] == 0) or
                    (x == 0   or g[y, x - 1] == 0) or
                    (x == W-1 or g[y, x + 1] == 0)):
                    cnt += 1

    out = np.empty((cnt, 2), dtype=np.int32)

    # Second pass: record boundary coordinates
    k = 0
    for y in range(H):
        for x in range(W):
            if g[y, x] == 1:
                if ((y == 0   or g[y - 1, x] == 0) or
                    (y == H-1 or g[y + 1, x] == 0) or
                    (x == 0   or g[y, x - 1] == 0) or
                    (x == W-1 or g[y, x + 1] == 0)):
                    out[k, 0] = y
                    out[k, 1] = x
                    k += 1

    return out


@njit(cache=True)
def _degree_njit(g: np.ndarray, y: int, x: int) -> int:
    """
    Compute the 4-neighbor degree (number of foreground neighbors) of a pixel.

    Parameters
    ----------
    g : (H, W) uint8 array
        Binary mask.
    y, x : int
        Pixel coordinates.

    Returns
    -------
    d : int
        Number of 4-neighbor foreground neighbors.
    """
    H, W = g.shape
    d = 0
    if y > 0   and g[y - 1, x] == 1:
        d += 1
    if y < H-1 and g[y + 1, x] == 1:
        d += 1
    if x > 0   and g[y, x - 1] == 1:
        d += 1
    if x < W-1 and g[y, x + 1] == 1:
        d += 1
    return d


@njit(cache=True)
def _exposure_njit(g: np.ndarray, y: int, x: int) -> int:
    """
    Compute the 4-neighbor exposure of a foreground pixel.

    Exposure is the number of directions in which the neighbor is either
    background (0) or out of bounds. Larger exposure means the pixel is
    more "exposed" on the outer rim.

    Parameters
    ----------
    g : (H, W) uint8 array
        Binary mask.
    y, x : int
        Pixel coordinates.

    Returns
    -------
    e : int
        Exposure count in {0, 1, 2, 3, 4}.
    """
    H, W = g.shape
    e = 0
    if y == 0   or g[y - 1, x] == 0:
        e += 1
    if y == H-1 or g[y + 1, x] == 0:
        e += 1
    if x == 0   or g[y, x - 1] == 0:
        e += 1
    if x == W-1 or g[y, x + 1] == 0:
        e += 1
    return e


@njit(cache=True)
def _is_connected_after_remove_njit(
    g: np.ndarray,
    y: int,
    x: int,
    buf_seen: np.ndarray,
    qy: np.ndarray,
    qx: np.ndarray,
) -> bool:
    """
    Check whether the foreground in g remains 4-connected after removing
    a single foreground pixel at (y, x).

    This function temporarily sets g[y, x] = 0, performs a BFS over
    foreground pixels using preallocated buffers, and then restores g[y, x].
    It is intended to be called repeatedly inside a deletion loop, so
    allocations are done outside and passed in as buffers.

    Parameters
    ----------
    g : (H, W) uint8 array
        Binary mask. Modified temporarily but restored before return.
    y, x : int
        Candidate pixel to remove (must currently be 1).
    buf_seen : (H, W) uint8 array
        Work buffer for marking visited pixels.
    qy, qx : 1D arrays of length H*W
        Work buffers implementing the BFS queue.

    Returns
    -------
    bool
        True if the remaining foreground pixels form a single 4-connected
        component after removal; False otherwise.
    """
    H, W = g.shape
    if g[y, x] == 0:
        return False

    total = int(g.sum()) - 1
    if total <= 0:
        return False

    # Temporarily remove the pixel
    g[y, x] = 0

    # Find a starting foreground pixel
    sy = -1
    sx = -1
    for i in range(H):
        found = False
        for j in range(W):
            if g[i, j] == 1:
                sy = i
                sx = j
                found = True
                break
        if found:
            break

    if sy < 0:
        g[y, x] = 1
        return False

    # BFS using qy / qx arrays as the queue
    # Reset visited buffer
    for i in range(H * W):
        buf_seen.ravel()[i] = 0

    head = 0
    tail = 0
    qy[tail] = sy
    qx[tail] = sx
    tail += 1
    buf_seen[sy, sx] = 1
    cnt = 1

    while head < tail:
        cy = qy[head]
        cx = qx[head]
        head += 1

        # 4-neighbor traversal
        if cy + 1 < H and g[cy + 1, cx] == 1 and buf_seen[cy + 1, cx] == 0:
            buf_seen[cy + 1, cx] = 1
            qy[tail] = cy + 1
            qx[tail] = cx
            tail += 1
            cnt += 1
            if cnt == total:
                g[y, x] = 1
                return True

        if cy - 1 >= 0 and g[cy - 1, cx] == 1 and buf_seen[cy - 1, cx] == 0:
            buf_seen[cy - 1, cx] = 1
            qy[tail] = cy - 1
            qx[tail] = cx
            tail += 1
            cnt += 1
            if cnt == total:
                g[y, x] = 1
                return True

        if cx + 1 < W and g[cy, cx + 1] == 1 and buf_seen[cy, cx + 1] == 0:
            buf_seen[cy, cx + 1] = 1
            qy[tail] = cy
            qx[tail] = cx + 1
            tail += 1
            cnt += 1
            if cnt == total:
                g[y, x] = 1
                return True

        if cx - 1 >= 0 and g[cy, cx - 1] == 1 and buf_seen[cy, cx - 1] == 0:
            buf_seen[cy, cx - 1] = 1
            qy[tail] = cy
            qx[tail] = cx - 1
            tail += 1
            cnt += 1
            if cnt == total:
                g[y, x] = 1
                return True

    g[y, x] = 1
    return cnt == total

def _delete_boundary_pixels(
    g: np.ndarray,
    n_delete: int,
    alpha: float,
    beta: float,
    verbose: bool = False,
) -> int:
    """
    Delete boundary pixels from a binary aggregate while preserving
    global 4-connectivity.

    Pixels are removed using a prioritized scoring rule until `n_delete`
    pixels have been successfully deleted or no safe removable pixels remain.

    Deletion strategy
    -----------------
    1) Restrict candidates to boundary pixels (4-neighbor definition).
    2) Prefer "leaf" pixels (degree = 1), i.e., endpoints of thin branches.
    3) From the leaf set (or full boundary set if no leaves exist),
       keep only those pixels whose removal does NOT break connectivity.
       Connectivity is verified using `_is_connected_after_remove_njit`.
    4) Among the remaining safe candidates, choose the pixel with the
       minimal score:
           score = alpha * r^2 - beta * exposure
       where:
          - r^2 = squared distance to the current center of mass
          - exposure = # of 4-neighbors that are background or boundary
       Larger exposure → more "outer rim" like.
       Smaller score is preferred for deletion.
    5) Delete the chosen pixel and repeat.

    Parameters
    ----------
    g : (H, W) uint8 array
        Binary mask of the aggregate. Modified in-place.
    n_delete : int
        Number of pixels requested to be deleted.
    alpha : float
        Weight on r^2 (distance to centroid) in the score.
    beta : float
        Weight on exposure term in the score.
    verbose : bool
        If True, print brief diagnostic messages.

    Returns
    -------
    deleted : int
        Number of pixels actually deleted (may be < n_delete).
    """
    deleted = 0
    H, W = g.shape

    # Preallocated buffers for Numba connectivity checks
    buf_seen = np.zeros_like(g, dtype=np.uint8)
    qy = np.empty(H * W, dtype=np.int32)
    qx = np.empty(H * W, dtype=np.int32)

    while deleted < n_delete:
        ys, xs = np.nonzero(g)
        if ys.size == 0:
            break

        # Compute centroid of current foreground
        comx = float(xs.mean())
        comy = float(ys.mean())

        # Step 1: boundary pixels
        bps = _boundary_points_njit(g)
        if bps.shape[0] == 0:
            if verbose:
                print(f"[Delete] no boundary points at {deleted}/{n_delete}")
            break

        # Step 2: leaf-pixel filtering
        leaf_mask = np.zeros(bps.shape[0], dtype=np.uint8)
        for i in range(bps.shape[0]):
            y, x = int(bps[i, 0]), int(bps[i, 1])
            if _degree_njit(g, y, x) == 1:
                leaf_mask[i] = 1

        cand = bps[leaf_mask == 1] if leaf_mask.sum() > 0 else bps

        # Step 3: filter for connectivity safety
        safe = []
        for i in range(cand.shape[0]):
            y, x = int(cand[i, 0]), int(cand[i, 1])
            if _is_connected_after_remove_njit(g, y, x, buf_seen, qy, qx):
                safe.append((y, x))

        if not safe:
            if verbose:
                print(f"[Delete] stopped early at {deleted}/{n_delete} (no safe boundary)")
            break

        # Step 4: choose minimum-score candidate
        best_sc = None
        best_pt = None
        for (y, x) in safe:
            r2 = (x - comx) ** 2 + (y - comy) ** 2
            sc = alpha * r2 - beta * _exposure_njit(g, y, x)
            if (best_sc is None) or (sc < best_sc):
                best_sc = sc
                best_pt = (y, x)

        # Step 5: remove the chosen pixel
        yb, xb = best_pt
        g[yb, xb] = 0
        deleted += 1

    return deleted


# -----------------------------
# Visualization (grid version)
# -----------------------------

def plot_aggregate_grid(grid: np.ndarray, origin: Tuple[int, int]) -> None:
    """
    Visualize a binary aggregate grid as an occupancy map.

    Parameters
    ----------
    grid : (H, W) uint8 array
        Binary mask of the aggregate.
    origin : (x_min, y_min)
        Original coordinate offset returned by the lattice generator.
        Only used for labeling.
    """
    H, W = grid.shape
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.imshow(grid, origin='lower', interpolation='nearest', alpha=0.35)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_title("2D Lattice Aggregate")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Minor grid overlay (one tick per cell)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', linewidth=0.3)

    plt.tight_layout()
    plt.show()


def plot_rg_vs_n_2d(
    Ns: np.ndarray,
    Rgs: np.ndarray,
    Df_est: float,
    slope: float,
) -> None:
    """
    Plot Rg vs N on a log–log scale for the 2D lattice aggregate.

    Parameters
    ----------
    Ns : array
        Sequence of particle counts during growth.
    Rgs : array
        Corresponding radii of gyration.
    Df_est : float
        Estimated fractal dimension from log-log fit.
    slope : float
        Fitted slope of log(Rg) vs. log(N).
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.loglog(Ns, Rgs, marker='o', linestyle='-', linewidth=1, markersize=3)

    ax.set_xlabel("N (number of particles)")
    ax.set_ylabel("Rg (radius of gyration, 2D)")
    ax.set_title(
        f"[2D Lattice] Rg(N) log-log  |  slope≈{slope:.3f}  =>  Df_est≈{Df_est:.3f}"
    )

    plt.tight_layout()
    plt.show()
