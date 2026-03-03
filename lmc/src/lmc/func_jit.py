# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:16:11 2025

@author: px2030
"""

from __future__ import annotations
import numpy as np
from numba import njit

@njit(cache=True)
def float_gcd(a: float, b: float, rtol: float = 1e-3, atol: float = 1e-8) -> float:
    """
    Compute a floating-point version of the greatest common divisor (GCD).
    
    This function repeatedly applies the Euclidean algorithm (using the remainder)
    until the remainder is sufficiently small compared to the tolerance thresholds.
    
    Parameters
    ----------
    a : float
        First input value.
    b : float
        Second input value.
    rtol : float, optional
        Relative tolerance for convergence (default = 1e-3).
    atol : float, optional
        Absolute tolerance for convergence (default = 1e-8).
    
    Returns
    -------
    float
        Approximate GCD of a and b.
    """
    t = abs(a) if abs(a) < abs(b) else abs(b)
    aa, bb = a, b
    while abs(bb) > rtol * t + atol:
        tmp = bb
        bb = aa % bb
        aa = tmp
    return aa

@njit(cache=True)
def uf_find(parent: np.ndarray, x: np.int32) -> np.int32:
    # Path compression: follow parent pointers until reaching root
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit(cache=True)
def uf_union(parent: np.ndarray, rank: np.ndarray, a: np.int32, b: np.int32):
    # Union by rank: attach smaller tree under larger one
    ra = uf_find(parent, a); rb = uf_find(parent, b)
    if ra == rb: return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1
       
@njit(cache=True)
def uf_label_bool(b: np.ndarray) -> np.ndarray:
    """
    Connected-component labeling using Union–Find (Disjoint Set Union, DSU).
    
    Each "True" cell in the boolean grid `b` belongs to a component. 
    Components are identified by connecting adjacent cells (up and left neighbors).
    Union–Find data structure is used to efficiently group them.
    
    Parameters
    ----------
    b : np.ndarray of shape (H, W), dtype=bool or int
        Binary array where nonzero values represent "occupied" cells.
    
    Returns
    -------
    np.ndarray of shape (H, W), dtype=int
        Labeled array where each connected region of occupied cells
        gets a unique positive integer label (1,2,...,K).
        Empty cells are labeled as 0.
    """
    H, W = b.shape
    n = H * W
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)

    # Connect each cell with its top/left neighbors if both are occupied
    for y in range(H):
        for x in range(W):
            if b[y, x]:
                i = y * W + x
                if y > 0 and b[y-1, x]:
                    uf_union(parent, rank, i, (y-1) * W + x)
                if x > 0 and b[y, x-1]:
                    uf_union(parent, rank, i, y * W + (x-1))

    # Assign compact labels (1,2,3,...)
    lab = np.zeros(n, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    label = 0
    for i in range(n):
        y = i // W; x = i - y * W
        if b[y, x]:
            r = uf_find(parent, i)
            if lab[r] == 0:
                label += 1
                lab[r] = label
            out[i] = lab[r]

    return out.reshape(H, W)


@njit(cache=True)
def compress_count(cell_lab: np.ndarray, M: np.ndarray):
    """
    Compress connected-component labels into compact indices and count material types.
    
    `cell_lab` comes from `uf_label_bool` applied on an expanded grid 
    and then restricted to material cells only. Thus it has the same shape as `M`.
    
    Because Union-Find labels may be sparse (e.g. {1,5,9}), this function
    remaps them into a contiguous index range {0,...,K-1}.
    It also counts how many cells of material 1 and material 2 belong to each fragment.
    
    Parameters
    ----------
    cell_lab : np.ndarray of shape (H, W), dtype=int
        Connected-component labels (possibly sparse).
    M : np.ndarray of shape (H, W), dtype=int
        Material grid: 0 = empty, 1 = material type A, 2 = material type B.
    
    Returns
    -------
    labels : np.ndarray of shape (H, W), dtype=int
        Compact fragment labels: -1 for empty cells, 0..K-1 for fragment IDs.
    cnt1 : np.ndarray of shape (K,), dtype=int
        Number of cells of material 1 in each fragment.
    cnt2 : np.ndarray of shape (K,), dtype=int
        Number of cells of material 2 in each fragment.
    """
    H, W = M.shape
    Lmax = 0
    for y in range(H):
        for x in range(W):
            if M[y, x] > 0:
                v = cell_lab[y, x]
                if v > Lmax:
                    Lmax = v
    if Lmax <= 0:
        labels = -np.ones((H, W), dtype=np.int32)
        return labels, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    # Map sparse labels into compact indices
    mapping = -np.ones(Lmax + 1, dtype=np.int32)
    K = 0
    for y in range(H):
        for x in range(W):
            if M[y, x] > 0:
                v = cell_lab[y, x]
                if v > 0 and mapping[v] == -1:
                    mapping[v] = K
                    K += 1

    # Build final labels and count material distribution
    labels = -np.ones((H, W), dtype=np.int32)
    cnt1 = np.zeros(K, dtype=np.int32)
    cnt2 = np.zeros(K, dtype=np.int32)
    for y in range(H):
        for x in range(W):
            if M[y, x] > 0:
                v = cell_lab[y, x]
                k = mapping[v]
                labels[y, x] = k
                if M[y, x] == 1:
                    cnt1[k] += 1
                else:
                    cnt2[k] += 1
    return labels, cnt1, cnt2


@njit(cache=True)
def to_old_G_layout_jit(M: np.ndarray, Hbond: np.ndarray, Vbond: np.ndarray) -> np.ndarray:
    """
    Convert the compact representation (M, Hbond, Vbond) into the old-style "big grid" layout G.
    
    The big grid G is a (2H+1) x (2W+1) array with the following convention:
      - G[even, even] = 0 : junction points (grid corners).
      - G[odd, odd]  = material cells (1,2 or -1 for empty).
      - G[odd, even] = horizontal bonds (11,12,22 or -1 if broken/missing).
      - G[even, odd] = vertical bonds (11,12,22 or -1 if broken/missing).
      - Any other entries default to -1 (unused).
    
    This format is compatible with older visualization/analysis functions
    that expect a full "expanded" grid with both materials and bonds.
    
    Parameters
    ----------
    M : ndarray (H, W), dtype=uint8
        Material grid: 0=empty, 1=material A, 2=material B.
    Hbond : ndarray (H, W-1), dtype=int16
        Horizontal bonds between neighboring cells.
    Vbond : ndarray (H-1, W), dtype=int16
        Vertical bonds between neighboring cells.
    
    Returns
    -------
    G : ndarray (2H+1, 2W+1), dtype=int16
        Expanded grid representation.
    """
    H, W = M.shape
    G = np.empty((2*H+1, 2*W+1), dtype=np.int16)
    for y in range(2*H+1):
        for x in range(2*W+1):
            G[y, x] = -1
    for y in range(0, 2*H+1, 2):
        for x in range(0, 2*W+1, 2):
            G[y, x] = 0
    for y in range(H):
        for x in range(W):
            v = M[y, x]
            G[2*y+1, 2*x+1] = -1 if v == 0 else v
    if W >= 2:
        for y in range(H):
            for x in range(W-1):
                G[2*y+1, 2*x+2] = Hbond[y, x]
    if H >= 2:
        for y in range(H-1):
            for x in range(W):
                G[2*y+2, 2*x+1] = Vbond[y, x]
    return G


@njit(cache=True)
def build_big_grid_mask(M: np.ndarray,
                        Hbond: np.ndarray,
                        Vbond: np.ndarray) -> np.ndarray:
    """
    Build a binary "mask grid" representing materials and intact bonds.
    
    This mask is used for connected-component labeling (fragment detection).
    Its resolution is (2H-1, 2W-1), so that:
      - Positions [0::2, 0::2] correspond to material cells.
      - Positions [0::2, 1::2] correspond to horizontal bonds.
      - Positions [1::2, 0::2] correspond to vertical bonds.
    
    Cells and bonds are marked as 1 if present/intact, 0 if empty/broken.
    
    Parameters
    ----------
    M : ndarray (H, W), dtype=uint8
        Material grid: 0=empty, 1=material A, 2=material B.
    Hbond : ndarray (H, W-1), dtype=int16
        Horizontal bonds (>=0 if intact, -1 if broken).
    Vbond : ndarray (H-1, W), dtype=int16
        Vertical bonds (>=0 if intact, -1 if broken).
    
    Returns
    -------
    R : ndarray (2H-1, 2W-1), dtype=uint8
        Binary mask grid (1 = occupied cell or intact bond, 0 = empty).
    """
    H, W = M.shape
    RH = 2*H - 1 if H > 0 else 1
    RW = 2*W - 1 if W > 0 else 1
    R = np.zeros((RH, RW), dtype=np.uint8)
    for y in range(H):
        by = 2*y
        for x in range(W):
            if M[y, x] > 0:
                R[by, 2*x] = 1
    if W >= 2:
        for y in range(H):
            by = 2*y
            for x in range(W-1):
                if Hbond[y, x] != -1:
                    R[by, 2*x+1] = 1
    if H >= 2:
        for y in range(H-1):
            by = 2*y + 1
            for x in range(W):
                if Vbond[y, x] != -1:
                    R[by, 2*x] = 1
    return R


@njit(cache=True)
def _choose_index(weights: np.ndarray, K: int, u: float) -> int:
    """
    Select an index from [0, K-1] according to given weights and a random number u.
    
    This is essentially weighted random sampling (roulette-wheel selection):
      - If all weights are <= 0, fall back to uniform random choice.
      - Otherwise, probabilities are proportional to weights.
      - u is assumed to be a uniform random number in [0, 1).
    
    Parameters
    ----------
    weights : ndarray (K,), dtype=float
        Nonnegative weights for each candidate.
    K : int
        Number of candidates.
    u : float
        Uniform random number in [0, 1).
    
    Returns
    -------
    int
        Selected index in [0, K-1].
    """
    s = 0.0
    for k in range(K):
        s += weights[k]
    if s <= 0.0:
        # Fallback: uniform choice
        t = u * K
        idx = 0; acc = 1.0
        while idx < K-1 and t >= acc:
            idx += 1; acc += 1.0
        return idx
    # Roulette-wheel selection
    t = u * s; acc = 0.0
    for k in range(K):
        acc += weights[k]
        if t <= acc:
            return k
    return K-1


@njit(cache=True)
def run_one_fracture_kernel(Hbond, Vbond, H, W, str11, str12, str22, gamma, int_bre_len,
                            allow_loops, r0, c0, max_path,
                            urand):
    # Preallocate path recording buffers (upper bound length = max_path).
    # Each broken bond step will append one entry to these arrays.
    axis_arr = np.empty(max_path, np.int16)   # 0=Hbond, 1=Vbond for each broken step
    i_arr    = np.empty(max_path, np.int32)   # row index in the chosen bond grid
    j_arr    = np.empty(max_path, np.int32)   # col index in the chosen bond grid
    old_arr  = np.empty(max_path, np.int16)   # original bond type (11/12/22) before breaking

    used   = 0          # how many bonds are actually broken and recorded
    energy = 0.0        # accumulated fracture energy (sum of strengths of broken bonds)

    # Crack tip starts at junction (r0, c0)
    r = r0
    c = c0
    prev_dir = -1       # last chosen direction (0 U, 1 D, 2 L, 3 R); -1 means "none yet"

    # Visited junctions by THIS crack (to prevent self-loops if allow_loops==0)
    visited = np.zeros((H+1, W+1), np.uint8)
    if allow_loops == 0:
        visited[r, c] = 1  # mark the start node as visited

    ui = 0  # cursor into urand[] (pre-generated uniforms)

    # Return True if at junction (rr,cc) there exists some *other* incident edge
    # (not the opposite to the last step) that is already broken (-1), meaning
    # we should stop because we hit/merged into an existing crack network.
    def other_broken(rr, cc, ignore_dir):
        if rr > 0 and 1 <= cc <= W-1:
            if ignore_dir != 0 and Hbond[rr-1, cc-1] == -1: return True  # Up edge broken
        if rr < H and 1 <= cc <= W-1:
            if ignore_dir != 1 and Hbond[rr, cc-1] == -1: return True    # Down edge broken
        if cc > 0 and 1 <= rr <= H-1:
            if ignore_dir != 2 and Vbond[rr-1, cc-1] == -1: return True  # Left edge broken
        if cc < W and 1 <= rr <= H-1:
            if ignore_dir != 3 and Vbond[rr-1, cc] == -1: return True    # Right edge broken
        return False

    # Enumerate all intact bonds incident to junction (rr,cc) and write them into
    # the provided small arrays:
    #   types[k]    ∈ {11,12,22}
    #   loc_axis[k] ∈ {0,1}   (0=Hbond grid, 1=Vbond grid)
    #   loc_i[k], loc_j[k]    index within that grid
    #   dirs[k]     ∈ {0,1,2,3}  (0=Up, 1=Down, 2=Left, 3=Right)
    # Return K = #candidates.
    def gather(rr, cc, types, loc_axis, loc_i, loc_j, dirs):
        k = 0
        # Up: Hbond[rr-1, cc-1]
        if rr > 0 and 1 <= cc <= W-1:
            t = Hbond[rr-1, cc-1]
            if t != -1:
                types[k] = t; loc_axis[k] = 0; loc_i[k] = rr-1; loc_j[k] = cc-1; dirs[k] = 0; k += 1
        # Down: Hbond[rr, cc-1]
        if rr < H and 1 <= cc <= W-1:
            t = Hbond[rr, cc-1]
            if t != -1:
                types[k] = t; loc_axis[k] = 0; loc_i[k] = rr;   loc_j[k] = cc-1; dirs[k] = 1; k += 1
        # Left: Vbond[rr-1, cc-1]
        if cc > 0 and 1 <= rr <= H-1:
            t = Vbond[rr-1, cc-1]
            if t != -1:
                types[k] = t; loc_axis[k] = 1; loc_i[k] = rr-1; loc_j[k] = cc-1; dirs[k] = 2; k += 1
        # Right: Vbond[rr-1, cc]
        if cc < W and 1 <= rr <= H-1:
            t = Vbond[rr-1, cc]
            if t != -1:
                types[k] = t; loc_axis[k] = 1; loc_i[k] = rr-1; loc_j[k] = cc;   dirs[k] = 3; k += 1
        return k

    # Small fixed-size buffers for at most 4 incident edges at a junction
    types    = np.empty(4, np.int16)
    loc_axis = np.empty(4, np.int16)
    loc_i    = np.empty(4, np.int32)
    loc_j    = np.empty(4, np.int32)
    dirs     = np.empty(4, np.int16)
    weights  = np.empty(4, np.float64)

    # Collect initial outgoing intact bonds at the start junction
    K = gather(r, c, types, loc_axis, loc_i, loc_j, dirs)
    if K == 0:
        # No way to go: return immediately with "complete=1" and zero steps
        return r, c, 0, energy, 0, axis_arr, i_arr, j_arr, old_arr

    # Build sampling weights for the *first* choice:
    # - if random_first==1: uniform over the K candidates (w=1.0)
    # - else: inverse of bond strength (weaker bond => larger probability).
    for k in range(K):
        t = types[k]
        w = (1.0 / max(str11,1e-12) if t==11
                   else (1.0 / max(str12,1e-12) if t==12
                         else (1.0 / max(str22,1e-12) if t==22 else 0.0)))
        weights[k] = w

    # Sample one candidate direction index using a pre-generated uniform urand[ui]
    ksel = _choose_index(weights, K, urand[ui] if ui < urand.size else 0.5)
    ui += 1
    prev_dir = int(dirs[ksel])  # remember the chosen direction for directional bias ahead

    # ---- Straight-ahead initial phase (go straight for up to int_bre_len steps) ----
    for _ in range(int_bre_len):
        # Re-collect intact incident bonds at the current junction
        K = gather(r, c, types, loc_axis, loc_i, loc_j, dirs)
    
        # Pick the candidate that keeps the same direction as 'prev_dir'
        take = -1
        for kk in range(K):
            if dirs[kk] == prev_dir:
                take = kk
                break
        # If no edge continues straight, exit the straight-run phase
        if take == -1:
            break
    
        # Loop-prevention: forbid revisiting the next junction of THIS crack
        if allow_loops == 0:
            dr = -1 if prev_dir == 0 else (1 if prev_dir == 1 else 0)    # row delta
            dc = -1 if prev_dir == 2 else (1 if prev_dir == 3 else 0)    # col delta
            r_next = r + dr
            c_next = c + dc
            if visited[r_next, c_next] == 1:
                # continuing straight would self-loop -> leave straight phase,
                # and try general selection among other directions
                break
    
        # Break the selected bond in-place and record it
        ax = int(loc_axis[take]); ii = int(loc_i[take]); jj = int(loc_j[take])
        if ax == 0:
            old = Hbond[ii, jj]; Hbond[ii, jj] = -1   # break a horizontal bond
        else:
            old = Vbond[ii, jj]; Vbond[ii, jj] = -1   # break a vertical bond
        axis_arr[used] = ax; i_arr[used] = ii; j_arr[used] = jj; old_arr[used] = old; used += 1
    
        # Add energy by the type of the broken bond
        t = types[take]
        energy += (str11 if t==11 else (str12 if t==12 else (str22 if t==22 else 0.0)))
    
        # Advance crack tip to the next junction in the same direction
        dr = -1 if prev_dir == 0 else (1 if prev_dir == 1 else 0)
        dc = -1 if prev_dir == 2 else (1 if prev_dir == 3 else 0)
        r += dr; c += dc
    
        # Mark newly reached junction as visited (for self-loop prevention)
        if allow_loops == 0:
            visited[r, c] = 1
    
        # If required to stop at boundary (external start), then stop once boundary is hit
        if (r == 0 or r == H or c == 0 or c == W):
            return r, c, 1, energy, used, axis_arr, i_arr, j_arr, old_arr
    
        # If another incident bond at this junction is already broken (not the opposite edge),
        # we have met/merged an existing crack; stop here.
        opp = 1 if prev_dir == 0 else (0 if prev_dir == 1 else (3 if prev_dir == 2 else 2))
        if other_broken(r, c, opp):
            return r, c, 1, energy, used, axis_arr, i_arr, j_arr, old_arr
    
    
    # ---- General growth phase (biased random walk with gamma) ----
    complete = 0
    while True:
        # Gather all intact exits at the current junction
        K = gather(r, c, types, loc_axis, loc_i, loc_j, dirs)
    
        # If dead-end or a single dangling edge, treat as completion
        if K <= 1:
            complete = 0
            break
    
        # Build weights: inverse strength; multiply by gamma if continuing prev_dir
        # Skip any move that would revisit a 'visited' node
        wsum = 0.0
        for k in range(K):
            d = int(dirs[k])
            dr = -1 if d == 0 else (1 if d == 1 else 0)
            dc = -1 if d == 2 else (1 if d == 3 else 0)
            r_next = r + dr; c_next = c + dc
    
            if allow_loops == 0 and visited[r_next, c_next] == 1:
                weights[k] = 0.0
                continue
    
            t = types[k]
            w = (1.0 / max(str11,1e-12) if t==11 else
                 (1.0 / max(str12,1e-12) if t==12 else
                  (1.0 / max(str22,1e-12) if t==22 else 0.0)))
            if dirs[k] == prev_dir:
                w *= gamma
            weights[k] = w
            wsum += w
            
        if wsum <= 0.0:
            # every direction would self-loop (or no valid exits) -> stop
            complete = 0
            break
        # Sample a direction index with pre-generated uniform random
        ksel = _choose_index(weights, K, urand[ui] if ui < urand.size else 0.5)
        ui += 1
    
        # Break the chosen bond in-place and record the step
        ax = int(loc_axis[ksel]); ii = int(loc_i[ksel]); jj = int(loc_j[ksel])
        if ax == 0:
            old = Hbond[ii, jj]; Hbond[ii, jj] = -1
        else:
            old = Vbond[ii, jj]; Vbond[ii, jj] = -1
        axis_arr[used] = ax; i_arr[used] = ii; j_arr[used] = jj; old_arr[used] = old; used += 1
    
        # Energy increment by broken bond type
        t = types[ksel]
        energy += (str11 if t==11 else (str12 if t==12 else (str22 if t==22 else 0.0)))
    
        # Update last direction, advance crack tip to the next junction
        prev_dir = int(dirs[ksel])
        dr = -1 if prev_dir == 0 else (1 if prev_dir == 1 else 0)
        dc = -1 if prev_dir == 2 else (1 if prev_dir == 3 else 0)
        r += dr; c += dc
    
        # Mark junction as visited to prevent self-loops (if enabled)
        if allow_loops == 0:
            visited[r, c] = 1
    
        # Stop if we reached the outer boundary (always stop in general phase)
        if (r == 0 or r == H or c == 0 or c == W):
            complete = 1
            break
    
        # Stop if this junction has another already-broken incident edge
        # (i.e., we merged into an existing crack network)
        opp = 1 if prev_dir == 0 else (0 if prev_dir == 1 else (3 if prev_dir == 2 else 2))
        if other_broken(r, c, opp):
            complete = 1
            break
    
    # Return end junction, completion flag, energy, written length, and recorded path arrays
    return r, c, complete, energy, used, axis_arr, i_arr, j_arr, old_arr
