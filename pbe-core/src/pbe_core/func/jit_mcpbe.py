import math
import numpy as np
from numba import njit, prange

# external JIT kernels (must be available & nopython-compatible)
from pbe_core.func.jit_kernel_agg import calc_beta as _kb_beta
from pbe_core.func.jit_kernel_break import breakage_func_1d as _kb_break1d, breakage_func_2d as _kb_break2d


# -----------------------------
# Agglomeration kernels
# -----------------------------
@njit(parallel=True, fastmath=True)
def nb_rebuild_ragg(COLEVAL: int, CORR_BETA: float, G: float, R: np.ndarray) -> np.ndarray:
    """Parallel r_i = sum_j beta(i,j). R is radii array (X/2) of length a."""
    a = R.shape[0]
    r = np.zeros(a, dtype=np.float64)
    for i in prange(a):
        s = 0.0
        for j in range(a):
            if j != i:
                s += _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
        r[i] = s
    return r


@njit(parallel=True, fastmath=True)
def nb_rebuild_ragg_weighted(
    COLEVAL: int,
    CORR_BETA: float,
    G: float,
    R: np.ndarray,
    W: np.ndarray,
    DELTA: np.ndarray,
) -> np.ndarray:
    """Parallel weighted r_i for WMCPBE, including self-agglomeration.

    For j != i, the contribution is W_i * W_j * beta(i,j).
    For j == i, one selected batch of size delta_i can collide with the
    remaining represented mass in the same compute particle only when
    W_i > 2 * delta_i. In that case the self contribution is
    W_i * (W_i - delta_i) * beta(i,i).
    """
    a = R.shape[0]
    r = np.zeros(a, dtype=np.float64)
    for i in prange(a):
        Wi = W[i]
        if Wi <= 0.0:
            continue
        s = 0.0
        for j in range(a):
            if j == i:
                continue
            Wj = W[j]
            if Wj <= 0.0:
                continue
            s += Wj * _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
        delta_i = DELTA[i]
        if delta_i > 0.0 and Wi > 2.0 * delta_i:
            s += (Wi - delta_i) * _kb_beta(COLEVAL, CORR_BETA, G, R, i, i)
        val = Wi * s
        r[i] = val if val > 0.0 else 0.0
    return r


@njit(parallel=True, fastmath=True)
def nb_rebuild_ragg_weighted_pair_delta(
    COLEVAL: int,
    CORR_BETA: float,
    G: float,
    R: np.ndarray,
    W: np.ndarray,
    DELTA: np.ndarray,
) -> np.ndarray:
    """Parallel corrected weighted r_i for WMCPBE agglomeration.

    Each pair contribution is scaled by the effective pair batch size
    delta_ij = min(delta_i, delta_j). Because delta_i is already capped by
    dW_const, this is equivalent to min(delta_i, delta_j, dW_const).
    """
    a = R.shape[0]
    r = np.zeros(a, dtype=np.float64)

    for i in prange(a):
        Wi = W[i]
        if Wi <= 0.0:
            continue
        delta_i = DELTA[i]
        if delta_i <= 0.0:
            continue

        s = 0.0
        for j in range(a):
            if j == i:
                continue
            Wj = W[j]
            if Wj <= 0.0:
                continue
            delta_j = DELTA[j]
            if delta_j <= 0.0:
                continue

            bij = _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
            if bij <= 0.0:
                continue
            pair_delta = delta_i
            if delta_j < pair_delta:
                pair_delta = delta_j
            s += Wj * bij / pair_delta

        if Wi > 2.0 * delta_i:
            bij_self = _kb_beta(COLEVAL, CORR_BETA, G, R, i, i)
            if bij_self > 0.0:
                s += (Wi - delta_i) * bij_self / delta_i

        val = Wi * s
        r[i] = val if val > 0.0 else 0.0
    return r

@njit(fastmath=True)
def nb_pick_partner(
    i: int,
    COLEVAL: int,
    CORR_BETA: float,
    G: float,
    R: np.ndarray,
    V0: np.ndarray,
    V1: np.ndarray,
    dim: int,
    alpha1d: float,
    alpha4: np.ndarray,  # length 4 when dim==2, ignored otherwise
    SIZEEVAL: int,
    X_SEL: float,
    Y_SEL: float,
    Vmean2: float,
    u_sel: float,
    u_acc: float,
) -> int:
    """Build β(i,·), sample j, compute alpha, accept/reject. Return j or -1."""
    a = R.shape[0]
    n = a - 1
    if n <= 0:
        return -1

    betas = np.empty(n, dtype=np.float64)
    js = np.empty(n, dtype=np.int64)

    kk = 0
    for j in range(a):
        if j == i:
            continue
        js[kk] = j
        betas[kk] = _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
        kk += 1

    # sample j
    tot = 0.0
    for t in range(n):
        tot += betas[t]
    if tot <= 0.0:
        sel = int(u_sel * n)
        if sel >= n:
            sel = n - 1
        j = js[sel]
    else:
        thresh = u_sel * tot
        acc = 0.0
        j = js[n - 1]
        for t in range(n):
            acc += betas[t]
            if acc > thresh:
                j = js[t]
                break

    # compute alpha
    if dim == 1:
        alpha = alpha1d
        Vi = V0[i]
        Vj = V0[j]
    else:
        Vi0 = V0[i]
        Vi1 = V1[i]
        Vti = Vi0 + Vi1
        Vj0 = V0[j]
        Vj1 = V1[j]
        Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            alpha = 0.0
        else:
            p0 = (Vi0 / Vti) * (Vj0 / Vtj)
            p1 = (Vi0 / Vti) * (Vj1 / Vtj)
            p2 = (Vi1 / Vti) * (Vj0 / Vtj)
            p3 = (Vi1 / Vti) * (Vj1 / Vtj)
            alpha = p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3]
        Vi = Vti
        Vj = Vtj

    if SIZEEVAL == 2:
        Xi = 2.0 * R[i]
        Xj = 2.0 * R[j]
        lam = Xi / Xj if Xi < Xj else Xj / Xi
        if Vmean2 > 0.0 and Vi > 0.0 and Vj > 0.0:
            alpha_corr = math.exp(-(X_SEL) * (1.0 - lam) * (1.0 - lam)) / (((Vi * Vj) / Vmean2) ** (Y_SEL))
            alpha *= alpha_corr

    if alpha < 0.0:
        alpha = 0.0
    elif alpha > 1.0:
        alpha = 1.0

    if u_acc >= alpha:
        return -1
    return j


@njit(fastmath=True)
def nb_pick_partner_weighted(
    i: int,
    COLEVAL: int,
    CORR_BETA: float,
    G: float,
    R: np.ndarray,
    W: np.ndarray,
    DELTA: np.ndarray,
    V0: np.ndarray,
    V1: np.ndarray,
    dim: int,
    alpha1d: float,
    alpha4: np.ndarray,  # length 4 when dim==2, ignored otherwise
    SIZEEVAL: int,
    X_SEL: float,
    Y_SEL: float,
    Vmean2: float,
    u_sel: float,
    u_acc: float,
):
    """Weighted partner sampling for WMCPBE with self-agglomeration.

    For j != i, the sampling weight is W_j * beta(i,j).
    For j == i, one batch of size delta_i is already selected by choosing i,
    so self-agglomeration is only possible when W_i > 2 * delta_i and the
    partner weight becomes (W_i - delta_i) * beta(i,i).

    Returns (j, partner_weight_selected), or (-1, 0.0) on reject/failure.
    """
    a = R.shape[0]
    nmax = a
    if nmax <= 0:
        return -1, 0.0

    weights = np.empty(nmax, dtype=np.float64)
    js = np.empty(nmax, dtype=np.int64)
    kk = 0
    tot = 0.0
    for j in range(a):
        if j == i:
            Wi = W[i]
            delta_i = DELTA[i]
            if delta_i <= 0.0 or Wi <= 2.0 * delta_i:
                continue
            bij = _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
            if bij <= 0.0:
                continue
            wij = (Wi - delta_i) * bij
            if wij <= 0.0:
                continue
            js[kk] = j
            weights[kk] = wij
            tot += wij
            kk += 1
            continue
        Wj = W[j]
        if Wj <= 0.0:
            continue
        bij = _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
        if bij <= 0.0:
            continue
        wij = Wj * bij
        if wij <= 0.0:
            continue
        js[kk] = j
        weights[kk] = wij
        tot += wij
        kk += 1

    if kk <= 0 or tot <= 0.0:
        return -1, 0.0

    # sample j by weighted CDF
    thresh = u_sel * tot
    acc = 0.0
    sel = kk - 1
    for t in range(kk):
        acc += weights[t]
        if acc > thresh:
            sel = t
            break
    j = js[sel]
    wjbeta = weights[sel]

    # compute alpha
    if dim == 1:
        alpha = alpha1d
        Vi = V0[i]
        Vj = V0[j]
    else:
        Vi0 = V0[i]
        Vi1 = V1[i]
        Vti = Vi0 + Vi1
        Vj0 = V0[j]
        Vj1 = V1[j]
        Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            alpha = 0.0
        else:
            p0 = (Vi0 / Vti) * (Vj0 / Vtj)
            p1 = (Vi0 / Vti) * (Vj1 / Vtj)
            p2 = (Vi1 / Vti) * (Vj0 / Vtj)
            p3 = (Vi1 / Vti) * (Vj1 / Vtj)
            alpha = p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3]
        Vi = Vti
        Vj = Vtj

    if SIZEEVAL == 2:
        Xi = 2.0 * R[i]
        Xj = 2.0 * R[j]
        lam = Xi / Xj if Xi < Xj else Xj / Xi
        if Vmean2 > 0.0 and Vi > 0.0 and Vj > 0.0:
            alpha_corr = math.exp(-(X_SEL) * (1.0 - lam) * (1.0 - lam)) / (((Vi * Vj) / Vmean2) ** (Y_SEL))
            alpha *= alpha_corr

    if alpha < 0.0:
        alpha = 0.0
    elif alpha > 1.0:
        alpha = 1.0

    if u_acc >= alpha:
        return -1, 0.0
    return j, wjbeta


@njit(fastmath=True)
def nb_pick_partner_weighted_pair_delta(
    i: int,
    COLEVAL: int,
    CORR_BETA: float,
    G: float,
    R: np.ndarray,
    W: np.ndarray,
    DELTA: np.ndarray,
    PARTNER_TOTAL: float,
    V0: np.ndarray,
    V1: np.ndarray,
    dim: int,
    alpha1d: float,
    alpha4: np.ndarray,  # length 4 when dim==2, ignored otherwise
    SIZEEVAL: int,
    X_SEL: float,
    Y_SEL: float,
    Vmean2: float,
    u_sel: float,
    u_acc: float,
):
    """Pair-delta corrected partner sampling for WMCPBE agglomeration.

    For j != i, the sampling weight is W_j * beta(i,j) / delta_ij.
    For j == i, the sampling weight is (W_i - delta_ii) * beta(i,i) / delta_ii,
    where delta_ij = min(delta_i, delta_j). PARTNER_TOTAL is r_i / W_i.

    Returns (j, corrected_partner_weight_selected), or (-1, 0.0).
    """
    a = R.shape[0]
    if a <= 0 or i < 0 or i >= a:
        return -1, 0.0

    partner_total = PARTNER_TOTAL
    if partner_total <= 0.0:
        return -1, 0.0
    delta_i = DELTA[i]
    if delta_i <= 0.0:
        return -1, 0.0

    thresh = u_sel * partner_total
    acc = 0.0
    selected_j = -1
    selected_w = 0.0
    last_j = -1
    last_w = 0.0
    for j in range(a):
        if j == i:
            Wi = W[i]
            if Wi <= 2.0 * delta_i:
                continue
            bij = _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
            if bij <= 0.0:
                continue
            wij = (Wi - delta_i) * bij / delta_i
            if wij <= 0.0:
                continue
            acc += wij
            last_j = j
            last_w = wij
            if acc > thresh:
                selected_j = j
                selected_w = wij
                break
            continue

        Wj = W[j]
        if Wj <= 0.0:
            continue
        delta_j = DELTA[j]
        if delta_j <= 0.0:
            continue

        bij = _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
        if bij <= 0.0:
            continue
        pair_delta = delta_i
        if delta_j < pair_delta:
            pair_delta = delta_j
        wij = Wj * bij / pair_delta
        if wij <= 0.0:
            continue
        acc += wij
        last_j = j
        last_w = wij
        if acc > thresh:
            selected_j = j
            selected_w = wij
            break

    if selected_j < 0:
        selected_j = last_j
        selected_w = last_w
    if selected_j < 0 or selected_w <= 0.0:
        return -1, 0.0
    j = selected_j
    wjbeta_over_delta = selected_w

    if dim == 1:
        alpha = alpha1d
        Vi = V0[i]
        Vj = V0[j]
    else:
        Vi0 = V0[i]
        Vi1 = V1[i]
        Vti = Vi0 + Vi1
        Vj0 = V0[j]
        Vj1 = V1[j]
        Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            alpha = 0.0
        else:
            p0 = (Vi0 / Vti) * (Vj0 / Vtj)
            p1 = (Vi0 / Vti) * (Vj1 / Vtj)
            p2 = (Vi1 / Vti) * (Vj0 / Vtj)
            p3 = (Vi1 / Vti) * (Vj1 / Vtj)
            alpha = p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3]
        Vi = Vti
        Vj = Vtj

    if SIZEEVAL == 2:
        Xi = 2.0 * R[i]
        Xj = 2.0 * R[j]
        lam = Xi / Xj if Xi < Xj else Xj / Xi
        if Vmean2 > 0.0 and Vi > 0.0 and Vj > 0.0:
            alpha_corr = math.exp(-(X_SEL) * (1.0 - lam) * (1.0 - lam)) / (((Vi * Vj) / Vmean2) ** (Y_SEL))
            alpha *= alpha_corr

    if alpha < 0.0:
        alpha = 0.0
    elif alpha > 1.0:
        alpha = 1.0

    if u_acc >= alpha:
        return -1, 0.0
    return j, wjbeta_over_delta


# -----------------------------
# Fenwick (BIT) kernels
# -----------------------------
@njit(fastmath=True)
def nb_fenwick_add(tree: np.ndarray, n: int, idx: int, delta: float):
    i = idx + 1
    while i <= n:
        tree[i] += delta
        i += i & -i


@njit(fastmath=True)
def nb_fenwick_build(tree: np.ndarray, n: int, w: np.ndarray):
    for i in range(n):
        nb_fenwick_add(tree, n, i, w[i])


@njit(fastmath=True)
def nb_fenwick_update(tree: np.ndarray, n: int, w: np.ndarray, idx: int, new_w: float) -> float:
    delta = new_w - w[idx]
    if delta != 0.0:
        w[idx] = new_w
        nb_fenwick_add(tree, n, idx, delta)
    return delta


@njit(fastmath=True)
def nb_fenwick_prefix_sum_1based(tree: np.ndarray, i1: int) -> float:
    """Fenwick prefix sum on 1-based index: sum[1..i1]."""
    s = 0.0
    i = i1
    while i > 0:
        s += tree[i]
        i -= i & -i
    return s


@njit(fastmath=True)
def nb_fenwick_append_arrays(tree: np.ndarray, w: np.ndarray, n: int, new_w: float):
    """Create resized Fenwick arrays for append and set the new tail node locally."""
    new_n = n + 1

    new_tree = np.zeros(new_n + 1, dtype=np.float64)
    for i in range(n + 1):
        new_tree[i] = tree[i]

    new_weights = np.empty(new_n, dtype=np.float64)
    for i in range(n):
        new_weights[i] = w[i]
    new_weights[n] = new_w

    # Fenwick node value at index new_n (1-based)
    lowbit = new_n & -new_n
    l = new_n - lowbit + 1
    old_sum_right = nb_fenwick_prefix_sum_1based(tree, n)
    old_sum_left = nb_fenwick_prefix_sum_1based(tree, l - 1)
    new_tree[new_n] = (old_sum_right - old_sum_left) + new_w

    return new_tree, new_weights


@njit(fastmath=True)
def nb_fenwick_remove_swap_last(tree: np.ndarray, w: np.ndarray, n: int, idx: int) -> float:
    """Remove with swap-last semantics in-place on old-size arrays; returns removed last weight."""
    last = n - 1

    if idx != last:
        old_idx_w = w[idx]
        moved = w[last]
        delta = moved - old_idx_w
        if delta != 0.0:
            w[idx] = moved
            nb_fenwick_add(tree, n, idx, delta)

    removed = w[last]
    if removed != 0.0:
        nb_fenwick_add(tree, n, last, -removed)
    return removed


# -----------------------------
# Breakage CDF builders (two-level sampling)
# -----------------------------
@njit(fastmath=True)
def _build_table_1d_jit(rel: np.ndarray, v: float, q: float, bf: int):
    n = rel.shape[0]
    pdf = np.empty(n, dtype=np.float64)
    for k in range(n):
        x = rel[k]
        pdf[k] = _kb_break1d(x, 1.0, v, q, bf)
    s = 0.0
    for k in range(n):
        s += pdf[k]
    cdf = np.empty(n, dtype=np.float64)
    if s <= 0.0:
        inv = 1.0 / n
        acc = 0.0
        for k in range(n):
            acc += inv
            cdf[k] = acc
    else:
        invs = 1.0 / s
        acc = 0.0
        for k in range(n):
            acc += pdf[k] * invs
            cdf[k] = acc
    return cdf  # monotonically non-decreasing in [0,1]


@njit(fastmath=True)
def _build_tables_2d_jit(rel1: np.ndarray, rel3: np.ndarray, v: float, q: float, bf: int):
    n1 = rel1.shape[0]
    n3 = rel3.shape[0]
    rowsum = np.empty(n1, dtype=np.float64)
    # first pass: row sums
    for i in range(n1):
        s = 0.0
        x1 = rel1[i]
        for j in range(n3):
            x3 = rel3[j]
            s += _kb_break2d(x3, x1, 1.0, 1.0, v, q, bf)
        rowsum[i] = s
    # total sum
    total = 0.0
    for i in range(n1):
        total += rowsum[i]
    # rowsum CDF
    rowsum_cdf = np.empty(n1, dtype=np.float64)
    if total <= 0.0:
        inv = 1.0 / n1
        acc = 0.0
        for i in range(n1):
            acc += inv
            rowsum_cdf[i] = acc
    else:
        invt = 1.0 / total
        acc = 0.0
        for i in range(n1):
            acc += rowsum[i] * invt
            rowsum_cdf[i] = acc
    # per-row cdf
    row_cdf = np.empty((n1, n3), dtype=np.float64)
    for i in range(n1):
        s = rowsum[i]
        acc = 0.0
        if s <= 0.0:
            inv = 1.0 / n3
            for j in range(n3):
                acc += inv
                row_cdf[i, j] = acc
        else:
            invs = 1.0 / s
            x1 = rel1[i]
            for j in range(n3):
                x3 = rel3[j]
                acc += _kb_break2d(x3, x1, 1.0, 1.0, v, q, bf) * invs
                row_cdf[i, j] = acc
    return rowsum_cdf, row_cdf
