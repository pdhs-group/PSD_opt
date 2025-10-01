import math
import numpy as np
from numba import njit, prange

# external JIT kernels (must be available & nopython-compatible)
from optframework.utils.func.jit_kernel_agg import calc_beta as _kb_beta
from optframework.utils.func.jit_kernel_break import breakage_func_1d as _kb_break1d, breakage_func_2d as _kb_break2d


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
