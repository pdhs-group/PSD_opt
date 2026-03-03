from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple, Sequence

import numpy as np
from numba import njit, prange

# External base class with physics parameters and config loader
from pbe_core.base.base_solver import BaseSolver

import pbe_core.func.jit_kernel_agg as JKA  # provides calc_beta, calc_F_M_*, ...
import pbe_core.func.jit_kernel_break as JKB  # provides breakage_func_*, calc_B_R_*
from pbe_core.func.jit_kernel_agg import calc_beta as _kb_beta
from pbe_core.func.jit_kernel_break import breakage_func_2d as _kb_break2d, breakage_func_1d as _kb_break1d

# -----------------------------
# Utilities
# -----------------------------
@njit(parallel=True)
def nb_rebuild_ragg(COLEVAL, CORR_BETA, G, R):
    a = R.shape[0]
    r = np.zeros(a, dtype=np.float64)
    for i in prange(a):
        s = 0.0
        for j in range(a):
            if j != i:
                s += _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
        r[i] = s
    return r

@njit
def nb_pick_partner(i, COLEVAL, CORR_BETA, G, R, V0, V1, dim, alpha1d, alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2, u_sel, u_acc):
    a = R.shape[0]
    n = a - 1
    if n <= 0:
        return -1
    betas = np.empty(n, dtype=np.float64)
    js = np.empty(n, dtype=np.int64)
    k = 0
    for j in range(a):
        if j == i:
            continue
        js[k] = j
        betas[k] = _kb_beta(COLEVAL, CORR_BETA, G, R, i, j)
        k += 1
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
    if dim == 1:
        alpha = alpha1d
        Vi = V0[i]; Vj = V0[j]
    else:
        Vi0 = V0[i]; Vi1 = V1[i]; Vti = Vi0 + Vi1
        Vj0 = V0[j]; Vj1 = V1[j]; Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            alpha = 0.0
        else:
            p0 = (Vi0 / Vti) * (Vj0 / Vtj)
            p1 = (Vi0 / Vti) * (Vj1 / Vtj)
            p2 = (Vi1 / Vti) * (Vj0 / Vtj)
            p3 = (Vi1 / Vti) * (Vj1 / Vtj)
            alpha = p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3]
        Vi = Vti; Vj = Vtj
    if SIZEEVAL == 2:
        Xi = 2.0 * R[i]; Xj = 2.0 * R[j]
        lam = Xi / Xj if Xi < Xj else Xj / Xi
        if Vmean2 > 0.0 and Vi > 0.0 and Vj > 0.0:
            alpha_corr = math.exp(-X_SEL * (1.0 - lam) * (1.0 - lam)) / (((Vi * Vj) / Vmean2) ** Y_SEL)
            alpha *= alpha_corr
    if alpha < 0.0:
        alpha = 0.0
    elif alpha > 1.0:
        alpha = 1.0
    if u_acc >= alpha:
        return -1
    return j

@njit
def nb_fenwick_add(tree, n, idx, delta):
    i = idx + 1
    while i <= n:
        tree[i] += delta
        i += i & -i

@njit
def nb_fenwick_build(tree, n, w):
    for i in range(n):
        nb_fenwick_add(tree, n, i, w[i])

@njit
def nb_fenwick_update(tree, n, w, idx, new_w):
    delta = new_w - w[idx]
    if delta != 0.0:
        w[idx] = new_w
        nb_fenwick_add(tree, n, idx, delta)
    return delta

@njit(parallel=False, fastmath=True)
def _build_table_1d_jit(rel, v, q, bf):
    n = rel.size
    cdf = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        val = _kb_break1d(rel[i], 1.0, v, q, bf)
        cdf[i] = val
        s += val
    if s > 0.0:
        invs = 1.0 / s
        acc = 0.0
        for i in range(n):
            acc += cdf[i] * invs
            cdf[i] = acc
        cdf[n-1] = 1.0
    return s, cdf

@njit(parallel=True, fastmath=True)
def _build_tables_2d_jit(rel1, rel3, v, q, bf):
    N1 = rel1.size
    N3 = rel3.size
    rowsum = np.zeros(N3, dtype=np.float64)
    row_cdf = np.zeros((N3, N1), dtype=np.float64)
    for i in prange(N3):
        s = 0.0
        for j in range(N1):
            val = _kb_break2d(rel1[j], rel3[i], 1.0, 1.0, v, q, bf)
            row_cdf[i, j] = val
            s += val
        rowsum[i] = s
        if s > 0.0:
            invs = 1.0 / s
            acc = 0.0
            for j in range(N1):
                acc += row_cdf[i, j] * invs
                row_cdf[i, j] = acc
            row_cdf[i, N1-1] = 1.0
    return rowsum, row_cdf

class FenwickSampler:
    __slots__ = ("_n", "_cap", "_tree", "_w", "_total")

    def __init__(self, weights: np.ndarray, capacity: Optional[int] = None):
        w = np.asarray(weights, dtype=float).ravel()
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        n = w.size
        cap = max(n, 1 if capacity is None else int(capacity))
        self._n = n
        self._cap = cap
        self._w = np.zeros(self._cap, dtype=float)
        if n:
            self._w[:n] = w
        self._tree = np.zeros(self._cap + 1, dtype=float)
        self._total = float(np.sum(w))
        # build tree for first n entries
        if self._n > 0:
            nb_fenwick_build(self._tree, self._n, self._w)

    def _grow(self):
        new_cap = max(2 * self._cap, 1)
        old_n = self._n
        old_w = self._w[:old_n].copy()
        self._cap = new_cap
        self._w = np.zeros(self._cap, dtype=float)
        if old_n:
            self._w[:old_n] = old_w
        self._tree = np.zeros(self._cap + 1, dtype=float)
        for i in range(old_n):
            wi = self._w[i]
            if wi:
                self._add(i, wi)

    def _add(self, idx: int, delta: float):
        nb_fenwick_add(self._tree, self._n, idx, delta)

    def total(self) -> float:
        return self._total

    def update(self, idx: int, new_weight: float):
        new_w = float(new_weight)
        delta = nb_fenwick_update(self._tree, self._n, self._w, idx, new_w)
        if delta:
            self._total += delta
    def append(self, weight: float):
        if self._n >= self._cap:
            self._grow()
        self._w[self._n] = float(weight)
        self._add(self._n, self._w[self._n])
        self._total += self._w[self._n]
        self._n += 1
    def prefix_sum_search(self, s: float) -> int:
        if s<0 or s>=self._total:raise ValueError("s must be in [0,total)")
        i=0
        bit=1<<(self._cap.bit_length()-1) if self._cap>0 else 0
        while bit:
            nxt=i+bit
            if nxt<=self._n and self._tree[nxt]<=s:
                s-=self._tree[nxt];i=nxt
            bit>>=1
        return i

    def sample(self, rng: np.random.Generator) -> int:
        if self._total <= 0 or self._n <= 0:
            raise ValueError("Total weight is non-positive; cannot sample.")
        u = rng.random() * self._total
        idx = self.prefix_sum_search(u)
        # ensure idx < n (could equal n when capacity > n and u ~ total)
        return min(idx, self._n - 1)

    def asarray(self) -> np.ndarray:
        return self._w[: self._n].copy()

# -----------------------------
# Solver
# -----------------------------

class MCPBESolver(BaseSolver):  # Monte Carlo PBE Solver with capacity buffers and CDF cache
    def __init__(
        self,
        dim: int = 2,
        t_total: int = 601,
        t_write: int = 10,
        t_vec: Optional[np.ndarray] = None,
        verbose: bool = False,
        load_attr: bool = True,
        config_path: Optional[str] = None,
        init: bool = True,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        self._init_base_parameters(dim, t_total, t_write, t_vec)
        self.c = np.full(dim, 0.1e-2)
        self.x = np.full(dim, 1e-6)
        self.x2 = np.full(dim, 1e-6)
        self.a0 = 1e3
        self.CDF_method = "disc"
        self.VERBOSE = verbose
        self.PGV = np.full(dim, 'mono')
        self.SIG = np.full(dim, 0.1)
        self.V_flat: Optional[np.ndarray] = None
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir, "config", "MCPBE_config.py")
        if load_attr:
            self._load_attributes(config_path)
        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
        if init:
            self._initialize_particles()
            self._initialize_samplers()
            self._bf_ready = False
        self._bf_cache = {}

    def _validate_input_arrays(self):
        dim = self.dim
        def _len(name: str) -> int:
            v = getattr(self, name, None)
            try:
                return len(v)
            except Exception:
                return -1
        for name in ("c", "x", "PGV", "SIG"):
            L = _len(name)
            if L != dim:
                raise ValueError(f"`{name}` must be a 1D array (sequence) of length dim={dim}, got length {L}.")

    def _growth_factor(self) -> float:
        T = float(getattr(self, 't_total', 1.0))
        t = float(getattr(self, '_elapsed', 0.0))
        r = 1.0 - min(max(t / max(T, 1e-12), 0.0), 1.0)
        f = 1.1 + 0.9 * r  # in [1.1, 2.0]
        return float(min(2.0, max(1.1, f)))

    def _compute_frag_num(self):
        v = float(getattr(self, 'pl_v', 1.0))
        bf = int(getattr(self, 'BREAKFVAL', 1))
        if self.dim == 1:
            if bf == 1: p = 4.0
            elif bf == 2: p = 2.0
            elif bf == 3: p = v
            elif bf == 4: p = (v + 1.0) / max(v, 1e-12)
            elif bf == 5: p = (v + 2.0) / max(v, 1e-12)
            else: raise ValueError(f"Unsupported BREAKFVAL={bf} for 1D.")
        else:
            if bf == 1: p = 4.0
            elif bf == 2: p = 2.0
            elif bf == 3: raise ValueError("BREAKFVAL=3 (product function) not implemented for 2D!")
            elif bf == 4: p = (v + 1.0) / max(v, 1e-12)
            elif bf == 5: p = (2.0 * v + 1.0) * (v + 2.0) / (2.0 * max(v, 1e-12) * (v + 1.0))
            else: raise ValueError(f"Unsupported BREAKFVAL={bf} for 2D.")
        if p <= 1.0:
            raise ValueError(f"Expected number of fragments p={p:.3f} must be > 1. Check BREAKFVAL/pl_v.")
        self.frag_num = float(p)

    def _initialize_particles(self, init_Vc=True, V_flat=None):
        dim = self.dim
        self._validate_input_arrays()
        if init_Vc:
            self.c = np.asarray(self.c, dtype=float)
            self.x = np.asarray(self.x, dtype=float)
            self.PGV = np.asarray(self.PGV)
            self.SIG = np.asarray(self.SIG, dtype=float)
            self.v = (self.x ** 3) * math.pi / 6.0
            self.n = np.round(self.c / self.v)
            self.n0 = float(np.sum(self.n))
            if self.n0 <= 0:
                raise ValueError("Total primary particle count `n0` must be > 0 (check c and x).")
            self.Vc = self.a0 / self.n0
            self.a = np.round(self.n * self.Vc).astype(int)
        total_cols = int(np.sum(self.a))
        if total_cols <= 0:
            raise ValueError("No particles to initialize (sum(a) == 0). Check c/x/PGV/SIG.")
        if V_flat is None:
            V_init = np.zeros((dim + 1, total_cols), dtype=float)
            cnt = 0
            for i in range(dim):
                ai = int(self.a[i])
                if ai <= 0: continue
                p = str(self.PGV[i])
                if p == 'mono':
                    V_init[i, cnt:cnt + ai] = self.v[i]
                elif p == 'norm':
                    mu = self.v[i]; sig = float(self.SIG[i]) * mu
                    V_init[i, cnt:cnt + ai] = self._rng.normal(mu, sig, ai)
                elif p == 'weibull':
                    V_init[i, cnt:cnt + ai] = self._rng.weibull(2.0, ai) * (self.SIG[i] * self.v[i])
                else:
                    raise ValueError(f"Unsupported PGV[{i}]='{p}'. Use 'mono' | 'norm' | 'weibull'.")
                cnt += ai
            V_init[-1, :] = np.sum(V_init[:dim, :], axis=0)
            keep = V_init[-1, :] > 0.0
            V_init = V_init[:, keep]
        else:
            V_init = V_flat
        a0_eff = V_init.shape[1]
        if a0_eff <= 0:
            raise ValueError("No particles initialized after filtering non-positive volumes.")
        cap = max(a0_eff + max(8, a0_eff // 10), 16)
        self._cap = int(cap)
        self.V_flat = np.zeros((dim + 1, self._cap), dtype=float)
        self.V_flat[:, :a0_eff] = V_init
        self.a_tot = a0_eff
        self.X = np.zeros(self._cap, dtype=float)
        self.X[:a0_eff] = self._vol2diam(self.V_flat[-1, :a0_eff])
        self.t = [0.0]
        if self.t_vec is None:
            steps = max(1, int(self.t_total // max(1, self.t_write)))
            self.t_vec = np.linspace(0.0, float(self.t_total), steps + 1)
        self._compute_frag_num()
        self.V0 = self.V_flat[:, :self.a_tot].copy(); self.X0 = self.X[:self.a_tot].copy()
        self.V0_save = [self.V0.copy()]
        self.V_save = [self.V_flat[:, :self.a_tot].copy()]
        self.Vc_save = [float(self.Vc)]
        self.step = 1

    def _initialize_samplers(self):
        pt = getattr(self, 'process_type', 'agglomeration')
        if pt in ('agglomeration', 'mix'):
            self._rebuild_all_propensities()
            if not hasattr(self, '_r_agg') or self._r_agg is None or self._r_agg.shape[0] < self._cap:
                buf = np.zeros(self._cap, dtype=float)
                if hasattr(self, '_r_agg') and self._r_agg is not None:
                    buf[:self.a_tot] = self._r_agg[:self.a_tot]
                self._r_agg = buf
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        else:
            self._r_agg = np.zeros(self._cap, dtype=float)
            self._agg_sampler = None
        if pt in ('breakage', 'mix'):
            self._calc_break_rates_full()
            if not hasattr(self, '_break_rate') or self._break_rate is None or self._break_rate.shape[0] < self._cap:
                br = np.zeros(self._cap, dtype=float)
                if hasattr(self, '_break_rate') and self._break_rate is not None:
                    br[:self.a_tot] = self._break_rate[:self.a_tot]
                self._break_rate = br
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
        else:
            self._break_rate = np.zeros(self._cap, dtype=float)
            self._break_sampler = None

    def _ensure_capacity_for(self, extra: int):
        need = self.a_tot + int(extra)
        if self._cap >= need:
            return
        old_cap = self._cap
        factor = self._growth_factor()
        new_cap = int(max(math.ceil(old_cap * factor), need))
        V_new = np.zeros((self.dim + 1, new_cap), dtype=float)
        X_new = np.zeros(new_cap, dtype=float)
        V_new[:, :self.a_tot] = self.V_flat[:, :self.a_tot]
        X_new[:self.a_tot] = self.X[:self.a_tot]
        self.V_flat = V_new; self.X = X_new; self._cap = new_cap
        # extend auxiliary arrays if present
        if hasattr(self, '_r_agg') and self._r_agg is not None:
            r_new = np.zeros(new_cap, dtype=float)
            r_new[:self.a_tot] = self._r_agg[:self.a_tot]
            self._r_agg = r_new
        if hasattr(self, '_break_rate') and self._break_rate is not None:
            b_new = np.zeros(new_cap, dtype=float)
            b_new[:self.a_tot] = self._break_rate[:self.a_tot]
            self._break_rate = b_new
        # print expansion info
        print(f"[MC-PBE] Capacity grown at t={getattr(self,'_elapsed',0.0):.4g} after {getattr(self,'_iter_count',0)} events: cap {old_cap} -> {new_cap} (x{new_cap/max(old_cap,1):.2f})")

    # -------------------------
    # Physics hooks (with external kernel integration)
    # -------------------------
    def _vol2diam(self, V: np.ndarray) -> np.ndarray:
        return (6.0 * V / math.pi) ** (1.0 / 3.0)

    def _alpha_ccm(self, idx1: int, idx2: int) -> float:
        """Collision efficiency based on component fractions (2D case).
        Mirrors legacy `calc_alpha_ccm_jit`.
        """
        if self.dim == 1:
            return float(np.sum(self.alpha_prim)) if np.ndim(self.alpha_prim) else float(self.alpha_prim)
        V = self.V_flat
        P = np.array([
            V[0, idx1] / V[-1, idx1] * V[0, idx2] / V[-1, idx2],
            V[0, idx1] / V[-1, idx1] * V[1, idx2] / V[-1, idx2],
            V[1, idx1] / V[-1, idx1] * V[0, idx2] / V[-1, idx2],
            V[1, idx1] / V[-1, idx1] * V[1, idx2] / V[-1, idx2],
        ])
        return float(np.sum(P * self.alpha_prim))

    def _beta(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        if JKA is not None and hasattr(JKA, 'calc_beta'):
            return float(JKA.calc_beta(self.COLEVAL, self.CORR_BETA, getattr(self, 'G', 1.0), self.X / 2.0, i, j))
        raise RuntimeError("jit_kernel_agg.calc_beta is required but not available.")
    def _break_rate_single(self, idx: int) -> float:
        """Compute single-particle breakage rate for particle `idx` using external JIT if available."""
        if JKB is not None:
            try:
                V = np.ascontiguousarray(self.V_flat[-1, :], dtype=np.float64)
                if self.dim == 1 and hasattr(JKB, 'calc_break_rate_1d'):
                    return float(JKB.calc_break_rate_1d(
                        V,
                        float(getattr(self, 'pl_P1', 1.0)),
                        float(getattr(self, 'pl_P2', 1.0)),
                        float(getattr(self, 'G', 1.0)),
                        int(getattr(self, 'BREAKRVAL', 1)),
                        int(idx),
                    ))
                if self.dim == 2 and hasattr(JKB, 'calc_break_rate_2d_flat'):
                    V1 = np.ascontiguousarray(self.V_flat[0, :], dtype=np.float64)
                    V3 = np.ascontiguousarray(self.V_flat[1, :], dtype=np.float64)
                    return float(JKB.calc_break_rate_2d_flat(
                        V,
                        V1,
                        V3,
                        float(getattr(self, 'G', 1.0)),
                        float(getattr(self, 'pl_P1', 1.0)),
                        float(getattr(self, 'pl_P2', 1.0)),
                        float(getattr(self, 'pl_P3', 1.0)),
                        float(getattr(self, 'pl_P4', 1.0)),
                        int(getattr(self, 'BREAKRVAL', 1)),
                        int(getattr(self, 'BREAKFVAL', 1)),
                        int(idx),
                    ))
            except Exception:
                pass
    def _rebuild_all_propensities(self):
        a = self.a_tot
        if a <= 0:
            if not hasattr(self, '_r_agg') or self._r_agg is None or self._r_agg.shape[0] < self._cap:
                self._r_agg = np.zeros(self._cap, dtype=float)
            else:
                self._r_agg[:] = 0.0
            self._calc_break_rates_full()
            return
        if _kb_beta is None:
            raise RuntimeError("jit_kernel_agg.calc_beta is required for JIT rebuild.")
        R = (self.X[:a] * 0.5).astype(np.float64)
        r = nb_rebuild_ragg(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, 'G', 1.0)), R)
        if not hasattr(self, '_r_agg') or self._r_agg is None or self._r_agg.shape[0] < self._cap:
            self._r_agg = np.zeros(self._cap, dtype=float)
        self._r_agg[:a] = r
        if self._r_agg.shape[0] > a:
            self._r_agg[a:] = 0.0
        self._calc_break_rates_full()


    def _calc_break_rates_full(self):
        a = self.a_tot
        if not hasattr(self,'_break_rate') or self._break_rate is None or self._break_rate.shape[0] < self._cap:
            self._break_rate = np.zeros(self._cap, dtype=float)
        if JKB is None:
            raise RuntimeError("jit_kernel_break is required but not available.")
        self.V = self.V_flat[-1, :a]
        self.B_R = np.zeros(a, dtype=float)
        if self.dim == 1 and hasattr(JKB, 'calc_B_R_1d'):
            JKB.calc_B_R_1d(self)
        elif self.dim == 2 and hasattr(JKB, 'calc_B_R_2d_flat'):
            JKB.calc_B_R_2d_flat(self)
        else:
            raise RuntimeError("Required breakage kernels not found in jit_kernel_break.")
        self._break_rate[:a] = np.asarray(self.B_R, dtype=float)
        if self._break_rate.shape[0] > a:
            self._break_rate[a:] = 0.0

    def _dt_agg(self) -> float:
        a = self.a_tot
        if a < 2:
            return float('inf')
        sum_r = float(np.sum(self._r_agg[:a]))
        if sum_r <= 0:
            return float('inf')
        return 2.0 * float(self.Vc) * (a - 1) / (a * sum_r)

    def _dt_break(self) -> float:
        a = self.a_tot
        if a <= 0:
            return float('inf')
        # total propensity is sum of rates; using sampler if available avoids array sums
        total = None
        if hasattr(self, '_break_sampler') and self._break_sampler is not None:
            total = self._break_sampler.total()
        if total is None:
            total = float(np.sum(self._break_rate[:a]))
        if total <= 0:
            return float('inf')
        return 1.0 / total
    def _maybe_double_control_volume(self, elapsed_time: float, iter_count: int):
        if getattr(self, 'process_type', 'agglomeration') not in ('agglomeration', 'mix'): return
        if getattr(self, 'a0', 0) <= 0: return
        if self.a_tot > self.a0 / 2: return
        old_a = self.a_tot; old_Vc = float(self.Vc)
        self.Vc *= 2.0
        V_dup = np.concatenate((self.V_flat[:, :self.a_tot], self.V_flat[:, :self.a_tot]), axis=1)
        X_dup = np.concatenate((self.X[:self.a_tot], self.X[:self.a_tot]))
        self.a_tot = V_dup.shape[1]
        if self._cap < self.a_tot:
            self._cap = int(self.a_tot * 1.2) + 8
            V_new = np.zeros((self.dim+1, self._cap), dtype=float)
            X_new = np.zeros(self._cap, dtype=float)
            V_new[:, :self.a_tot] = V_dup; X_new[:self.a_tot] = X_dup
            self.V_flat = V_new; self.X = X_new
        else:
            self.V_flat[:, :self.a_tot] = V_dup; self.X[:self.a_tot] = X_dup
        if hasattr(self, 'V0') and isinstance(self.V0, np.ndarray):
            self.V0 = np.concatenate((self.V0, self.V0), axis=1)
        self._rebuild_all_propensities(); self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot], capacity=self._cap)
        pt=getattr(self,'process_type','agglomeration')
        if pt in ('breakage','mix'):
            self._calc_break_rates_full(); self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot], capacity=self._cap)
        print(f"[MC-PBE] Control volume doubled at t={elapsed_time:.6g} after {iter_count} events: a_tot {old_a} -> {self.a_tot}, Vc {old_Vc:.6g} -> {self.Vc:.6g}")
        
    # -------------------------
    # Main solve loop (no post-processing)
    # -------------------------
    def solve(self, maxiter: int = int(1e8)):
        t0 = time.time()
        count = 0
        t_write = float(self.t_write)
        pt = getattr(self, 'process_type', 'agglomeration')
        timer_agg = 0.0
        timer_break = 0.0
        dtd_agg = self._dt_agg() if pt in ('agglomeration', 'mix') else float('inf')
        dtd_break = self._dt_break() if pt in ('breakage', 'mix') else float('inf')
        timer_agg += dtd_agg
        timer_break += dtd_break
        if self.VERBOSE:
            if np.isfinite(dtd_agg): print(f"Initial dt_agg = {dtd_agg:.3e} s")
            if np.isfinite(dtd_break): print(f"Initial dt_break = {dtd_break:.3e} s")
        next_save_idx = 1 if len(self.t_vec) > 1 else 0
        self._elapsed = 0.0
        self._iter_count = 0
        while self.t[-1] <= float(self.t_vec[-1]) and count < maxiter:
            self._elapsed = self.t[-1]
            self._iter_count = count
            if getattr(self, 'process_type', 'agglomeration') == 'agglomeration':
                self._do_one_agg()
                elapsed_time = timer_agg
                dtd_agg = self._dt_agg()
                timer_agg += dtd_agg
            elif getattr(self, 'process_type', 'agglomeration') == 'breakage':
                self._do_one_break()
                elapsed_time = timer_break
                dtd_break = self._dt_break()
                timer_break += dtd_break
            else:  # mixed
                if timer_agg <= timer_break:
                    self._do_one_agg()
                    elapsed_time = timer_agg
                    dtd_agg = self._dt_agg()
                    timer_agg += dtd_agg
                else:
                    self._do_one_break()
                    elapsed_time = timer_break
                    dtd_break = self._dt_break()
                    timer_break += dtd_break

            self.t.append(elapsed_time)
            self._maybe_double_control_volume(elapsed_time, count)
            # save at requested times
            while next_save_idx < len(self.t_vec) and elapsed_time >= self.t_vec[next_save_idx]:
                self.V_save.append(self.V_flat[:, :self.a_tot].copy())
                self.Vc_save.append(float(self.Vc))
                self.step += 1
                next_save_idx += 1

            count += 1
            if self.a_tot < 2 and getattr(self, 'process_agg', True):
                break

        self.MACHINE_TIME = time.time() - t0
        print(f'## The calculation took {self.MACHINE_TIME:.2f}s ##')
        return self
    def solve_repeats(self,N:int=5,base_seed:int=42,seeds:Optional[Sequence[int]]=None,maxiter:int=int(1e8)):
        if seeds is None:
            master=np.random.SeedSequence(base_seed);seeds=master.spawn(N)
        if len(seeds)!=N:raise ValueError("Length of seeds must equal N.")
        if not hasattr(self,'_bf_cache'): self._bf_cache={}
        results=[]
        for k in range(N):
            seed_k=seeds[k]
            if isinstance(seed_k,np.random.SeedSequence):self._rng=np.random.default_rng(seed_k);seed_info={"base_seed":base_seed,"spawn_key":tuple(seed_k.spawn_key)}
            else:self._rng=np.random.default_rng(int(seed_k));seed_info={"seed":int(seed_k)}
            self.V_flat=None;self._initialize_particles();self._initialize_samplers()
            self.solve(maxiter=maxiter)
            mu,tv=self.calc_moments_over_time(normalize=True);results.append({"seed_info":seed_info,"t_vec":tv,"moments":mu})
        return results
    def _do_one_agg(self):
        a = self.a_tot
        if a < 2:
            return
        i = self._agg_sampler.sample(self._rng)
        R = (self.X[:a] * 0.5).astype(np.float64)
        if self.dim == 1:
            alpha1d = float(self.alpha_prim if np.ndim(self.alpha_prim) == 0 else np.mean(self.alpha_prim))
            alpha4 = np.zeros(4, dtype=np.float64)
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = np.zeros_like(V0)
        else:
            alpha1d = 1.0
            ap = np.asarray(self.alpha_prim, dtype=np.float64)
            alpha4 = ap if ap.size == 4 else np.ones(4, dtype=np.float64)
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = self.V_flat[1, :a].astype(np.float64)
        SIZEEVAL = int(getattr(self, 'SIZEEVAL', 1))
        X_SEL = float(getattr(self, 'X_SEL', 0.31))
        Y_SEL = float(getattr(self, 'Y_SEL', 1.06))
        Vmean2 = float(np.mean(self.V0[-1, :]) ** 2) if hasattr(self, 'V0') else float(np.mean(self.V_flat[-1, :a]) ** 2)
        u_sel = float(self._rng.random()); u_acc = float(self._rng.random())
        j = nb_pick_partner(i, int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, 'G', 1.0)), R, V0, V1, int(self.dim), float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2, u_sel, u_acc)
        if j < 0 or j == i:
            return
        Vi = self.V_flat[: self.dim, i].copy()
        Vj = self.V_flat[: self.dim, j].copy()
        Vnew = Vi + Vj
        Xnew = float(self._vol2diam(np.sum(Vnew)))
        for m in range(a):
            if m == i or m == j:
                continue
            beta_m_i = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, 'G', 1.0)), R, m, i)
            beta_m_j = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, 'G', 1.0)), R, m, j)
            Xi_old = self.X[i]
            self.X[i] = Xnew
            R[i] = 0.5 * Xnew
            beta_m_new = _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, 'G', 1.0)), R, m, i)
            self.X[i] = Xi_old
            R[i] = 0.5 * Xi_old
            rm_new = self._r_agg[m] - beta_m_i - beta_m_j + beta_m_new
            self._r_agg[m] = rm_new
            self._agg_sampler.update(m, rm_new)
        self.V_flat[: self.dim, i] = Vnew
        self.V_flat[-1, i] = float(np.sum(Vnew))
        self.X[i] = Xnew
        last = a - 1
        i_after = j if (j != last and i == last) else i
        self._remove_particle_column(j)
        i = i_after
        rk = 0.0
        for m in range(self.a_tot):
            if m == i:
                continue
            rk += _kb_beta(int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, 'G', 1.0)), (self.X[:self.a_tot] * 0.5).astype(np.float64), i, m)
        self._r_agg[i] = rk
        self._agg_sampler.update(i, rk)
        if getattr(self, 'process_break', False):
            self._calc_break_rates_full()
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])


    def _do_one_break(self):
        a = self.a_tot
        if a < 1: return
        self._ensure_break_sampler()
        if self._break_sampler is None or self._break_sampler.total() <= 0: return
        k = self._break_sampler.sample(self._rng)

        # expected number of fragments -> integer via stochastic rounding
        fn = float(getattr(self, 'frag_num', 2.0))
        fl = int(math.floor(fn)); ce = int(math.ceil(fn))
        if fl <= 1:
            raise ValueError("Expected number of fragments <= 1. Check BREAKFVAL/pl_v.")
        if fl == ce:
            n = fl
        else:
            delta = fn - fl
            n = ce if (self._rng.random() < delta) else fl
        # remaining volume vector for the selected particle
        Vrem = self.V_flat[: self.dim, k].copy()
        frags = []
        for _ in range(n - 1):
            f = self._produce_one_frag_from_remaining(Vrem)
            Vrem = Vrem - f
            Vrem = np.maximum(Vrem, 0.0)
            frags.append(f)
        # assign remainder to k, append the other (n-1) fragments
        self.V_flat[: self.dim, k] = Vrem
        self.V_flat[-1, k] = float(np.sum(Vrem))
        self.X[k] = float(self._vol2diam(self.V_flat[-1, k]))
        br_k = self._break_rate_single(k)
        if hasattr(self, '_break_rate') and self._break_rate.size >= a: self._break_rate[k] = br_k
        if self._break_sampler is not None: self._break_sampler.update(k, br_k)

        a_before = self.a_tot
        for f in frags:
            self._append_particle_column(f)
            new_idx = self.a_tot - 1
            br_new = self._break_rate_single(new_idx)
            # capacity buffers ensure room; assign directly
            if not hasattr(self, '_break_rate') or self._break_rate.shape[0] < self._cap:
                self._break_rate = np.pad(self._break_rate, (0, self._cap - self._break_rate.shape[0]))
            self._break_rate[new_idx] = br_new
            if self._break_sampler is not None:
                self._break_sampler.append(br_new)

        if getattr(self, 'process_type', 'agglomeration') == 'mix':
            a_after = self.a_tot
            n_new = a_after - a_before
            if n_new > 0:
                if self._r_agg.size < self.a_tot:
                    self._r_agg = np.concatenate((self._r_agg, np.zeros(self.a_tot - self._r_agg.size, dtype=float)))
                # update existing m != k by adding contributions from new particles
                for m in range(a_before):
                    if m == k: continue
                    inc = 0.0
                    for new_idx in range(a_before, a_after):
                        inc += self._beta(m, new_idx)
                    self._r_agg[m] += inc
                    self._agg_sampler.update(m, self._r_agg[m])
                # update r for k
                rk = 0.0
                for m in range(self.a_tot):
                    if m != k: rk += self._beta(k, m)
                self._r_agg[k] = rk
                self._agg_sampler.update(k, rk)
                # set r for each new particle and append to sampler
                for new_idx in range(a_before, a_after):
                    rnew = 0.0
                    for m in range(self.a_tot):
                        if m != new_idx: rnew += self._beta(new_idx, m)
                    self._r_agg[new_idx] = rnew
                    self._agg_sampler.append(rnew)

    def _remove_particle_column(self, j: int):
        a = self.a_tot
        if j < 0 or j >= a:
            raise IndexError("column index out of range")
        if a <= 1:
            self.a_tot = max(0, a - 1)
            if self._agg_sampler is not None:
                self._agg_sampler = FenwickSampler(np.zeros(0))
            if self._break_sampler is not None:
                self._break_sampler = FenwickSampler(np.zeros(0))
            return
        last = a - 1
        if j != last:
            self.V_flat[:, [j, last]] = self.V_flat[:, [last, j]]
            self.X[j], self.X[last] = self.X[last], self.X[j]
            if hasattr(self, '_r_agg') and self._r_agg is not None and self._r_agg.shape[0] >= a:
                self._r_agg[j], self._r_agg[last] = self._r_agg[last], self._r_agg[j]
            if hasattr(self, '_break_rate') and self._break_rate is not None and self._break_rate.shape[0] >= a:
                self._break_rate[j], self._break_rate[last] = self._break_rate[last], self._break_rate[j]
        self.a_tot = last
        # zero freed slot to avoid stale contributions
        self.V_flat[:, self.a_tot:self.a_tot+1] = 0.0
        self.X[self.a_tot:self.a_tot+1] = 0.0
        if hasattr(self,'_r_agg') and self._r_agg is not None and self._r_agg.shape[0]>self.a_tot:
            self._r_agg[self.a_tot:self.a_tot+1] = 0.0
        if hasattr(self,'_break_rate') and self._break_rate is not None and self._break_rate.shape[0]>self.a_tot:
            self._break_rate[self.a_tot:self.a_tot+1] = 0.0
        if self._agg_sampler is not None:
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        if self._break_sampler is not None:
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])

    def _append_particle_column(self, frag_vols: np.ndarray):
        frag_vols = np.asarray(frag_vols, dtype=float)
        if frag_vols.shape != (self.dim,):
            raise ValueError("frag_vols must have shape (dim,)")
        self._ensure_capacity_for(1)
        idx = self.a_tot
        Vnew = float(np.sum(frag_vols))
        self.V_flat[: self.dim, idx] = frag_vols
        self.V_flat[-1, idx] = Vnew
        self.X[idx] = self._vol2diam(Vnew)
        self.a_tot += 1
    def _ensure_break_sampler(self):
        if getattr(self, 'process_type', 'agglomeration') not in ('breakage', 'mix'): return
        if not hasattr(self, '_break_sampler') or self._break_sampler is None:
            if np.sum(self._break_rate[:self.a_tot]) > 0:
                self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot], capacity=self._cap)
            return
        if self._break_sampler.total() <= 0 and np.sum(self._break_rate[:self.a_tot]) > 0:
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot], capacity=self._cap)
        return
        if self._break_sampler.total() <= 0 and np.sum(self._break_rate) > 0:
            self._break_sampler = FenwickSampler(self._break_rate)
        return
    def _build_break_function(self, num_points: int = 1000):
        kb = JKB
        key=(int(self.dim),int(num_points),int(getattr(self,'BREAKFVAL',1)),float(getattr(self,'pl_v',1.0)),float(getattr(self,'pl_q',1.0)))
        if hasattr(self,'_bf_cache') and key in self._bf_cache:
            val=self._bf_cache[key]
            if self.dim==1:
                self._bf_rel_1d,self._bf_cdf_1d=val
            else:
                self._bf_rel1_2d,self._bf_rel3_2d,self._bf_rowsum_cdf,self._bf_row_cdf=val
            self._bf_ready=True; return
        self._bf_ready = True
        if self.dim == 1:
            if kb is None or not hasattr(kb, 'breakage_func_1d'):
                raise RuntimeError("breakage_func_1d not available in jit_kernel_break; no fallback.")
            rel = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
            s, cdf = _build_table_1d_jit(rel, float(getattr(self, 'pl_v', 1.0)), float(getattr(self, 'pl_q', 1.0)), int(getattr(self, 'BREAKFVAL', 1)))
            if s <= 0.0:
                raise RuntimeError("1D breakage pdf sums to zero.")
            self._bf_rel_1d = rel
            self._bf_cdf_1d = cdf
        else:
            if kb is None or not hasattr(kb, 'breakage_func_2d'):
                raise RuntimeError("breakage_func_2d not available in jit_kernel_break; no fallback.")
            rel1 = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
            rel3 = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
            rowsum, row_cdf = _build_tables_2d_jit(rel1, rel3, float(getattr(self, 'pl_v', 1.0)), float(getattr(self, 'pl_q', 1.0)), int(getattr(self, 'BREAKFVAL', 1)))
            rowsum_total = float(np.sum(rowsum))
            if rowsum_total <= 0.0:
                raise RuntimeError("Row sums of 2D pdf are zero.")
            rowsum_cdf = np.cumsum(rowsum / rowsum_total)
            rowsum_cdf[-1] = 1.0
            self._bf_rel1_2d = rel1
            self._bf_rel3_2d = rel3
            self._bf_rowsum_cdf = rowsum_cdf
            self._bf_row_cdf = row_cdf
    def _discrete_sample(self, x: np.ndarray, cdf: np.ndarray) -> float:
        u = self._rng.random()
        k = int(np.searchsorted(cdf, u, side='right'))
        k = min(max(k, 0), x.size - 1)
        return float(x[k])
    
    def _discrete_sample_2d(self) -> Tuple[float, float]:
        u1 = self._rng.random()
        i = int(np.searchsorted(self._bf_rowsum_cdf, u1, side='right'))
        i = min(max(i, 0), self._bf_rel1_2d.size - 1)
        u2 = self._rng.random()
        row_cdf = self._bf_row_cdf[i]
        j = int(np.searchsorted(row_cdf, u2, side='right'))
        j = min(max(j, 0), self._bf_rel3_2d.size - 1)
        return float(self._bf_rel1_2d[i]), float(self._bf_rel3_2d[j])

    def _produce_one_frag_from_remaining(self, Vrem: np.ndarray) -> np.ndarray:
        if not getattr(self, '_bf_ready', False):
            self._build_break_function()
        Vrem = np.asarray(Vrem, dtype=float)
        if self.dim == 1:
            Vtot = float(np.sum(Vrem))
            rel = self._discrete_sample(self._bf_rel_1d, self._bf_cdf_1d)
            vol = max(0.0, min(Vtot, rel * Vtot))
            return np.array([vol], dtype=float)
        r1, r3 = self._discrete_sample_2d()
        v1 = max(0.0, min(Vrem[0], r1 * Vrem[0]))
        v3 = max(0.0, min(Vrem[1], r3 * Vrem[1]))
        return np.array([v1, v3], dtype=float)
  
    def calc_moments_over_time(self,max_i:int=2,max_j:int=2,normalize:bool=True):
        T=min(len(self.V_save),len(self.Vc_save),len(self.t_vec))
        mu=np.zeros((max_i+1,max_j+1,T),dtype=float)
        for t in range(T):
            Vc=float(self.Vc_save[t]) if (normalize and self.Vc_save) else 1.0
            if self.dim==1:
                V=np.asarray(self.V_save[t][0,:],dtype=float)
                for i in range(max_i+1): mu[i,0,t]=np.sum(np.power(V,i))/Vc
            else:
                V1=np.asarray(self.V_save[t][0,:],dtype=float)
                V3=np.asarray(self.V_save[t][1,:],dtype=float)
                for i in range(max_i+1):
                    Vi=np.power(V1,i)
                    for j in range(max_j+1): mu[i,j,t]=np.dot(Vi,np.power(V3,j))/Vc
        return mu,self.t_vec[:T]
        out=np.zeros((T,max_i+1,max_j+1),dtype=float)
        for t in range(T):
            V1=self.V_save[t][0,:];V3=self.V_save[t][1,:]
            for i in range(max_i+1):
                Vi=V1**i
                for j in range(max_j+1): out[t,i,j]=np.sum(Vi*(V3**j))
            if normalize and self.Vc_save and self.Vc_save[t]>0: out[t,:,:]/=float(self.Vc_save[t])
        return out,self.t_vec[:T]
