# -*- coding: utf-8 -*-
"""
PSD-driven on-lattice aggregate growth (MVP)
- Online PSD scheduler + online PSD-driven contact growth
- Candidate generation via morphological dilation (implemented by shifts + integral images)
- Df-convergence via Metropolis acceptance on Rg/slope + soft PSD quota penalty
- Two-material (A/B) composition respected via per-material area budgets
- Minimal fallbacks: skip oversized particles; end-fill with small pixels if needed

Dependencies: numpy, matplotlib
Reuses utilities from mptsa2d.py: radius_of_gyration_2d, estimate_fractal_dimension_2d
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import math
from numba import njit

# Reuse existing utilities
from mptsa2d import radius_of_gyration_2d, estimate_fractal_dimension_2d

import matplotlib.pyplot as plt
Material = str  # "A" or "B"

@dataclass
class PSD:
    """Particle size distribution in *number* space.
    - `diam_um`: array of particle diameters (µm), ascending.
    - `prob`: number-PSD (discrete probability for each diameter entry), will be discretized to square side length `s`.

    Note: If your experimental data is cumulative number distribution Q0(d), use
    `PSD.from_cumulative(diam_um, Q0)` or the loader `load_psd_from_npz(..., cumulative=True)`.
    """
    diam_um: np.ndarray
    prob: np.ndarray

    @classmethod
    def from_cumulative(cls, diam_um: np.ndarray, q0: np.ndarray, *, percent_ok: bool = True) -> "PSD":
        """Build a differential *number-PSD* from cumulative number distribution Q0.
        Assumptions:
        - `diam_um` and `q0` have the same length, sorted by diameter.
        - `q0` is non-decreasing in [0, 1] (if in %, values >1.5 will be divided by 100).
        - Each `q0[i]` is the cumulative fraction up to *and including* `diam_um[i]`.
        """
        d = np.asarray(diam_um, dtype=float).ravel()
        q = np.asarray(q0, dtype=float).ravel()
        if len(d) != len(q):
            raise ValueError("diam_um and q0 must have the same length for cumulative conversion")
        # sort by diameter just in case
        order = np.argsort(d)
        d = d[order]
        q = q[order]
        if percent_ok and np.nanmax(q) > 1.5:
            q = q / 100.0
        # clamp & enforce monotonic non-decreasing
        q = np.clip(q, 0.0, 1.0)
        q = np.maximum.accumulate(q)
        # diff to obtain discrete number-PSD
        p = np.diff(np.concatenate([[0.0], q]))
        p = np.clip(p, 0.0, None)
        S = p.sum()
        if not np.isfinite(S) or S <= 0:
            # fall back to uniform if numerical issues
            p = np.ones_like(p) / len(p)
        else:
            p = p / S
        # Representative diameter per bin: use the provided upper edge `d`
        return cls(diam_um=d, prob=p)

    def to_side_hist(self, cell_um: float, s_min: int = 1, s_max: int = 64) -> Dict[int, float]:
        assert len(self.diam_um) == len(self.prob) and len(self.prob) > 0
        p = np.array(self.prob, dtype=float)
        p = p / p.sum()
        s = np.clip(np.rint(self.diam_um / max(cell_um, 1e-12)).astype(int), s_min, s_max)
        hist: Dict[int, float] = {}
        for si, pi in zip(s, p):
            hist[si] = hist.get(si, 0.0) + float(pi)
        # normalize again
        Z = sum(hist.values())
        return {k: v / Z for k, v in hist.items() if v > 0}


@dataclass
class OnlinePSDScheduler:
    Np: int
    frac_A: float
    psd_A: PSD
    psd_B: PSD
    cell_um: float = 1.0
    s_min: int = 1
    s_max: int = 64
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    # --- new switches ---
    integer_quota: bool = False               # use integer targets via Hamilton apportionment
    quota_method: str = "largest_remainder"

    # internal state
    target_counts: Dict[Tuple[Material, int], float] = field(init=False, default_factory=dict)
    placed_counts: Dict[Tuple[Material, int], int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # area budgets by material
        A_area = int(round(self.frac_A * self.Np))
        B_area = int(self.Np - A_area)

        # discretize PSDs to side-length histograms (number fraction by size, per material)
        pA = self.psd_A.to_side_hist(self.cell_um, self.s_min, self.s_max)
        pB = self.psd_B.to_side_hist(self.cell_um, self.s_min, self.s_max)

        # Expected number of particles for each material: area = n * E[s^2]  =>  n = area / E[s^2]
        Es2_A = sum(pA[s] * (s ** 2) for s in pA) if pA else 1.0
        Es2_B = sum(pB[s] * (s ** 2) for s in pB) if pB else 1.0
        nA = A_area / Es2_A
        nB = B_area / Es2_B

        if self.integer_quota and self.quota_method == "largest_remainder":
            # Hamilton (largest remainder) apportionment to integer target counts
            def apportion(probs: Dict[int, float], total_n: float) -> Dict[int, int]:
                if total_n <= 0 or not probs:
                    return {}
                keys = sorted(probs.keys())
                fracs = np.array([probs[k] for k in keys], dtype=float)
                Z = fracs.sum()
                fracs = fracs / (Z if Z > 0 else 1.0)
                Ntot = int(max(0, round(total_n)))
                raw = fracs * Ntot
                base = np.floor(raw).astype(int)
                remainder = raw - base
                to_assign = Ntot - int(base.sum())
                order = np.argsort(-remainder)
                for i in range(int(to_assign)):
                    base[order[i % len(keys)]] += 1
                return {k: int(v) for k, v in zip(keys, base)}

            A_counts = apportion(pA, nA)
            B_counts = apportion(pB, nB)
            for s, c in A_counts.items():
                self.target_counts[("A", s)] = float(c)
            for s, c in B_counts.items():
                self.target_counts[("B", s)] = float(c)
        else:
            # fractional targets
            for s, p in pA.items():
                self.target_counts[("A", s)] = nA * p
            for s, p in pB.items():
                self.target_counts[("B", s)] = nB * p

        # initialize placed counts
        for key in list(self.target_counts.keys()):
            self.placed_counts[key] = 0

        # store for reporting
        self._pA_hist = pA
        self._pB_hist = pB
        self._A_area = A_area
        self._B_area = B_area

    def total_area_placed(self) -> int:
        return sum((s ** 2) * c for (m, s), c in self.placed_counts.items())

    def area_remaining(self) -> int:
        return max(0, self.Np - self.total_area_placed())

    def deficits(self) -> Dict[Tuple[Material, int], float]:
        # deficit in NUMBER of particles (not area)
        return {k: max(0.0, self.target_counts.get(k, 0.0) - self.placed_counts.get(k, 0)) for k in self.target_counts}

    def next_item(self) -> Optional[Tuple[Material, int]]:
        """Sample next (material, s) proportional to positive deficit; if no deficit, fall back to smallest s to fill residual area."""
        rem = self.area_remaining()
        if rem <= 0:
            return None
        d = self.deficits()
        keys = [k for k, v in d.items() if v > 1e-6]
        if keys:
            weights = np.array([d[k] for k in keys], dtype=float)
            if weights.sum() <= 0:
                keys = []
        if not keys:
            # no deficit left: fill with s_min pixels of the material that still has area budget, else alternate
            # compute material areas placed
            areaA = sum((s ** 2) * self.placed_counts.get(("A", s), 0) for s in range(self.s_min, self.s_max + 1))
            areaB = sum((s ** 2) * self.placed_counts.get(("B", s), 0) for s in range(self.s_min, self.s_max + 1))
            if areaA < self._A_area and areaB < self._B_area:
                mat = "A" if self.rng.random() < 0.5 else "B"
            elif areaA < self._A_area:
                mat = "A"
            elif areaB < self._B_area:
                mat = "B"
            else:
                mat = "A" if self.rng.random() < 0.5 else "B"
            s = self.s_min
            # guard big residuals by enlarging s greedily (still simple)
            while (s + 1) <= self.s_max and (s + 1) ** 2 <= rem:
                s += 1
            return (mat, s)
        probs = weights / weights.sum()
        idx = int(self.rng.choice(len(keys), p=probs))
        return keys[idx]

    def record_placed(self, item: Tuple[Material, int]):
        self.placed_counts[item] = self.placed_counts.get(item, 0) + 1

    # reporting
    def achieved_psd_by_number(self) -> Dict[Material, Dict[int, float]]:
        res = {"A": {}, "B": {}}
        for (m, s), c in self.placed_counts.items():
            res[m][s] = res[m].get(s, 0) + int(c)
        for m in res:
            total = sum(res[m].values()) or 1
            for s in list(res[m].keys()):
                res[m][s] = res[m][s] / total
        return res

    def input_psd_by_number(self) -> Dict[Material, Dict[int, float]]:
        return {"A": dict(self._pA_hist), "B": dict(self._pB_hist)}

    def composition_area(self) -> Dict[Material, float]:
        areaA = sum((s ** 2) * self.placed_counts.get(("A", s), 0) for s in range(self.s_min, self.s_max + 1))
        areaB = sum((s ** 2) * self.placed_counts.get(("B", s), 0) for s in range(self.s_min, self.s_max + 1))
        total = areaA + areaB
        if total <= 0:
            return {"A": 0.0, "B": 0.0}
        return {"A": areaA / total, "B": areaB / total}


@dataclass
class GrowthParams:
    Np: int
    Df: float
    k: float = 1.0
    cell_um: float = 1.0
    adaptive_bias: float = 2.0
    w_rg: float = 1.0
    w_slope: float = 0.25
    w_quota: float = 0.1
    num_trials: int = 16
    seed: int = 42
    # cooling schedule
    T0: float = 0.5
    Tmin: float = 0.05
    # grid padding for growth
    pad_margin: int = 24  # keep this many empty cells around current bbox
    # --- new switches ---
    bidirectional_quota_penalty: bool = False  # penalize both over- and under-quota if True
    endfill_by_psd_deficit: bool = False       # end-fill by PSD deficits if True

@dataclass
class GrowthResult:
    grid: np.ndarray           # occupancy 0/1
    labels: np.ndarray         # 0 for A, 1 for B, -1 for empty
    origin: Tuple[int, int]    # (x_min, y_min)
    Ns: List[int]
    Rgs: List[float]
    Df_est: float
    slope: float
    scheduler: OnlinePSDScheduler
    placed_particles: List[Tuple[Material, int, Tuple[int, int]]]

def integral_image(arr: np.ndarray) -> np.ndarray:
    # arr: H x W (0/1)
    return arr.cumsum(axis=0).cumsum(axis=1)

# @njit(cache=True, fastmath=True)
# def rect_sum(ii: np.ndarray, top: int, left: int, h: int, w: int) -> int:
#     i2 = top + h - 1
#     j2 = left + w - 1
#     s = ii[i2, j2]
#     if top > 0: s -= ii[top - 1, j2]
#     if left > 0: s -= ii[i2, left - 1]
#     if top > 0 and left > 0: s += ii[top - 1, left - 1]
#     return int(s)

# @njit(cache=True, fastmath=True)
# def rect_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
#     A = ii[y1, x1]
#     B = ii[y0, x1]
#     C = ii[y1, x0]
#     D = ii[y0, x0]
#     return int(A - B - C + D)

def ensure_margin(grid: np.ndarray, labels: np.ndarray, margin: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    # returns possibly padded grid/labels and pad offsets (dx, dy) applied to origin
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        # empty -> ensure minimal canvas
        H, W = grid.shape
        need = max(margin, 8)
        pad = need
        new = np.pad(grid, ((pad, pad), (pad, pad)))
        lab = np.pad(labels, ((pad, pad), (pad, pad)), constant_values=-1)
        return new, lab, (pad, pad)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    H, W = grid.shape
    top = max(0, margin - y0)
    left = max(0, margin - x0)
    bottom = max(0, (y1 + margin + 1) - H)
    right = max(0, (x1 + margin + 1) - W)
    if top or left or bottom or right:
        grid2 = np.pad(grid, ((top, bottom), (left, right)))
        labels2 = np.pad(labels, ((top, bottom), (left, right)), constant_values=-1)
        return grid2, labels2, (left, top)  # origin shift by (left, top)
    else:
        return grid, labels, (0, 0)


def crop_to_bbox(grid: np.ndarray, labels: np.ndarray, margin: int = 0) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Crop grid/labels to the tight bounding box around occupied cells, with optional integer margin.
    Returns (cropped_grid, cropped_labels, (dx, dy)) where (dx, dy) is the top-left shift applied.
    """
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return grid, labels, (0, 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    if margin > 0:
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(grid.shape[0] - 1, y1 + margin)
        x1 = min(grid.shape[1] - 1, x1 + margin)
    grid2 = grid[y0:y1+1, x0:x1+1]
    labels2 = labels[y0:y1+1, x0:x1+1]
    return grid2, labels2, (x0, y0)


def neighbor_touch_mask(occ: np.ndarray) -> np.ndarray:
    # mask of empty cells that are 4-neighbor to occ
    up = np.zeros_like(occ); up[1:, :] = occ[:-1, :]
    down = np.zeros_like(occ); down[:-1, :] = occ[1:, :]
    left = np.zeros_like(occ); left[:, 1:] = occ[:, :-1]
    right = np.zeros_like(occ); right[:, :-1] = occ[:, 1:]
    neighbor = up | down | left | right
    return (~occ) & neighbor

@njit(cache=True, fastmath=True)
def _rect_sum_njit(ii: np.ndarray, top: int, left: int, h: int, w: int) -> int:
    """
    与原 rect_sum 一致（无 padding 的积分图四点求和）。
    """
    i2 = top + h - 1
    j2 = left + w - 1
    s = ii[i2, j2]
    if top > 0:
        s -= ii[top - 1, j2]
    if left > 0:
        s -= ii[i2, left - 1]
    if top > 0 and left > 0:
        s += ii[top - 1, left - 1]
    return int(s)

@njit(cache=True)
def _scan_candidates_ii(ii_occ: np.ndarray, ii_touch: np.ndarray, H: int, W: int, s: int):
    """
    两遍扫描：先计数，后填充；返回两个等长数组 ys, xs。
    """
    # pass 1: count
    cnt = 0
    for y in range(0, H - s + 1):
        for x in range(0, W - s + 1):
            if _rect_sum_njit(ii_occ, y, x, s, s) != 0:
                continue
            if _rect_sum_njit(ii_touch, y, x, s, s) <= 0:
                continue
            cnt += 1

    ys = np.empty(cnt, dtype=np.int32)
    xs = np.empty(cnt, dtype=np.int32)

    # pass 2: fill
    k = 0
    for y in range(0, H - s + 1):
        for x in range(0, W - s + 1):
            if _rect_sum_njit(ii_occ, y, x, s, s) != 0:
                continue
            if _rect_sum_njit(ii_touch, y, x, s, s) <= 0:
                continue
            ys[k] = y
            xs[k] = x
            k += 1
    return ys, xs


def candidate_top_lefts_for_s(occ: np.ndarray, s: int) -> List[Tuple[int, int]]:
    """
    与原函数同名同参。
    - 先构建积分图：占用 ii_occ；触边掩膜 ii_touch
    - 调用 numba 内核扫描可行 top-left
    """
    H, W = occ.shape
    if H < s or W < s:
        return []

    # 与原实现一致：整型积分图
    ii_occ = integral_image(occ.astype(np.int32))
    touch = neighbor_touch_mask(occ).astype(np.int32)
    ii_touch = integral_image(touch)

    ys, xs = _scan_candidates_ii(ii_occ, ii_touch, int(H), int(W), int(s))
    if ys.size == 0:
        return []
    # 打包为 List[Tuple[int,int]]（与原返回值一致）
    return list(zip(ys.tolist(), xs.tolist()))

@njit(cache=True, fastmath=True)
def block_center(x: int, y: int, s: int) -> Tuple[float, float]:
    # For grid indexing (y, x), center is at x + (s-1)/2, y + (s-1)/2
    return (x + 0.5 * (s - 1), y + 0.5 * (s - 1))

def compute_beta(N: int, Rg: float, Df: float, k: float, cell_um: float, adaptive_bias: float) -> float:
    a = 0.6 * (k ** (1.0 / max(Df, 1e-6)))
    Rg_target = a * cell_um * (N ** (1.0 / max(Df, 1e-6)))
    ratio = max(Rg_target / max(Rg, 1e-12), 0.1)
    if ratio > 1.0:
        beta = 1.0 + adaptive_bias * np.arctan(ratio - 1.0)
    else:
        beta = 1.0 + 0.5 * (ratio - 1.0)
    return float(np.clip(beta, 0.2, 2.0))

def rg_after_placing(occ: np.ndarray, block: Tuple[int, int, int], cell_um: float) -> float:
    """Compute Rg after placing sxs block at (y, x)."""
    y, x, s = block
    ys, xs = np.nonzero(occ)
    if len(xs) == 0:
        # block alone
        # exact Rg of an sxs square about its centroid: use discrete approx by sampling pixels
        # but MVP: treat as sampling pixels, ok
        pts_y, pts_x = np.mgrid[y:y+s, x:x+s]
        pts = np.stack([pts_x.ravel(), pts_y.ravel()], axis=1).astype(float)
        return radius_of_gyration_2d(pts) * cell_um
    # existing points
    pts = np.stack([xs, ys], axis=1).astype(float)
    # concatenate block pixels
    by, bx = np.mgrid[y:y+s, x:x+s]
    bpts = np.stack([bx.ravel(), by.ravel()], axis=1).astype(float)
    allpts = np.concatenate([pts, bpts], axis=0)
    return radius_of_gyration_2d(allpts) * cell_um

@njit(cache=True, fastmath=True)
def _weights_for_candidates(
    cands_y: np.ndarray,
    cands_x: np.ndarray,
    s: int,
    com_x: float,
    com_y: float,
    cell_um: float,
    beta: float,
    sector_hits: np.ndarray  # shape=(K,)
) -> np.ndarray:
    """
    纯数值内核：为一批候选计算采样权重。
    - 温和化向外偏置： (1 + atan(r/scale)) ** beta
    - 角度均衡： 1/(1+sector_hits[sector])
    """
    K = sector_hits.shape[0]
    out = np.empty(cands_x.size, dtype=np.float64)
    inv_sector = 1.0 / (2.0 * np.pi / K)
    scale = cell_um * 5.0  # 与之前建议一致；不改变结果可把它当常数

    for i in range(cands_x.size):
        x = cands_x[i]
        y = cands_y[i]
        # block_center 内联
        cx = x + 0.5 * s
        cy = y + 0.5 * s
        dx = cx - com_x
        dy = cy - com_y
        r = (dx*dx + dy*dy) ** 0.5
        theta = math.atan2(dy, dx)
        sector = int(math.floor((theta + math.pi) * inv_sector)) % K
        angle_boost = 1.0 / (1.0 + sector_hits[sector])
        radial = (1.0 + math.atan(r / scale)) ** beta
        out[i] = radial * angle_boost
    return out

def choose_candidate_by_weight(
    occ: np.ndarray,
    cands: List[Tuple[int,int]],
    s: int,
    cell_um: float,
    beta: float,
    rng: np.random.Generator,
    K: int = 12,
    sector_usage: Optional[np.ndarray] = None
) -> Optional[Tuple[int,int]]:
    """
    与原函数同名同参：返回按权重随机挑选的候选点。
    - 权重计算用 numba 内核加速
    - sector_usage 可传入全局的扇区计数（用于角度均衡）；不传则内部用全 0
    """
    if not cands:
        return None

    ys_occ, xs_occ = np.nonzero(occ)
    if xs_occ.size == 0:
        idx = int(rng.integers(0, len(cands)))
        return cands[idx]

    com_x = float(xs_occ.mean())
    com_y = float(ys_occ.mean())

    # 将 list[tuple] 拆成两个数组，便于传入 njit 内核
    cands_arr = np.asarray(cands, dtype=np.int32)
    cands_y = cands_arr[:, 0]
    cands_x = cands_arr[:, 1]

    if sector_usage is None or getattr(sector_usage, "shape", (0,))[0] != K:
        sector_hits = np.zeros(K, dtype=np.float64)
    else:
        sector_hits = np.asarray(sector_usage, dtype=np.float64)

    w = _weights_for_candidates(cands_y, cands_x, int(s), com_x, com_y,
                                float(cell_um), float(beta), sector_hits)
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        idx = int(rng.integers(0, len(cands)))
        return cands[idx]
    probs = w / w_sum
    i = int(rng.choice(len(cands), p=probs))
    return (int(cands_y[i]), int(cands_x[i]))

def accept_energy(
        Ns: List[int],
        Rgs: List[float],
        coords: np.ndarray,
        coords_new: np.ndarray,
        N_new: int,
        Rg_new: float,
        Df: float,
        k: float,
        cell_um: float,
        w_rg: float,
        w_slope: float,
        w_quota: float,
        quota_penalty: float,
        w_aniso: float = 1.0,
        Lmax: float = 0.75
    ) -> Tuple[float, float]:
    """Return (E_base, E_new) normalized energies; safe when data points are few."""
    def linear_anisotropy(coords: np.ndarray) -> float:
        """计算线性各向异性指标 L = λ_max / (λ_max + λ_min)。"""
        if coords.shape[0] < 3:
            return 0.5  # 几乎是点或线时默认最小各向异性
        cov = np.cov(coords.T)
        eigvals = np.linalg.eigvalsh(cov)
        lam_max, lam_min = np.max(eigvals), np.min(eigvals)
        return lam_max / (lam_max + lam_min + 1e-12)
    eps = 1e-9
    a = 0.6 * (k ** (1.0 / max(Df, 1e-6)))
    # ---------- 基础 Rg 能量 ----------
    Rg_target_base = a * cell_um * ((Ns[-1] if Ns else 1) ** (1.0 / max(Df, 1e-6)))
    Rg_base = (Rgs[-1] if Rgs else 0.0)
    e_rg_base = ((Rg_base - Rg_target_base) / (Rg_target_base + eps)) ** 2 if Ns else 0.0

    if len(Ns) >= 3:
        _, slope_base = estimate_fractal_dimension_2d(np.array(Ns, dtype=float), np.array(Rgs, dtype=float))
        e_slope_base = (slope_base - 1.0 / max(Df, 1e-6)) ** 2
    else:
        e_slope_base = 0.0

    # ---------- 新点 Rg 能量 ----------
    Rg_target_new = a * cell_um * (N_new ** (1.0 / max(Df, 1e-6)))
    e_rg_new = ((Rg_new - Rg_target_new) / (Rg_target_new + eps)) ** 2
    Ns2 = Ns + [N_new]
    Rgs2 = Rgs + [Rg_new]
    if len(Ns2) >= 3:
        _, slope_new = estimate_fractal_dimension_2d(np.array(Ns2, dtype=float), np.array(Rgs2, dtype=float))
        e_slope_new = (slope_new - 1.0 / max(Df, 1e-6)) ** 2
    else:
        e_slope_new = 0.0

    # ---------- 各向异性惩罚 ----------
    L_base = linear_anisotropy(coords)
    L_new = linear_anisotropy(coords_new)
    e_aniso_base = max(0.0, L_base - Lmax) ** 2
    e_aniso_new = max(0.0, L_new - Lmax) ** 2

    # ---------- 总能量 ----------
    E_base = w_rg * e_rg_base + w_slope * e_slope_base + w_aniso * e_aniso_base
    E_new = w_rg * e_rg_new + w_slope * e_slope_new + w_quota * (quota_penalty ** 2) + w_aniso * e_aniso_new
    return E_base, E_new

def growth_mvp(params: GrowthParams, scheduler: OnlinePSDScheduler) -> GrowthResult:
    rng = np.random.default_rng(params.seed)

    # unify RNG across scheduler and growth for full reproducibility
    scheduler.rng = rng

    # create initial grid with one seed pixel at center
    size0 = max(3, 2 * params.pad_margin + 3)
    grid = np.zeros((size0, size0), dtype=bool)
    labels = np.full((size0, size0), -1, dtype=int)
    cy, cx = size0 // 2, size0 // 2
    grid[cy, cx] = True
    labels[cy, cx] = 0  # seed as material A by convention
    # angle sector usage for angle balancing
    K_sectors = 12
    sector_usage = np.zeros(K_sectors, dtype=float)
    # account seed in scheduler so total area meets Np exactly
    try:
        scheduler.record_placed(("A", 1))
    except Exception:
        pass
    origin = (-(size0 // 2), -(size0 // 2))  # logical origin; not used heavily

    Ns: List[int] = [1]
    Rgs: List[float] = [radius_of_gyration_2d(np.array([[cx, cy]], dtype=float)) * params.cell_um]
    placed_particles: List[Tuple[Material, int, Tuple[int, int]]] = []

    # place until area budget is met
    defer_list: List[Tuple[Material, int]] = []
    fail_counter: Dict[Tuple[Material, int], int] = {}
    while scheduler.area_remaining() > 0:
        # keep margin
        grid, labels, shift = ensure_margin(grid, labels, params.pad_margin)
        if shift != (0, 0):
            origin = (origin[0] - shift[0], origin[1] - shift[1])

        item = scheduler.next_item()
        if item is None:
            break
        mat, s = item
        # prevent overshoot of area budget
        rem_area = scheduler.area_remaining()
        if s * s > rem_area:
            s = max(1, int(math.floor(math.sqrt(rem_area))))
            item = (mat, s)

        # candidate generation
        cands = candidate_top_lefts_for_s(grid, s)
        if not cands:
            # backoff and try later
            key = (mat, s)
            fail_counter[key] = fail_counter.get(key, 0) + 1
            if fail_counter[key] >= 20:
                defer_list.append(item)
            continue

        # sample a handful by outward bias
        beta = compute_beta(Ns[-1], Rgs[-1], params.Df, params.k, params.cell_um, params.adaptive_bias)
        # for MVP: pick num_trials random weighted candidates, evaluate and keep the best by Metropolis
        idxs = np.arange(len(cands))
        # sample with replacement according to weights
        picks: List[int] = []
        for _ in range(params.num_trials):
            i = choose_candidate_by_weight(grid, cands, s, params.cell_um, beta, rng,
                                           K=K_sectors, sector_usage=sector_usage)
            if i is None:
                break
            # convert candidate to index
            # we need index; just append position tuple
            picks.append(cands.index(i))
        if not picks:
            # fallback: pick one uniformly
            picks = [int(rng.integers(0, len(cands)))]

        # baseline energy
        Ebest = float("inf")
        best_block = None
        Ebase_best = None
        for pi in picks:
            (y, x) = cands[pi]
            # quota penalty according to config
            placed_now = scheduler.placed_counts.get(item, 0)
            target = scheduler.target_counts.get(item, 0.0)
            if params.bidirectional_quota_penalty:
                diff = (placed_now + 1) - target
                quota_penalty = abs(diff) / max(1.0, target)
            else:
                overfill = max(0.0, (placed_now + 1) - target)
                quota_penalty = overfill / max(1.0, target)
            # energy
            Rg_new = rg_after_placing(grid, (y, x, s), params.cell_um)
            N_new = Ns[-1] + s * s
            coords = np.stack(np.nonzero(grid)[::-1], axis=1).astype(float)  # (x,y)
            cx_try, cy_try = block_center(x, y, s)
            coords_new = np.vstack([coords,
                        np.tile(np.array([[cx_try, cy_try]], dtype=float), (s*s, 1))])
            Ebase, Enew = accept_energy(Ns, Rgs, coords, coords_new, 
                                        N_new, Rg_new, params.Df, params.k, params.cell_um,
                                        params.w_rg, params.w_slope, params.w_quota, quota_penalty)
            dE = Enew - Ebase
            # Metropolis acceptance probability if we were to choose this
            # we don't accept here; only record lowest Enew (or negative ΔE)
            if Enew < Ebest:
                Ebest = Enew
                best_block = (y, x, s)
                Ebase_best = Ebase

        # temperature schedule
        progress = min(1.0, Ns[-1] / max(1, params.Np))
        T = params.T0 * ((params.Tmin / max(params.T0, 1e-12)) ** progress)

        # evaluate best_block acceptance
        assert best_block is not None
        yb, xb, sb = best_block
        Rg_new = rg_after_placing(grid, (yb, xb, sb), params.cell_um)
        N_new = Ns[-1] + sb * sb
        # recompute energies for this block
        placed_now = scheduler.placed_counts.get(item, 0)
        target = scheduler.target_counts.get(item, 0.0)
        if params.bidirectional_quota_penalty:
            diff = (placed_now + 1) - target
            quota_penalty = abs(diff) / max(1.0, target)
        else:
            overfill = max(0.0, (placed_now + 1) - target)
            quota_penalty = overfill / max(1.0, target)
            
        coords = np.stack(np.nonzero(grid)[::-1], axis=1).astype(float)  # (x,y)
        cx_try, cy_try = block_center(xb, yb, sb)
        coords_new = np.vstack([
            coords,
            np.tile(np.array([[cx_try, cy_try]], dtype=float), (sb * sb, 1))
        ])
        Ebase, Enew = accept_energy(Ns, Rgs, coords, coords_new, 
                                    N_new, Rg_new, params.Df, params.k, params.cell_um,
                                    params.w_rg, params.w_slope, params.w_quota, quota_penalty)
        dE = Enew - Ebase
        accept = (dE <= 0) or (np.exp(-dE / max(T, 1e-9)) > rng.random())
        if not accept:
            # soft reject: occasionally we still place to avoid stalling, with small prob
            if rng.random() < 0.05:
                accept = True
        if not accept:
            # backoff and try later
            key = (mat, s)
            fail_counter[key] = fail_counter.get(key, 0) + 1
            if fail_counter[key] >= 20:
                defer_list.append(item)
            continue

        yy_c, xx_c = np.nonzero(grid)  # pre-commit centroid
        com_x, com_y = float(xx_c.mean()), float(yy_c.mean())
        cxp, cyp = block_center(xb, yb, sb)
        theta_p = math.atan2(cyp - com_y, cxp - com_x)
        sector_p = int(math.floor((theta_p + math.pi) / (2 * math.pi / K_sectors))) % K_sectors
        sector_usage[sector_p] += 1.0
        # commit placement
        grid[yb:yb+sb, xb:xb+sb] = True
        labels[yb:yb+sb, xb:xb+sb] = 0 if mat == "A" else 1
        scheduler.record_placed(item)
        Ns.append(N_new)
        Rgs.append(Rg_new)
        placed_particles.append((mat, sb, (xb, yb)))

    # finalize: if residual area remains (due to skipping), fill with s=1 pixels alternating materials to match composition
    residual = scheduler.area_remaining()
    if residual > 0:
        if params.endfill_by_psd_deficit:
            # PSD-deficit-driven end-fill
            def current_deficits():
                dmap = scheduler.deficits()  # positive deficits only
                return sorted(dmap.items(), key=lambda kv: (kv[1], kv[0][1]), reverse=True)

            attempts_without_progress = 0
            while residual > 0 and attempts_without_progress < 500:
                progress = False
                for (mat_s, deficit) in current_deficits():
                    if deficit <= 1e-9:
                        continue
                    mat, s = mat_s
                    if s * s > residual:
                        continue
                    cands = candidate_top_lefts_for_s(grid, s)
                    if not cands:
                        continue
                    beta = compute_beta(Ns[-1], Rgs[-1], params.Df, params.k, params.cell_um, params.adaptive_bias)
                    pos = choose_candidate_by_weight(grid, cands, s, params.cell_um, beta, rng)
                    if pos is None:
                        continue
                    yb, xb = pos
                    grid[yb:yb+s, xb:xb+s] = True
                    labels[yb:yb+s, xb:xb+s] = 0 if mat == "A" else 1
                    # update sector usage for angle balancing
                    yy_c, xx_c = np.nonzero(grid)
                    com_x, com_y = float(xx_c.mean()), float(yy_c.mean())
                    cxp, cyp = block_center(xb, yb, s)
                    theta_p = math.atan2(cyp - com_y, cxp - com_x)
                    sector_p = int(math.floor((theta_p + math.pi) / (2 * math.pi / K_sectors))) % K_sectors
                    sector_usage[sector_p] += 1.0
                    scheduler.record_placed((mat, s))
                    Ns.append(Ns[-1] + s * s)
                    if (residual % 16) == 0:
                        yy, xx = np.nonzero(grid)
                        Rgs.append(radius_of_gyration_2d(np.stack([xx, yy], axis=1).astype(float)) * params.cell_um)
                    residual -= s * s
                    progress = True
                    if residual <= 0:
                        break
                if not progress:
                    attempts_without_progress += 1
                    # fallback: try s=1 near boundary
                    touch = neighbor_touch_mask(grid)
                    ys, xs = np.nonzero(touch)
                    if len(xs) > 0 and residual > 0:
                        comp = scheduler.composition_area()
                        targetA = scheduler._A_area / scheduler.Np
                        errA = targetA - comp["A"]
                        mat = "A" if (errA > 0) else "B"
                        idx = int(rng.integers(0, len(xs)))
                        y, x = ys[idx], xs[idx]
                        if not grid[y, x]:
                            grid[y, x] = True
                            labels[y, x] = 0 if mat == "A" else 1
                            scheduler.record_placed((mat, 1))
                            Ns.append(Ns[-1] + 1)
                            residual -= 1
            # ensure final Rg appended
            yy, xx = np.nonzero(grid)
            Rg_final = radius_of_gyration_2d(np.stack([xx, yy], axis=1).astype(float)) * params.cell_um
            if len(Rgs) == len(Ns) - 1:
                Rgs.append(Rg_final)
        else:
            # original simple s=1 end-fill
            while residual > 0:
                grid, labels, shift = ensure_margin(grid, labels, params.pad_margin)
                if shift != (0, 0):
                    origin = (origin[0] - shift[0], origin[1] - shift[1])
                touch = neighbor_touch_mask(grid)
                ys, xs = np.nonzero(touch)
                if len(xs) == 0:
                    grid, labels, _ = ensure_margin(grid, labels, params.pad_margin + 8)
                    touch = neighbor_touch_mask(grid)
                    ys, xs = np.nonzero(touch)
                    if len(xs) == 0:
                        break
                comp = scheduler.composition_area()
                targetA = scheduler._A_area / scheduler.Np
                errA = targetA - comp["A"]
                mat = "A" if (errA > 0) else "B"
                idx = int(rng.integers(0, len(xs)))
                y, x = ys[idx], xs[idx]
                if grid[y, x]:
                    continue
                grid[y, x] = True
                labels[y, x] = 0 if mat == "A" else 1
                scheduler.record_placed((mat, 1))
                Ns.append(Ns[-1] + 1)
                if (residual % 32) == 0:
                    yy, xx = np.nonzero(grid)
                    Rgs.append(radius_of_gyration_2d(np.stack([xx, yy], axis=1).astype(float)) * params.cell_um)
                residual -= 1
            yy, xx = np.nonzero(grid)
            Rg_final = radius_of_gyration_2d(np.stack([xx, yy], axis=1).astype(float)) * params.cell_um
            if len(Rgs) == len(Ns) - 1:
                Rgs.append(Rg_final)

    # estimate Df
    Df_est, slope = estimate_fractal_dimension_2d(np.array(Ns, dtype=float), np.array(Rgs, dtype=float))

    # --- crop to occupied bbox (remove outer padding) ---
    grid, labels, shift_final = crop_to_bbox(grid, labels, margin=0)
    origin = (origin[0] + shift_final[0], origin[1] + shift_final[1])

    return GrowthResult(
        grid=grid.astype(np.uint8),
        labels=labels.astype(np.int8),
        origin=(origin[0], origin[1]),
        Ns=Ns,
        Rgs=Rgs,
        Df_est=Df_est,
        slope=slope,
        scheduler=scheduler,
        placed_particles=placed_particles,
    )

# ---------- Convenience: evaluation and plotting ----------
def compare_psd_input_vs_achieved(scheduler: OnlinePSDScheduler) -> Dict[str, Dict]:
    """Return per-material dict with target number-PSD (by s), achieved number-PSD (by s), and L1 distance."""
    tgt = scheduler.input_psd_by_number()
    ach = scheduler.achieved_psd_by_number()
    out = {}
    for m in ["A", "B"]:
        # align keys
        keys = sorted(set(list(tgt[m].keys()) + list(ach[m].keys())))
        tv = 0.0
        rows = []
        for s in keys:
            p_t = float(tgt[m].get(s, 0.0))
            p_a = float(ach[m].get(s, 0.0))
            tv += 0.5 * abs(p_t - p_a)
            rows.append((s, p_t, p_a))
        out[m] = {"table": rows, "total_variation": tv}
    return out

def plot_labels_grid(grid: np.ndarray, labels: np.ndarray, origin: Tuple[int,int]):
    """
    Show occupancy (grid) with material labels overlay.
    - empty = background
    - A (0) and B (1) are colored differently
    """
    H, W = grid.shape
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # base: occupancy
    ax.imshow(grid, origin='lower', interpolation='nearest', alpha=0.35)

    # overlay: labels where occupied
    show = np.ma.masked_where(labels < 0, labels)
    ax.imshow(show, origin='lower', interpolation='nearest')

    x_min, y_min = origin
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_title("Lattice aggregate with A/B labels")
    ax.set_xlabel(f"x (offset {x_min})")
    ax.set_ylabel(f"y (offset {y_min})")

    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', linewidth=0.3)

    plt.tight_layout()
    plt.show()

def load_psd_from_npz(npz_path: str, diam_key: str, psd_key: str) -> PSD:
    """Load PSD from an .npz file.
    Parameters
    ----------
    npz_path : str
        Path to the .npz file.
    diam_key : str
        Key for the diameter array (µm), length N.
    psd_key : str
        Key for the PSD array, length N. If `cumulative=True`, this is *Q0* (cumulative undersize).
    cumulative : bool, default True
        Whether the provided PSD is cumulative (Q0). If False, it is treated as differential number-PSD (probabilities or %).
    """
    data = np.load(npz_path)
    if diam_key not in data or psd_key not in data:
        raise KeyError(f"Keys '{diam_key}' and/or '{psd_key}' not found in npz file")
    d = np.asarray(data[diam_key], dtype=float).ravel()
    q = np.asarray(data[psd_key], dtype=float).ravel()
    # sort by diameter
    order = np.argsort(d)
    d = d[order]
    q = q[order]
    return PSD.from_cumulative(d, q)


def save_example_psd_npz(npz_path: str) -> None:
    """Create a small example .npz containing *cumulative* number distributions Q0 for A/B.
    Arrays written: 'diam_A', 'Q0_A', 'diam_B', 'Q0_B'."""
    d = np.arange(1.0, 21.0, 1.0)
    # synthetic differential PSDs
    pA = np.exp(-0.2 * (d - 1.0))
    pA /= pA.sum()
    pB = np.exp(-0.5 * ((d - 10.0) ** 2) / (2.0 ** 2))
    pB /= pB.sum()
    # to cumulative Q0 in [0,1]
    Q0_A = np.cumsum(pA); Q0_A /= Q0_A[-1]
    Q0_B = np.cumsum(pB); Q0_B /= Q0_B[-1]
    np.savez(npz_path, diam_A=d, Q0_A=Q0_A, diam_B=d, Q0_B=Q0_B)

# ------------- example usage -------------
if __name__ == "__main__":
    # --- Example pipeline using .npz with cumulative Q0 ---
    cell_um = 1.0
    npz_path = "psd_example.npz"

    # 1) Save synthetic example cumulative PSDs to .npz
    save_example_psd_npz(npz_path)

    # 2) Load PSDs from .npz as *cumulative* Q0
    psdA = load_psd_from_npz(npz_path, diam_key="diam_A", psd_key="Q0_A")
    psdB = load_psd_from_npz(npz_path, diam_key="diam_B", psd_key="Q0_B")

    Np = 2000
    params = GrowthParams(
        Np=Np, Df=1.6, k=1.0, cell_um=cell_um,
        w_rg=1.0, w_slope=0.25, w_quota=0.1,
        num_trials=24, seed=42, pad_margin=16,
        # True时，对“超配/欠配”都惩罚；即 |placed+1 - target| / max(1, target)
        bidirectional_quota_penalty=True,
        # False时，一律使用s=1补充缺口，True时，收尾阶段按 PSD 缺口优先
        endfill_by_psd_deficit=True,
    )
    scheduler = OnlinePSDScheduler(Np=Np, frac_A=0.6, psd_A=psdA, psd_B=psdB, 
                                   cell_um=cell_um, s_min=1, s_max=20,
                                   # Hamilton 最大余数法
                                   integer_quota=True)

    result = growth_mvp(params, scheduler)

    print(f"[Result] N={int(result.grid.sum())}, Df_est≈{result.Df_est:.3f}, slope≈{result.slope:.3f}, comp={scheduler.composition_area()}")
    cmp = compare_psd_input_vs_achieved(result.scheduler)
    for m in ["A", "B"]:
        print(f"Material {m}: TV distance (number-PSD) = {cmp[m]['total_variation']:.3f}")
        # print table head
        head = sorted(cmp[m]['table'], key=lambda t: t[0])[:5]
        print(" ", head, "...")
    plot_labels_grid(result.grid, result.labels, result.origin)
