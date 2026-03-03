# -*- coding: utf-8 -*-
"""
Utilities shared by samplers in the fracture simulator.

This file intentionally groups:
  * Geometry helpers (bond <-> junction conversion; neighbor enumeration);
  * Lightweight predicates (e.g., has_unbroken_incident);
  * Two boundary trackers:
      - ExteriorUF: union-find over junctions with a virtual EXTERIOR node;
      - BoundaryMask: fast boolean frontier used by boundary-only bond samplers.

Shape conventions (single source of truth)
------------------------------------------
Let the junction lattice be (H+1) x (W+1).
Bonds live on the edges between junctions with arrays:

  Hbond: shape (H,   W-1)   # horizontal edges within rows
  Vbond: shape (H-1, W)     # vertical edges within columns

Indexing (axis, i, j) and endpoints (corrected mapping):
  axis = 0  -> Hbond[i, j] between junctions (i, j+1) and (i+1, j+1)  # vertical
  axis = 1  -> Vbond[i, j] between junctions (i+1, j) and (i+1, j+1)  # horizontal
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from ..func_jit import uf_find, uf_union

# ---------------------------------------------------------------------------
# Geometry helpers (JIT-safe 版本)
# ---------------------------------------------------------------------------

@njit(cache=True)
def jid_of(r: int, c: int, W: int) -> int:
    return int(r) * (int(W) + 1) + int(c)

@njit(cache=True)
def rc_of(jid: int, W: int) -> Tuple[int, int]:
    # Numba 支持返回扁平 tuple；这里是简单的 div 与 mod
    j = int(jid); w1 = int(W) + 1
    return j // w1, j % w1

@njit(cache=True)
def bond_endpoints_flat_jit(H: int, W: int, axis: int, i: int, j: int) -> Tuple[int, int, int, int]:
    # 扁平返回，避免嵌套 tuple
    if axis == 0:  # Hbond[i,j] between (i, j+1) and (i+1, j+1)
        return i, j + 1, i + 1, j + 1
    else:          # Vbond[i,j] between (i+1, j) and (i+1, j+1)
        return i + 1, j, i + 1, j + 1

def bond_endpoints(H: int, W: int, axis: int, i: int, j: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # Python 包装，继续提供旧接口 ((r0,c0),(r1,c1))
    r0, c0, r1, c1 = bond_endpoints_flat_jit(H, W, int(axis), int(i), int(j))
    return (int(r0), int(c0)), (int(r1), int(c1))

def incident_bonds_keys(r: int, c: int, H: int, W: int) -> Iterator[Tuple[int, int, int]]:
    r, c, H, W = int(r), int(c), int(H), int(W)
    if r > 0 and 1 <= c <= W - 1:   # Up: Hbond[r-1, c-1]
        yield (0, r - 1, c - 1)
    if r < H and 1 <= c <= W - 1:   # Down: Hbond[r, c-1]
        yield (0, r, c - 1)
    if c > 0 and 1 <= r <= H - 1:   # Left: Vbond[r-1, c-1]
        yield (1, r - 1, c - 1)
    if c < W and 1 <= r <= H - 1:   # Right: Vbond[r-1, c]
        yield (1, r - 1, c)

def incident_intact_bonds(Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int) -> List[Tuple[int, int, int, int]]:
    H = int(Hbond.shape[0]); W = int(Vbond.shape[1])
    out: List[Tuple[int, int, int, int]] = []
    for axis, i, j in incident_bonds_keys(r, c, H, W):
        t = int(Hbond[i, j]) if axis == 0 else int(Vbond[i, j])
        if t != -1:
            out.append((axis, i, j, t))
    return out

@njit(cache=True)
def has_unbroken_incident_jit(Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int) -> bool:
    H = int(Hbond.shape[0]); W = int(Vbond.shape[1])
    if r > 0 and 1 <= c <= W - 1 and int(Hbond[r - 1, c - 1]) != -1: return True
    if r < H and 1 <= c <= W - 1 and int(Hbond[r,     c - 1]) != -1: return True
    if c > 0 and 1 <= r <= H - 1 and int(Vbond[r - 1, c - 1]) != -1: return True
    if c < W and 1 <= r <= H - 1 and int(Vbond[r - 1, c    ]) != -1: return True
    return False

def has_unbroken_incident(Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int) -> bool:
    return bool(has_unbroken_incident_jit(Hbond, Vbond, int(r), int(c)))

def dedup_rcs(rcs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set(); out: List[Tuple[int, int]] = []
    for r, c in rcs:
        rc = (int(r), int(c))
        if rc in seen: continue
        seen.add(rc); out.append(rc)
    return out

def neighbors4(r: int, c: int, H: int, W: int) -> List[Tuple[int, int]]:
    r = int(r); c = int(c); H = int(H); W = int(W)
    out = [(r, c)]
    if r > 0: out.append((r - 1, c))
    if r < H: out.append((r + 1, c))
    if c > 0: out.append((r, c - 1))
    if c < W: out.append((r, c + 1))
    # 去重保持顺序
    seen = set(); res: List[Tuple[int, int]] = []
    for rc in out:
        if rc in seen: continue
        seen.add(rc); res.append(rc)
    return res

# ---------------------------------------------------------------------------
# UF 辅助 JIT 内核
# ---------------------------------------------------------------------------

@njit(cache=True)
def is_exterior_connected_jit(parent: np.ndarray, ext_id: int, r: int, c: int, W: int) -> bool:
    return uf_find(parent, jid_of(int(r), int(c), int(W))) == uf_find(parent, int(ext_id))

@njit(cache=True)
def uf_connect_frame_to_exterior_jit(parent: np.ndarray, rank: np.ndarray, H: int, W: int, ext_id: int) -> None:
    H = int(H); W = int(W); E = int(ext_id)
    # 顶/底
    for c in range(W + 1):
        uf_union(parent, rank, jid_of(0, c, W), E)
        uf_union(parent, rank, jid_of(H, c, W), E)
    # 左/右
    for r in range(H + 1):
        uf_union(parent, rank, jid_of(r, 0, W), E)
        uf_union(parent, rank, jid_of(r, W, W), E)

@njit(cache=True)
def uf_union_bond_jit(parent: np.ndarray, rank: np.ndarray, H: int, W: int, axis: int, i: int, j: int) -> None:
    r0, c0, r1, c1 = bond_endpoints_flat_jit(H, W, axis, i, j)
    uf_union(parent, rank, jid_of(r0, c0, W), jid_of(r1, c1, W))

@njit(cache=True)
def uf_union_broken_from_arrays_jit(parent: np.ndarray, rank: np.ndarray,
                                    Hbond: np.ndarray, Vbond: np.ndarray, H: int, W: int) -> None:
    # Hbond 扫描
    for i in range(Hbond.shape[0]):
        for j in range(Hbond.shape[1]):
            if int(Hbond[i, j]) == -1:
                uf_union_bond_jit(parent, rank, H, W, 0, i, j)
    # Vbond 扫描
    for i in range(Vbond.shape[0]):
        for j in range(Vbond.shape[1]):
            if int(Vbond[i, j]) == -1:
                uf_union_bond_jit(parent, rank, H, W, 1, i, j)

@njit(cache=True)
def uf_union_broken_incident_jit(parent: np.ndarray, rank: np.ndarray,
                                 Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int, H: int, W: int) -> None:
    # 直接四向检查（避免 Python 生成器）
    if r > 0 and 1 <= c <= W - 1 and int(Hbond[r - 1, c - 1]) == -1:
        uf_union_bond_jit(parent, rank, H, W, 0, r - 1, c - 1)
    if r < H and 1 <= c <= W - 1 and int(Hbond[r,     c - 1]) == -1:
        uf_union_bond_jit(parent, rank, H, W, 0, r,     c - 1)
    if c > 0 and 1 <= r <= H - 1 and int(Vbond[r - 1, c - 1]) == -1:
        uf_union_bond_jit(parent, rank, H, W, 1, r - 1, c - 1)
    if c < W and 1 <= r <= H - 1 and int(Vbond[r - 1, c    ]) == -1:
        uf_union_bond_jit(parent, rank, H, W, 1, r - 1, c    )

# ---------------------------------------------------------------------------
# Eligible 掩码（批量 JIT）
# ---------------------------------------------------------------------------

@njit(cache=True)
def compute_eligible_mask_jit(Hbond: np.ndarray, Vbond: np.ndarray,
                              parent: np.ndarray, W: int, ext_id: int) -> np.ndarray:
    H = int(Hbond.shape[0]); W = int(W)
    mask = np.zeros((H + 1, W + 1), dtype=np.bool_)
    for r in range(H + 1):
        for c in range(W + 1):
            # 先判外连通，再做 incident intact 检查，少读几次内存
            if is_exterior_connected_jit(parent, ext_id, r, c, W):
                if has_unbroken_incident_jit(Hbond, Vbond, r, c):
                    mask[r, c] = True
    return mask

def compute_eligible_mask(Hbond: np.ndarray, Vbond: np.ndarray, uf: "ExteriorUF") -> np.ndarray:
    return compute_eligible_mask_jit(Hbond, Vbond, uf.parent, uf.W, uf.ext_id)

# ---------------------------------------------------------------------------
# Exterior Union-Find over junction lattice
# ---------------------------------------------------------------------------

class ExteriorUF:
    """
    Union-Find that treats the rectangular frame as connected to a virtual EXTERIOR.
    It supports incremental union of newly broken bonds (endpoints get merged).
    """

    def __init__(self, H: int, W: int, M: Optional[np.ndarray] = None) -> None:
        self.H, self.W = int(H), int(W)
        n = (self.H + 1) * (self.W + 1) + 1  # +1 for EXTERIOR
        # 统一 int32，配合 JIT 内核
        self.parent = np.arange(n, dtype=np.int32)
        self.rank   = np.zeros(n, dtype=np.int32)
        self.ext_id = np.int32(n - 1)
        # 连接外框
        uf_connect_frame_to_exterior_jit(self.parent, self.rank, self.H, self.W, self.ext_id)

    # -- helpers --

    def _connect_frame_to_exterior(self) -> None:
        # 若仍需旧接口，直接走 JIT 内核
        uf_connect_frame_to_exterior_jit(self.parent, self.rank, self.H, self.W, self.ext_id)

    def union_bond(self, axis: int, i: int, j: int) -> None:
        uf_union_bond_jit(self.parent, self.rank, self.H, self.W, int(axis), int(i), int(j))

    def union_broken_from_arrays(self, Hbond: np.ndarray, Vbond: np.ndarray) -> None:
        uf_union_broken_from_arrays_jit(self.parent, self.rank, Hbond, Vbond, self.H, self.W)

    def union_broken_incident(self, Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int) -> None:
        uf_union_broken_incident_jit(self.parent, self.rank, Hbond, Vbond, int(r), int(c), self.H, self.W)

    def union_many_from_path_info(self, path_info: Sequence[Tuple[int, int, int, int]]) -> None:
        # 维持 Python 接口（list[tuple]），逐条 union；如需更快可在调用处打包成数组走 JIT 批量核
        for axis, i, j, old_t in path_info:
            if int(old_t) == -1:
                continue
            uf_union_bond_jit(self.parent, self.rank, self.H, self.W, int(axis), int(i), int(j))

    # -- queries --

    def is_exterior_connected(self, r: int, c: int) -> bool:
        return bool(is_exterior_connected_jit(self.parent, self.ext_id, int(r), int(c), self.W))

