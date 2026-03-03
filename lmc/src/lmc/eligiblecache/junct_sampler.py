# -*- coding: utf-8 -*-
"""
Junction-domain samplers:
  - JunctUniSampler: globally uniform over eligible junctions;
  - JunctUniBoundarySampler: uniform over junctions that are exterior-connected
    via a dynamic union-find (ExteriorUF).

Both classes share the JunctionSamplerBase from ._base and use helpers in ._utils.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np

from ._base import JunctionSamplerBase
from ._utils import (
    jid_of, 
    ExteriorUF,
    has_unbroken_incident,
    neighbors4,
    compute_eligible_mask,
)

# ----------------------------------------------------------------------------
# Global uniform sampler over junctions
# ----------------------------------------------------------------------------

class JunctUniSampler(JunctionSamplerBase):
    """
    Eligibility: any junction with at least one intact incident bond.
    """

    def __init__(self, H: int, W: int) -> None:
        super().__init__(H, W)

    def build_initial(self, Hbond: np.ndarray, Vbond: np.ndarray) -> None:
        """
        Fill the set with all eligible junctions.
        """
        H = int(Hbond.shape[0])
        W = int(Vbond.shape[1])
        assert H == self.H and W == self.W, "Shape mismatch."
        for r in range(H + 1):
            for c in range(W + 1):
                if has_unbroken_incident(Hbond, Vbond, r, c):
                    self._add_id(jid_of(r, c, self.W))

    # -- JunctionSamplerBase hooks --

    def _is_eligible(self, Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int) -> bool:
        return has_unbroken_incident(Hbond, Vbond, r, c)

    def _build_touch_set(self, rcs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # small 4-neighborhood around touched nodes
        H, W = self.H, self.W
        seen = set()
        out: List[Tuple[int, int]] = []
        for r, c in rcs:
            for rc in neighbors4(int(r), int(c), H, W):
                if rc in seen: continue
                seen.add(rc)
                out.append(rc)
        return out


# ----------------------------------------------------------------------------
# Boundary-connected uniform sampler over junctions
# ----------------------------------------------------------------------------

class JunctUniBoundarySampler(JunctionSamplerBase):
    """
    Eligibility: (has_unbroken_incident) AND (ExteriorUF.is_exterior_connected).
    The ExteriorUF:
      * initially connects frame to EXTERIOR and unions all broken bonds;
      * incrementally unions newly broken incident bonds near the touch set.
    """

    def __init__(self, H: int, W: int) -> None:
        super().__init__(H, W)
        self.uf = ExteriorUF(H, W)

    # public helper (optional, for callers that have a path_info list)
    def on_bonds_broken(self, path_info: Sequence[Tuple[int, int, int, int]]) -> None:
        """
        Optionally union all endpoints of newly broken bonds before a refresh.
        If your caller only uses `recompute_at`, you may skip calling this; the
        sampler will also union around the touched junctions in `_pre_refresh_hook`.
        """
        self.uf.union_many_from_path_info(path_info)

    def build_initial(self, Hbond: np.ndarray, Vbond: np.ndarray) -> None:
        H = int(Hbond.shape[0])
        W = int(Vbond.shape[1])
        assert H == self.H and W == self.W, "Shape mismatch."
    
        # 先把“外框灌水”，再让水沿已断裂的 bond 自然贯通
        self.uf.union_broken_from_arrays(Hbond, Vbond)
    
        # 统一口径：eligible := 外连通 ∧ 有完整相邻键
        eligible = compute_eligible_mask(Hbond, Vbond, self.uf)
    
        # 把所有 eligible junction 加入集合（mask/ids/pos 三件套在基类里维护）
        for r in range(H + 1):
            for c in range(W + 1):
                if eligible[r, c]:
                    self._add_id(jid_of(r, c, self.W))

    # -- JunctionSamplerBase hooks --

    def _pre_refresh_hook(self,
                          Hbond: np.ndarray,
                          Vbond: np.ndarray,
                          touch: Sequence[Tuple[int, int]]) -> None:
        # Newly broken bonds near the touch set should be unioned so connectivity is up-to-date.
        for r, c in touch:
            self.uf.union_broken_incident(Hbond, Vbond, int(r), int(c))

    def _is_eligible(self, Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int) -> bool:
        return has_unbroken_incident(Hbond, Vbond, r, c) and self.uf.is_exterior_connected(r, c)

    def _build_touch_set(self, rcs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        H, W = self.H, self.W
        seen = set()
        out: List[Tuple[int, int]] = []
        for r, c in rcs:
            for rc in neighbors4(int(r), int(c), H, W):
                if rc in seen: continue
                seen.add(rc)
                out.append(rc)
        return out
