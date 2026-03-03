# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:16:37 2025

@author: px2030
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict
from .meta import GridMeta
from .func_jit import float_gcd

class GridFactory:
    def _choose_dims(self, total_units: int, aspect_ratio: float) -> Tuple[int, int]:
        if total_units <= 0: return 0, 0
        ar = float(max(aspect_ratio, 1e-9))
        if ar >= 1.0:
            W = int(np.ceil(np.sqrt(total_units * ar))); H = int(np.ceil(total_units / W))
        else:
            H = int(np.ceil(np.sqrt(total_units / ar))); W = int(np.ceil(total_units / H))
        while H * W < total_units:
            if ar >= 1.0: W += 1
            else: H += 1
        return H, W

    def make(self, A: float, X1: float, X2: float, *, A0: float = 0.0,
             aspect_ratio: float = 1.0, int_bre: float = 0.0, seed: int | None = None):
        X1 = float(X1); X2 = float(X2)
        s = X1 + X2
        if s <= 0: raise ValueError("X1+X2 must be > 0")
        X1 /= s; X2 /= s
        if A0 == 0 or (A0 > (A * X1) / 2 and A0 > (A * X2) / 2):
            A0 = float_gcd(A * X1, A * X2)
        N1 = int((A * X1) // A0); N2 = int((A * X2) // A0)
        R1 = (A * X1) % A0; R2 = (A * X2) % A0
        total_units = N1 + N2
        H, W = self._choose_dims(total_units, aspect_ratio)
        if int_bre <= 0: int_bre_len = 1
        else: int_bre_len = int(np.ceil(max(H, W) * float(int_bre)))
        rng = np.random.default_rng(seed)
        flat = np.empty(total_units, dtype=np.uint8)
        flat[:N1] = 1; flat[N1:] = 2; rng.shuffle(flat)
        M = np.zeros((H, W), dtype=np.uint8)
        if total_units > 0: M.flat[:total_units] = flat
        # Hbond = np.full((H, max(W - 1, 0)), -1, dtype=np.int16)
        # if W >= 2:
        #     left = M[:, :-1]; right = M[:, 1:]
        #     both = (left > 0) & (right > 0)
        #     same1 = both & (left == 1) & (right == 1)
        #     same2 = both & (left == 2) & (right == 2)
        #     diff = both & ~(same1 | same2)
        #     Hbond[same1] = 11; Hbond[same2] = 22; Hbond[diff] = 12
        # Vbond = np.full((max(H - 1, 0), W), -1, dtype=np.int16)
        # if H >= 2:
        #     up = M[:-1, :]; dn = M[1:, :]
        #     both = (up > 0) & (dn > 0)
        #     same1 = both & (up == 1) & (dn == 1)
        #     same2 = both & (up == 2) & (dn == 2)
        #     diff = both & ~(same1 | same2)
        #     Vbond[same1] = 11; Vbond[same2] = 22; Vbond[diff] = 12
        # n11 = int((Hbond == 11).sum() + (Vbond == 11).sum())
        # n12 = int((Hbond == 12).sum() + (Vbond == 12).sum())
        # n22 = int((Hbond == 22).sum() + (Vbond == 22).sum())
        Hbond, Vbond, bond_counts = self._build_bonds_from_M(M)
        meta = GridMeta(H=H, W=W, A0=float(A0), N1=N1, N2=N2,
                        R=(float(R1), float(R2)), total_units=total_units,
                        aspect_ratio=float(aspect_ratio),
                        int_bre=float(int_bre), int_bre_len=int(int_bre_len))
        return M, Hbond, Vbond, meta, bond_counts
    
    @staticmethod
    def _build_bonds_from_M(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """
        给定内部表示的 M(0=空,1=A,2=B)，生成 Hbond/Vbond 以及计数。
        """
        H, W = M.shape
        Hbond = np.full((H, max(W - 1, 0)), -1, dtype=np.int16)
        if W >= 2:
            left = M[:, :-1]
            right = M[:, 1:]
            both = (left > 0) & (right > 0)
            same1 = both & (left == 1) & (right == 1)
            same2 = both & (left == 2) & (right == 2)
            diff  = both & ~(same1 | same2)
            Hbond[same1] = 11
            Hbond[same2] = 22
            Hbond[diff]  = 12
    
        Vbond = np.full((max(H - 1, 0), W), -1, dtype=np.int16)
        if H >= 2:
            up = M[:-1, :]
            dn = M[1:, :]
            both = (up > 0) & (dn > 0)
            same1 = both & (up == 1) & (dn == 1)
            same2 = both & (up == 2) & (dn == 2)
            diff  = both & ~(same1 | same2)
            Vbond[same1] = 11
            Vbond[same2] = 22
            Vbond[diff]  = 12
    
        n11 = int((Hbond == 11).sum() + (Vbond == 11).sum())
        n12 = int((Hbond == 12).sum() + (Vbond == 12).sum())
        n22 = int((Hbond == 22).sum() + (Vbond == 22).sum())
        return Hbond, Vbond, {11: n11, 12: n12, 22: n22}

    def make_from_array(
        self,
        mat: np.ndarray,
        *,
        a_code: int = 0,
        b_code: int = 1,
        empty_code: int = -1,
        A0: float = 1.0,
        int_bre: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, GridMeta, Dict[int, int]]:
        if mat.ndim != 2:
            raise ValueError("mat must be 2D array")
        H, W = mat.shape

        M = np.zeros((H, W), dtype=np.uint8)
        M[mat == a_code] = 1
        M[mat == b_code] = 2
        M[mat == empty_code] = 0

        Hbond, Vbond, bond_counts = self._build_bonds_from_M(M)

        N1 = int((M == 1).sum())
        N2 = int((M == 2).sum())
        total_units = N1 + N2
        aspect_ratio = (float(W) / float(H)) if H > 0 else 1.0

        if int_bre <= 0:
            int_bre_len = 1
        else:
            int_bre_len = int(np.ceil(max(H, W) * float(int_bre)))

        meta = GridMeta(
            H=H, W=W, A0=float(A0),
            N1=N1, N2=N2, R=(0.0, 0.0),
            total_units=total_units,
            aspect_ratio=float(aspect_ratio),
            int_bre=float(int_bre), int_bre_len=int(int_bre_len)
        )
        return M, Hbond, Vbond, meta, bond_counts