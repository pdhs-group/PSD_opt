
# -*- coding: utf-8 -*-
"""
Base scaffolding for start-junction sampling in the LMC fracture simulator.

This module intentionally concentrates:
  - lightweight type aliases (BondKey / PathInfo);
  - a performant bucket container for bonds;
  - abstract base classes for junction-based samplers and bond-weighted samplers.

Implementation notes
--------------------
* Axis convention:
    axis = 0 -> Hbond[i, j]  (between junctions (i, j) and (i, j+1))
    axis = 1 -> Vbond[i, j]  (between junctions (i, j) and (i+1, j))

* Bond types: {11, 12, 22}; -1 means broken.

* This file has no geometry helpers (e.g., bond_endpoints, boundary tests).
  Those belong in `._utils` so both samplers can import a single source of truth.

Author: refactor scaffold generated with user's consolidation plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Protocol, runtime_checkable

import numpy as np

# from ._utils import (
#     jid_of,
#     rc_of,
# )


# -------------------------
# Type aliases
# -------------------------

Axis = int                               # 0 (Hbond) or 1 (Vbond)
BondKey = Tuple[Axis, int, int]          # (axis, i, j)
PathInfoItem = Tuple[Axis, int, int, int]  # (axis, i, j, old_type)
PathInfo = List[PathInfoItem]            # a crack path as a list of items


# -------------------------
# Lightweight containers
# -------------------------

@dataclass
class _Bucket:
    """
    A compact "swap-pop" set-like container with O(1) add/discard/sample.

    Keys are (axis, i, j). Internally maintains:
      - a dense list `items`;
      - a hash map `pos` from key -> index in `items`.

    Methods:
      add(key) / add_many(keys)
      discard(key) -> bool
      sample_uniform(rng) -> key
      __len__()
      __contains__(key)
      clear()
    """
    items: List[BondKey]
    pos: Dict[BondKey, int]

    def __init__(self) -> None:
        self.items = []
        self.pos = {}

    def add(self, key: BondKey) -> None:
        if key in self.pos:
            return
        self.pos[key] = len(self.items)
        self.items.append(key)

    def add_many(self, keys: Iterable[BondKey]) -> None:
        # Still relies on hashing to deduplicate, but reduces Python overhead.
        p = self.pos
        it = self.items
        for k in keys:
            if k in p:
                continue
            p[k] = len(it)
            it.append(k)

    def discard(self, key: BondKey) -> bool:
        """
        Remove a key if present; return True if removed, False otherwise.
        """
        p = self.pos.pop(key, None)
        if p is None:
            return False
        last = self.items[-1]
        if p != len(self.items) - 1:
            self.items[p] = last
            self.pos[last] = p
        self.items.pop()
        return True

    def sample_uniform(self, rng: np.random.Generator) -> BondKey:
        if not self.items:
            raise RuntimeError("Bucket is empty.")
        idx = int(rng.integers(0, len(self.items)))
        return self.items[idx]

    def __len__(self) -> int:  # noqa: D401
        """Number of keys in the bucket."""
        return len(self.items)

    def __contains__(self, key: BondKey) -> bool:
        return key in self.pos

    def clear(self) -> None:
        self.items.clear()
        self.pos.clear()


# -------------------------
# Junction-set mixin & base
# -------------------------

class _JunctionSetMixin:
    """
    Mix-in providing a uniform set container over junctions with:
      - boolean mask (H+1, W+1),
      - dense numpy array `ids` of flattened junction indices,
      - dense array `pos` from flat index -> position in `ids` (or -1 if absent).

    Compared with the old dict-based version:
      - avoids Python dict overhead;
      - supports bulk reset from a boolean mask;
      - avoids redundant jid -> (r,c) conversion when caller already has (r,c).
    """

    def __init__(self, H: int, W: int) -> None:
        self.H = int(H)
        self.W = int(W)
        self._W1 = self.W + 1
        self._n_junctions = (self.H + 1) * (self.W + 1)

        self.mask = np.zeros((self.H + 1, self.W + 1), dtype=bool)
        self.ids = np.empty(0, dtype=np.int32)

        # dense position table: pos[jid] = index in ids, or -1 if absent
        self.pos = np.full(self._n_junctions, -1, dtype=np.int32)

    # ---- internal dense-array helpers ----

    def _append_jid(self, jid: int) -> None:
        jid32 = np.int32(jid)
        if self.ids.size == 0:
            self.ids = np.array([jid32], dtype=np.int32)
        else:
            self.ids = np.append(self.ids, jid32)

    def _swap_pop_id_at(self, p: int) -> int:
        p = int(p)
        last = int(self.ids[-1])
        if p != self.ids.size - 1:
            self.ids[p] = last

        if self.ids.size == 1:
            self.ids = np.empty(0, dtype=np.int32)
        else:
            self.ids = self.ids[:-1].copy()

        return last

    # ---- helpers over flat indices / (r,c) ----

    def _add_rc(self, r: int, c: int) -> None:
        r = int(r)
        c = int(c)
        if self.mask[r, c]:
            return

        jid = r * self._W1 + c
        self.pos[jid] = self.ids.size
        self._append_jid(jid)
        self.mask[r, c] = True

    def _remove_rc(self, r: int, c: int) -> None:
        r = int(r)
        c = int(c)
        if not self.mask[r, c]:
            return

        jid = r * self._W1 + c
        p = int(self.pos[jid])
        if p < 0:
            return

        last = self._swap_pop_id_at(p)
        if p != self.ids.size:
            self.pos[last] = p

        self.pos[jid] = -1
        self.mask[r, c] = False

    def _add_id(self, jid: int) -> None:
        jid = int(jid)
        if self.pos[jid] >= 0:
            return

        self.pos[jid] = self.ids.size
        self._append_jid(jid)

        r = jid // self._W1
        c = jid % self._W1
        self.mask[r, c] = True

    def _remove_id(self, jid: int) -> None:
        jid = int(jid)
        p = int(self.pos[jid])
        if p < 0:
            return

        last = self._swap_pop_id_at(p)
        if p != self.ids.size:
            self.pos[last] = p

        self.pos[jid] = -1

        r = jid // self._W1
        c = jid % self._W1
        self.mask[r, c] = False

    def _reset_from_mask(self, mask_new: np.ndarray) -> None:
        """
        Bulk rebuild the set from a boolean mask, avoiding Python per-node add calls.
        """
        if mask_new.shape != self.mask.shape:
            raise ValueError(
                f"mask shape mismatch: got {mask_new.shape}, expected {self.mask.shape}"
            )

        mask_new = np.asarray(mask_new, dtype=bool)

        # overwrite mask
        self.mask[:, :] = mask_new

        # rebuild ids densely as a 1D int32 numpy array
        ids_arr = np.flatnonzero(mask_new.ravel(order="C")).astype(np.int32, copy=False)
        self.ids = ids_arr

        # rebuild dense position table
        self.pos.fill(-1)
        if ids_arr.size > 0:
            self.pos[ids_arr] = np.arange(ids_arr.size, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.ids.size)

    def _sample_uniform_junction(self, rng: np.random.Generator) -> Tuple[int, int]:
        if self.ids.size == 0:
            raise RuntimeError("No available start junction.")
        k = int(rng.integers(0, self.ids.size))
        jid = int(self.ids[k])
        return jid // self._W1, jid % self._W1


class JunctionSamplerBase(_JunctionSetMixin):
    """
    Abstract base for **junction-domain** samplers (e.g., uniform over eligible junctions).

    Subclasses must at least implement:
      - `_is_eligible(Hbond, Vbond, r, c) -> bool`
        (e.g., boundary-connected & has-unbroken-incident);
      - optionally `_build_touch_set(rcs)` to enlarge local refresh scope;
      - optionally `_pre_refresh_hook(Hbond, Vbond, touch)` for side-effects
        before eligibility recomputation (e.g., union-find updates).

    Public API:
      - `sample(rng) -> (r, c)`
      - `recompute_at(Hbond, Vbond, rcs)`
    """

    # ---- overridable hooks ----
    def _is_eligible(self, Hbond: np.ndarray, Vbond: np.ndarray, r: int, c: int) -> bool:
        raise NotImplementedError

    def _build_touch_set(self, rcs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Given a list of touched junctions, return the (possibly enlarged) set to refresh.
        Default: deduplicate input as-is.
        """
        seen = set()
        out: List[Tuple[int, int]] = []
        for r, c in rcs:
            rc = (int(r), int(c))
            if rc in seen:
                continue
            seen.add(rc)
            out.append(rc)
        return out

    def _pre_refresh_hook(self,
                          Hbond: np.ndarray,
                          Vbond: np.ndarray,
                          touch: Sequence[Tuple[int, int]]) -> None:
        """
        Give subclasses a chance to perform side-effects before the eligibility
        update (e.g., union across newly broken incident bonds).
        Default: no-op.
        """
        return None

    # ---- public API ----
    def sample(self, rng: np.random.Generator) -> Tuple[int, int]:
        return self._sample_uniform_junction(rng)

    def recompute_at(self,
                 Hbond: np.ndarray,
                 Vbond: np.ndarray,
                 rcs: Sequence[Tuple[int, int]]) -> None:
        """
        Local incremental refresh after an **accepted** crack.
        1) Build the refresh scope (subclasses may enlarge).
        2) Run pre-refresh hook (e.g., union of broken incident bonds).
        3) Recompute eligibility and update the set.
    
        Uses _add_rc/_remove_rc directly to avoid redundant jid<->(r,c) conversion.
        """
        if not rcs:
            return
        touch = self._build_touch_set(rcs)
        if not touch:
            return
    
        self._pre_refresh_hook(Hbond, Vbond, touch)
    
        for r, c in touch:
            r = int(r)
            c = int(c)
            now = bool(self._is_eligible(Hbond, Vbond, r, c))
            if now and not self.mask[r, c]:
                self._add_rc(r, c)
            elif (not now) and self.mask[r, c]:
                self._remove_rc(r, c)


# -------------------------
# Bond-weighted base
# -------------------------

class BondWeightedBase:
    """
    Abstract base for **bond-domain** weighted samplers.

    Core responsibilities:
      - manage three buckets for 11/12/22 bonds;
      - sample a bond type by weighted counts;
      - sample a specific bond uniformly from the chosen bucket;
      - defer endpoint selection (bond->junction) to subclasses;
      - provide update hooks when bonds are broken/restored.

    Subclasses must implement:
      - `build_initial(Hbond, Vbond, *args, **kwargs)` to fill buckets;
      - `_choose_endpoint(axis, i, j, rng) -> (r, c)` endpoint strategy;
      - optionally `after_bonds_deleted(path_info)` to expand candidates (e.g., boundary growth).

    Public API:
      - `is_empty()`
      - `sample(rng)`
      - `on_bonds_broken(path_info)`
      - `on_bonds_restored(path_info)` (optional; only if you used early removal)
    """

    def __init__(self,
                 H: int,
                 W: int,
                 *,
                 STR: Optional[Sequence[float]] = None,
                 weights: Optional[Sequence[float]] = None) -> None:
        """
        Parameters
        ----------
        STR : [s11, s12, s22], optional
            Strengths per bond type. If provided (preferred), internal weights
            are set to 1/STR[i] with a small eps guard.
        weights : [w11, w12, w22], optional
            Directly specify type-weights. Overrides STR if both are given.
        """
        self.H, self.W = int(H), int(W)

        if weights is not None:
            w11, w12, w22 = (float(weights[0]), float(weights[1]), float(weights[2]))
        elif STR is not None:
            s11, s12, s22 = (float(STR[0]), float(STR[1]), float(STR[2]))
            eps = 1e-12
            w11, w12, w22 = (1.0 / max(s11, eps), 1.0 / max(s12, eps), 1.0 / max(s22, eps))
        else:
            w11 = w12 = w22 = 1.0

        self.w11, self.w12, self.w22 = w11, w12, w22

        # Three buckets for intact bonds
        self.b11 = _Bucket()
        self.b12 = _Bucket()
        self.b22 = _Bucket()

        # Optionally let subclasses keep direct views if useful
        self._Hbond_view: Optional[np.ndarray] = None
        self._Vbond_view: Optional[np.ndarray] = None

    # -- hooks for subclasses --

    def build_initial(self, Hbond: np.ndarray, Vbond: np.ndarray, *args, **kwargs) -> None:
        """
        Fill the buckets with initial intact bonds.
        Subclasses must implement their preferred construction (global scan or boundary-only scan, etc.).
        """
        raise NotImplementedError

    def _choose_endpoint(self, axis: Axis, i: int, j: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Convert a sampled bond to a start junction (r,c).
        Strategy varies by subclass (e.g., random end vs. bias to exterior).
        """
        raise NotImplementedError

    def after_bonds_deleted(self, path_info: PathInfo) -> None:
        """
        Optional: boundary-only samplers can grow the frontier here by inspecting `path_info`.
        """
        return None

    # -- utilities for subclasses --

    def set_views(self, Hbond: np.ndarray, Vbond: np.ndarray) -> None:
        """
        Keep array views handy (purely optional). Subclasses may rely on it.
        """
        self._Hbond_view = Hbond
        self._Vbond_view = Vbond

    # -- public API --

    def is_empty(self) -> bool:
        return (len(self.b11) + len(self.b12) + len(self.b22)) == 0

    def _sample_type(self, rng: np.random.Generator) -> int:
        n11, n12, n22 = len(self.b11), len(self.b12), len(self.b22)
        w11 = n11 * self.w11
        w12 = n12 * self.w12
        w22 = n22 * self.w22
        tot = w11 + w12 + w22
        if tot <= 0.0:
            non_empty = [t for t, c in ((11, n11), (12, n12), (22, n22)) if c > 0]
            if not non_empty:
                raise RuntimeError("No available bonds.")
            return int(non_empty[int(rng.integers(0, len(non_empty)))])
        u = float(rng.random()) * float(tot)
        if u < w11:
            return 11
        u -= w11
        if u < w12:
            return 12
        return 22

    def sample(self, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Sample a bond type by weight, then pick a bond uniformly from the chosen bucket,
        finally map it to a junction using subclass strategy.
        """
        if self.is_empty():
            raise RuntimeError("No available bonds to start from.")
        t = self._sample_type(rng)
        if   t == 11: axis, i, j = self.b11.sample_uniform(rng)
        elif t == 12: axis, i, j = self.b12.sample_uniform(rng)
        else:         axis, i, j = self.b22.sample_uniform(rng)
        r, c = self._choose_endpoint(int(axis), int(i), int(j), rng)
        return int(r), int(c)

    def on_bonds_broken(self, path_info: PathInfo) -> None:
        """
        Remove broken bonds from buckets after a crack is ACCEPTED.
        """
        for axis, i, j, old_t in path_info:
            key: BondKey = (int(axis), int(i), int(j))
            if   old_t == 11: self.b11.discard(key)
            elif old_t == 12: self.b12.discard(key)
            elif old_t == 22: self.b22.discard(key)
            # ignore -1 (already broken)
        # let subclass expand frontier / add new candidates if needed
        self.after_bonds_deleted(path_info)

    def on_bonds_restored(self, path_info: PathInfo) -> None:
        """
        Optional rollback API. Only needed if you remove early and later revert.
        With the "remove-on-accept" policy, this is usually unused.
        """
        for axis, i, j, old_t in path_info:
            if old_t not in (11, 12, 22):
                continue
            key: BondKey = (int(axis), int(i), int(j))
            if   old_t == 11: self.b11.add(key)
            elif old_t == 12: self.b12.add(key)
            elif old_t == 22: self.b22.add(key)

