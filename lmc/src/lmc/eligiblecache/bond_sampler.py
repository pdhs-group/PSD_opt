# -*- coding: utf-8 -*-
"""
Bond-domain weighted samplers:
  - BondWeightedSampler: global weighted-by-type over all intact bonds;
  - BondWeightedBoundarySampler: boundary/frontier-only weighted sampling, and
    a biased endpoint selection favoring boundary/exterior side.

Both classes share BondWeightedBase from ._base and use helpers in ._utils.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np

from ._base import BondWeightedBase, BondKey
from ._utils import (
    ExteriorUF,
    bond_endpoints,
    incident_intact_bonds,
    neighbors4,
    compute_eligible_mask,
)

# ----------------------------------------------------------------------------
# Global weighted sampler over all intact bonds
# ----------------------------------------------------------------------------

class BondWeightedSampler(BondWeightedBase):
    """
    Weighted-by-type global sampler.
    Endpoint strategy: choose uniformly among the two endpoints.
    """

    def __init__(self, H: int, W: int, *, STR=None, weights=None) -> None:
        super().__init__(H, W, STR=STR, weights=weights)

    def build_initial(self, Hbond: np.ndarray, Vbond: np.ndarray) -> None:
        self.set_views(Hbond, Vbond)
        # Hbond scan
        Hb = Hbond
        for i in range(Hb.shape[0]):
            for j in range(Hb.shape[1]):
                t = int(Hb[i, j])
                if t == -1: continue
                key: BondKey = (0, i, j)
                if   t == 11: self.b11.add(key)
                elif t == 12: self.b12.add(key)
                elif t == 22: self.b22.add(key)
        # Vbond scan
        Vb = Vbond
        for i in range(Vb.shape[0]):
            for j in range(Vb.shape[1]):
                t = int(Vb[i, j])
                if t == -1: continue
                key: BondKey = (1, i, j)
                if   t == 11: self.b11.add(key)
                elif t == 12: self.b12.add(key)
                elif t == 22: self.b22.add(key)

    def _choose_endpoint(self, axis: int, i: int, j: int, rng: np.random.Generator) -> Tuple[int, int]:
        # Uniform over the two endpoints
        (r0, c0), (r1, c1) = bond_endpoints(self.H, self.W, axis, i, j)
        if int(rng.integers(0, 2)) == 0:
            return r0, c0
        else:
            return r1, c1


# ----------------------------------------------------------------------------
# Boundary/frontier-only weighted sampler (UF-aligned with JunctUniBoundarySampler)
# ----------------------------------------------------------------------------

class BondWeightedBoundarySampler(BondWeightedBase):
    """
    Candidate bonds: ONLY intact bonds that have at least one endpoint being an
    ELIGIBLE junction, where
        eligible(r,c) := uf.is_exterior_connected(r,c) AND has_unbroken_incident(r,c)

    Initialization:
      - ExteriorUF connects the rectangular frame to EXTERIOR (in __init__);
      - union_broken_from_arrays() propagates exterior connectivity along existing cracks;
      - compute_eligible_mask() determines eligible junctions;
      - Only bonds incident to eligible junctions are added to the weighted buckets.

    Incremental update (after a crack is accepted):
      - Base class removes broken bonds from buckets;
      - We union all new broken bonds' endpoints into UF so "water" propagates;
      - Recompute eligibility LOCALLY on a touched set (endpoints and their 4-neighbors);
      - For junctions that become newly-eligible, we add their incident intact bonds.

    Endpoint strategy:
      - Choose ONLY among eligible endpoints of the selected bond:
          * If both endpoints are eligible -> uniformly choose one;
          * If exactly one is eligible -> choose that one;
          * (A bond with no eligible endpoints should not be in buckets; we lazily prune.)
    """

    def __init__(self, H: int, W: int, *, STR=None, weights=None) -> None:
        super().__init__(H, W, STR=STR, weights=weights)
        self.uf = ExteriorUF(H, W)
        self._eligible: Optional[np.ndarray] = None  # shape (H+1, W+1), bool

    # --- helpers -------------------------------------------------------------

    def _add_bond_by_type(self, axis: int, i: int, j: int, t: int) -> None:
        """Add intact bond (axis,i,j) into the bucket corresponding to its type t."""
        key: BondKey = (axis, i, j)
        if   t == 11: self.b11.add(key)
        elif t == 12: self.b12.add(key)
        elif t == 22: self.b22.add(key)
        # if t == -1 (broken), caller should not call this

    def _endpoint_eligible(self, r: int, c: int) -> bool:
        elig = self._eligible
        return bool(elig[int(r), int(c)]) if elig is not None else False

    # --- required interface --------------------------------------------------

    def build_initial(self, Hbond: np.ndarray, Vbond: np.ndarray) -> None:
        """
        1) Set array views;
        2) Propagate exterior connectivity along existing cracks;
        3) Compute eligible junctions (UF-aligned);
        4) Add ONLY bonds that touch eligible junctions into buckets.
        """
        self.set_views(Hbond, Vbond)

        # Step 2: let "water" flow along current cracks
        self.uf.union_broken_from_arrays(Hbond, Vbond)

        # Step 3: compute eligible mask on junction lattice
        eligible = compute_eligible_mask(Hbond, Vbond, self.uf)
        self._eligible = eligible

        # Step 4: collect candidate bonds globally (intact bonds touching any eligible endpoint)
        # Hbond scan
        Hb = Hbond
        for i in range(Hb.shape[0]):
            for j in range(Hb.shape[1]):
                t = int(Hb[i, j])
                if t == -1:
                    continue
                (r0, c0), (r1, c1) = bond_endpoints(self.H, self.W, 0, i, j)
                if eligible[r0, c0] or eligible[r1, c1]:
                    self._add_bond_by_type(0, i, j, t)

        # Vbond scan
        Vb = Vbond
        for i in range(Vb.shape[0]):
            for j in range(Vb.shape[1]):
                t = int(Vb[i, j])
                if t == -1:
                    continue
                (r0, c0), (r1, c1) = bond_endpoints(self.H, self.W, 1, i, j)
                if eligible[r0, c0] or eligible[r1, c1]:
                    self._add_bond_by_type(1, i, j, t)

    def _choose_endpoint(self, axis: int, i: int, j: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Choose ONLY among eligible endpoints. Return (-1, -1) to signal "no eligible
        endpoint found" (caller may lazily remove the bond and retry).
        """
        (r0, c0), (r1, c1) = bond_endpoints(self.H, self.W, axis, i, j)
        e0 = self._endpoint_eligible(r0, c0)
        e1 = self._endpoint_eligible(r1, c1)
        if e0 and e1:
            return (r0, c0) if int(rng.integers(0, 2)) == 0 else (r1, c1)
        elif e0:
            return (r0, c0)
        elif e1:
            return (r1, c1)
        else:
            return (-1, -1)

    def sample(self, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Override to ensure we only return junctions on the UF-aligned frontier.
        If a sampled bond turns out to have no eligible endpoints, lazily prune it
        from the corresponding bucket and retry.
        """
        # We rely on the superclass for _sample_type() and _bucket access.
        # Retry a bounded number of times to avoid infinite loops if empty.
        for _ in range(1024):
            t = self._sample_type(rng)  # returns 11/12/22 or raises if empty
            # Select the corresponding bucket
            if   t == 11: bucket = self.b11
            elif t == 12: bucket = self.b12
            elif t == 22: bucket = self.b22
            else:         raise RuntimeError("Unexpected bond type sampled.")

            # If selected bucket is empty (race with deletions), resample type
            if len(bucket) == 0:
                continue

            axis, i, j = bucket.sample_uniform(rng)
            # Bond might have been broken since it was added; drop it if so
            Hb, Vb = self._Hbond_view, self._Vbond_view
            assert Hb is not None and Vb is not None
            t_now = int(Hb[i, j]) if axis == 0 else int(Vb[i, j])
            if t_now == -1:
                # broken -> remove and retry
                bucket.discard((axis, i, j))
                continue

            r, c = self._choose_endpoint(axis, i, j, rng)
            if r >= 0:
                return (r, c)
            else:
                # no eligible endpoint -> remove and retry
                bucket.discard((axis, i, j))
                continue

        # If we reached here, buckets are effectively empty under the new criterion
        raise RuntimeError("No eligible junction available under UF-aligned frontier.")

    # --- incremental maintenance --------------------------------------------

    def after_bonds_deleted(self, path_info: Sequence[Tuple[int, int, int, int]]) -> None:
        """
        Called by the base `on_bonds_broken` after broken bonds were removed from buckets.
        We:
          1) Union endpoints of newly broken bonds in UF (propagate exterior connectivity);
          2) Recompute eligibility LOCALLY on a touched set (endpoints +/- 4-neighborhood);
          3) For junctions newly turning eligible, add their incident intact bonds.
        """
        Hb, Vb = self._Hbond_view, self._Vbond_view
        assert Hb is not None and Vb is not None
        H = int(Hb.shape[0]); W = int(Vb.shape[1])

        # 1) propagate exterior connectivity along the newly accepted crack
        self.uf.union_many_from_path_info(path_info)

        # 2) build touched set: endpoints of the path plus their 4-neighbors
        touched_set = set()
        for axis, i, j, _old_t in path_info:
            (r0, c0), (r1, c1) = bond_endpoints(self.H, self.W, int(axis), int(i), int(j))
            for rc in neighbors4(int(r0), int(c0), H, W):
                touched_set.add(rc)
            for rc in neighbors4(int(r1), int(c1), H, W):
                touched_set.add(rc)

        # 3) recompute eligibility locally and expand candidates only around new-eligible junctions
        if self._eligible is None:
            self._eligible = np.zeros((H + 1, W + 1), dtype=bool)
        for (r, c) in touched_set:
            # recompute "eligible := UF-exterior-connected AND has-unbroken-incident"
            now = self.uf.is_exterior_connected(r, c)
            if now:
                # cheap guard: only check incident intact if exterior-connected holds
                # (saves a few array reads when now == False)
                now = False
                # fast check for any intact incident bond:
                # NOTE: inline of has_unbroken_incident for locality isn't necessary; reuse logic is fine
                # but here we do a tiny local check to reduce imports:
                # We still reuse the incident predicates via _utils if you prefer.
                # For clarity and consistency, we just reuse the existing function via compute_eligible_mask elsewhere.
                # Here, we keep it explicit to avoid extra imports inside hot path.
                # However, to stick to a single source of truth, we rely on incident_intact_bonds below:
                now = len(incident_intact_bonds(Hb, Vb, r, c)) > 0

            was = bool(self._eligible[r, c])
            if (not was) and now:
                # Became newly-eligible: add all incident INTACT bonds to the buckets
                for axis, i, j, t in incident_intact_bonds(Hb, Vb, r, c):
                    self._add_bond_by_type(axis, i, j, int(t))
                self._eligible[r, c] = True
            elif was and (not now):
                # Rare: lost eligibility (e.g., all incident bonds got broken).
                # We do not scan to remove bonds aggressively; lazy pruning happens in sample().
                self._eligible[r, c] = False
