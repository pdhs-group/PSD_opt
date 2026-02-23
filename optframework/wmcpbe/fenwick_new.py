import numpy as np
from optframework.utils.func.jit_mcpbe import (
    nb_fenwick_add,
    nb_fenwick_build,
    nb_fenwick_update,
    nb_fenwick_append_arrays,
    nb_fenwick_remove_swap_last,
    nb_fenwick_prefix_sum_1based,
)  # local numba kernels


class FenwickSampler:
    """Fenwick tree (BIT) for dynamic positive weights with sampling by prefix-sum.

    API:
      - __init__(weights: 1D array-like)
      - total() -> float
      - size()  -> int
      - update(idx, new_weight)
      - append(weight)
      - prefix_sum_search(s) -> idx   (0 <= s < total)
      - sample(rng) -> idx
      - asarray() -> copy of weights
    """
    __slots__ = ("_n", "_tree", "_w")

    def __init__(self, weights: np.ndarray):
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        self._n = int(w.size)
        self._w = w.copy()
        self._tree = np.zeros(self._n + 1, dtype=float)
        if self._n > 0:
            nb_fenwick_build(self._tree, self._n, self._w)

    # --- basic ops ---
    def size(self) -> int:
        return self._n

    def total(self) -> float:
        return self._tree_total()

    def _tree_total(self) -> float:
            if self._n <= 0:
                return 0.0
            return float(nb_fenwick_prefix_sum_1based(self._tree, self._n))

    def asarray(self) -> np.ndarray:
        return self._w.copy()

    # --- updates ---
    def _add(self, idx: int, delta: float):
        nb_fenwick_add(self._tree, self._n, idx, float(delta))

    def update(self, idx: int, new_weight: float):
        if idx < 0 or idx >= self._n:
            raise IndexError("FenwickSampler.update: idx out of range")
        # NEW: ensure non-negative
        new_w = float(new_weight)
        if new_w < 0.0:
            new_w = 0.0
        nb_fenwick_update(self._tree, self._n, self._w, idx, new_w)
    
    
    def append(self, weight: float):
        """Append a new item with given weight (>=0), updating BIT locally."""
        w = float(weight)
        if w < 0.0:
            w = 0.0

        old_n = self._n
        self._tree, self._w = nb_fenwick_append_arrays(self._tree, self._w, old_n, w)
        self._n = old_n + 1

    def remove(self, idx: int):
        """Remove one item by index using swap-with-last semantics, updating BIT locally."""
        n = self._n
        if idx < 0 or idx >= n:
            raise IndexError("FenwickSampler.remove: idx out of range")
        if n == 0:
            raise ValueError("FenwickSampler.remove on empty tree")

        nb_fenwick_remove_swap_last(self._tree, self._w, n, idx)

        new_n = n - 1
        self._n = new_n
        self._w = self._w[:new_n]
        self._tree = self._tree[: new_n + 1]

    # --- sampling ---
    def prefix_sum_search(self, s: float) -> int:
        """Return largest idx such that prefix_sum(idx) <= s (0-based)."""
        if self._n == 0:
            raise ValueError("FenwickSampler.prefix_sum_search on empty tree")
        t = self._tree_total()
        if s < 0.0 or s >= t:
            raise ValueError("s must be in [0,total)")
        i = 0
        bit = 1 << (self._n.bit_length() - 1)
        # walk the implicit binary lifting table
        while bit:
            nxt = i + bit
            if nxt <= self._n and self._tree[nxt] <= s:
                s -= self._tree[nxt]
                i = nxt
            bit >>= 1
        return i  # selected index

    def sample(self, rng) -> int:
        """Sample index according to weights / total."""
        if self._n == 0:
            raise ValueError("FenwickSampler.sample: empty or non-positive total")
        t = self._tree_total()
        if t <= 0.0:
            raise ValueError("FenwickSampler.sample: empty or non-positive total")
        u = float(rng.random()) * t
        #print("self._total : ", float(np.sum(self._w)))
        #print("tree total : ", self._tree_total())
        return self.prefix_sum_search(u)
