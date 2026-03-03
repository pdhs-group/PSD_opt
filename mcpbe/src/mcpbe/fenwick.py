import numpy as np
from pbe_core.func.jit_mcpbe import nb_fenwick_add, nb_fenwick_build, nb_fenwick_update, nb_fenwick_prefix_sum_1based  # local numba kernels


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
    __slots__ = ("_n", "_tree", "_w", "_total")

    def __init__(self, weights: np.ndarray):
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        self._n = int(w.size)
        self._w = w.copy()
        self._tree = np.zeros(self._n + 1, dtype=float)
        self._total = float(np.sum(w))
        if self._n > 0:
            nb_fenwick_build(self._tree, self._n, self._w)

    # --- basic ops ---
    def size(self) -> int:
        return self._n

    def total(self) -> float:
        return self._total
    
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
        new_w = float(new_weight)
        delta = nb_fenwick_update(self._tree, self._n, self._w, idx, new_w)
        if delta:
            self._total += delta

    def append(self, weight: float):
        """Append a new item with given weight (>=0)."""
        w = float(weight)
        # grow arrays by 1 (preserve existing tree structure)
        old_n = self._n
        self._n = old_n + 1
        # grow weight vector
        self._w = np.append(self._w, w)
        # grow tree with copy of old prefix structure
        new_tree = np.zeros(self._n + 1, dtype=float)
        new_tree[:old_n + 1] = self._tree
        self._tree = new_tree
        if w != 0.0:
            nb_fenwick_add(self._tree, self._n, old_n, w)
            self._total += w

    # --- sampling ---
    def prefix_sum_search(self, s: float) -> int:
        """Return largest idx such that prefix_sum(idx) <= s (0-based)."""
        if self._n == 0:
            raise ValueError("FenwickSampler.prefix_sum_search on empty tree")
        if s < 0.0 or s >= self._total:
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
        if self._n == 0 or self._total <= 0.0:
            raise ValueError("FenwickSampler.sample: empty or non-positive total")
        u = float(rng.random()) * self._total
        # print("self._total : ", self._total)
        # print("tree total : ", self._tree_total())
        return self.prefix_sum_search(u)

