import numpy as np
from optframework.utils.func.jit_mcpbe import nb_fenwick_add, nb_fenwick_build, nb_fenwick_update  # local numba kernels


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
            
    def swap(self, i: int, j: int):
        """Swap two indices' weights in-place (O(log n))."""
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError("FenwickSampler.swap: idx out of range")
        if i == j:
            return
        wi = float(self._w[i])
        wj = float(self._w[j])
        # update() will adjust tree and _w; total unchanged overall
        self.update(i, wj)
        self.update(j, wi)

    def pop_last(self) -> float:
        """Remove and return last item's weight (O(log n) + O(1) slice).

        Important: This maintains correctness by first updating last weight to 0
        in the *current* tree (size old_n), then truncating arrays.
        """
        if self._n <= 0:
            raise IndexError("FenwickSampler.pop_last: empty sampler")

        last = self._n - 1
        w_last = float(self._w[last])

        # Remove last weight contribution from the BIT
        if w_last != 0.0:
            self.update(last, 0.0)  # updates tree + total

        # Now physically shrink arrays
        self._n = last
        self._w = self._w[:last]
        self._tree = self._tree[: last + 1]  # tree length = n+1

        return w_last

    def remove_swap_last(self, idx: int) -> float:
        """Remove idx using swap-with-last strategy, return removed weight.

        This matches the common O(1) removal for unordered arrays:
            swap(idx, last); pop_last()
        """
        if idx < 0 or idx >= self._n:
            raise IndexError("FenwickSampler.remove_swap_last: idx out of range")
        last = self._n - 1
        if idx != last:
            self.swap(idx, last)
        return self.pop_last()

    def _prefix_sum_1based(self, i: int) -> float:
        """Fenwick prefix sum for 1-based index i (inclusive)."""
        s = 0.0
        while i > 0:
            s += float(self._tree[i])
            i -= i & -i
        return s
    
    def append(self, weight: float):
        """Append a new item with given weight (>=0), maintaining Fenwick invariants."""
        w_new = float(weight)
    
        old_n = self._n
        new_n = old_n + 1
        new_i = new_n  # 1-based index of appended element
    
        # --- grow weight vector ---
        self._w = np.append(self._w, w_new)
    
        # --- grow tree by 1 and copy old structure ---
        new_tree = np.zeros(new_n + 1, dtype=float)
        if old_n > 0:
            new_tree[: old_n + 1] = self._tree
        self._tree = new_tree
        self._n = new_n
    
        # --- initialize tree[new_i] correctly ---
        # tree[i] stores sum over [i-lowbit(i)+1, i]
        lowbit = new_i & -new_i
        left = new_i - lowbit + 1  # 1-based
    
        # old part is sum(left .. old_n) (since new element is at old_n+1)
        # IMPORTANT: correct formula uses (left-1), NOT (left-2)
        sum_upto_old = self._prefix_sum_1based(old_n) if old_n > 0 else 0.0
        sum_before_left = self._prefix_sum_1based(left - 1) if left > 1 else 0.0
        old_block_sum = sum_upto_old - sum_before_left
    
        self._tree[new_i] = old_block_sum + w_new
    
        # --- propagate ONLY w_new to ancestors above new_i ---
        j = new_i + lowbit
        while j <= new_n:
            self._tree[j] += w_new
            j += j & -j
    
        self._total += w_new


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
        return self.prefix_sum_search(u)
