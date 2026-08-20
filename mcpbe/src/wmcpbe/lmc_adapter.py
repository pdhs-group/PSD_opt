from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import math
from numba import njit
from lmc import LMCSimulator


# ==============================
# Base adapter for LMC tables
# ==============================
class LMCBaseAdapter:
    """
    Base class providing shared utilities for all LMC table-based solvers.

    Responsibilities
    ----------------
    - Store grids over (A, X1) and associated metadata.
    - Handle interpolation (nearest or bilinear) in (log A, X1).
    - Convert A values between table-scale (A0_tab) and runtime-scale (A0_run).
    - Provide PMF/CDF utilities (through JIT helpers).
    - Provide caching for computed lookups.
    - Provide a small-particle policy for enabling/disabling table use.

    Subclasses must:
      - Load their specific tables in __init__,
      - Call self._init_common(...) to register grids and metadata,
      - Implement query and sampling interfaces that rely on the utilities here.
    """

    def _init_common(
        self,
        A_grid: np.ndarray,
        X1_grid: np.ndarray,
        meta: Dict[str, Any],
        *,
        interp: str = "bilinear",
        A0_run: Optional[float] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize all common state needed by table-based LMC solvers.

        Parameters
        ----------
        A_grid : array-like
            1D array of A-values defining the row grid of the lookup table.
            Must be strictly positive.
        X1_grid : array-like
            1D array of X1-values defining the column grid of the lookup table.
        meta : dict
            Metadata associated with the table (e.g., NO_FRAG, A0, etc.).
        interp : {"nearest", "bilinear"}, optional
            Interpolation method. "nearest" uses nearest-neighbor lookup;
            "bilinear" performs bilinear interpolation in (log A, X1).
        A0_run : float or None, optional
            Runtime A0 (unit area per lattice cell). If None, A0 from meta
            is used.
        cache_enabled : bool, optional
            Enable or disable caching of interpolation results.
        """

        self.A_grid = np.asarray(A_grid, dtype=float)
        self.X1_grid = np.asarray(X1_grid, dtype=float)
        self.meta: Dict[str, Any] = meta if isinstance(meta, dict) else dict(meta)
        self.eps = 1e-16

        # Small-particle policy (matches behavior in the live solver)
        # "fallback": solver will ignore table eligibility checks
        # "disable":  eligibility checks are applied
        self.small_particle_policy = "fallback"
        self.delta_cells = 0.1  # safety margin identical to live solver

        # Default desired number of fragments expected by the table
        self.NO_FRAG = int(self.meta.get("NO_FRAG", 4))

        if np.any(self.A_grid <= 0):
            raise ValueError("A_grid must contain strictly positive values.")

        self.logA_grid = np.log(self.A_grid)

        if interp not in ("nearest", "bilinear"):
            raise ValueError("interp must be 'nearest' or 'bilinear'")
        self.interp = interp

        # Table-scale A0 (from metadata) vs runtime A0
        self.A0_tab = float(self.meta.get("A0", 1.0))
        self.A0_run = self.A0_tab if (A0_run is None) else float(A0_run)

        # Caching of table lookups
        self.cache_enabled = bool(cache_enabled)
        # key format: (round(logA, 6), round(X1, 6), interp_str, tag_int)
        self._cache: Dict[Tuple[float, float, str, int], Any] = {}

    # ------- Cache helpers (to be used by subclasses) -------
    def _cache_get(self, key):
        """Return cached result or None if caching disabled or missing."""
        if not self.cache_enabled:
            return None
        return self._cache.get(key)

    def _cache_set(self, key, value):
        """Store a value in cache, respecting the cache_enabled flag."""
        if not self.cache_enabled:
            return
        self._cache[key] = value

    def clear_cache(self):
        """Explicitly clear all cached interpolations."""
        self._cache.clear()

    def set_small_particle_policy(self, policy: str = "fallback", delta_cells: float = 0.1):
        """
        Configure the policy used to determine whether a particle is "large
        enough" to use table-based predictions.

        Parameters
        ----------
        policy : {"fallback", "disable"}
            - "fallback": table eligibility is ignored (always OK).
            - "disable": eligibility is enforced, based on NO_FRAG.
        delta_cells : float
            Safety margin used in the eligibility criterion.
        """
        if policy not in ("fallback", "disable"):
            raise ValueError("small_particle_policy must be 'fallback' or 'disable'")
        self.small_particle_policy = policy
        self.delta_cells = float(delta_cells)

    def eligible_for_tables(self, A_run: float) -> bool:
        """
        Decide whether a particle of area A_run is large enough to use
        the precomputed table.

        Logic
        -----
        Compute n_cells = A_run / A0_run, then check:

            floor(n_cells - delta_cells) >= NO_FRAG

        Returns
        -------
        bool
            True if the particle can produce at least NO_FRAG fragments
            under the table model.
        """
        n_cells = float(A_run) / max(self.A0_run, self.eps)
        NO_FRAG_raw = int(math.floor(max(n_cells - self.delta_cells, 0.0)))
        return (NO_FRAG_raw >= int(self.NO_FRAG))

    # ----- Bilinear interpolation in (log A, X1) -----
    def _neighbors_weights(self, A: float, X1: float):
        """
        Locate the bracketing indices in (log A, X1) and compute bilinear weights.

        Parameters
        ----------
        A : float
            Runtime A-value.
        X1 : float
            Runtime X1-value.

        Returns
        -------
        tuple
            (i0, i1, j0, j1, w00, w01, w10, w11)

            If nearest interpolation or degenerate case:
                (i0, i0, j0, j0, 1, 0, 0, 0)
            Otherwise, bilinear weights over the four neighbors.
        """
        logA = np.log(max(A, self.eps))

        # Locate A neighbors
        i1 = np.searchsorted(self.logA_grid, logA, side="right")
        i0 = max(0, i1 - 1)
        i1 = min(i1, self.logA_grid.size - 1)

        # Locate X1 neighbors
        j1 = np.searchsorted(self.X1_grid, X1, side="right")
        j0 = max(0, j1 - 1)
        j1 = min(j1, self.X1_grid.size - 1)

        # Nearest-neighbor or trivial case
        if self.interp == "nearest" or (i0 == i1 and j0 == j1):
            return (i0, i0, j0, j0, 1.0, 0.0, 0.0, 0.0)

        # Bilinear interpolation factors
        xA0, xA1 = self.logA_grid[i0], self.logA_grid[i1]
        tA = 0.0 if (xA1 <= xA0 + 1e-15) else (logA - xA0) / (xA1 - xA0)

        xX0, xX1 = self.X1_grid[j0], self.X1_grid[j1]
        tX = 0.0 if (xX1 <= xX0 + 1e-15) else (X1 - xX0) / (xX1 - xX0)

        # Weights in the usual (A0,X0):w00, (A0,X1):w01, ...
        w00 = (1.0 - tA) * (1.0 - tX)
        w01 = (1.0 - tA) * tX
        w10 = tA * (1.0 - tX)
        w11 = tA * tX

        return (i0, i1, j0, j1, w00, w01, w10, w11)

    # ----- Convert runtime A to table A keeping n = A / A0 invariant -----
    def _A_lookup(self, A_run: float) -> float:
        """
        Convert a runtime area A_run to the table-scale area.

        The unit conversion keeps the number of 'cells' invariant:

            A_run / A0_run  ==  A_tab / A0_tab

        Therefore:
            A_tab = A_run * (A0_tab / A0_run)
        """
        s = self.A0_tab / max(self.A0_run, self.eps)
        return float(A_run) * s


# ----- CDF/PMF helpers (Numba-accelerated) -----
@njit(fastmath=True)
def _pmf_from_cdf_1d(cdf: np.ndarray) -> np.ndarray:
    """
    Convert a 1D cumulative distribution function (CDF) into a PMF.
    """
    N = cdf.shape[0]
    p = np.empty_like(cdf)
    p[0] = cdf[0]
    for k in range(1, N):
        p[k] = cdf[k] - cdf[k - 1]
    return p


@njit(fastmath=True)
def _cdf_from_pmf_1d(p: np.ndarray) -> np.ndarray:
    """
    Convert a 1D PMF into a CDF, enforcing the last element to be exactly 1.
    """
    c = np.cumsum(p)
    if c[-1] != 1.0:
        c[-1] = 1.0
    return c


@njit(fastmath=True)
def _pmf2_from_twolevel(rowsum_cdf: np.ndarray, row_cdf: np.ndarray) -> np.ndarray:
    """
    Convert a hierarchical CDF representation into a full 2D PMF.

    Representation
    --------------
    rowsum_cdf[i]  = cumulative mass along row sums.
    row_cdf[i, j] = cumulative mass within row i.

    Returns
    -------
    P : (N, N) array
        Full 2D PMF normalized to 1.
    """
    N = rowsum_cdf.shape[0]
    P = np.zeros_like(row_cdf)

    # row marginals
    r = np.empty(N, dtype=np.float64)
    r[0] = rowsum_cdf[0]
    for i in range(1, N):
        r[i] = rowsum_cdf[i] - rowsum_cdf[i - 1]
    r = np.maximum(r, 0.0)

    # conditional PMFs per row
    for i in range(N):
        p_row = np.empty(N, dtype=np.float64)
        p_row[0] = row_cdf[i, 0]
        for j in range(1, N):
            p_row[j] = row_cdf[i, j] - row_cdf[i, j - 1]

        p_row = np.maximum(p_row, 0.0)
        s = p_row.sum()
        if s <= 0.0:
            p_row[:] = 1.0 / N
        else:
            p_row /= s

        P[i, :] = r[i] * p_row

    # normalize
    S = P.sum()
    if S > 0.0:
        P /= S
    return P


@njit(fastmath=True)
def _twolevel_from_pmf2(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D PMF into hierarchical CDF representation:
      - rowsum_cdf : cumulative sum of row marginals
      - row_cdf[i] : CDF of the conditional distribution in row i
    """
    N = P.shape[0]

    rowsum = P.sum(axis=1)
    tot = rowsum.sum()
    if tot <= 0.0:
        rowsum[:] = 1.0 / N
        tot = 1.0
    rowsum /= tot
    rowsum_cdf = np.cumsum(rowsum)

    row_cdf = np.zeros_like(P)
    for i in range(N):
        s = P[i, :].sum()
        if s <= 0.0:
            row_cdf[i, :] = np.cumsum(np.full(N, 1.0 / N))
        else:
            row_cdf[i, :] = np.cumsum(P[i, :] / s)

    rowsum_cdf[-1] = 1.0
    row_cdf[:, -1] = 1.0
    return rowsum_cdf, row_cdf

# ==============================
# Marginalè¡¨çš„é€‚é…å™¨
# ==============================
# ==============================
# Adapter for marginal LMC tables
# ==============================
class LMCTableAdapter(LMCBaseAdapter):
    """
    Adapter for marginal LMC tables produced by `preprocess_lmc_to_tables.py`.

    The .npz file is expected to contain:
      - Grids over (A, X1),
      - 1D marginal CDF tables,
      - 2D joint CDF tables in two-level (row-sum / row) representation,
      - Metadata describing A0, NO_FRAG, and other configuration.

    This adapter provides:
      - get_1d(A, X1): 1D CDF over normalized fragment sizes (marginal),
      - get_2d(A, X1): 2D hierarchical CDF for joint distributions.
    """

    def __init__(
        self,
        npz_path: str,
        interp: str = "bilinear",
        A0_run: float | None = None,
        cache_enabled: bool = True,
    ):
        """
        Load a marginal table (.npz) and initialize the base adapter.

        Parameters
        ----------
        npz_path : str
            Path to the .npz file generated by `preprocess_lmc_to_tables.py`.
        interp : {"nearest", "bilinear"}
            Interpolation mode in (log A, X1) space.
        A0_run : float or None
            Runtime A0 (area per cell). If None, A0 from the table metadata is used.
        cache_enabled : bool
            Whether to cache interpolation results by (A, X1, interp, tag).
        """
        d = np.load(npz_path, allow_pickle=True)

        A_grid = np.asarray(d["A_grid"], dtype=float)
        X1_grid = np.asarray(d["X1_grid"], dtype=float)
        meta = d["meta"].item() if isinstance(d["meta"], np.ndarray) else dict(d["meta"])

        super()._init_common(
            A_grid,
            X1_grid,
            meta,
            interp=interp,
            A0_run=A0_run,
            cache_enabled=cache_enabled,
        )

        # 1D marginal for relative fragment sizes
        self.rel1d = np.asarray(d["rel1d"], dtype=float)      # (N,)
        # Grid of 1D CDF arrays over (A, X1); object array of shape (nA, nX)
        self.cdf1d_grid = d["cdf1d_grid"]                     # object[nA, nX] of float64[N]
        self.zmin1d_grid = np.asarray(d["zmin1d_grid"], dtype=float)

        # 2D joint distribution support (e.g., for two material phases)
        self.rel1 = np.asarray(d["rel1"], dtype=float)        # (N,)
        self.rel3 = np.asarray(d["rel3"], dtype=float)        # (N,)
        # Hierarchical CDF representation: row sums and row-wise CDFs
        self.rowsum_cdf_grid = d["rowsum_cdf_grid"]           # object[nA, nX]
        self.row_cdf_grid = d["row_cdf_grid"]                 # object[nA, nX]
        self.zmin1_grid = np.asarray(d["zmin1_grid"], dtype=float)
        self.zmin3_grid = np.asarray(d["zmin3_grid"], dtype=float)

        # Expected number of fragments and observed sample counts
        self.p_expected_grid = np.asarray(d["p_expected_grid"], dtype=float)
        self.n_obs_grid = np.asarray(d["n_obs_grid"], dtype=np.int64)

        self.N = int(self.rel1.shape[0])

    # ------ Public queries ------

    def get_1d(self, A: float, X1: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Retrieve the 1D marginal CDF over relative fragment sizes for
        a given (A, X1) point, using interpolation in table space.

        Parameters
        ----------
        A : float
            Total particle area at runtime.
        X1 : float
            Mass fraction (or volume fraction) of phase 1 at runtime.

        Returns
        -------
        rel : (N,) array
            Grid of relative fragment sizes.
        cdf : (N,) array
            Interpolated CDF over rel.
        zmin1d : float
            Minimum relative size (runtime) for 1D sampling; typically
            A0_run / A, clipped to [0,1].
        p_expected : float
            Interpolated expected number of fragments in the table.
        """
        A_lookup = self._A_lookup(A)
        key = (
            round(float(np.log(max(A_lookup, self.eps))), 6),
            round(float(X1), 6),
            self.interp,
            1,  # tag for 1D
        )
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)

        def pmf_at(i, j):
            c = np.asarray(self.cdf1d_grid[i, j], dtype=np.float64)
            return _pmf_from_cdf_1d(c)

        # Bilinear interpolation of PMFs, then renormalize
        p = (
            w00 * pmf_at(i0, j0)
            + w01 * pmf_at(i0, j1)
            + w10 * pmf_at(i1, j0)
            + w11 * pmf_at(i1, j1)
        )
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s > 0.0:
            p /= s
        cdf = _cdf_from_pmf_1d(p)

        # Interpolate expected number of fragments
        pe = (
            w00 * self.p_expected_grid[i0, j0]
            + w01 * self.p_expected_grid[i0, j1]
            + w10 * self.p_expected_grid[i1, j0]
            + w11 * self.p_expected_grid[i1, j1]
        )

        # Minimal relative size given runtime A (in units of A0_run)
        zmin1d = float(np.clip(self.A0_run / max(A, 1e-20), 0.0, 1.0))

        out = (self.rel1d, cdf, zmin1d, float(pe))
        self._cache_set(key, out)
        return out

    def get_2d(
        self,
        A: float,
        X1: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """
        Retrieve the 2D joint distribution (in hierarchical CDF form) for
        a given (A, X1) point.

        The joint distribution is given on a discretized grid of
        (rel1, rel3), where rel1 and rel3 are relative fragment sizes for
        two different phases (e.g., two materials).

        Parameters
        ----------
        A : float
            Total particle area at runtime.
        X1 : float
            Mass or volume fraction of phase 1 at runtime.

        Returns
        -------
        rel1 : (N,) array
            Grid of relative sizes for phase 1.
        rel3 : (N,) array
            Grid of relative sizes for phase 3 (or complementary phase).
        rowsum_cdf : (N,) array
            CDF of row sums (marginal in one dimension).
        row_cdf : (N, N) array
            Row-wise CDFs representing the conditional distributions.
        zmin1 : float
            Minimal relative size for phase 1 (A * X1 scaled by A0_run).
        zmin3 : float
            Minimal relative size for phase 3 (A * (1 - X1) scaled by A0_run).
        p_expected : float
            Interpolated expected number of fragments from the table.
        """
        A_lookup = self._A_lookup(A)
        key = (
            round(float(np.log(max(A_lookup, self.eps))), 6),
            round(float(X1), 6),
            self.interp,
            2,  # tag for 2D
        )
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)

        def P_at(i, j):
            rowsum_cdf = np.asarray(self.rowsum_cdf_grid[i, j], dtype=np.float64)
            row_cdf = np.asarray(self.row_cdf_grid[i, j], dtype=np.float64)
            return _pmf2_from_twolevel(rowsum_cdf, row_cdf)

        # Bilinear interpolation of 2D PMFs, then renormalize
        P = (
            w00 * P_at(i0, j0)
            + w01 * P_at(i0, j1)
            + w10 * P_at(i1, j0)
            + w11 * P_at(i1, j1)
        )
        P = np.maximum(P, 0.0)
        S = P.sum()
        if S > 0.0:
            P /= S

        # Convert back to hierarchical (rowsum / row) CDF representation
        rowsum_cdf, row_cdf = _twolevel_from_pmf2(P)

        # Interpolate expected number of fragments
        pe = (
            w00 * self.p_expected_grid[i0, j0]
            + w01 * self.p_expected_grid[i0, j1]
            + w10 * self.p_expected_grid[i1, j0]
            + w11 * self.p_expected_grid[i1, j1]
        )

        # Minimal relative sizes for each phase, in units of A0_run
        denom1 = A * X1
        denom3 = A * (1.0 - X1)
        zmin1 = float(np.clip(self.A0_run / max(denom1, 1e-20), 0.0, 1.0)) if denom1 > self.eps else 0.0
        zmin3 = float(np.clip(self.A0_run / max(denom3, 1e-20), 0.0, 1.0)) if denom3 > self.eps else 0.0

        out = (self.rel1, self.rel3, rowsum_cdf, row_cdf, zmin1, zmin3, float(pe))
        self._cache_set(key, out)
        return out

# ==============================
# Rank-wise (Top-K + Tail) table adapter
# ==============================
class LMCRankAdapter(LMCBaseAdapter):
    """
    Adapter for rank-wise LMC tables produced by `preprocess_lmc_to_mcpbe_joint.py`.

    The .npz file contains, for each (A, X1):

      - Rank-wise 2D hierarchical CDF tables (Top-K ranks):
          rowsum_cdf_rank[r], row_cdf_rank[r]  for r = 1..K
      - Histogram of the number of fragments PN: pn_hist_grid
      - Tail-total CDF: tail_T_cdf_grid
      - (Optional) two-phase tail composition modeled as Beta(Î±, Î²)
      - Frequency adjustment vectors freq_adj_grid
      - M2_true_grid for diagnostics

    This adapter provides:

      - get_rank_tables(A, X1) ->
          rel1, rel3,
          rowsum_cdf_list[K], row_cdf_list[K],
          zmin1, zmin3,
          EN, tail_T_cdf, tail_mode, tail_alpha, tail_beta

      - sample_one_shot(A, X1, rng, N=None, K_use=None) ->
          (rA_list, rB_list)

        Single-shot sampling of N fragments (relative volumes for A/B):
        the last fragment is used as a remainder to guarantee conservation.
        K_use (if None) defaults to min(K, N-1).
    """

    def __init__(
        self,
        npz_path: str,
        interp: str = "bilinear",
        A0_run: float | None = None,
        cache_enabled: bool = True,
    ):
        d = np.load(npz_path, allow_pickle=True)
        A_grid = np.asarray(d["A_grid"], dtype=float)
        X1_grid = np.asarray(d["X1_grid"], dtype=float)
        meta = d["meta"].item() if isinstance(d["meta"], np.ndarray) else dict(d["meta"])

        super()._init_common(
            A_grid,
            X1_grid,
            meta,
            interp=interp,
            A0_run=A0_run,
            cache_enabled=cache_enabled,
        )

        self.rank_K = int(d["rank_K"])
        self.N = int(d["N_bins"])
        self.rel1 = np.asarray(d["rel1"], dtype=float)
        self.rel3 = np.asarray(d["rel3"], dtype=float)
        self.rel1d = np.asarray(d["rel1d"], dtype=float)
        self.zmin1d_grid = np.asarray(d["zmin1d_grid"], dtype=float)

        # Rank-wise grids (per rank r = 1..K)
        self.rowsum_cdf_rank: List[Any] = []
        self.row_cdf_rank: List[Any] = []
        self.cdf1d_rank: List[Any] = []
        for r in range(self.rank_K):
            self.rowsum_cdf_rank.append(d[f"rowsum_cdf_rank{r+1}_grid"])
            self.row_cdf_rank.append(d[f"row_cdf_rank{r+1}_grid"])
            self.cdf1d_rank.append(d[f"cdf1d_rank{r+1}_grid"])

        # Histogram of N fragments and rank observation counts
        self.pn_hist_grid = d["pn_hist_grid"]              # object[nA, nX] of int64[NO_FRAG+1]
        self.n_obs_ranks_grid = d["n_obs_ranks_grid"]      # object[nA, nX] of int64[K]

        # Tail distribution of total tail volume T
        self.tail_T_cdf_grid = d["tail_T_cdf_grid"]        # object[nA, nX] of float64[N]
        self.tail_mode = str(d["tail_mode"]) if "tail_mode" in d else "equal"
        self.tail_alpha_grid = d["tail_alpha_grid"] if "tail_alpha_grid" in d else None
        self.tail_beta_grid = d["tail_beta_grid"] if "tail_beta_grid" in d else None

        # Frequency adjustment and second moment diagnostics
        self.freq_adj_grid = d["freq_adj_grid"]
        self.M2_true_grid = d["M2_true_grid"]

    # ------- Internal: interpolate rank-wise rowsum/row CDFs -------
    def _interp_rank_tables(self, A: float, X1: float):
        """
        Bilinearly interpolate the rank-wise hierarchical CDFs over (A, X1).

        Returns
        -------
        rowsum_list : list of (N,) arrays
        rowcdf_list : list of (N, N) arrays
            For each rank r = 0..K-1:
              rowsum_list[r] = rowsum_cdf
              rowcdf_list[r] = row_cdf
        """
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A, X1)

        rowsum_list: List[np.ndarray] = []
        rowcdf_list: List[np.ndarray] = []

        for r in range(self.rank_K):

            def tab(i, j):
                rs = np.asarray(self.rowsum_cdf_rank[r][i, j], dtype=np.float64)
                rc = np.asarray(self.row_cdf_rank[r][i, j], dtype=np.float64)
                return rs, rc

            rs00, rc00 = tab(i0, j0)
            rs01, rc01 = tab(i0, j1)
            rs10, rc10 = tab(i1, j0)
            rs11, rc11 = tab(i1, j1)

            # Convex combination directly on CDF grids (numerically stable)
            rs = w00 * rs00 + w01 * rs01 + w10 * rs10 + w11 * rs11
            rc = w00 * rc00 + w01 * rc01 + w10 * rc10 + w11 * rc11

            rs = np.clip(rs, 0.0, 1.0)
            rs[-1] = 1.0
            rc = np.clip(rc, 0.0, 1.0)
            rc[:, -1] = 1.0

            rowsum_list.append(rs)
            rowcdf_list.append(rc)

        return rowsum_list, rowcdf_list

    def _interp_EN(self, A: float, X1: float) -> float:
        """
        Interpolate the expected number of fragments EN from the pn_hist grid.
        """
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A, X1)

        def EN_at(i, j):
            hist = np.asarray(self.pn_hist_grid[i, j], dtype=np.int64)
            tot = float(hist.sum()) if hist.size > 0 else 0.0
            if tot <= 0.0:
                # Fallback default (e.g. binary split)
                return 2.0
            idx = np.arange(hist.size, dtype=float)
            return float((idx * hist).sum() / tot)

        return (
            w00 * EN_at(i0, j0)
            + w01 * EN_at(i0, j1)
            + w10 * EN_at(i1, j0)
            + w11 * EN_at(i1, j1)
        )

    def _interp_tail(self, A: float, X1: float):
        """
        Interpolate the tail-total CDF and optional Beta parameters for
        the tail composition.
        """
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A, X1)

        def cdf_at(i, j):
            return np.asarray(self.tail_T_cdf_grid[i, j], dtype=np.float64)

        c00 = cdf_at(i0, j0)
        c01 = cdf_at(i0, j1)
        c10 = cdf_at(i1, j0)
        c11 = cdf_at(i1, j1)

        cdf = w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf[-1] = 1.0

        alpha = beta = None
        if (
            self.tail_mode == "beta"
            and (self.tail_alpha_grid is not None)
            and (self.tail_beta_grid is not None)
        ):
            a00 = float(self.tail_alpha_grid[i0, j0])
            a01 = float(self.tail_alpha_grid[i0, j1])
            a10 = float(self.tail_alpha_grid[i1, j0])
            a11 = float(self.tail_alpha_grid[i1, j1])

            b00 = float(self.tail_beta_grid[i0, j0])
            b01 = float(self.tail_beta_grid[i0, j1])
            b10 = float(self.tail_beta_grid[i1, j0])
            b11 = float(self.tail_beta_grid[i1, j1])

            alpha = w00 * a00 + w01 * a01 + w10 * a10 + w11 * a11
            beta = w00 * b00 + w01 * b01 + w10 * b10 + w11 * b11

        return cdf, self.tail_mode, alpha, beta

    def _interp_freq_adj(self, A_lookup: float, X1: float) -> np.ndarray:
        """
        Bilinearly interpolate the freq_adj vector (length K, non-negative,
        normalized to 1). If no freq_adj_grid is available, fall back to
        uniform weights.
        """
        K = int(self.rank_K)
        if (self.freq_adj_grid is None) or (K <= 0):
            # Backward compatibility: uniform weights
            return np.full(K, 1.0 / max(K, 1), dtype=np.float64)

        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)

        def fa(i, j):
            v = np.asarray(self.freq_adj_grid[i, j], dtype=np.float64)
            if v.size != K:
                # Fallback if shape is unexpected
                return np.full(K, 1.0 / max(K, 1), dtype=np.float64)
            v = np.maximum(v, 0.0)
            s = float(v.sum())
            if s <= 0.0:
                return np.full(K, 1.0 / max(K, 1), dtype=np.float64)
            return v / s

        f00 = fa(i0, j0)
        f01 = fa(i0, j1)
        f10 = fa(i1, j0)
        f11 = fa(i1, j1)

        f = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
        f = np.maximum(f, 0.0)
        s = float(f.sum())
        if s > 0.0:
            f /= s
        else:
            f[:] = 1.0 / max(K, 1)
        return f

    # ------- Public: rank-wise table query -------
    def get_rank_tables(self, A: float, X1: float):
        """
        Query all rank-wise tables and tail information at (A, X1).

        Returns
        -------
        rel1 : (N,) array
        rel3 : (N,) array
            Support grids for relative fragment sizes of the two phases.
        rowsum_list : list of (N,) arrays
        rowcdf_list : list of (N, N) arrays
            Rank-wise hierarchical CDFs: for each rank r:
              rowsum_list[r] = rowsum_cdf_r
              rowcdf_list[r] = row_cdf_r
        zmin1 : float
            Minimal relative size for phase 1 (A*X1 scaled by A0_run).
        zmin3 : float
            Minimal relative size for phase 3 (A*(1-X1) scaled by A0_run).
        EN : float
            Interpolated expected number of fragments.
        tail_T_cdf : (N,) array
            CDF of the total tail fraction T.
        tail_mode : {"equal", "beta", ...}
            Tail composition model used.
        tail_alpha, tail_beta : float or None
            Beta-distribution parameters for the tail A-fraction when
            tail_mode == "beta"; otherwise None.
        """
        A_lookup = self._A_lookup(A)
        key_base = (
            round(float(np.log(max(A_lookup, self.eps))), 6),
            round(float(X1), 6),
            self.interp,
        )

        # Determine single-phase vs two-phase (using denominators for stability)
        denom1 = A * X1
        denom3 = A * (1.0 - X1)
        pure_A = (X1 > 1.0 - self.eps)   # X1 â‰ˆ 1
        pure_B = (X1 < self.eps)         # X1 â‰ˆ 0

        if pure_A or pure_B:
            # Single-phase "pseudo-2D" representation
            tag = 201 if pure_A else 202
            key = key_base + (tag,)
            hit = self._cache_get(key)
            if hit is not None:
                return hit

            # Interpolate rank-wise 1D CDFs for each rank
            i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)
            N = self.N
            cdf1d_list = []

            for r in range(self.rank_K):

                def cdf_at(i, j):
                    return np.asarray(self.cdf1d_rank[r][i, j], dtype=np.float64)

                c00 = cdf_at(i0, j0)
                c01 = cdf_at(i0, j1)
                c10 = cdf_at(i1, j0)
                c11 = cdf_at(i1, j1)

                cdf = w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11
                cdf = np.clip(cdf, 0.0, 1.0)
                cdf[-1] = 1.0
                cdf1d_list.append(cdf)

            # Construct pseudo-2D representation depending on which phase is present
            if pure_A:
                # Phase A only: row dimension carries 1D distribution (rA),
                # column is fixed at j=0 (rB=0).
                rel1 = self.rel1d.copy()
                rel3 = np.zeros_like(self.rel1d)

                rowsum_list = [cdf.copy() for cdf in cdf1d_list]    # shape (N,)
                rowcdf_list = [
                    np.ones((N, N), dtype=float) for _ in range(self.rank_K)
                ]  # each row is degenerate at j=0

                zmin1 = (
                    float(np.clip(self.A0_run / max(denom1, 1e-20), 0.0, 1.0))
                    if denom1 > self.eps
                    else 0.0
                )
                zmin3 = 0.0
            else:
                # Phase B only: column dimension carries 1D distribution (rB),
                # row fixed at i=0 (rA=0).
                rel1 = np.zeros_like(self.rel1d)
                rel3 = self.rel1d.copy()

                rowsum_list = [
                    np.ones(N, dtype=float) for _ in range(self.rank_K)
                ]  # all mass at i=0
                rowcdf_list = []
                for cdf in cdf1d_list:
                    rc = np.tile(cdf, (N, 1))  # N identical rows
                    rowcdf_list.append(rc)

                zmin1 = 0.0
                zmin3 = (
                    float(np.clip(self.A0_run / max(denom3, 1e-20), 0.0, 1.0))
                    if denom3 > self.eps
                    else 0.0
                )

            # EN and tail remain interpolated from the full table
            EN = self._interp_EN(A, X1)
            tail_T_cdf, tail_mode, tail_alpha, tail_beta = self._interp_tail(A_lookup, X1)

            out = (
                rel1,
                rel3,
                rowsum_list,
                rowcdf_list,
                zmin1,
                zmin3,
                float(EN),
                tail_T_cdf,
                tail_mode,
                tail_alpha,
                tail_beta,
            )
            self._cache_set(key, out)
            return out

        # Mixed (two-phase) case: use full 2D tables
        key = key_base + (200,)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        rowsum_list, rowcdf_list = self._interp_rank_tables(A_lookup, X1)
        EN = self._interp_EN(A, X1)
        tail_T_cdf, tail_mode, tail_alpha, tail_beta = self._interp_tail(A_lookup, X1)

        zmin1 = (
            float(np.clip(self.A0_run / max(denom1, 1e-20), 0.0, 1.0))
            if denom1 > self.eps
            else 0.0
        )
        zmin3 = (
            float(np.clip(self.A0_run / max(denom3, 1e-20), 0.0, 1.0))
            if denom3 > self.eps
            else 0.0
        )

        out = (
            self.rel1,
            self.rel3,
            rowsum_list,
            rowcdf_list,
            zmin1,
            zmin3,
            float(EN),
            tail_T_cdf,
            tail_mode,
            tail_alpha,
            tail_beta,
        )
        self._cache_set(key, out)
        return out

    # ------- Optional: single-shot sampling (Top-K + tail) -------
    def sample_one_shot(
        self,
        A: float,
        X1: float,
        rng: np.random.Generator,
        N: Optional[int] = None,
        K_use: Optional[int] = None,
        tail_strategy: str = "equal",  # 'equal' | 'beta'
    ) -> Tuple[List[float], List[float]]:
        """
        Sample a two-phase fragment distribution (rA, rB) of length N in a
        single shot (Top-K ranks + tail + remainder).

        The last fragment is always used as a remainder to enforce exact
        conservation per phase.

        Parameters
        ----------
        A : float
            Total particle area at runtime.
        X1 : float
            Fraction of phase 1 at runtime.
        rng : np.random.Generator
            Random number generator.
        N : int or None
            Number of fragments to generate. If None, defaults to self.NO_FRAG
            (consistent with the online LMC semantics).
        K_use : int or None
            Number of Top-K ranks to sample explicitly. Defaults to
            min(K, N-1). The remaining (N-1-K_use) fragments are modeled
            as tail + remainder.
        tail_strategy : {"equal", "beta"}
            Strategy for distributing the tail mass between phases:
              - "equal": A and B share tail mass equally (pA = 0.5),
              - "beta":  use Beta(a_tail, b_tail) when available.

        Returns
        -------
        rA : list of float
        rB : list of float
            Relative volumes of phase A and B in each fragment.
            len(rA) == len(rB) == N, and:
              sum(rA) = 1, sum(rB) = 1 (up to numerical rounding).
        """
        (
            rel1,
            rel3,
            rowsum_list,
            rowcdf_list,
            zmin1,
            zmin3,
            EN,
            tail_T_cdf,
            tmode,
            a_tail,
            b_tail,
        ) = self.get_rank_tables(A, X1)

        Nbins = self.N

        # Default number of fragments, consistent with online LMC
        if N is None:
            N = max(2, int(self.NO_FRAG))

        # The last fragment is the remainder
        pick = max(0, N - 1)
        K_use = min(self.rank_K, pick) if (K_use is None) else min(int(K_use), pick)

        # ==== Use freq_adj when K_use < pick to choose which ranks to sample explicitly ====
        A_lookup = self._A_lookup(A)
        freq = self._interp_freq_adj(A_lookup, X1)  # length K, sum to 1
        # Pick the K_use most important ranks (larger freq), then sort them
        selected = np.argsort(-freq)[:K_use]
        selected = np.sort(selected)

        rA: List[float] = []
        rB: List[float] = []

        # ==== 1. Explicitly sample the selected K_use ranks ====
        for ridx in selected.tolist():
            rs = rowsum_list[ridx]
            rc = rowcdf_list[ridx]

            u1 = float(rng.random())
            i = int(np.searchsorted(rs, u1, side="right"))
            i = min(i, Nbins - 1)

            row = rc[i]
            u2 = float(rng.random())
            j = int(np.searchsorted(row, u2, side="right"))
            j = min(j, Nbins - 1)

            # Map to relative fractions
            rAi = float(rel1[i])
            rBj = float(rel3[j])

            # --- Local minimum-volume safeguard ---
            vt_rel = rAi * X1 + rBj * (1.0 - X1)
            vt_rel_min = self.A0_run / max(A, 1e-20)
            if vt_rel < vt_rel_min:
                denom1 = A * X1
                denom3 = A * (1.0 - X1)
                if denom1 > self.eps and rAi <= 0.0:
                    rAi = max(rAi, zmin1)
                elif denom3 > self.eps and rBj <= 0.0:
                    rBj = max(rBj, zmin3)
                else:
                    need = vt_rel_min - vt_rel
                    if need > 0.0:
                        if denom1 > self.eps and (X1 >= 0.5 or denom3 <= self.eps):
                            rAi += need / max(X1, 1e-20)
                        elif denom3 > self.eps:
                            rBj += need / max(1.0 - X1, 1e-20)

            rA.append(rAi)
            rB.append(rBj)

        # ==== 2. Sample the tail for the remaining (pick - K_use) fragments ====
        tail_count = pick - K_use
        if tail_count > 0:
            uT = float(rng.random())
            kT = int(np.searchsorted(tail_T_cdf, uT, side="right"))
            kT = min(kT, Nbins - 1)
            T = (kT + 0.5) / Nbins
            T = max(0.0, min(1.0, T))

            if (
                tail_strategy == "beta"
                and tmode == "beta"
                and (a_tail is not None)
                and (b_tail is not None)
                and a_tail > 0
                and b_tail > 0
            ):
                pA = float(rng.beta(a_tail, b_tail))
            else:
                pA = 0.5

            tiny = T / max(1, tail_count)
            for _ in range(tail_count):
                rA.append(pA * tiny)
                rB.append((1.0 - pA) * tiny)

        # ==== 3. Pre-normalize if the explicit ranks exceeded total mass ====
        sumA = float(np.sum(rA))
        sumB = float(np.sum(rB))
        if (sumA > 1.0) or (sumB > 1.0):
            gA = 1.0 / max(sumA, 1e-20)
            gB = 1.0 / max(sumB, 1e-20)
            g = min(gA, gB)
            for t in range(len(rA)):
                rA[t] *= g
                rB[t] *= g

        # ==== 4. Add final remainder fragment to ensure exact conservation ====
        sA = float(np.sum(rA))
        sB = float(np.sum(rB))
        rA_last = max(0.0, 1.0 - sA)
        rB_last = max(0.0, 1.0 - sB)
        rA.append(rA_last)
        rB.append(rB_last)

        # ==== 5. Clip bounds [0,1] ====
        rA = [max(0.0, min(1.0, x)) for x in rA]
        rB = [max(0.0, min(1.0, y)) for y in rB]

        # ==== 6. Global minimum-volume safeguard ====
        vt_rel_min = self.A0_run / max(A, 1e-20)
        X3 = 1.0 - X1
        vt_rel = np.array(rA) * X1 + np.array(rB) * X3
        too_small = vt_rel < vt_rel_min
        if np.any(too_small):
            for idx in np.where(too_small)[0]:
                need = vt_rel_min - vt_rel[idx]
                if X1 >= 0.5:
                    rA[idx] += need / max(X1, 1e-20)
                else:
                    rB[idx] += need / max(X3, 1e-20)

            # Renormalize again to conserve each phase
            sA = float(np.sum(rA))
            sB = float(np.sum(rB))
            gA = 1.0 / max(sA, 1e-20)
            gB = 1.0 / max(sB, 1e-20)
            g = min(gA, gB)
            for t in range(len(rA)):
                rA[t] *= g
                rB[t] *= g

        # ==== 7. Final per-phase normalization (sum(rA) = sum(rB) = 1) ====
        sumA_final = float(np.sum(rA))
        sumB_final = float(np.sum(rB))
        scaleA = (1.0 / sumA_final) if sumA_final > 0 else 0.0
        scaleB = (1.0 / sumB_final) if sumB_final > 0 else 0.0
        for t in range(len(rA)):
            rA[t] *= scaleA
            rB[t] *= scaleB

        return rA, rB

# ==============================
# Small-particle policy signals
# ==============================
class LMCLiveFallback(Exception):
    """
    Raised when the parent particle does not contain enough lattice cells
    to perform an online LMC breakage according to the current settings.
    The PBE solver is expected to apply its own fallback strategy
    (e.g. table/rank-based model or uniform splitting).
    """
    pass

class LMCLiveDisable(Exception):
    """
    Raised when the parent particle does not contain enough lattice cells
    to perform an online LMC breakage, and the policy is to mark this
    particle as non-breakable (breakage probability = 0).
    """
    pass

class LMCLiveAdapter:
    """
    Online LMC adapter that samples one full set of fragments per call.

    Features
    --------
    - Dynamic convergence of NO_FRAG:
        NO_FRAG_eff = min(NO_FRAG, floor(A / A0_run - delta_cells)), but â‰¥ 2.
    - If A / A0_run < 2:
        * small_particle_policy = 'fallback':
              raise LMCLiveFallback, so the solver can fall back to a
              rank/table-based model.
        * small_particle_policy = 'disable':
              raise LMCLiveDisable, so the solver can treat this particle
              as unbreakable in the current step.
    """

    def __init__(self) -> None:
        # LMC parameters (can be overridden in configure_simulator)
        self.STR = np.array([1.0, 1.0, 1.0], dtype=float)
        self.NO_FRAG = 4
        self.gamma = 1.0
        self.allow_loops = False
        self.accept_all_cracks = False
        self.use_weighted_start = True
        self.aspect_ratio = 1.0
        self.int_bre = 0.0
        # Runtime cell area (we treat volume numerically as area here)
        self.A0_run = 1.0

        # Small-particle handling strategy and safety margin
        # 'fallback' -> raise LMCLiveFallback, 'disable' -> raise LMCLiveDisable
        self.small_particle_policy = "fallback"
        self.delta_cells = 0.1  # safety margin on A / A0_run
        self.warn_pool_out_of_bounds = True

        self._sim: Optional[LMCSimulator] = None

    def configure_simulator(
        self,
        *,
        STR: Optional[np.ndarray] = None,
        NO_FRAG: Optional[int] = None,
        gamma: Optional[float] = None,
        allow_loops: Optional[bool] = None,
        accept_all_cracks: Optional[bool] = None,
        use_weighted_start: Optional[bool] = None,
        aspect_ratio: Optional[float] = None,
        int_bre: Optional[float] = None,
        A0_run: Optional[float] = None,
        small_particle_policy: Optional[str] = None,  # 'fallback' | 'disable'
        delta_cells: Optional[float] = None,
        pool_dir: Optional[str] = None,
        Df: Optional[float] = None,
        MAS: Optional[float] = None,
        warn_pool_out_of_bounds: Optional[bool] = None,
        rebuild: bool = True,
    ) -> None:
        """
        Configure the internal LMCSimulator and various runtime settings.

        Parameters
        ----------
        STR : (3,) array-like, optional
            Relative strengths for the three bond directions in the lattice.
        NO_FRAG : int, optional
            Target number of fragments in each breakage event (upper bound,
            dynamically reduced if the particle is too small).
        gamma : float, optional
            Energy scaling factor in the LMC model.
        allow_loops : bool, optional
            If True, allow crack loops in the LMC simulation.
        accept_all_cracks : bool, optional
            If True, accept all cracks without additional energy filtering.
        use_weighted_start : bool, optional
            If True, use weighted starting points for crack growth.
        aspect_ratio : float, optional
            Aspect ratio of the initial aggregate grid.
        int_bre : float, optional
            Internal breakage parameter coupled to the LMC energy.
        A0_run : float, optional
            Runtime lattice cell area (or effective area per unit).
        small_particle_policy : {"fallback", "disable"}, optional
            Behavior when the particle is too small for online LMC:
            'fallback' raises LMCLiveFallback, 'disable' raises LMCLiveDisable.
        delta_cells : float, optional
            Safety margin subtracted from A / A0_run before computing
            effective NO_FRAG.
        pool_dir : str, optional
            Directory containing pre-generated aggregate pools for
            mc_breakage_from_pool.
        Df : float, optional
            Fractal dimension tag used when selecting aggregates from the pool.
        MAS : float, optional
            MischgÃ¼te (mixing quality) tag used for selecting aggregates.
        warn_pool_out_of_bounds : bool, optional
            Print a one-time warning when the requested normalized area or
            X1 lies outside the selected aggregate pool coverage.
        rebuild : bool, default True
            If True or if no simulator exists yet, build a new LMCSimulator.
        """
        if STR is not None:
            self.STR = np.asarray(STR, dtype=float)
        if NO_FRAG is not None:
            self.NO_FRAG = int(NO_FRAG)
        if gamma is not None:
            self.gamma = float(gamma)
        if allow_loops is not None:
            self.allow_loops = bool(allow_loops)
        if accept_all_cracks is not None:
            self.accept_all_cracks = bool(accept_all_cracks)
        if use_weighted_start is not None:
            self.use_weighted_start = bool(use_weighted_start)
        if aspect_ratio is not None:
            self.aspect_ratio = float(aspect_ratio)
        if int_bre is not None:
            self.int_bre = float(int_bre)
        if A0_run is not None:
            self.A0_run = float(A0_run)
        if Df is not None:
            self.Df = float(Df)
        if MAS is not None:
            self.MAS = float(MAS)
        if warn_pool_out_of_bounds is not None:
            self.warn_pool_out_of_bounds = bool(warn_pool_out_of_bounds)
        if small_particle_policy is not None:
            if small_particle_policy not in ("fallback", "disable"):
                raise ValueError("small_particle_policy must be 'fallback' or 'disable'")
            self.small_particle_policy = small_particle_policy
        if pool_dir is not None:
            self.pool_dir = pool_dir
        if delta_cells is not None:
            self.delta_cells = float(delta_cells)

        if rebuild or (self._sim is None):
            self._sim = LMCSimulator(
                STR=self.STR,
                NO_FRAG=self.NO_FRAG,
                gamma=self.gamma,
                allow_loops=self.allow_loops,
                accept_all_cracks=self.accept_all_cracks,
                use_weighted_start=self.use_weighted_start,
                plotter=None,
                pool_dir=self.pool_dir,
                warn_pool_out_of_bounds=self.warn_pool_out_of_bounds,
            )

    def _AX1_from_Vparent(self, V_parent: np.ndarray) -> Tuple[float, float]:
        """
        Convert parent phase volumes V_parent into (A, X1).

        Cases
        -----
        - len(V_parent) == 1:
            Single-phase parent, treated as A = V_parent[0], X1 = 1.
        - len(V_parent) == 2:
            Two-phase parent, A = V_A + V_B, X1 = V_A / (V_A + V_B).
        """
        if V_parent.size == 1:
            A = float(V_parent[0])
            X1 = 1.0
        else:
            VA, VB = float(V_parent[0]), float(V_parent[1])
            A = VA + VB
            X1 = (VA / A) if A > 0 else 0.5
        return A, X1

    def sample_one_shot(
        self,
        V_parent: np.ndarray,
        rng: np.random.Generator,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Perform a single LMC breakage for one parent particle and return
        the resulting fragments.

        Parameters
        ----------
        V_parent : array-like, shape (1,) or (2,)
            Parent phase volumes:
              - [V]   for single-phase,
              - [V_A, V_B] for two-phase.
        rng : np.random.Generator
            Random number generator used to derive an integer seed.
        seed : int or None, optional
            If provided, overrides rng-based seeding for reproducibility.

        Returns
        -------
        frags : list of np.ndarray
            List of fragment volumes. Each element is either shape (1,)
            or (2,), matching the phase count of the parent.
        energy : float
            LMC energy of the accepted crack configuration.

        Raises
        ------
        LMCLiveFallback
            If the particle is too small for online LMC and
            small_particle_policy == "fallback".
        LMCLiveDisable
            If the particle is too small for online LMC and
            small_particle_policy == "disable".
        """
        if self._sim is None:
            self.configure_simulator(rebuild=True)

        V_parent = np.asarray(V_parent, dtype=float)
        if V_parent.ndim != 1 or V_parent.size not in (1, 2):
            raise ValueError("V_parent must be a 1D array of length 1 or 2.")

        A, X1 = self._AX1_from_Vparent(V_parent)

        if A <= 0.0:
            # Degenerate case: non-positive volume, return unchanged
            return [V_parent.copy()], 0.0

        # Dynamic NO_FRAG based on available cell count
        n_cells = A / self.A0_run
        NO_FRAG_raw = int(np.floor(max(n_cells - self.delta_cells, 0.0)))

        # Small-particle check (compared against configured NO_FRAG)
        if NO_FRAG_raw < self.NO_FRAG:
            if self.small_particle_policy == "fallback":
                raise LMCLiveFallback()
            else:
                raise LMCLiveDisable()

        NO_FRAG_eff = min(self.NO_FRAG, NO_FRAG_raw)

        # Temporarily override simulator NO_FRAG
        old_nf = self._sim.NO_FRAG
        self._sim.NO_FRAG = int(NO_FRAG_eff)

        seed_use = int(seed) if (seed is not None) else int(rng.integers(0, 2**31 - 1))

        # Legacy online LMC path (kept for reference):
        # F = self._sim.mc_breakage_repeat(
        #     A=A, X1=X1, X2=(1.0 - X1),
        #     N_GRIDS=1, N_FRACS=1,
        #     A0=self.A0_run,
        #     aspect_ratio=self.aspect_ratio,
        #     int_bre=self.int_bre,
        #     seed=seed_use,
        #     plot_each=False,
        # )

        # Current path: use aggregate pool + mc_breakage_from_pool
        F = self._sim.mc_breakage_from_pool(
            pool_dir=self.pool_dir,
            Df=self.Df,
            MAS=self.MAS,
            A=A,
            X1=X1,
            N_GRIDS=1,
            N_FRACS=1,
            A0=self.A0_run,
            int_bre=self.int_bre,
            interp="bilinear",  # "knn" or "bilinear"
            seed=seed_use,
            plot_each=False,
        )

        # Restore original NO_FRAG
        self._sim.NO_FRAG = old_nf

        if F.size == 0:
            # No valid breakage (e.g. no crack found): return parent unchanged
            return [V_parent.copy()], 0.0

        VT = F[:, 0]
        valid = VT > 0.0
        energy = float(F[0, 3])

        frags: List[np.ndarray] = []

        # Single-phase case
        if V_parent.size == 1:
            parts = VT[valid].astype(float)
            s = float(np.sum(parts))
            if s <= 0.0:
                return [V_parent.copy()], energy

            # Scale to parent volume (mass conservation)
            parts *= (V_parent[0] / s)
            for p in parts:
                frags.append(np.array([float(p)], dtype=float))
            return frags, energy

        # Two-phase case
        else:
            VA_arr = F[valid, 1].astype(float)
            VB_arr = F[valid, 2].astype(float)
            if VA_arr.size == 0:
                return [V_parent.copy()], energy

            # NOTE: optional scaling for phase-wise exact conservation
            # has been commented out in the original code. If strict
            # projection is needed, it can be re-enabled.

            for a, b in zip(VA_arr, VB_arr):
                frags.append(np.array([float(a), float(b)], dtype=float))

            return frags, energy

# ==============================
# Copula-based (Top-K stick-breaking) adapter
# ==============================
try:
    import pyvinecopulib as pv
    _PV_OK = True
except Exception:
    _PV_OK = False

try:
    from scipy.stats import beta as sp_beta
    from scipy.stats import norm as sp_norm
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

_C_EPS = 1e-12
_C_CLIP = 1e-6


def _c_clip01(x: np.ndarray, eps: float = _C_CLIP) -> np.ndarray:
    """Numerically safe clipping to (eps, 1-eps)."""
    return np.clip(x, eps, 1.0 - eps)


class _C_ECDFMarginal:
    """
    Simple empirical CDF marginal defined on discrete support xs with
    cumulative probabilities ps.

    Used for:
    - cdf(x):   interpolate ps on the grid xs,
    - ppf(u):   inverse-CDF via interpolation on (ps, xs).
    """

    def __init__(self, xs: np.ndarray, ps: np.ndarray):
        self.xs = np.asarray(xs, dtype=float)
        self.ps = np.asarray(ps, dtype=float)
        if self.xs.ndim != 1 or self.ps.ndim != 1:
            raise ValueError("ECDF marginal expects 1D xs, ps.")
        if self.xs.size != self.ps.size:
            raise ValueError("ECDF marginal: xs and ps must have same length.")
        # Optional: could enforce ps[0] ~ 0, ps[-1] ~ 1 by extending tails.
        # Currently left as-is.

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the empirical CDF at x by linear interpolation.
        """
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.xs, self.ps, left=self.ps[0], right=self.ps[-1])

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """
        Evaluate the empirical inverse-CDF at u by interpolation.

        Values of u are clipped to [ps[0], ps[-1]].
        """
        u = np.asarray(u, dtype=float)
        u = np.clip(u, self.ps[0], self.ps[-1])
        return np.interp(u, self.ps, self.xs)


class _C_BetaMarginal:
    """
    Beta distribution marginal used in the training pipeline.

    Provides:
      - cdf(x),
      - ppf(u),
      - pdf(x),

    falling back to numeric integration / bisection if SciPy is not available.
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if _SCIPY_OK:
            return sp_beta.cdf(x, self.alpha, self.beta, loc=0.0, scale=1.0)
        # Fallback: approximate CDF by numerical integration of PDF
        xs = np.linspace(0.0, 1.0, 2049)
        pdf = self.pdf(xs)
        c = np.cumsum(pdf)
        c /= c[-1]
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(xs, np.clip(x, 0.0, 1.0), side="right")
        idx = np.clip(idx, 1, xs.size - 1)
        w = (x - xs[idx - 1]) / (xs[idx] - xs[idx - 1] + _C_EPS)
        return c[idx - 1] * (1 - w) + c[idx] * w

    def ppf(self, u: np.ndarray) -> np.ndarray:
        if _SCIPY_OK:
            return sp_beta.ppf(_c_clip01(u), self.alpha, self.beta, loc=0.0, scale=1.0)
        # Fallback: bisection on [0,1]
        u = _c_clip01(u)
        lo = np.zeros_like(u)
        hi = np.ones_like(u)
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            cm = self.cdf(mid)
            lo = np.where(cm < u, mid, lo)
            hi = np.where(cm >= u, mid, hi)
        return 0.5 * (lo + hi)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if _SCIPY_OK:
            return sp_beta.pdf(x, self.alpha, self.beta, loc=0.0, scale=1.0)
        # Fallback: direct Beta(a,b) density
        import math
        from math import lgamma
        a, b = self.alpha, self.beta
        x = np.clip(x, 0.0, 1.0)
        B = math.exp(lgamma(a) + lgamma(b) - lgamma(a + b))
        return np.where(
            (x > 0) & (x < 1),
            x**(a - 1) * (1 - x) ** (b - 1) / (B + _C_EPS),
            0.0,
        )


class _C_GaussianCopula:
    """
    Simple Gaussian copula wrapper.

    Parameters
    ----------
    R : (d, d) array
        Correlation matrix (or covariance-like). Eigenvalues are clamped
        to keep it positive definite before Cholesky factorization.
    """

    def __init__(self, R: np.ndarray):
        R = np.asarray(R, dtype=float)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-6, None)
        self.R = (V * w) @ V.T
        self.L = np.linalg.cholesky(self.R)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample n points from the Gaussian copula and return U in (0,1)^d.
        """
        Z = rng.standard_normal(size=(int(n), self.R.shape[0])) @ self.L.T
        if _SCIPY_OK:
            U = sp_norm.cdf(Z)
        else:
            U = 0.5 * (1.0 + np.erf(Z / np.sqrt(2.0)))
        return _c_clip01(U, 1e-12)


class _C_VineCopula:
    """
    Wrapper around pyvinecopulib's Vinecop model stored as JSON.
    """

    def __init__(self, json_blob: str):
        if not _PV_OK:
            raise RuntimeError("pyvinecopulib not available.")
        self.model = pv.Vinecop.from_json(json_blob)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample n points from the vine copula and return U in (0,1)^d.
        """
        seed = int(rng.integers(0, 2**31 - 1))
        return self.model.simulate(int(n), seeds=[seed])


class LMCCopulaAdapter(LMCBaseAdapter):
    """
    Adapter for copula-based Top-K stick-breaking models.

    It reads an `lmc_copula_grid.npz` file produced by the new copula
    preprocessing script. For each (A, X1) grid cell there may be a model,
    stored in `self.models[i, j]`, describing:

      - pure-phase model (is_pure = True):
          * K-dimensional copula over Y_1..Y_K (stick-breaking logits),
          * only one phase is present (A or B),
          * Y is transformed via inverse stick-breaking into fragment
            sizes z_1..z_K and remainder.

      - mixed-phase model (is_pure = False):
          * K-dimensional copula over Y_1..Y_K,
          * for each rank k, an independent marginal pA_k for the A-fraction,
          * no copula is used for pA itself; they are sampled independently.

    This adapter samples a single breakage event and returns (rA, rB)
    arrays of relative volumes, using the same interface as
    `LMCRankAdapter.sample_one_shot`.
    """

    def __init__(
        self,
        npz_path: str,
        interp: str = "bilinear",
        A0_run: float | None = None,
        cache_enabled: bool = True,
    ):
        d = np.load(npz_path, allow_pickle=True)
        A_grid = np.asarray(d["A_grid"], dtype=float)
        X1_grid = np.asarray(d["X1_grid"], dtype=float)
        meta = d["meta"].item() if isinstance(d["meta"], np.ndarray) else dict(d["meta"])

        super()._init_common(
            A_grid,
            X1_grid,
            meta,
            interp=interp,
            A0_run=A0_run,
            cache_enabled=cache_enabled,
        )

        # models: object[nA, nX] of dict or None
        self.models = d["models"]
        # Global K is a default; each cell may also store its own K.
        self.K = int(self.meta.get("stick_breaking_K", 4))

    # ---------- helpers ----------
    @staticmethod
    def _inv_stick_breaking(Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Inverse stick-breaking transform.

        Given Y in [0,1]^K, produce:
          z_k = Y_k * remaining_mass, with remaining_mass updated each step.

        Returns
        -------
        z : (K,) array
            First K stick components.
        remain : float
            Remaining tail fraction after K sticks.
        """
        K = Y.size
        z = np.zeros(K, dtype=float)
        remain = 1.0
        for k in range(K):
            yk = float(np.clip(Y[k], 0.0, 1.0))
            z[k] = yk * remain
            remain = max(0.0, remain - z[k])
        return z, float(remain)

    def _pick_cell_model(
        self,
        A_lookup: float,
        X1: float,
        want_pure: bool,
        rng: np.random.Generator,
    ):
        """
        Pick one model cell among the 4 neighbors around (A_lookup, X1),
        using the bilinear weights as probabilities.

        Preference is given to cells whose is_pure flag matches
        `want_pure`. If none match, falls back to cells with mismatched
        type.

        Returns
        -------
        (i, j, w, model_obj) or None
            Indices, weight, and the model object stored in `self.models`.
        """
        i0, i1, j0, j1, w00, w01, w10, w11 = self._neighbors_weights(A_lookup, X1)
        cand = [
            (i0, j0, w00),
            (i0, j1, w01),
            (i1, j0, w10),
            (i1, j1, w11),
        ]

        same_type: List[Tuple[int, int, float, Dict[str, Any]]] = []
        other_type: List[Tuple[int, int, float, Dict[str, Any]]] = []

        for (i, j, w) in cand:
            obj = self.models[i, j]
            if obj is None:
                continue
            is_pure_cell = bool(obj.get("is_pure", False))
            if is_pure_cell == want_pure:
                same_type.append((i, j, w, obj))
            else:
                other_type.append((i, j, w, obj))

        def _pick(lst):
            ws = np.array([x[2] for x in lst], dtype=float)
            ws = np.maximum(ws, 0.0)
            if ws.sum() <= 0.0:
                ws[:] = 1.0
            ws /= ws.sum()
            idx = int(rng.choice(len(lst), p=ws))
            return lst[idx]

        if same_type:
            return _pick(same_type)
        if other_type:
            return _pick(other_type)
        return None

    def _sample_from_cell(
        self,
        model_obj: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Sample (Y, pA) from a single grid cell model.

        Steps
        -----
        1) Reconstruct Y marginals (ECDF or Beta) and sample Y via:
           - draw U_y from a K-dimensional copula (vine or Gaussian),
           - apply inverse marginals Y_k = F_k^{-1}(U_y[k]).
        2) If the cell is mixed-phase (is_pure = False) and pA_marginals
           are provided, sample pA_k independently per rank from those
           marginals. If pA_marginals is missing, pA is returned as None.

        Parameters
        ----------
        model_obj : dict
            Model specification for a given (A, X1) cell.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        Y : (K,) array
            Stick-breaking parameters Y_k in [0,1].
        pA : (K,) array or None
            A-fraction per rank (for mixed-phase cells), or None if absent.
        is_pure_cell : bool
            True if the model is pure-phase in this cell.
        """
        is_pure_cell = bool(model_obj.get("is_pure", False))
        K = int(model_obj.get("K", self.K))

        # 1) Restore Y marginals
        y_mj = model_obj["y_marginals"]
        y_marginals = []
        for m in y_mj:
            mtype = m.get("type", "beta")
            if mtype == "ecdf":
                xs = np.asarray(m["xs"], dtype=float)
                ps = np.asarray(m["ps"], dtype=float)
                y_marginals.append(_C_ECDFMarginal(xs, ps))
            else:
                # Backward-compatible Beta format
                y_marginals.append(_C_BetaMarginal(float(m["alpha"]), float(m["beta"])))
        if len(y_marginals) != K:
            raise RuntimeError(f"cell: y_marginals length {len(y_marginals)} != K {K}")

        # 2) Restore Y copula
        ycop = model_obj["y_copula"]
        if ycop.get("type") == "vine":
            if not _PV_OK:
                raise RuntimeError("vine model present but pyvinecopulib not installed.")
            backend = _C_VineCopula(ycop["json"])
            U_y = backend.sample(1, rng).reshape(-1)
        else:
            R = np.asarray(ycop["R"], dtype=float)
            backend = _C_GaussianCopula(R)
            U_y = backend.sample(1, rng).reshape(-1)

        # 3) Invert marginals to obtain Y
        Y = np.zeros(K, dtype=float)
        for d in range(K):
            Y[d] = y_marginals[d].ppf(np.array([U_y[d]], dtype=float))[0]
        Y = _c_clip01(Y)

        # 4) pA part (mixed-phase only; independent 1D marginals per rank)
        if is_pure_cell:
            return Y, None, True

        pA_list_raw = model_obj.get("pA_marginals", None)
        if pA_list_raw is None:
            # No pA stored; the caller will use global X1 and feasibility.
            return Y, None, False

        pA = np.zeros(K, dtype=float)
        for d, m in enumerate(pA_list_raw):
            mtype = m.get("type", "beta")
            u = rng.random()
            if mtype == "ecdf":
                xs = np.asarray(m["xs"], dtype=float)
                ps = np.asarray(m["ps"], dtype=float)
                mm = _C_ECDFMarginal(xs, ps)
                pA[d] = mm.ppf(np.array([u], dtype=float))[0]
            else:
                bm = _C_BetaMarginal(float(m["alpha"]), float(m["beta"]))
                pA[d] = bm.ppf(np.array([u], dtype=float))[0]
        pA = _c_clip01(pA)

        return Y, pA, False

    # ---------- public API ----------
    def sample_one_shot(
        self,
        A: float,
        X1: float,
        rng: np.random.Generator,
        N: Optional[int] = None,
        K_use: Optional[int] = None,
        tail_strategy: str = "equal",
    ) -> Tuple[List[float], List[float]]:
        """
        Sample a single breakage event using the copula-based Top-K
        stick-breaking model.

        Parameters
        ----------
        A : float
            Total parent volume (treated as area in the lattice model).
        X1 : float
            Fraction of phase 1 (A-phase) of the parent.
        rng : np.random.Generator
            Random number generator.
        N : int or None
            Desired number of fragments. If None, defaults to self.NO_FRAG.
            The first (N-1) fragments are generated explicitly, the last
            one is used as a remainder to guarantee conservation.
        K_use : int or None
            Number of Top-K sticks used explicitly. Defaults to
            min(self.K, N-1). The rest of the mass is treated as tail.
        tail_strategy : str
            Currently kept for API symmetry with rank-based models.
            For now, only 'equal' semantics are effectively used.

        Returns
        -------
        rA : list of float
        rB : list of float
            Relative volumes of phase A and B in each fragment. The lists
            have length N and (up to numerical precision):
                sum(rA) = sum(rB) = 1.
        """
        if N is None:
            N = max(2, int(self.NO_FRAG))
        pick = max(0, N - 1)

        X1 = float(np.clip(X1, 0.0, 1.0))
        pure_A_req = (X1 >= 1.0 - self.eps)
        pure_B_req = (X1 <= self.eps)
        want_pure = pure_A_req or pure_B_req

        # Small-particle policy: optionally disable tables when too few cells
        if self.small_particle_policy == "disable":
            if not self.eligible_for_tables(A):
                if pure_A_req:
                    return [1.0], [0.0]
                elif pure_B_req:
                    return [0.0], [1.0]
                else:
                    return [X1], [1.0 - X1]

        A_lookup = self._A_lookup(A)
        picked = self._pick_cell_model(A_lookup, X1, want_pure, rng)
        if picked is None:
            # No usable cell found â†’ fallback: equal splitting
            z = np.full(pick, 1.0 / max(N, 1), dtype=float)
            if pure_A_req:
                rA = z.tolist()
                rB = [0.0] * pick
            elif pure_B_req:
                rA = [0.0] * pick
                rB = z.tolist()
            else:
                rA = (X1 * z / max(X1, 1e-12)).tolist()
                rB = ((1.0 - X1) * z / max(1.0 - X1, 1e-12)).tolist()
            rA.append(max(0.0, 1.0 - float(np.sum(rA))))
            rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB

        i_sel, j_sel, w_sel, model_obj = picked

        try:
            Y, pA, is_pure_cell = self._sample_from_cell(model_obj, rng)
        except Exception:
            # Any failure inside the cell â†’ fallback: equal splitting
            z = np.full(pick, 1.0 / max(N, 1), dtype=float)
            if pure_A_req:
                rA = z.tolist()
                rB = [0.0] * pick
            elif pure_B_req:
                rA = [0.0] * pick
                rB = z.tolist()
            else:
                rA = (X1 * z / max(X1, 1e-12)).tolist()
                rB = ((1.0 - X1) * z / max(1.0 - X1, 1e-12)).tolist()
            rA.append(max(0.0, 1.0 - float(np.sum(rA))))
            rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB

        # Inverse stick-breaking: first K sticks and tail mass T
        z_all, T = self._inv_stick_breaking(Y)
        if K_use is None:
            K_use = min(self.K, pick)
        else:
            K_use = min(int(K_use), pick)
        z = z_all[:K_use]

        rA: List[float] = []
        rB: List[float] = []

        if is_pure_cell or pure_A_req or pure_B_req:
            # Pure-phase case
            if pure_A_req or (is_pure_cell and X1 >= 0.5):
                for zk in z:
                    rA.append(float(zk))
                    rB.append(0.0)
            else:
                for zk in z:
                    rA.append(0.0)
                    rB.append(float(zk))
        else:
            # Mixed-phase case: pA is drawn from independent marginals
            X3 = 1.0 - X1
            for idx in range(K_use):
                zk = float(np.clip(z[idx], 0.0, 1.0))
                if zk <= 0.0:
                    rA.append(0.0)
                    rB.append(0.0)
                    continue
                if pA is None:
                    # No pA model provided: fall back to global X1
                    p = float(np.clip(X1, 0.0, 1.0))
                else:
                    p = float(np.clip(pA[idx], 0.0, 1.0))

                # Feasible region for p given zk, X1, X3
                Lk = max(0.0, 1.0 - X3 / max(zk, _C_EPS))
                Uk = min(1.0, X1 / max(zk, _C_EPS))
                p = float(np.clip(p, Lk, Uk))

                # Convert to rA, rB so that summed over fragments they match phase totals
                rA.append(zk * p / max(X1, _C_EPS))
                rB.append(zk * (1.0 - p) / max(X3, _C_EPS))

        # Tail mass (remaining T) is split evenly across the remaining fragments
        tail_count = pick - K_use
        if tail_count > 0:
            t = float(T) / max(1, tail_count)
            if pure_A_req:
                rA.extend([t] * tail_count)
                rB.extend([0.0] * tail_count)
            elif pure_B_req:
                rA.extend([0.0] * tail_count)
                rB.extend([t] * tail_count)
            elif is_pure_cell:
                rA.extend([t] * tail_count)
                rB.extend([0.0] * tail_count)
            else:
                for _ in range(tail_count):
                    # By default, split tail mass equally in A/B, then normalize
                    rA.append(0.5 * t / max(X1, 1e-12))
                    rB.append(0.5 * t / max(1.0 - X1, 1e-12))

        # Append final remainder fragment for exact conservation
        sA = float(np.sum(rA))
        sB = float(np.sum(rB))
        rA.append(max(0.0, 1.0 - sA))
        rB.append(max(0.0, 1.0 - sB))

        # Minimum volume safeguard + normalization
        vt_rel_min = self.A0_run / max(A, 1e-20)
        X3 = 1.0 - X1
        vt_rel = np.array(rA) * X1 + np.array(rB) * X3
        too_small = vt_rel < vt_rel_min
        if np.any(too_small):
            for idx in np.where(too_small)[0]:
                need = vt_rel_min - vt_rel[idx]
                if X1 >= 0.5:
                    rA[idx] += need / max(X1, 1e-12)
                else:
                    rB[idx] += need / max(X3, 1e-12)
            sA = float(np.sum(rA))
            sB = float(np.sum(rB))
            gA = 1.0 / max(sA, 1e-20)
            gB = 1.0 / max(sB, 1e-20)
            for t in range(len(rA)):
                rA[t] *= gA
                rB[t] *= gB

        # Final enforcement sum(rA) = sum(rB) = 1 (up to rounding)
        sA = float(np.sum(rA))
        sB = float(np.sum(rB))
        if abs(sA - 1.0) > 1e-12:
            gA = 1.0 / max(sA, 1e-20)
            for t in range(len(rA)):
                rA[t] *= gA
        if abs(sB - 1.0) > 1e-12:
            gB = 1.0 / max(sB, 1e-20)
            for t in range(len(rB)):
                rB[t] *= gB

        return rA, rB

# -----------------------------------------------
#  Flow-based (conditional RealNVP) é€‚é…å™¨ï¼ˆpure / mix åŒæ¨¡åž‹ + K-1 ç»´ï¼‰
# -----------------------------------------------
try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


class LMCFlowAdapter(LMCBaseAdapter):
    """
    ä½¿ç”¨ç¦»çº¿è®­ç»ƒå¥½çš„æ¡ä»¶æµæ¨¡åž‹ï¼Œåˆ†åˆ«é’ˆå¯¹çº¯å‡€ç‰©å’Œæ··åˆç‰©å»ºç«‹ä¸¤å¥—æ¨¡åž‹ï¼š
      - pure æ¨¡åž‹ï¼štarget_dim = K-1ï¼Œåªé¢„æµ‹å‰ K-1 å—çš„ stick-breaking ä½“ç§¯åˆ†å¸ƒ
      - mix  æ¨¡åž‹ï¼štarget_dim = 2*(K-1)ï¼Œé¢„æµ‹å‰ K-1 å—çš„ stick-breaking + å„å—çš„ pA
    adapter è´Ÿè´£ï¼š
      1) æ ¹æ® X1 åˆ¤å®šç”¨å“ªå¥—æ¨¡åž‹
      2) ç”¨å‰©ä½™é‡è¡¥ç¬¬ K å—
      3) å¯¹æ··åˆç‰©åšå¯è¡ŒåŸŸæŠ•å½±å¹¶æ‹†æˆ rA / rB
    è°ƒç”¨æŽ¥å£ä¿æŒå’Œ rank / copula ä¸€è‡´ï¼š
        rA, rB = flow.sample_one_shot(A, X1, rng, N=None)
    """

    def __init__(self,
                 pure_model_path: str = None,
                 mix_model_path: str = None,
                 *,
                 interp: str = "bilinear",
                 A0_run: float = None,
                 cache_enabled: bool = True):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch is required for LMCFlowAdapter.")

        # æˆ‘ä»¬ä»ç„¶éœ€è¦ç½‘æ ¼ä¿¡æ¯ï¼ˆA_grid / X1_gridï¼‰æ¥å– meta å’Œ NO_FRAGï¼Œ
        # ä½† flow è®­ç»ƒæ˜¯å…¨å±€çš„ï¼Œè¿™é‡Œå°±åšä¸€ä¸ªæœ€å°ç½‘æ ¼
        A_grid = np.array([1.0], dtype=float)
        X1_grid = np.array([0.0, 1.0], dtype=float)
        meta = {"NO_FRAG": 4, "A0": 1.0}
        super()._init_common(A_grid, X1_grid, meta,
                             interp=interp, A0_run=A0_run, cache_enabled=cache_enabled)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pure_model = None
        self.pure_meta = None
        self.mix_model = None
        self.mix_meta = None

        if pure_model_path is not None:
            self.pure_model, self.pure_meta = self._load_flow_model(pure_model_path)
        if mix_model_path is not None:
            self.mix_model, self.mix_meta = self._load_flow_model(mix_model_path)

        if self.pure_meta is None and self.mix_meta is None:
            raise ValueError("LMCFlowAdapter needs at least one of pure_model_path / mix_model_path.")

        # å–ä¸€ä¸ª K åŸºå‡†
        if self.pure_meta is not None:
            self.K = int(self.pure_meta["K"])
        else:
            self.K = int(self.mix_meta["K"])

    # ====== ä¸‹é¢æ˜¯å’Œè®­ç»ƒè„šæœ¬åŒæž„çš„å‡ ä¸ªå°æ¨¡å— ======
    class _CondMLP(torch.nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, n_layers: int = 3):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(n_layers - 1):
                layers.append(torch.nn.Linear(d, hidden))
                layers.append(torch.nn.ReLU())
                d = hidden
            layers.append(torch.nn.Linear(d, out_dim))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class _RealNVPCoupling(torch.nn.Module):
        def __init__(self, dim: int, cond_dim: int, mask: torch.Tensor, hidden: int = 128):
            super().__init__()
            self.dim = dim
            self.cond_dim = cond_dim
            self.register_buffer("mask", mask)
            in_net = dim + cond_dim
            self.s_net = LMCFlowAdapter._CondMLP(in_net, dim, hidden=hidden)
            self.t_net = LMCFlowAdapter._CondMLP(in_net, dim, hidden=hidden)
            self.max_s = 2.0  # å¯ä»¥å’Œè®­ç»ƒæ—¶çš„ä¿æŒä¸€è‡´

        def forward(self, x, cond):
            m = self.mask
            x_masked = x * m
            inp = torch.cat([x_masked, cond], dim=1)
            s = self.s_net(inp).tanh() * self.max_s
            t = self.t_net(inp)
            s = s * (1.0 - m)
            t = t * (1.0 - m)
            y = x_masked + (1.0 - m) * (x * torch.exp(s) + t)
            logdet = ((1.0 - m) * s).sum(dim=1)
            return y, logdet

        def inverse(self, y, cond):
            m = self.mask
            y_masked = y * m
            inp = torch.cat([y_masked, cond], dim=1)
            s = self.s_net(inp).tanh() * self.max_s
            t = self.t_net(inp)
            s = s * (1.0 - m)
            t = t * (1.0 - m)
            x = y_masked + (1.0 - m) * ((y - t) * torch.exp(-s))
            logdet = -((1.0 - m) * s).sum(dim=1)
            return x, logdet

    class _CondRealNVP(torch.nn.Module):
        def __init__(self, dim: int, cond_dim: int, n_flows: int = 6, hidden: int = 128):
            super().__init__()
            masks = []
            for i in range(n_flows):
                if i % 2 == 0:
                    m = torch.cat([torch.ones(dim // 2), torch.zeros(dim - dim // 2)])
                else:
                    m = torch.cat([torch.zeros(dim // 2), torch.ones(dim - dim // 2)])
                masks.append(m)
            self.flows = torch.nn.ModuleList([
                LMCFlowAdapter._RealNVPCoupling(dim, cond_dim, mask=m, hidden=hidden) for m in masks
            ])
            self.dim = dim
            self.cond_dim = cond_dim
    
            # base dist = N(0,1)
            self.register_buffer("base_mu", torch.zeros(dim))
            self.register_buffer("base_logstd", torch.zeros(dim))
    
        def fwd(self, x, cond):
            logdet_sum = torch.zeros(x.size(0), device=x.device)
            h = x
            for flow in self.flows:
                h, logdet = flow(h, cond)
                logdet_sum = logdet_sum + logdet
            return h, logdet_sum
    
        def inv(self, z, cond):
            h = z
            logdet_sum = torch.zeros(z.size(0), device=z.device)
            for flow in reversed(self.flows):
                h, logdet = flow.inverse(h, cond)
                logdet_sum = logdet_sum + logdet
            return h, logdet_sum
    
        def log_prob(self, x, cond):
            z, logdet = self.fwd(x, cond)
            log_base = -0.5 * ((z - self.base_mu) ** 2 / torch.exp(self.base_logstd * 2) + math.log(2 * math.pi)).sum(dim=1)
            return log_base + logdet
    
        def sample(self, n: int, cond: torch.Tensor):
            # cond: (n, cond_dim)
            z = torch.randn(n, self.dim, device=cond.device)
            x, _ = self.inv(z, cond)
            return x

    # ====== åŠ è½½æ¨¡åž‹ ======
    def _load_flow_model(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        mk = ck["model_kwargs"]
        model = LMCFlowAdapter._CondRealNVP(
            dim=int(meta["target_dim"]),
            cond_dim=int(meta["cond_dim"]),
            n_flows=int(mk.get("n_flows", 6)),
            hidden=int(mk.get("hidden", 256)),
        ).to(self.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, meta

    @staticmethod
    def _is_pure_x1(x1: float, eps: float = 1e-6) -> bool:
        return (x1 <= eps) or (x1 >= 1.0 - eps)

    # ====== å…¬å…±çš„é‡‡æ ·å…¥å£ ======
    def sample_one_shot(
        self,
        A: float,
        X1: float,
        rng: np.random.Generator,
        N: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        ä¸€æ¬¡æ€§è¿”å›ž N ä¸ªç¢Žç‰‡ï¼ˆå‰ N-1 ç”±æ¨¡åž‹é¢„æµ‹ï¼Œæœ€åŽ 1 ä¸ªä¸ºä½™é‡ï¼‰ã€‚
        å…¼å®¹ pure/mix åŒæ¨¡åž‹ï¼ˆK-1 / 2*(K-1) ç»´ï¼‰ã€‚
        """
        if N is None:
            N = max(2, int(self.NO_FRAG))
        pick = max(1, N - 1)  # åªé¢„æµ‹å‰ N-1
    
        A = float(A)
        A_lookup = self._A_lookup(A)
        X1 = float(np.clip(X1, 0.0, 1.0))
        is_pure_req = (X1 <= self.eps) or (X1 >= 1.0 - self.eps)
    
        # é€‰æ¨¡åž‹ï¼šçº¯å‡€ä¼˜å…ˆ pureï¼Œå¦åˆ™ mixï¼›ç¼ºå“ªå¥—å°±ç”¨å¦ä¸€å¥—å…œåº•
        if is_pure_req:
            model, meta = (self.pure_model, self.pure_meta) if (self.pure_model is not None) else (self.mix_model, self.mix_meta)
        else:
            model, meta = (self.mix_model, self.mix_meta) if (self.mix_model is not None) else (self.pure_model, self.pure_meta)
    
        if model is None:
            # æžç«¯å…œåº•ï¼šå‡åˆ† + ä½™é‡
            z = np.full(pick, 1.0 / max(N, 1), dtype=float)
            rA = (X1 * z / max(X1, 1e-12)).tolist()
            rB = ((1.0 - X1) * z / max(1.0 - X1, 1e-12)).tolist()
            rA.append(max(0.0, 1.0 - float(np.sum(rA))))
            rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB
    
        # æ¡ä»¶å‘é‡ï¼ˆä¸Žè®­ç»ƒå®Œå…¨ä¸€è‡´ï¼‰ï¼š[logA, X1]
        cond_np = np.array([np.log(max(A_lookup, 1e-8)), X1], dtype=np.float32)[None, :]  # (1, 2)
        cond_t  = torch.from_numpy(cond_np).to(self.device)                         # (1, cond_dim)
    
        # é‡‡æ ·æ— ç•Œå˜é‡ï¼šä¸€æ¬¡åªè¦ 1 æ¡ï¼ˆä¸€ä¸ª K-1 æˆ– 2*(K-1) å‘é‡ï¼‰
        with torch.no_grad():
            x_u = model.sample(n=1, cond=cond_t)   # (1, target_dim)
            x_u = x_u[0]                           # (target_dim,)
    
        # ä»Žæ— ç•Œç©ºé—´æ˜ å›ž (0,1)
        if meta.get("support_transform") == "logit":
            x = torch.sigmoid(x_u).cpu().numpy()
        else:
            x = x_u.cpu().numpy()
            x = np.clip(x, 1e-6, 1.0 - 1e-6)
    
        K   = int(meta["K"])
        Km1 = K - 1
    
        # çº¯å‡€ç‰©ï¼šåªå­¦ Y[:K-1]ï¼Œstick-breaking å¾—åˆ° zï¼Œå†æŒ‰ç›¸åˆ«æ”¾åˆ° rA æˆ– rBï¼›æœ«å—ç”¨ä½™é‡è¡¥é½
        if meta.get("model_kind") == "pure":
            Y = np.zeros(K, dtype=float)
            Y[:Km1] = x[:Km1]
            z_all= self._inv_stick_breaking(Y)   # æ­£ç¡®è§£åŒ…
            z_use = z_all[:pick]
    
            rA, rB = [], []
            if X1 >= 0.5:  # çº¯ A
                rA.extend([float(zk) for zk in z_use])
                rB.extend([0.0] * len(z_use))
                rA.append(max(0.0, 1.0 - float(np.sum(rA))))  # æœ«å—ä½™é‡
                rB.append(0.0)
            else:          # çº¯ B
                rA.extend([0.0] * len(z_use))
                rB.extend([float(zk) for zk in z_use])
                rA.append(0.0)
                rB.append(max(0.0, 1.0 - float(np.sum(rB))))
            return rA, rB
    
        # æ··åˆç‰©ï¼šå­¦ [Y[:K-1], pA[:K-1]]
        Y = np.zeros(K, dtype=float)
        Y[:Km1] = x[:Km1]
        z_all= self._inv_stick_breaking(Y)
        z_use = z_all[:pick]
    
        pA = np.zeros(K, dtype=float)
        pA[:Km1] = x[Km1: Km1 + Km1]
        pA[Km1] = X1  # æœ«å—ææ–™åˆ†æ•°å°±ç”¨æ¯ç²’çš„ X1
    
        rA: List[float] = []
        rB: List[float] = []
        X3 = 1.0 - X1
    
        # å¯è¡ŒåŸŸæŠ•å½± + ç›¸å†…å½’ä¸€
        for k in range(pick):
            zk = float(np.clip(z_use[k], 0.0, 1.0))
            if zk <= 0.0:
                rA.append(0.0); rB.append(0.0); continue
            pk = float(np.clip(pA[k], 0.0, 1.0))
            # å¯è¡ŒåŸŸï¼šVA â‰¤ A*X1, VB â‰¤ A*(1-X1)  â‡’  p âˆˆ [max(0,1 - X3/zk), min(1, X1/zk)]
            Lk = max(0.0, 1.0 - X3 / max(zk, 1e-12))
            Uk = min(1.0, X1 / max(zk, 1e-12))
            pk = float(np.clip(pk, Lk, Uk))
            rA.append(zk * pk / max(X1, 1e-12))
            rB.append(zk * (1.0 - pk) / max(X3, 1e-12))
    
        # æœ«å—ä¸¥æ ¼å®ˆæ’ï¼ˆå„ç›¸å•ç‹¬è¡¥ä½™é‡ï¼‰ï¼Œä¿æŒä¸Žå…¶å®ƒ adapter ä¸€è‡´
        sA = float(np.sum(rA)); sB = float(np.sum(rB))
        rA.append(max(0.0, 1.0 - sA))
        rB.append(max(0.0, 1.0 - sB))
    
        # å†åšä¸€æ¬¡ç›¸å†…å½’ä¸€ï¼Œé¿å…ç´¯è®¡è¯¯å·®
        sA = float(np.sum(rA)); sB = float(np.sum(rB))
        if sA > 0:
            gA = 1.0 / sA
            rA = [x * gA for x in rA]
        if sB > 0:
            gB = 1.0 / sB
            rB = [y * gB for y in rB]
    
        return rA, rB


    # ------- stick-breaking è¿˜åŽŸï¼Œè¿”å›žé•¿åº¦ K çš„ z å‘é‡ -------
    @staticmethod
    def _inv_stick_breaking(Y: np.ndarray) -> np.ndarray:
        K = Y.size
        z = np.zeros(K, dtype=float)
        remain = 1.0
        for k in range(K):
            yk = float(np.clip(Y[k], 0.0, 1.0))
            z[k] = yk * remain
            remain = max(0.0, remain - z[k])
        # ä¸ºäº†æ•°å€¼ä¿é™©ï¼ŒæŠŠæœ€åŽä¸€å—å†å¯¹é½ä¸€ä¸‹
        if remain > 1e-10:
            z[-1] += remain
        return z


