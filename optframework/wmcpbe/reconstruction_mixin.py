# reconstruction_mixin.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np


@dataclass
class _CellStats:
    """Per-cell aggregated stats used by CAM."""
    key: Tuple[int, ...]          # cell index tuple (i,) or (i,j)
    idx: np.ndarray               # indices in this cell
    M0: float                     # sum W
    mean_v: np.ndarray            # (dim,) mean coordinate per dimension: sum(W*v_d)/M0


class ReconstructionMixin:
    """
    Reconstruction / rebin / resampling mixin.

    This version implements CAM (Cell Average Method) reconstruction:

    - Build grid bins per component dimension (NOT using V_tot).
    - For each cell: compute mean coordinate (weighted by W) and M0 = sum(W).
    - Distribute that cell mass (M0) to neighboring pivot points using
      linear (1D) or bilinear (2D) weights.
    - Create new particles located at pivot points with weights from distribution.

    Requirements on host solver (provided by MCPBEBase and mixins):
      - Arrays: self.V_flat (shape (dim+1, cap)), self.W (shape (cap,)), self.X (shape (cap,))
      - Scalars: self.dim, self.a_tot, self._cap
      - Methods: self._ensure_capacity_for(extra), self._vol2diam(V), self._initialize_samplers()
    """

    # -----------------------------
    # Public knobs (safe defaults)
    # -----------------------------
    recon_enable: bool = True
    recon_method: str = "QMX"      # kept for compatibility; "CAM" or "RS", "2PM", "QMX"(Quantile MiX)
    recon_N_max: int = 15000        # trigger if a_tot exceeds this
    recon_every_events: int = 0    # optional periodic trigger; 0 disables

    # CAM/2PM grid controls
    recon_bins: int = 50           # number of bins per dimension (1D: ~bins, 2D: ~bins^2 cells)
    recon_grid_log: bool = True    # build edges in log-space (recommended for PSD-like scaling)

    # --- RS method controls (resampling) ---
    recon_RS_target: int = 2000          # target number of ACTIVE particles after recon (excluding protected)
    recon_RS_min_per_cell: int = 1       # minimum reps per occupied cell (keeps support)
    recon_RS_max_per_cell: int = 200     # safety cap to avoid huge replication in one cell

    # --- QMX method controls ---
    recon_QMX_q_small: float = 0.55     # small / mid 分界（加权分位数）
    recon_QMX_q_tail: float = 0.85      # mid / tail 分界（加权分位数）
    recon_QMX_small_method: str = "2PM"   # "2PM" recommended
    recon_QMX_mid_method: str = "RS"     # "RS" recommended (with C-1)
    recon_QMX_tail_method: str = "CAM"  # "CAM" or "2PM" or "RS" or "NONE"

    # tail protection (optional)
    recon_tail_protect: int = 50    # keep largest K particles (by V_tot) unchanged

    # internal counters
    _recon_count: int = 0
    _recon_last_iter: int = 0

    # =============================
    # Public entry points
    # =============================
    def maybe_reconstruct(self, iter_count: int, reason: str = "") -> bool:
        if not bool(getattr(self, "recon_enable", True)):
            return False

        a = int(getattr(self, "a_tot", 0))
        if a <= 0:
            return False

        Nmax = int(getattr(self, "recon_N_max", 0))
        every = int(getattr(self, "recon_every_events", 0))

        trig = False
        if Nmax > 0 and a > Nmax:
            trig = True
        if (not trig) and every > 0:
            last = int(getattr(self, "_recon_last_iter", 0))
            if (iter_count - last) >= every:
                trig = True

        if not trig:
            return False

        self.reconstruct(method=str(getattr(self, "recon_method", "CAM")), reason=reason, iter_count=iter_count)
        return True

    def reconstruct(self, method: str = "CAM", reason: str = "", iter_count: Optional[int] = None) -> None:
        method = str(method).upper().strip()
        if method not in ("CAM", "RS", "2PM", "QMX"):
            raise NotImplementedError(f"Reconstruction method '{method}' is not implemented in this file.")
    
        a = int(self.a_tot)
        if a <= 0:
            return
    
        dim = int(self.dim)
        Vcomp = np.asarray(self.V_flat[:dim, :a], dtype=float)  # (dim, a)
        Vtot = np.asarray(self.V_flat[-1, :a], dtype=float)
        W = np.asarray(self.W[:a], dtype=float)
    
        # valid particles: positive weights and positive coordinates
        valid = np.isfinite(W) & (W > 0.0) & np.isfinite(Vtot) & (Vtot > 0.0)
        for d in range(dim):
            valid &= np.isfinite(Vcomp[d]) & (Vcomp[d] >= 0.0)
    
        if not np.any(valid):
            return
    
        idx_all = np.nonzero(valid)[0]
    
        # optional tail protection: keep largest Vtot unchanged
        protected = self._recon_select_protected(idx_all, Vtot, W)
        mask_prot = np.zeros(a, dtype=bool)
        if protected.size > 0:
            mask_prot[protected] = True
    
        idx_work = idx_all[~mask_prot[idx_all]]
        if idx_work.size == 0:
            return
    
        if method == "CAM":
            self._reconstruct_cam(idx_work, protected, Vcomp, Vtot, W)
        elif method == "RS":
            self._reconstruct_rs(idx_work, protected, Vcomp, Vtot, W)
        elif method == "QMX":
            self._reconstruct_qmx(idx_work, protected, Vcomp, Vtot, W)
        else:
            self._reconstruct_b(idx_work, protected, Vcomp, Vtot, W)
    
        # counters
        self._recon_count = int(getattr(self, "_recon_count", 0)) + 1
        if iter_count is not None:
            self._recon_last_iter = int(iter_count)
            
    # =============================
    # Tail protection
    # =============================
    def _recon_select_protected(self, idx_valid: np.ndarray, Vtot: np.ndarray, W: np.ndarray) -> np.ndarray:
        K = int(getattr(self, "recon_tail_protect", 0))
        if K <= 0 or idx_valid.size == 0:
            return np.array([], dtype=int)
        K = min(K, idx_valid.size)
        vv = Vtot[idx_valid]
        order = np.argsort(vv)  # ascending
        return np.asarray(idx_valid[order[-K:]], dtype=int)
    
    def _apply_kernel_and_replace(
        self,
        method: str,
        idx_work: np.ndarray,
        protected: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
        *,
        mode: str = "full",
        N_target_override: Optional[int] = None,
    ) -> None:
        """
        Apply a kernel on idx_work, append protected as-is, then replace active particles.
    
        mode:
          - "full": kernel interprets target sizes as global reconstruction target
          - "subset": kernel interprets target sizes as subset reconstruction target
        """
        dim = int(self.dim)
        method = str(method).upper().strip()
    
        if method == "CAM":
            V_cols, W_out = self._kernel_cam(idx_work, Vcomp, W)
        elif method == "2PM":
            V_cols, W_out = self._kernel_2pm(idx_work, Vcomp, Vtot, W)
        elif method == "RS":
            V_cols, W_out = self._kernel_rs(
                idx_work, Vcomp, Vtot, W,
                mode=mode,
                N_target_override=N_target_override,
            )
        else:
            raise ValueError(f"Unknown kernel method: {method}")
    
        # append protected particles as-is
        for i in protected.tolist():
            V_cols.append(self.V_flat[:dim, i].copy())
            W_out.append(float(self.W[i]))
    
        if len(W_out) == 0:
            return
    
        self._recon_replace_active_particles(V_cols, np.asarray(W_out, dtype=float))
    
    def _bucket_by_cam_cells(
        self,
        idx: np.ndarray,
        Vcomp: np.ndarray,
        *,
        n_bins: Optional[int] = None,
        return_grid: bool = False,
    ) -> tuple[dict[tuple, np.ndarray], Optional[list[np.ndarray]], Optional[list[np.ndarray]]]:
        """
        Bucket indices by CAM-style cells in component-volume space.
    
        Parameters
        ----------
        idx : np.ndarray[int]
            Global indices of particles to bucket.
        Vcomp : np.ndarray[float] shape (dim, a_tot)
            Component volumes for all active particles.
        n_bins : Optional[int]
            Number of bins per dimension (defaults to self.recon_bins).
        return_grid : bool
            If True, also return (edges_list, piv_list). piv_list == edges_list in your node-based CAM.
    
        Returns
        -------
        buckets : dict[key -> np.ndarray[int]]
            key is (i,) in 1D or (i,j) in 2D; values are global indices belonging to that cell.
        edges_list : Optional[list[np.ndarray]]
        piv_list   : Optional[list[np.ndarray]]
        """
        dim = int(self.dim)
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return {}, ([] if return_grid else None), ([] if return_grid else None)
    
        if n_bins is None:
            n_bins = int(getattr(self, "recon_bins", 50))
    
        # Build grid on this subset only (same behavior as your current C/B kernels)
        edges_list, piv_list = self._cam_build_grid(Vcomp[:, idx], n_bins=int(n_bins))
    
        # Assign each particle to a cell
        cell_ids = self._cam_assign_cells(Vcomp[:, idx], edges_list)  # (dim, idx.size)
    
        # Group by cell key
        buckets: dict[tuple, list] = {}
        for local_k in range(idx.size):
            i = int(idx[local_k])  # global index
            key = tuple(int(cell_ids[d, local_k]) for d in range(dim))
            buckets.setdefault(key, []).append(i)
    
        # Convert lists to arrays for downstream speed/consistency
        buckets_arr: dict[tuple, np.ndarray] = {k: np.asarray(v, dtype=int) for k, v in buckets.items()}
    
        if return_grid:
            return buckets_arr, edges_list, piv_list
        return buckets_arr, None, None

            
    def _cam_cell_stats_from_buckets(
        self,
        buckets: dict[tuple, np.ndarray],
        Vcomp_all: np.ndarray,
        W_all: np.ndarray,
    ) -> List[_CellStats]:
        dim = int(self.dim)
        out: List[_CellStats] = []
        for key, idx in buckets.items():
            Wi = W_all[idx]
            M0 = float(np.sum(Wi))
            if not np.isfinite(M0) or M0 <= 0.0:
                continue
            mean_v = np.zeros(dim, dtype=float)
            for d in range(dim):
                mean_v[d] = float(np.sum(Wi * Vcomp_all[d, idx]) / M0)
            out.append(_CellStats(key=key, idx=idx, M0=M0, mean_v=mean_v))
        return out
# %% CAM
    # =============================
    # CAM grid + assignment
    # =============================
    def _reconstruct_cam(
        self,
        idx_work: np.ndarray,
        protected: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> None:
        # CAM kernel does not need Vtot; keep signature for compatibility
        self._apply_kernel_and_replace(
            "CAM", idx_work, protected, Vcomp, Vtot, W, mode="full"
        )

    def _kernel_cam(self, idx: np.ndarray, Vcomp: np.ndarray, W: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        dim = int(self.dim)
        n_bins = int(getattr(self, "recon_bins", 50))
    
        buckets, edges_list, piv_list = self._bucket_by_cam_cells(idx, Vcomp, n_bins=n_bins, return_grid=True)
        cells = self._cam_cell_stats_from_buckets(buckets, Vcomp, W)
    
        pivot_weight: dict[tuple, float] = {}
        for st in cells:
            if st.M0 > 0.0:
                self._cam_distribute_to_pivots(st.mean_v, st.M0, piv_list, pivot_weight)
    
        V_cols: list[np.ndarray] = []
        W_out: list[float] = []
        for key in sorted(pivot_weight.keys()):
            w = float(pivot_weight[key])
            if w <= 0.0 or (not np.isfinite(w)):
                continue
            vcol = np.zeros(dim, dtype=float)
            for d in range(dim):
                vcol[d] = float(piv_list[d][key[d]])
            V_cols.append(vcol)
            W_out.append(w)
    
        return V_cols, W_out

    def _cam_build_grid(self, Vcomp_work: np.ndarray, n_bins: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Build per-dimension grid nodes (NOT centers).
        - edges_list[d] are grid NODES with length n_bins+1
        - piv_list[d] is kept as an alias to edges_list[d] for downstream code
          (so we don't have to rename many variables).
    
        Special handling when a dimension contains zeros:
          - node[0] = 0
          - remaining nodes (1..n_bins) are log-spaced from (min_positive/2) to xmax
            to keep log resolution while safely including 0.
        """
        dim, n = Vcomp_work.shape
        n_bins = max(4, int(n_bins))
        use_log_global = bool(getattr(self, "recon_grid_log", True))
    
        edges_list: List[np.ndarray] = []
        piv_list: List[np.ndarray] = []
    
        for d in range(dim):
            x_all = np.asarray(Vcomp_work[d], dtype=float)
            x_all = x_all[np.isfinite(x_all) & (x_all >= 0.0)]  # allow zeros
            if x_all.size == 0:
                nodes = np.linspace(0.0, 1.0, n_bins + 1)
                edges_list.append(nodes)
                piv_list.append(nodes)
                continue
    
            xmax = float(np.max(x_all))
            if not np.isfinite(xmax) or xmax <= 0.0:
                # everything is 0
                nodes = np.linspace(0.0, 1.0, n_bins + 1)
                edges_list.append(nodes)
                piv_list.append(nodes)
                continue
    
            x_pos = x_all[x_all > 0.0]
            has_zero = (x_pos.size < x_all.size)
    
            if use_log_global and x_pos.size > 0:
                xmin_pos = float(np.min(x_pos))
                # log start
                if has_zero:
                    x_start = 0.5 * xmin_pos
                    x_start = max(x_start, xmin_pos * 1e-6)  # safety
                    # nodes: [0, logspace(x_start..xmax) with n_bins points]
                    log_nodes = np.exp(np.linspace(np.log(x_start), np.log(xmax), n_bins))
                    nodes = np.empty(n_bins + 1, dtype=float)
                    nodes[0] = 0.0
                    nodes[1:] = log_nodes
                else:
                    xmin = float(np.min(x_all))
                    if xmax <= xmin * (1.0 + 1e-12):
                        xmax = xmin * 1.01
                    nodes = np.exp(np.linspace(np.log(max(xmin, 1e-300)), np.log(xmax), n_bins + 1))
            else:
                # linear nodes (either recon_grid_log disabled or no positive values)
                xmin = float(np.min(x_all))
                if xmax <= xmin * (1.0 + 1e-12):
                    xmax = xmin + 1.0 if xmin == 0.0 else xmin * 1.01
                nodes = np.linspace(xmin, xmax, n_bins + 1)
    
            # make strictly nondecreasing and enforce last > second last
            nodes = np.maximum.accumulate(nodes)
            if nodes[-1] <= nodes[-2] * (1.0 + 1e-12):
                nodes[-1] = nodes[-2] * (1.0 + 1e-6) if nodes[-2] > 0 else nodes[-2] + 1e-12
    
            edges_list.append(nodes)
            piv_list.append(nodes)  # alias: downstream uses "piv" but now piv == nodes
    
        return edges_list, piv_list

    def _cam_assign_cells(self, Vcomp_work: np.ndarray, edges_list: List[np.ndarray]) -> np.ndarray:
        """
        Return cell index array shape (dim, n_work).
        Each cell index i means x is in [node[i], node[i+1]] where node == edges_list[d].
    
        For node-based CAM, edges_list[d] has length n_bins+1, so valid cell indices are 0..n_bins-1.
        """
        dim, n = Vcomp_work.shape
        cell = np.zeros((dim, n), dtype=int)
    
        for d in range(dim):
            nodes = edges_list[d]
            nb = len(nodes) - 1  # number of cells
            x = np.asarray(Vcomp_work[d], dtype=float)
    
            # digitize against nodes: returns k such that nodes[k-1] <= x < nodes[k]
            # Convert to cell index = k-1, then clamp to [0, nb-1]
            bid = np.digitize(x, nodes, right=False) - 1
            bid = np.clip(bid, 0, nb - 1).astype(int)
            cell[d] = bid
    
        return cell

    def _cam_cell_stats(
        self,
        idx_work: np.ndarray,
        cell_ids: np.ndarray,
        Vcomp_all: np.ndarray,
        W_all: np.ndarray,
    ) -> List[_CellStats]:
        """
        Aggregate M0 and mean_v per occupied cell.
        mean_v is computed in the original coordinate (component volumes).
        """
        dim = int(self.dim)
        # group indices by cell key
        buckets: dict[Tuple[int, ...], List[int]] = {}
        for local_k in range(idx_work.size):
            i = int(idx_work[local_k])
            key = tuple(int(cell_ids[d, local_k]) for d in range(dim))
            buckets.setdefault(key, []).append(i)

        out: List[_CellStats] = []
        for key, idx_list in buckets.items():
            idx = np.asarray(idx_list, dtype=int)
            Wi = W_all[idx]
            M0 = float(np.sum(Wi))
            if not np.isfinite(M0) or M0 <= 0.0:
                continue
            mean_v = np.zeros(dim, dtype=float)
            for d in range(dim):
                mean_v[d] = float(np.sum(Wi * Vcomp_all[d, idx]) / M0)
            out.append(_CellStats(key=key, idx=idx, M0=M0, mean_v=mean_v))
        return out

    # =============================
    # CAM distribution to pivots
    # =============================
    def _cam_find_bracketing_pivots(self, x: float, nodes: np.ndarray) -> Tuple[int, int, float]:
        """
        Node-based bracketing:
          find i such that nodes[i] <= x <= nodes[i+1], return (i, i+1, t) with
            x = (1-t)*nodes[i] + t*nodes[i+1], t in [0,1].
    
        Boundary behavior:
          - x <= nodes[0]  -> (0,0,0)  (all weight to node 0)
          - x >= nodes[-1] -> (nb,nb,0) where nb=len(nodes)-1 (all weight to last node)
        This makes boundary cases naturally degenerate (your point 2).
        """
        nb = int(nodes.size) - 1  # number of cells
        if nb <= 0:
            return 0, 0, 0.0
    
        x = float(x)
        if not np.isfinite(x):
            return 0, 0, 0.0
    
        if x <= float(nodes[0]):
            return 0, 0, 0.0
        if x >= float(nodes[-1]):
            return nb, nb, 0.0
    
        # Find i so that nodes[i] <= x < nodes[i+1]
        i = int(np.searchsorted(nodes, x, side="right") - 1)
        i = max(0, min(i, nb - 1))
    
        x0 = float(nodes[i])
        x1 = float(nodes[i + 1])
        den = x1 - x0
        if den <= 0.0:
            return i, i, 0.0
    
        t = (x - x0) / den
        t = float(np.clip(t, 0.0, 1.0))
    
        # If exactly on boundary, t hits 0 or 1 -> naturally degenerates
        return i, i + 1, t

    def _cam_distribute_to_pivots(
        self,
        mean_v: np.ndarray,
        M0: float,
        piv_list: List[np.ndarray],
        pivot_weight: dict,
    ) -> None:
        """
        Distribute cell total weight M0 from mean coordinate to neighboring pivots:
        - 1D: 2 pivots
        - 2D: 4 pivots (bilinear)
        - dim>2: not implemented here (extend later if needed)
        """
        dim = int(self.dim)
        if dim == 1:
            piv0 = piv_list[0]
            i0, i1, t = self._cam_find_bracketing_pivots(float(mean_v[0]), piv0)

            w0 = (1.0 - t) * M0
            w1 = t * M0

            if w0 > 0.0:
                pivot_weight[(i0,)] = pivot_weight.get((i0,), 0.0) + w0
            if w1 > 0.0:
                pivot_weight[(i1,)] = pivot_weight.get((i1,), 0.0) + w1
            return

        if dim == 2:
            pivx, pivy = piv_list[0], piv_list[1]
            ix0, ix1, tx = self._cam_find_bracketing_pivots(float(mean_v[0]), pivx)
            iy0, iy1, ty = self._cam_find_bracketing_pivots(float(mean_v[1]), pivy)

            # bilinear weights
            w00 = (1.0 - tx) * (1.0 - ty) * M0
            w10 = tx * (1.0 - ty) * M0
            w01 = (1.0 - tx) * ty * M0
            w11 = tx * ty * M0

            if w00 > 0.0:
                pivot_weight[(ix0, iy0)] = pivot_weight.get((ix0, iy0), 0.0) + w00
            if w10 > 0.0:
                pivot_weight[(ix1, iy0)] = pivot_weight.get((ix1, iy0), 0.0) + w10
            if w01 > 0.0:
                pivot_weight[(ix0, iy1)] = pivot_weight.get((ix0, iy1), 0.0) + w01
            if w11 > 0.0:
                pivot_weight[(ix1, iy1)] = pivot_weight.get((ix1, iy1), 0.0) + w11
            return

        raise NotImplementedError("CAM distribution currently supports dim=1 or dim=2 only.")

    # =============================
    # Replace active particles
    # =============================
    def _recon_replace_active_particles(self, Vcomp_cols: List[np.ndarray], W_new: np.ndarray) -> None:
        """
        Replace active slice with provided columns and weights, then rebuild X and samplers.

        Vcomp_cols: list of arrays shape (dim,)
        W_new: shape (n_new,)
        """
        dim = int(self.dim)
        n_new = int(len(Vcomp_cols))
        if n_new != int(W_new.shape[0]):
            raise ValueError("Vcomp_cols and W_new length mismatch.")
        if n_new <= 0:
            return

        # ensure capacity
        extra = max(0, n_new - int(self._cap))
        if extra > 0:
            self._ensure_capacity_for(extra)
        if int(self._cap) < n_new:
            self._ensure_capacity_for(n_new - int(self.a_tot))

        V_flat_new = np.zeros((dim + 1, int(self._cap)), dtype=float)
        W_buf = np.zeros(int(self._cap), dtype=float)
        X_buf = np.zeros(int(self._cap), dtype=float)

        for j, vcomp in enumerate(Vcomp_cols):
            vcomp = np.asarray(vcomp, dtype=float)
            if vcomp.shape != (dim,):
                raise ValueError(f"Representative column must have shape ({dim},), got {vcomp.shape}.")
            V_flat_new[:dim, j] = vcomp
            V_flat_new[-1, j] = float(np.sum(vcomp))
            W_buf[j] = float(W_new[j])

        X_buf[:n_new] = self._vol2diam(V_flat_new[-1, :n_new])

        # swap in
        self.V_flat[:, :] = 0.0
        self.V_flat[:, :n_new] = V_flat_new[:, :n_new]
        self.W[:] = 0.0
        self.W[:n_new] = W_buf[:n_new]
        self.X[:] = 0.0
        self.X[:n_new] = X_buf[:n_new]
        self.a_tot = n_new

        # rebuild propensities & samplers
        self._initialize_samplers()
# %% RS
    def _reconstruct_rs(
        self,
        idx_work: np.ndarray,
        protected: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> None:
        # full reconstruction semantics for C: use recon_RS_target
        self._apply_kernel_and_replace(
            "RS", idx_work, protected, Vcomp, Vtot, W, mode="full"
        )
        
    def _kernel_rs(
        self,
        idx: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
        *,
        mode: str = "subset",
        N_target_override: Optional[int] = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Cell-based resampling with C-1 two-weight correction on M1 (Vtot).
        Returns representatives (V_cols, W_list) WITHOUT writing back to self.
    
        mode:
          - "full": N_target comes from recon_RS_target (or override)
          - "subset": N_target comes from override, or a scaled recon_RS_target
                     (recommended when called from E mid-partition)
        """
        dim = int(self.dim)
        n_bins = int(getattr(self, "recon_bins", 50))
        buckets, _, _ = self._bucket_by_cam_cells(idx, Vcomp, n_bins=n_bins, return_grid=False)
    
        # allocate reps by M0 proportion
        M0_total = float(np.sum(W[idx]))
        if M0_total <= 0.0 or not np.isfinite(M0_total):
            return [], []
    
        # ----- choose N_target -----
        if N_target_override is not None:
            N_target = int(max(1, N_target_override))
        else:
            base = int(getattr(self, "recon_RS_target", 2000))
            base = max(1, base)
    
            if str(mode).lower() == "full":
                # classic behavior
                N_target = base
            else:
                # subset behavior: scale by subset weight fraction
                # default: N_target_subset = base * (M0_subset / M0_active)
                # This prevents mid partition from being over/under sampled.
                # We approximate M0_active by sum(W[idx_valid]) == sum(W over idx passed in by caller) when caller is E,
                # but for safety we also try to read 'a_tot' slice.
                M0_subset = float(np.sum(W[idx]))
                if not np.isfinite(M0_subset) or M0_subset <= 0.0:
                    N_target = max(1, int(round(0.1 * base)))
                else:
                    # attempt "active" weight from current active slice (valid assumed)
                    a = int(getattr(self, "a_tot", 0))
                    if a > 0:
                        W_active = np.asarray(self.W[:a], dtype=float)
                        m = np.isfinite(W_active) & (W_active > 0.0)
                        M0_active = float(np.sum(W_active[m]))
                    else:
                        M0_active = M0_subset
    
                    if not np.isfinite(M0_active) or M0_active <= 0.0:
                        M0_active = M0_subset
    
                    frac = float(M0_subset / M0_active)
                    frac = float(np.clip(frac, 1e-6, 1.0))
                    N_target = int(max(1, round(base * frac)))
    
        min_pc = int(getattr(self, "recon_RS_min_per_cell", 1))
        max_pc = int(getattr(self, "recon_RS_max_per_cell", 200))
        rng = self._get_rng()
    
        
    
        # compute per-cell M0
        cell_keys = list(buckets.keys())
        cell_M0 = np.array([float(np.sum(W[buckets[k]])) for k in cell_keys], dtype=float)
        mask = (cell_M0 > 0) & np.isfinite(cell_M0)
        cell_keys = [cell_keys[i] for i in range(len(cell_keys)) if mask[i]]
        cell_M0 = cell_M0[mask]
        if cell_M0.size == 0:
            return [], []
    
        frac = cell_M0 / float(np.sum(cell_M0))
        n_cell = np.floor(frac * N_target).astype(int)
        n_cell = np.maximum(n_cell, min_pc)
        n_cell = np.minimum(n_cell, max_pc)
    
        # adjust total to match N_target
        s = int(np.sum(n_cell))
        residual = frac * N_target - np.floor(frac * N_target)
        if s < N_target:
            add = N_target - s
            order = np.argsort(-residual)
            for j in range(add):
                n_cell[int(order[j % len(order)])] += 1
        elif s > N_target:
            sub = s - N_target
            order = np.argsort(residual)
            j = 0
            while sub > 0 and j < 100000:
                ii = int(order[j % len(order)])
                if n_cell[ii] > min_pc:
                    n_cell[ii] -= 1
                    sub -= 1
                j += 1
            j = 0
            while sub > 0 and j < 100000:
                ii = int(order[j % len(order)])
                if n_cell[ii] > 1:
                    n_cell[ii] -= 1
                    sub -= 1
                j += 1
    
        V_cols: list[np.ndarray] = []
        W_out: list[float] = []
    
        for k, M0c, nc in zip(cell_keys, cell_M0, n_cell):
            idc = buckets[k]
            if idc.size == 0 or M0c <= 0 or int(nc) <= 0:
                continue
    
            M1c_target = float(np.sum(W[idc] * Vtot[idc]))
    
            if idc.size == 1:
                V_cols.append(Vcomp[:, idc[0]].copy())
                W_out.append(float(M0c))
                continue
    
            if int(nc) == 1:
                # exact mean particle for this cell (keeps M0/M1)
                Wi = W[idc]
                M0 = float(np.sum(Wi))
                mean_comp = np.zeros(dim, dtype=float)
                for d in range(dim):
                    mean_comp[d] = float(np.sum(Wi * Vcomp[d, idc]) / M0) if M0 > 0 else 0.0
                V_cols.append(mean_comp)
                W_out.append(float(M0c))
                continue
    
            chosen_local = self._systematic_resample(W[idc], int(nc), rng)
            chosen_idx = idc[chosen_local]
            w = np.full(int(nc), float(M0c) / float(nc), dtype=float)
    
            ok = self._correct_two(Vtot[chosen_idx].astype(float, copy=False), w, M1c_target)
            if not ok:
                Wi = W[idc]
                M0 = float(np.sum(Wi))
                mean_comp = np.zeros(dim, dtype=float)
                for d in range(dim):
                    mean_comp[d] = float(np.sum(Wi * Vcomp[d, idc]) / M0) if M0 > 0 else 0.0
                V_cols.append(mean_comp)
                W_out.append(float(M0c))
                continue
    
            for jj, ii in enumerate(chosen_idx):
                V_cols.append(Vcomp[:, ii].copy())
                W_out.append(float(w[jj]))
    
        return V_cols, W_out
    
    # two-weight correction helper
    def _correct_two(self, vtot_samp: np.ndarray, w: np.ndarray, M1_target: float) -> bool:
        if w.size < 2:
            return False
        M1_now = float(np.sum(w * vtot_samp))
        err = M1_target - M1_now
        if abs(err) <= 1e-14 * (abs(M1_target) + 1.0):
            return True

        order = np.argsort(vtot_samp)
        cand = []
        cand.append((int(order[-1]), int(order[0])))
        if w.size >= 3:
            cand.append((int(order[-1]), int(order[1])))
            cand.append((int(order[-2]), int(order[0])))

        for a, b in cand:
            Va = float(vtot_samp[a]); Vb = float(vtot_samp[b])
            den = Va - Vb
            if den == 0.0 or not np.isfinite(den):
                continue
            d = err / den
            wa = w[a] + d
            wb = w[b] - d
            if wa >= 0.0 and wb >= 0.0 and np.isfinite(wa) and np.isfinite(wb):
                w[a] = wa; w[b] = wb
                return True
        return False
    
    def _get_rng(self) -> np.random.Generator:
        # Try to reuse solver RNG if you have one; otherwise make a local default.
        rng = getattr(self, "rng", None)
        if isinstance(rng, np.random.Generator):
            return rng
        rng = getattr(self, "_rng", None)
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng()
    
    def _systematic_resample(self, p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Systematic resampling for categorical distribution p (sum p=1).
        Returns indices in [0, len(p)-1], length n, with replacement.
        """
        p = np.asarray(p, dtype=float)
        p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)
        s = float(np.sum(p))
        if s <= 0.0:
            # fallback: uniform
            p = np.ones_like(p) / float(p.size)
        else:
            p = p / s
    
        cdf = np.cumsum(p)
        cdf[-1] = 1.0  # guard
    
        u0 = rng.random() / float(n)
        u = u0 + (np.arange(n, dtype=float) / float(n))
    
        return np.searchsorted(cdf, u, side="left").astype(int)
# %% 2PM
    def _reconstruct_b(
        self,
        idx_work: np.ndarray,
        protected: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> None:
        self._apply_kernel_and_replace(
            "2PM", idx_work, protected, Vcomp, Vtot, W, mode="full"
        )

    def _kernel_2pm(self, idx: np.ndarray, Vcomp: np.ndarray, Vtot: np.ndarray, W: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        dim = int(self.dim)
        n_bins = int(getattr(self, "recon_bins", 50))
        buckets, _, _ = self._bucket_by_cam_cells(idx, Vcomp, n_bins=n_bins, return_grid=False)
    
        V_cols: list[np.ndarray] = []
        W_out: list[float] = []
    
        for _, idc in buckets.items():
            Wi = W[idc]
            M0 = float(np.sum(Wi))
            if M0 <= 0 or not np.isfinite(M0):
                continue
            xt = Vtot[idc]
            M1 = float(np.sum(Wi * xt))
            M2 = float(np.sum(Wi * xt * xt))
    
            # composition ratio preserves per-dim first moments (handles zeros naturally)
            M1_tot = M1
            M1_d = np.array([float(np.sum(Wi * Vcomp[d, idc])) for d in range(dim)], dtype=float)
            if np.isfinite(M1_tot) and M1_tot > 0:
                ratio = np.maximum(np.where(np.isfinite(M1_d), M1_d, 0.0) / M1_tot, 0.0)
                s = float(np.sum(ratio))
                if s > 1.0 + 1e-12:
                    ratio /= s
            else:
                ratio = np.zeros(dim, dtype=float)
    
            reps = self._two_point(M0, M1, M2)
            for x, w in reps:
                if w <= 0 or not np.isfinite(w):
                    continue
                x = float(max(x, 0.0))
                vcol = ratio * x
                vcol = np.maximum(np.where(np.isfinite(vcol), vcol, 0.0), 0.0)
                s = float(np.sum(vcol))
                if s > 0:
                    vcol *= (x / s)
                else:
                    if x > 0:
                        jmax = int(np.argmax(ratio)) if ratio.size else 0
                        vcol = np.zeros(dim, dtype=float)
                        vcol[jmax] = x
                V_cols.append(vcol)
                W_out.append(float(w))
    
        return V_cols, W_out
    
    def _two_point(self, M0: float, M1: float, M2: float) -> list[tuple[float, float]]:
        if (not np.isfinite(M0)) or M0 <= 0:
            return []
        mean = float(M1 / M0) if (np.isfinite(M1) and M0 > 0) else 0.0
        mean = max(mean, 0.0)
        m2b = float(M2 / M0) if np.isfinite(M2) else mean * mean
        var = max(m2b - mean * mean, 0.0)
        d = math.sqrt(var)
        x1 = mean - d
        x2 = mean + d
        if x2 <= 0:
            return [(0.0, M0)]
        if x1 >= 0:
            return [(x1, 0.5 * M0), (x2, 0.5 * M0)]
        # fallback {0,x2} keeps M0/M1
        xh = max(x2, 1e-300)
        wh = float(M1 / xh) if np.isfinite(M1) else 0.0
        w0 = M0 - wh
        if w0 >= 0 and wh >= 0 and np.isfinite(w0) and np.isfinite(wh):
            return [(0.0, w0), (xh, wh)]
        return [(mean, M0)]
# %% QMX
    def _weighted_quantile(self, x: np.ndarray, w: np.ndarray, q: float) -> float:
        """Weighted quantile for q in [0,1]."""
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(m):
            return float(np.nan)
        x = x[m]; w = w[m]
        order = np.argsort(x)
        x = x[order]; w = w[order]
        cw = np.cumsum(w)
        cw /= cw[-1]
        return float(np.interp(q, cw, x))
    
    
    def _partition_by_vtot(self, idx: np.ndarray, Vtot: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Split idx into small/mid/tail by weighted quantiles of Vtot."""
        q_small = float(getattr(self, "recon_QMX_q_small", 0.70))
        q_tail  = float(getattr(self, "recon_QMX_q_tail", 0.95))
        q_small = float(np.clip(q_small, 0.0, 1.0))
        q_tail  = float(np.clip(q_tail,  0.0, 1.0))
        if q_tail < q_small:
            q_tail = q_small
    
        vt = Vtot[idx]
        wt = W[idx]
        t1 = self._weighted_quantile(vt, wt, q_small)
        t2 = self._weighted_quantile(vt, wt, q_tail)
    
        # fallback if nan
        if not np.isfinite(t1):
            t1 = float(np.nanmin(vt))
        if not np.isfinite(t2):
            t2 = float(np.nanmax(vt))
    
        small = idx[vt <= t1]
        mid   = idx[(vt > t1) & (vt <= t2)]
        tail  = idx[vt > t2]
        return small, mid, tail, float(t1), float(t2)
    
    
    def _reconstruct_qmx(
        self,
        idx_work: np.ndarray,
        protected: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> None:
        """
        E: partition + mixed kernels
          - (optional) extra tail protection top-K by Vtot
          - small/mid/tail partitions by weighted quantiles of Vtot
          - apply different kernels on each partition and merge
        """
    
        dim = int(self.dim)
        if idx_work.size == 0:
            return
    
        small, mid, tail, t1, t2 = self._partition_by_vtot(idx_work, Vtot, W)
    
        m_small = str(getattr(self, "recon_QMX_small_method", "2PM")).upper()
        m_mid   = str(getattr(self, "recon_QMX_mid_method",   "RS")).upper()
        m_tail  = str(getattr(self, "recon_QMX_tail_method",  "CAM")).upper()
    
        V_cols_all: list[np.ndarray] = []
        W_all: list[float] = []
    
        # apply kernel on each partition
        if small.size > 0:
            vcols, wlist = self._apply_kernel(m_small, small, Vcomp, Vtot, W)
            V_cols_all.extend(vcols); W_all.extend(wlist)
    
        if mid.size > 0:
            vcols, wlist = self._apply_kernel(m_mid, mid, Vcomp, Vtot, W)
            V_cols_all.extend(vcols); W_all.extend(wlist)
    
        if tail.size > 0 and m_tail != "NONE":
            vcols, wlist = self._apply_kernel(m_tail, tail, Vcomp, Vtot, W)
            V_cols_all.extend(vcols); W_all.extend(wlist)
    
        # append protected as-is
        for i in protected.tolist():
            V_cols_all.append(self.V_flat[:dim, i].copy())
            W_all.append(float(self.W[i]))
    
        if len(W_all) == 0:
            return
    
        self._recon_replace_active_particles(V_cols_all, np.asarray(W_all, dtype=float))
    
    
    def _apply_kernel(
        self,
        method: str,
        idx: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
        *,
        mode: str = "subset",
        N_target_override: Optional[int] = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        method = str(method).upper()
        if method == "CAM":
            return self._kernel_cam(idx, Vcomp, W)
        if method == "RS":
            return self._kernel_rs(idx, Vcomp, Vtot, W, mode=mode, N_target_override=N_target_override)
        if method == "2PM":
            return self._kernel_2pm(idx, Vcomp, Vtot, W)
        raise ValueError(f"Unknown kernel method: {method}")