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
    recon_enable: bool = False
    recon_method: str = "RS"      # kept for compatibility; "CAM" or "RS", "2PM", "4PM", "4PMC", "QMX"(Quantile MiX)
    recon_N_max: int = 4000        # trigger if a_tot exceeds this
    recon_every_events: int = 0    # optional periodic trigger; 0 disables

    # CAM/2PM grid controls
    recon_bins: int = 500          # number of bins per dimension (1D: ~bins, 2D: ~bins^2 cells)
    recon_grid_log: bool = True    # build edges in log-space (recommended for PSD-like scaling)

    # --- RS method controls (resampling) ---
    recon_RS_target: int = 2000         # target number of ACTIVE particles after recon (excluding protected)
    recon_RS_min_per_cell: int = 1       # minimum reps per occupied cell (keeps support)
    recon_RS_max_per_cell: int = 200     # safety cap to avoid huge replication in one cell

    # --- QMX method controls ---
    recon_QMX_q_small: float = 0.65     # small / mid åˆ†ç•Œï¼ˆåŠ æƒåˆ†ä½æ•°ï¼‰
    recon_QMX_q_tail: float = 0.95      # mid / tail åˆ†ç•Œï¼ˆåŠ æƒåˆ†ä½æ•°ï¼‰
    recon_QMX_small_method: str = "2PM"   # "2PM" recommended
    recon_QMX_mid_method: str = "RS"     # "RS" recommended (with C-1)
    recon_QMX_tail_method: str = "CAM"  # "CAM" or "2PM" or "RS" or "NONE"

    # tail protection (optional)
    recon_tail_protect: int = 100    # keep largest K particles (by V_tot) unchanged

    # internal counters
    _recon_count: int = 0
    _recon_last_iter: int = 0
    _recon_cooldown_until_iter: int = -1
    recon_cooldown_events: int = 50
    recon_4pm_eps_w: float = 1e-14
    recon_4pm_cond_max: float = 1e12
    recon_4pmc_eps_var: float = 1e-30

    # =============================
    # Public entry points
    # =============================
    def maybe_reconstruct(self, iter_count: int, reason: str = "") -> bool:
        if not self.recon_enable:
            return False

        a = int(self.a_tot)
        if a <= 0:
            return False
        # Cooldown guard (works even when recon_every_events == 0)
        until = int(self._recon_cooldown_until_iter)
        if until >= 0 and int(iter_count) <= until:
            return False
    
        Nmax = int(self.recon_N_max)
        every = int(self.recon_every_events)

        trig = False
        if Nmax > 0 and a > Nmax:
            trig = True
        if (not trig) and every > 0:
            last = self._recon_last_iter
            if (iter_count - last) >= every:
                trig = True

        if not trig:
            return False
        
        method = str(self.recon_method)
        self.reconstruct(method=method, reason=reason, iter_count=iter_count)
        return True

    def reconstruct(self, method: str = "CAM", reason: str = "", iter_count: Optional[int] = None) -> None:
        method = str(method).upper().strip()
        if method not in ("CAM", "RS", "2PM", "QMX", "4PM", "4PMC"):
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
        protected = self._recon_select_protected(idx_all, Vtot)
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
        elif method == "4PM":
            self._reconstruct_4pm(idx_work, protected, Vcomp, Vtot, W)
        elif method == "4PMC":
            self._reconstruct_4pmc(idx_work, protected, Vcomp, Vtot, W)
        else:
            self._reconstruct_2pm(idx_work, protected, Vcomp, Vtot, W)
    
        # counters
        self._recon_count += 1
        if iter_count is not None:
            self._recon_last_iter = int(iter_count)
            # post-reconstruct safety + cooldown trigger ----
            self._post_reconstruct_safety(
                iter_count=int(iter_count),
                reason=reason,
                post_count=int(self.a_tot),
                threshold_factor=0.9,
            )
            
    # =============================
    # Tail protection
    # =============================
    def _recon_select_protected(self, idx_valid: np.ndarray, Vtot: np.ndarray) -> np.ndarray:
        K = self.recon_tail_protect
        if K <= 0 or idx_valid.size == 0:
            return np.array([], dtype=int)
        K = min(K, idx_valid.size)
        vv = Vtot[idx_valid]
        topk = np.argpartition(vv, -K)[-K:]
        return np.asarray(idx_valid[topk], dtype=int)
    # =============================
    # help functions
    # =============================
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
        elif method == "4PM":
            V_cols, W_out = self._kernel_4pm(idx_work, Vcomp, Vtot, W)
        elif method == "4PMC":
            V_cols, W_out = self._kernel_4pmc(idx_work, Vcomp, Vtot, W)
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
            n_bins = self.recon_bins
    
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

    def _merge_point_weights(
        self,
        V_cols: List[np.ndarray],
        W_out: List[float],
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Merge representatives with identical coordinates by summing their weights.
        """
        dim = int(self.dim)
        point_weight: dict[tuple[float, ...], float] = {}
        for vcol, w in zip(V_cols, W_out):
            wf = float(w)
            if wf <= 0.0 or (not np.isfinite(wf)):
                continue
            key = tuple(float(x) for x in np.asarray(vcol, dtype=float).reshape(dim))
            if key in point_weight:
                point_weight[key] += wf
            else:
                point_weight[key] = wf

        V_cols_merged: list[np.ndarray] = []
        W_out_merged: list[float] = []
        for key in sorted(point_weight.keys()):
            wf = float(point_weight[key])
            if wf <= 0.0 or (not np.isfinite(wf)):
                continue
            V_cols_merged.append(np.asarray(key, dtype=float))
            W_out_merged.append(wf)
        return V_cols_merged, W_out_merged
    
    def _post_reconstruct_safety(
        self,
        *,
        iter_count: int,
        reason: str = "",
        post_count: Optional[int] = None,
        threshold_factor: float = 0.9,
    ) -> bool:
        """
        Post-reconstruction safety + cooldown trigger.
    
        Logic:
          - Let N_post be particle count after reconstruction.
          - If N_post * threshold_factor > recon_N_max:
              * raise a RuntimeError
              * trigger a cooldown window to prevent near-dead-loop recon
              * return True  (cooldown triggered)
            else:
              return False
    
        Requires:
          - self.recon_N_max: int
          - self.recon_cooldown_events: int (recommended default in class: 50)
          - self._recon_cooldown_until_iter: int
        """
        N_post = int(post_count)
        N_max = int(self.recon_N_max)
        if N_max <= 0:
            return False  # nothing to do
    
        if float(N_post) * float(threshold_factor) > float(N_max):
            cooldown = int(self.recon_cooldown_events)
            if cooldown < 0:
                raise ValueError("`recon_cooldown_events` must be non-negative.")
    
            # set cooldown-until (inclusive)
            self._recon_cooldown_until_iter = int(iter_count) + int(cooldown)

            raise RuntimeError(
                f"[ReconstructionSafety] Reconstruction output is too close to/above recon_N_max: "
                f"N_post={N_post}, recon_N_max={N_max}, factor={threshold_factor}. "
                f"Cooldown was set for {cooldown} events (until iter={self._recon_cooldown_until_iter}). "
                f"Reason='{reason}'. Adjust recon_N_max or reconstruction parameters."
            )
    
        return False
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
        n_bins = int(self.recon_bins)
    
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
        use_log_global = self.recon_grid_log
    
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
                key = (i0,)
                if key in pivot_weight:
                    pivot_weight[key] += w0
                else:
                    pivot_weight[key] = w0
            if w1 > 0.0:
                key = (i1,)
                if key in pivot_weight:
                    pivot_weight[key] += w1
                else:
                    pivot_weight[key] = w1
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
                key = (ix0, iy0)
                if key in pivot_weight:
                    pivot_weight[key] += w00
                else:
                    pivot_weight[key] = w00
            if w10 > 0.0:
                key = (ix1, iy0)
                if key in pivot_weight:
                    pivot_weight[key] += w10
                else:
                    pivot_weight[key] = w10
            if w01 > 0.0:
                key = (ix0, iy1)
                if key in pivot_weight:
                    pivot_weight[key] += w01
                else:
                    pivot_weight[key] = w01
            if w11 > 0.0:
                key = (ix1, iy1)
                if key in pivot_weight:
                    pivot_weight[key] += w11
                else:
                    pivot_weight[key] = w11
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
        extra = n_new - int(self._cap)
        if extra > 0:
            self._ensure_capacity_for(extra)

        self.V_flat[:, :] = 0.0
        self.W[:] = 0.0
        self.X[:] = 0.0

        for j, vcomp in enumerate(Vcomp_cols):
            vcomp = np.asarray(vcomp, dtype=float)
            if vcomp.shape != (dim,):
                raise ValueError(f"Representative column must have shape ({dim},), got {vcomp.shape}.")
            self.V_flat[:dim, j] = vcomp
            self.V_flat[-1, j] = float(np.sum(vcomp))
            self.W[j] = float(W_new[j])

        self.X[:n_new] = self._vol2diam(self.V_flat[-1, :n_new])
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
        n_bins = self.recon_bins
        buckets, _, _ = self._bucket_by_cam_cells(idx, Vcomp, n_bins=n_bins, return_grid=False)
    
        # allocate reps by M0 proportion
        M0_total = float(np.sum(W[idx]))
        if M0_total <= 0.0 or not np.isfinite(M0_total):
            return [], []
    
        # ----- choose N_target -----
        if N_target_override is not None:
            N_target = int(max(1, N_target_override))
        else:
            base = self.recon_RS_target
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
                    # active weight from current active slice (valid assumed)
                    a = int(self.a_tot)
                    W_active = np.asarray(self.W[:a], dtype=float)
                    m = np.isfinite(W_active) & (W_active > 0.0)
                    M0_active = float(np.sum(W_active[m]))
    
                    if not np.isfinite(M0_active) or M0_active <= 0.0:
                        M0_active = M0_subset
    
                    frac = float(M0_subset / M0_active)
                    frac = float(np.clip(frac, 1e-6, 1.0))
                    N_target = int(max(1, round(base * frac)))
    
        min_pc = max(0, int(self.recon_RS_min_per_cell))
        max_pc = max(min_pc, int(self.recon_RS_max_per_cell))
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
    
            if dim == 1:
                target = np.array([M1c_target], dtype=float)
                samp = Vtot[chosen_idx].astype(float, copy=False)[None, :]
            else:
                target = np.array(
                    [float(np.sum(W[idc] * Vcomp[d, idc])) for d in range(dim)],
                    dtype=float,
                )
                samp = Vcomp[:, chosen_idx].astype(float, copy=False)

            ok = self._correct_two(samp, w, target)
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
    def _correct_two(self, v_samp: np.ndarray, w: np.ndarray, M1_target: np.ndarray) -> bool:
        """
        Adjust sampled weights to match per-dimension first moments while keeping
        total weight unchanged.

        1D uses the original two-weight correction.
        2D uses a three-weight correction that enforces:
          sum(delta_w) = 0
          sum(delta_w * v1) = err1
          sum(delta_w * v2) = err2
        """
        v = np.asarray(v_samp, dtype=float)
        tgt = np.asarray(M1_target, dtype=float).reshape(-1)

        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.ndim != 2 or v.shape[1] != w.size:
            return False

        dim = int(v.shape[0])
        if tgt.size != dim:
            return False

        if dim == 1:
            return self._correct_two_1d(v[0], w, float(tgt[0]))
        if dim == 2:
            return self._correct_two_2d(v, w, tgt)
        return False

    def _correct_two_1d(self, vtot_samp: np.ndarray, w: np.ndarray, M1_target: float) -> bool:
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

    def _correct_two_2d(self, vcomp_samp: np.ndarray, w: np.ndarray, M1_target: np.ndarray) -> bool:
        if w.size < 3:
            return False

        M_now = np.sum(vcomp_samp * w[None, :], axis=1)
        err = np.asarray(M1_target, dtype=float) - M_now
        tol = 1e-14 * (float(np.max(np.abs(M1_target))) + 1.0)
        if float(np.max(np.abs(err))) <= tol:
            return True

        x = np.asarray(vcomp_samp[0], dtype=float)
        y = np.asarray(vcomp_samp[1], dtype=float)

        order_x = np.argsort(x)
        order_y = np.argsort(y)
        candidate_triplets: list[tuple[int, int, int]] = []

        def add_triplet(a: int, b: int, c: int) -> None:
            trip = (int(a), int(b), int(c))
            if len({trip[0], trip[1], trip[2]}) < 3:
                return
            if trip not in candidate_triplets:
                candidate_triplets.append(trip)

        add_triplet(order_x[-1], order_x[0], order_y[-1])
        add_triplet(order_x[-1], order_x[0], order_y[0])
        add_triplet(order_y[-1], order_y[0], order_x[-1])
        add_triplet(order_y[-1], order_y[0], order_x[0])

        idx_sorted = np.argsort(-(x - np.mean(x)) ** 2 - (y - np.mean(y)) ** 2)
        max_candidates = min(w.size, 8)
        seed = idx_sorted[:max_candidates].tolist()
        for ia in range(len(seed)):
            for ib in range(ia + 1, len(seed)):
                for ic in range(ib + 1, len(seed)):
                    add_triplet(seed[ia], seed[ib], seed[ic])

        for a, b, c in candidate_triplets:
            A = np.array([
                [1.0, 1.0, 1.0],
                [x[a], x[b], x[c]],
                [y[a], y[b], y[c]],
            ], dtype=float)
            rhs = np.array([0.0, err[0], err[1]], dtype=float)
            if not np.all(np.isfinite(A)):
                continue
            cnd = float(np.linalg.cond(A))
            if (not np.isfinite(cnd)) or cnd > 1e12:
                continue
            delta = np.linalg.solve(A, rhs)

            wa = w[a] + float(delta[0])
            wb = w[b] + float(delta[1])
            wc = w[c] + float(delta[2])
            if (
                wa >= 0.0 and wb >= 0.0 and wc >= 0.0
                and np.isfinite(wa) and np.isfinite(wb) and np.isfinite(wc)
            ):
                w[a] = wa
                w[b] = wb
                w[c] = wc
                return True
        return False
    
    def _get_rng(self) -> np.random.Generator:
        return self._rng
    
    def _systematic_resample(self, p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Systematic resampling for categorical distribution p (sum p=1).
        Returns indices in [0, len(p)-1], length n, with replacement.
        """
        p = np.asarray(p, dtype=float)
        p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)
        s = float(np.sum(p))
        if s <= 0.0:
            raise ValueError("Systematic resampling requires at least one positive probability.")
        p = p / s
    
        cdf = np.cumsum(p)
        cdf[-1] = 1.0  # guard
    
        u0 = rng.random() / float(n)
        u = u0 + (np.arange(n, dtype=float) / float(n))
    
        return np.searchsorted(cdf, u, side="left").astype(int)
# %% 2PM
    def _reconstruct_2pm(
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
        n_bins = self.recon_bins
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
            M1_d = np.array([float(np.sum(Wi * Vcomp[d, idc])) for d in range(dim)], dtype=float)
            if np.isfinite(M1) and M1 > 0:
                ratio = np.maximum(np.where(np.isfinite(M1_d), M1_d, 0.0) / M1, 0.0)
                s = float(np.sum(ratio))
                if s > 1.0 + 1e-12:
                    ratio /= s
            else:
                ratio = np.zeros(dim, dtype=float)
    
            self._append_two_point_reps(M0, M1, M2, ratio, V_cols, W_out)
    
        return V_cols, W_out

    def _append_two_point_reps(
        self,
        M0: float,
        M1: float,
        M2: float,
        ratio: np.ndarray,
        V_cols: list[np.ndarray],
        W_out: list[float],
    ) -> None:
        dim = int(self.dim)
        for x, w in self._two_point(M0, M1, M2):
            if w <= 0 or not np.isfinite(w):
                continue
            x = float(max(x, 0.0))
            vcol = ratio * x
            vcol = np.maximum(np.where(np.isfinite(vcol), vcol, 0.0), 0.0)
            s = float(np.sum(vcol))
            if s > 0:
                vcol *= (x / s)
            elif x > 0:
                jmax = int(np.argmax(ratio)) if ratio.size else 0
                vcol = np.zeros(dim, dtype=float)
                vcol[jmax] = x
            V_cols.append(vcol)
            W_out.append(float(w))
    
    def _two_point(self, M0: float, M1: float, M2: float) -> list[tuple[float, float]]:
        if (not np.isfinite(M0)) or M0 <= 0:
            raise ValueError("Two-point reconstruction requires a positive finite M0.")
        if (not np.isfinite(M1)) or (not np.isfinite(M2)):
            raise ValueError("Two-point reconstruction requires finite M1 and M2.")
        mean = float(M1 / M0)
        mean = max(mean, 0.0)
        m2b = float(M2 / M0)
        var = max(m2b - mean * mean, 0.0)
        d = math.sqrt(var)
        x1 = mean - d
        x2 = mean + d
        if x2 <= 0:
            return [(0.0, M0)]
        if x1 >= 0:
            return [(x1, 0.5 * M0), (x2, 0.5 * M0)]
        # Boundary representation {0,x2} keeps M0/M1 when the lower node is negative.
        xh = max(x2, 1e-300)
        wh = float(M1 / xh)
        w0 = M0 - wh
        if w0 >= 0 and wh >= 0 and np.isfinite(w0) and np.isfinite(wh):
            return [(0.0, w0), (xh, wh)]
        raise ValueError("Two-point reconstruction produced negative or non-finite weights.")
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
        q_small = self.recon_QMX_q_small
        q_tail  = self.recon_QMX_q_tail
        q_small = float(np.clip(q_small, 0.0, 1.0))
        q_tail  = float(np.clip(q_tail,  0.0, 1.0))
        if q_tail < q_small:
            q_tail = q_small
    
        vt = Vtot[idx]
        wt = W[idx]
        t1 = self._weighted_quantile(vt, wt, q_small)
        t2 = self._weighted_quantile(vt, wt, q_tail)
    
        if not np.isfinite(t1):
            raise ValueError("QMX small quantile is not finite.")
        if not np.isfinite(t2):
            raise ValueError("QMX tail quantile is not finite.")
    
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
    
        small, mid, tail, _, _ = self._partition_by_vtot(idx_work, Vtot, W)
    
        m_small = str(self.recon_QMX_small_method).upper()
        m_mid   = str(self.recon_QMX_mid_method).upper()
        m_tail  = str(self.recon_QMX_tail_method).upper()
    
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
        if method == "4PM":
            return self._kernel_4pm(idx, Vcomp, Vtot, W)
        if method == "4PMC":
            return self._kernel_4pmc(idx, Vcomp, Vtot, W)
        raise ValueError(f"Unknown kernel method: {method}")
        
# %% 4PM
    def _reconstruct_4pm(
        self,
        idx_work: np.ndarray,
        protected: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> None:
        self._apply_kernel_and_replace("4PM", idx_work, protected, Vcomp, Vtot, W, mode="full")

    def _reconstruct_4pmc(
        self,
        idx_work: np.ndarray,
        protected: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> None:
        self._apply_kernel_and_replace("4PMC", idx_work, protected, Vcomp, Vtot, W, mode="full")
        
    def _kernel_4pm(
        self,
        idx: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        4PM (2D): per occupied CAM-cell, represent the cell by its 4 corner points and weights
                  that match M0, M1_v1, M1_v2, and M2_(Vtot) where Vtot=v1+v2.
        1D or unsupported dimensions fall back to 2PM. Individual ill-conditioned
        cells also fall back to 2PM.
        """
        dim = int(self.dim)
        if dim != 2:
            return self._kernel_2pm(idx, Vcomp, Vtot, W)
    
        n_bins = self.recon_bins
        buckets, edges_list, _piv_list = self._bucket_by_cam_cells(idx, Vcomp, n_bins=n_bins, return_grid=True)
        nodes_x = np.asarray(edges_list[0], dtype=float)  # length n_bins+1
        nodes_y = np.asarray(edges_list[1], dtype=float)
    
        # tolerances
        eps_w = float(self.recon_4pm_eps_w)       # allow tiny negatives
        cond_max = float(self.recon_4pm_cond_max)  # conditioning guard
    
        V_cols: list[np.ndarray] = []
        W_out: list[float] = []
    
        for key, idc in buckets.items():
            if idc.size == 0:
                continue
            Wi = np.asarray(W[idc], dtype=float)
            M0 = float(np.sum(Wi))
            if (not np.isfinite(M0)) or M0 <= 0.0:
                continue
    
            # cell index in each dim (0..n_bins-1)
            ix = int(key[0])
            iy = int(key[1])
    
            # clamp (safety)
            ix = max(0, min(ix, nodes_x.size - 2))
            iy = max(0, min(iy, nodes_y.size - 2))
    
            x0 = float(nodes_x[ix]); x1 = float(nodes_x[ix + 1])
            y0 = float(nodes_y[iy]); y1 = float(nodes_y[iy + 1])
    
            # target moments
            M1x = float(np.sum(Wi * Vcomp[0, idc]))
            M1y = float(np.sum(Wi * Vcomp[1, idc]))
            xt = np.asarray(Vtot[idc], dtype=float)
            M2t = float(np.sum(Wi * xt * xt))  # one second-moment constraint on Vtot
    
            # 4 corners (v1,v2) and their Vtot^2
            corners = np.array([
                [x0, y0],
                [x1, y0],
                [x0, y1],
                [x1, y1],
            ], dtype=float)
            t2 = np.square(np.sum(corners, axis=1))  # (v1+v2)^2
    
            # solve A w = b with w >= 0
            A = np.array([
                [1.0, 1.0, 1.0, 1.0],
                [corners[0, 0], corners[1, 0], corners[2, 0], corners[3, 0]],
                [corners[0, 1], corners[1, 1], corners[2, 1], corners[3, 1]],
                [t2[0], t2[1], t2[2], t2[3]],
            ], dtype=float)
            b = np.array([M0, M1x, M1y, M2t], dtype=float)

            cnd = float(np.linalg.cond(A))
            use_fallback = False
            if (not np.isfinite(cnd)) or (cnd > cond_max):
                use_fallback = True
            else:
                wsol = np.linalg.solve(A, b)
                if not np.all(np.isfinite(wsol)):
                    use_fallback = True
                else:
                    # allow tiny negatives within eps, but reject meaningful negatives
                    if float(np.min(wsol)) < -abs(eps_w) * (abs(M0) + 1.0):
                        use_fallback = True

            if use_fallback:
                xt_cell = np.asarray(Vtot[idc], dtype=float)
                M1 = float(np.sum(Wi * xt_cell))
                M2 = float(np.sum(Wi * xt_cell * xt_cell))
                M1_d = np.array([float(np.sum(Wi * Vcomp[d, idc])) for d in range(dim)], dtype=float)
                if np.isfinite(M1) and M1 > 0:
                    ratio = np.maximum(np.where(np.isfinite(M1_d), M1_d, 0.0) / M1, 0.0)
                    s = float(np.sum(ratio))
                    if s > 1.0 + 1e-12:
                        ratio /= s
                else:
                    ratio = np.zeros(dim, dtype=float)
                self._append_two_point_reps(M0, M1, M2, ratio, V_cols, W_out)
                continue
    
            # accept solution: clamp tiny negatives to 0 (preserve near-exact moments)
            wsol = np.maximum(wsol, 0.0)
    
            # optional: tiny renormalization to keep M0 consistent after clamp
            sw = float(np.sum(wsol))
            if sw <= 0.0 or (not np.isfinite(sw)):
                xt_cell = np.asarray(Vtot[idc], dtype=float)
                M1 = float(np.sum(Wi * xt_cell))
                M2 = float(np.sum(Wi * xt_cell * xt_cell))
                M1_d = np.array([float(np.sum(Wi * Vcomp[d, idc])) for d in range(dim)], dtype=float)
                if np.isfinite(M1) and M1 > 0:
                    ratio = np.maximum(np.where(np.isfinite(M1_d), M1_d, 0.0) / M1, 0.0)
                    s = float(np.sum(ratio))
                    if s > 1.0 + 1e-12:
                        ratio /= s
                else:
                    ratio = np.zeros(dim, dtype=float)
                self._append_two_point_reps(M0, M1, M2, ratio, V_cols, W_out)
                continue
            if abs(sw - M0) > 1e-12 * (abs(M0) + 1.0):
                wsol *= (M0 / sw)
    
            # emit corners with weights
            for k in range(4):
                wk = float(wsol[k])
                if wk <= 0.0 or (not np.isfinite(wk)):
                    continue
                V_cols.append(corners[k].copy())
                W_out.append(wk)

        return self._merge_point_weights(V_cols, W_out)

    def _kernel_4pmc(
        self,
        idx: np.ndarray,
        Vcomp: np.ndarray,
        Vtot: np.ndarray,
        W: np.ndarray,
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        4PMC (2D): per occupied CAM-cell, build a closed-form 2x2 quadrature from
        local mean, marginal variances, and correlation. This matches
        M00/M10/M01/M20/M02/M11 when the symmetric nodes are admissible.

        Unsupported dimensions fall back to 2PM.
        """
        dim = int(self.dim)
        if dim != 2:
            return self._kernel_2pm(idx, Vcomp, Vtot, W)

        n_bins = self.recon_bins
        buckets, _, _ = self._bucket_by_cam_cells(idx, Vcomp, n_bins=n_bins, return_grid=False)

        eps_var = float(self.recon_4pmc_eps_var)
        V_cols: list[np.ndarray] = []
        W_out: list[float] = []

        for _, idc in buckets.items():
            Wi = np.asarray(W[idc], dtype=float)
            M0 = float(np.sum(Wi))
            if (not np.isfinite(M0)) or M0 <= 0.0:
                continue

            x = np.asarray(Vcomp[0, idc], dtype=float)
            y = np.asarray(Vcomp[1, idc], dtype=float)

            M10 = float(np.sum(Wi * x))
            M01 = float(np.sum(Wi * y))
            M20 = float(np.sum(Wi * x * x))
            M02 = float(np.sum(Wi * y * y))
            M11 = float(np.sum(Wi * x * y))

            mux = float(M10 / M0)
            muy = float(M01 / M0)
            varx = max(float(M20 / M0) - mux * mux, 0.0)
            vary = max(float(M02 / M0) - muy * muy, 0.0)

            # Degenerate cells are represented with lower-order closures.
            if varx <= eps_var and vary <= eps_var:
                V_cols.append(np.array([max(mux, 0.0), max(muy, 0.0)], dtype=float))
                W_out.append(M0)
                continue

            if varx <= eps_var:
                reps_y = self._two_point(M0, M01, M02)
                x_fix = max(mux, 0.0)
                for yk, wk in reps_y:
                    if wk <= 0.0 or (not np.isfinite(wk)):
                        continue
                    V_cols.append(np.array([x_fix, max(float(yk), 0.0)], dtype=float))
                    W_out.append(float(wk))
                continue

            if vary <= eps_var:
                reps_x = self._two_point(M0, M10, M20)
                y_fix = max(muy, 0.0)
                for xk, wk in reps_x:
                    if wk <= 0.0 or (not np.isfinite(wk)):
                        continue
                    V_cols.append(np.array([max(float(xk), 0.0), y_fix], dtype=float))
                    W_out.append(float(wk))
                continue

            sigx = math.sqrt(varx)
            sigy = math.sqrt(vary)
            cov = float(M11 / M0) - mux * muy
            rho = float(np.clip(cov / max(sigx * sigy, eps_var), -1.0, 1.0))

            x1 = mux - sigx
            x2 = mux + sigx
            y1 = muy - sigy
            y2 = muy + sigy

            if x1 < 0.0 or y1 < 0.0:
                xt_cell = np.asarray(Vtot[idc], dtype=float)
                M1 = float(np.sum(Wi * xt_cell))
                M2 = float(np.sum(Wi * xt_cell * xt_cell))
                M1_d = np.array([M10, M01], dtype=float)
                if np.isfinite(M1) and M1 > 0:
                    ratio = np.maximum(M1_d / M1, 0.0)
                    s = float(np.sum(ratio))
                    if s > 1.0 + 1e-12:
                        ratio /= s
                else:
                    ratio = np.zeros(dim, dtype=float)
                self._append_two_point_reps(M0, M1, M2, ratio, V_cols, W_out)
                continue

            nodes = [
                np.array([x1, y1], dtype=float),
                np.array([x1, y2], dtype=float),
                np.array([x2, y1], dtype=float),
                np.array([x2, y2], dtype=float),
            ]
            weights = [
                0.25 * M0 * (1.0 + rho),
                0.25 * M0 * (1.0 - rho),
                0.25 * M0 * (1.0 - rho),
                0.25 * M0 * (1.0 + rho),
            ]
            weights_arr = np.asarray(weights, dtype=float)
            if (not np.all(np.isfinite(weights_arr))) or float(np.min(weights_arr)) < 0.0:
                xt_cell = np.asarray(Vtot[idc], dtype=float)
                M1 = float(np.sum(Wi * xt_cell))
                M2 = float(np.sum(Wi * xt_cell * xt_cell))
                M1_d = np.array([M10, M01], dtype=float)
                if np.isfinite(M1) and M1 > 0:
                    ratio = np.maximum(M1_d / M1, 0.0)
                    s = float(np.sum(ratio))
                    if s > 1.0 + 1e-12:
                        ratio /= s
                else:
                    ratio = np.zeros(dim, dtype=float)
                self._append_two_point_reps(M0, M1, M2, ratio, V_cols, W_out)
                continue

            for vcol, wk in zip(nodes, weights):
                wk = float(wk)
                if wk <= 0.0 or (not np.isfinite(wk)):
                    continue
                V_cols.append(vcol)
                W_out.append(wk)

        return self._merge_point_weights(V_cols, W_out)
