from __future__ import annotations
from typing import Any, Callable, Tuple, Dict, Optional, List

import numpy as np
import math

from .grid import GridFactory
from .eligiblecache import (
    JunctUniSampler,
    JunctUniBoundarySampler,
    BondWeightedSampler,
    BondWeightedBoundarySampler,
)
from .meta import GridMeta
from .visualize import Plotter
from .agg_pool_npz_sqlite import AggPool
from .log_helper import PoolMemoryLogger

# numba kernels
from .func_jit import (
    run_one_fracture_kernel,  # single-fracture propagation kernel
    uf_label_bool,            # connected-component labeling on the opened junction/bond mask
    compress_count,           # compress labels and count A/B cells per fragment
    build_big_grid_mask,
)

class LMCSimulator:
    """
    Monte Carlo lattice fracture simulator (compact-grid variant).
    Holds the current grid in attributes and exposes high-level APIs:
      - generate_grid(...)
      - run_one_fracture(...)
      - analyze_fragments_compact(...)
      - simulate_until_fragments(...)
      - mc_breakage_repeat(...)
    """

    def __init__(self,
                 STR: np.ndarray,
                 NO_FRAG: int,
                 gamma: float = 1.0,
                 allow_loops: bool = True,
                 accept_all_cracks: bool = False,   # incremental mode: accept ineffective cracks without rollback
                 use_weighted_start: bool = False,
                 plotter: Plotter | None = None,
                 pool_dir: str | None = None,
                 warn_pool_out_of_bounds: bool = True) -> None:
        self.STR = np.asarray(STR, dtype=float)  # [11,12,22]
        self.NO_FRAG = int(NO_FRAG)
        self.gamma = float(gamma)
        self.allow_loops = bool(allow_loops)
        self.accept_all_cracks = bool(accept_all_cracks)
        if self.allow_loops and accept_all_cracks:
            print("[Warning]")
        self.use_weighted_start = bool(use_weighted_start)
        self.warn_pool_out_of_bounds = bool(warn_pool_out_of_bounds)

        # runtime state (set by generate_grid)
        self.M: np.ndarray | None = None
        self.Hbond: np.ndarray | None = None
        self.Vbond: np.ndarray | None = None
        self.meta: GridMeta | None = None

        # helpers
        self.grid_factory = GridFactory()
        self.plotter = plotter if plotter is not None else Plotter()

        # start-junction cache (will be built on simulate)
        self._cache = None
        # for test
        self.rollback_cnt = 0
        self.inter_start_cnt = 0
        self._pool_debug_logger = PoolMemoryLogger()

        self.agg_pool = (
            AggPool(pool_dir, warn_out_of_bounds=self.warn_pool_out_of_bounds)
            if pool_dir is not None
            else None
        )

    def close(self) -> None:
        if self.agg_pool is not None:
            self.agg_pool.close_pool_cache()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------
    # Grid management
    # ------------------------------
    def generate_grid(self,
                      A: float,
                      X1: float,
                      X2: float,
                      *,
                      A0: float = 0.0,
                      aspect_ratio: float = 1.0,
                      int_bre: float = 0.0,
                      seed: int | None = None) -> Dict[int, int]:
        """
        Build a new compact grid and set self.M/Hbond/Vbond/meta accordingly.
        Returns bond count dict {11:n11, 12:n12, 22:n22}.
        """
        M, Hb, Vb, meta, bond_counts = self.grid_factory.make(
            A=A, X1=X1, X2=X2, A0=A0,
            aspect_ratio=aspect_ratio, int_bre=int_bre, seed=seed
        )
        self.M = M
        self.Hbond = Hb
        self.Vbond = Vb
        self.meta = meta
        self._cache = None  # reset cache (grid changed)
        return bond_counts

    def generate_grid_udp(self,
                        mat: np.ndarray,
                        *,
                        a_code: int = 0,
                        b_code: int = 1,
                        empty_code: int = -1,
                        A0: float = 1.0,
                        int_bre: float = 0.0) -> Dict[int, int]:
        M, Hb, Vb, meta, bond_counts = self.grid_factory.make_from_array(mat, 
           a_code=a_code, b_code=b_code, 
           empty_code=empty_code, A0=A0, int_bre=int_bre)
        self.M = M
        self.Hbond = Hb
        self.Vbond = Vb
        self.meta = meta
        self._cache = None  # reset cache (grid changed)
        return bond_counts
        
    # ------------------------------
    # Fragment analysis (moved from fragments.py)
    # ------------------------------
    def analyze_fragments_compact(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return:
          labels : (H,W) int32, -1 for empty, 0..K-1 for fragment id on occupied cells
          cnt1   : (K,) #cells of material 1 per fragment
          cnt2   : (K,) #cells of material 2 per fragment
        """
        if self.M is None or self.Hbond is None or self.Vbond is None or self.meta is None:
            raise RuntimeError("Grid is not initialized. Call generate_grid() first.")
        M, Hbond, Vbond = self.M, self.Hbond, self.Vbond
        H, W = M.shape
        if H == 0 or W == 0:
            return np.empty((0, 0), dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

        R = build_big_grid_mask(M, Hbond, Vbond)
        Rlab = uf_label_bool((R == 1))
        cell_lab = Rlab[0::2, 0::2].astype(np.int32, copy=False)
        labels, cnt1, cnt2 = compress_count(cell_lab, M.astype(np.uint8, copy=False))
        return labels, cnt1, cnt2

    # ------------------------------
    # Fracture (single run)
    # ------------------------------
    @staticmethod
    def _bond_endpoints(H: int, W: int, loc: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        axis, i, j = loc
        if axis == 0:  # Hbond[i,j] between (i,j) and (i,j+1)
            return i, j, i, j + 1
        else:          # Vbond[i,j] between (i,j) and (i+1,j)
            return i, j, i + 1, j

    def run_one_fracture(self,
                         rng: Optional[np.random.Generator] = None,
                         record_path: bool = True) -> Tuple[Tuple[int, int], bool, float, List[Tuple[int, int, int, int]]]:
        """
        Launch one crack growth on the current grid (in-place).
        Returns:
          (r_end,c_end), complete_flag, energy, path_info[(axis,i,j,old_type), ...]
        """
        if self.M is None or self.Hbond is None or self.Vbond is None or self.meta is None:
            raise RuntimeError("Grid is not initialized. Call generate_grid() first.")
        if rng is None:
            rng = np.random.default_rng()

        H, W = self.meta.H, self.meta.W
        # choose start junction via cache (uniform over all with incident unbroken bonds)
        if self._cache is None:
            if self.accept_all_cracks:
                if not self.use_weighted_start:
                    self._cache = JunctUniSampler(self.meta.H, self.meta.W)
                else:
                    self._cache = BondWeightedSampler(self.meta.H, self.meta.W, STR=self.STR)
            else:
                if not self.use_weighted_start:
                    self._cache = JunctUniBoundarySampler(self.meta.H, self.meta.W)
                else:
                    self._cache = BondWeightedBoundarySampler(self.meta.H, self.meta.W, STR=self.STR)
            self._cache.build_initial(self.Hbond, self.Vbond)
        r0, c0 = self._cache.sample(rng)
        # self.internal_start = not (r0 == 0 or r0 == H or c0 == 0 or c0 == W)
        # if self.internal_start:
        #     self.inter_start_cnt += 1
            # print(f"[TEST] Start point r0={r0}, c0={c0}")
        self.r0, self.c0 = r0, c0
        # random stream for the kernel
        n_bonds = int((self.Hbond != -1).sum() + (self.Vbond != -1).sum())
        max_path = max(1, n_bonds)
        urand = rng.random(max_path + 8)

        r_end, c_end, comp_i, energy, used, A, I, J, Old = run_one_fracture_kernel(
            self.Hbond, self.Vbond, H, W,
            float(self.STR[0]), float(self.STR[1]), float(self.STR[2]),
            float(self.gamma), int(self.meta.int_bre_len),
            1 if self.allow_loops else 0, 
            int(r0), int(c0), int(max_path),
            urand.astype(np.float64),
        )

        path_info: List[Tuple[int, int, int, int]] = []
        if record_path and used > 0:
            for t in range(int(used)):
                path_info.append((int(A[t]), int(I[t]), int(J[t]), int(Old[t])))

        return (int(r_end), int(c_end)), bool(comp_i), float(energy), path_info

    def _emit_crack_step_trace(
        self,
        *,
        callback: Callable[..., None],
        Hbond_before: np.ndarray,
        Vbond_before: np.ndarray,
        path_info: List[Tuple[int, int, int, int]],
        crack_groups_before: List[List[List[Tuple[int, int, int]]]],
        current_group_before: List[List[Tuple[int, int, int]]],
        crack_index: int,
        accepted: bool,
        fragment_count_before: int,
        fragment_count_after: int,
        crack_energy: float,
        trace_stride: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.M is None:
            raise RuntimeError("Grid is not initialized. Call generate_grid() first.")
        if not path_info:
            return

        stride = max(1, int(trace_stride))
        Hb_frame = Hbond_before.copy()
        Vb_frame = Vbond_before.copy()
        path_so_far: List[Tuple[int, int, int]] = []
        energy_so_far = 0.0

        for step_idx, (axis, ii, jj, old_type) in enumerate(path_info):
            if axis == 0:
                Hb_frame[ii, jj] = -1
            else:
                Vb_frame[ii, jj] = -1

            path_so_far.append((int(axis), int(ii), int(jj)))
            if int(old_type) == 11:
                energy_so_far += float(self.STR[0])
            elif int(old_type) == 12:
                energy_so_far += float(self.STR[1])
            elif int(old_type) == 22:
                energy_so_far += float(self.STR[2])

            is_last = step_idx == len(path_info) - 1
            if (step_idx % stride) != 0 and not is_last:
                continue

            crack_groups_frame = [
                [list(path) for path in group]
                for group in crack_groups_before
            ]
            active_group = [list(path) for path in current_group_before]
            active_group.append(path_so_far.copy())
            crack_groups_frame.append(active_group)

            callback(
                simulator=self,
                M=self.M,
                Hbond=Hb_frame.copy(),
                Vbond=Vb_frame.copy(),
                crack_paths=crack_groups_frame,
                path_so_far=path_so_far.copy(),
                step_index=int(step_idx),
                step_number=int(step_idx + 1),
                total_steps=int(len(path_info)),
                crack_index=int(crack_index),
                accepted=bool(accepted),
                fragment_count_before=int(fragment_count_before),
                fragment_count_after=int(fragment_count_after),
                old_type=int(old_type),
                crack_energy=float(crack_energy),
                energy_so_far=float(energy_so_far),
                context=dict(context or {}),
            )

    # ------------------------------
    # Fracture until target fragments
    # ------------------------------
    def simulate_until_fragments(self,
                                 NO_FRAG: Optional[int] = None,
                                 *,
                                 seed: Optional[int] = None,
                                 max_steps: Optional[int] = None,
                                 plot_intermediate: bool = False,
                                 plot_final: bool = False,
                                 track_crack_paths: bool = True,
                                 crack_step_callback: Optional[Callable[..., None]] = None,
                                 trace_rejected_cracks: bool = False,
                                 trace_stride: int = 1,
                                 crack_step_context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[List[Tuple[int, int, int]]]]:
        """
        Keep breaking bonds (with rollback if no new fragment formed) until we reach NO_FRAG.
        Returns:
          labels, cnt1, cnt2, total_energy, crack_paths(list of paths, each path: [(axis,i,j), ...])

        If crack_step_callback is provided, accepted crack paths are replayed
        step by step on copies of the pre-crack bond arrays and emitted through
        the callback. Rejected cracks are skipped unless trace_rejected_cracks
        is True.
        """
        if self.M is None or self.Hbond is None or self.Vbond is None or self.meta is None:
            raise RuntimeError("Grid is not initialized. Call generate_grid() first.")
        rng = np.random.default_rng(seed)

        if self.accept_all_cracks:
            if not self.use_weighted_start:
                self._cache = JunctUniSampler(self.meta.H, self.meta.W)
            else:
                self._cache = BondWeightedSampler(self.meta.H, self.meta.W, STR=self.STR)
        else:
            if not self.use_weighted_start:
                self._cache = JunctUniBoundarySampler(self.meta.H, self.meta.W)
            else:
                self._cache = BondWeightedBoundarySampler(self.meta.H, self.meta.W, STR=self.STR)
        self._cache.build_initial(self.Hbond, self.Vbond)

        labels, cnt1, cnt2 = self.analyze_fragments_compact()
        target = max(1, int(NO_FRAG) if NO_FRAG is not None else self.NO_FRAG)
        energy_total = 0.0
        need_crack_paths = bool(
            track_crack_paths
            or plot_intermediate
            or plot_final
            or crack_step_callback is not None
        )
        crack_groups: List[List[List[Tuple[int, int, int]]]] = []
        current_group: List[List[Tuple[int, int, int]]] = []

        if max_steps is None:
            n_bonds = int((self.Hbond != -1).sum() + (self.Vbond != -1).sum())
            max_steps = max(target * 4, int(1.1 * max(1, n_bonds)))

        steps = 0
        while (cnt1.size + 0) < target and steps < max_steps:
            steps += 1
            trace_Hbond_before = None
            trace_Vbond_before = None
            trace_crack_groups_before: List[List[List[Tuple[int, int, int]]]] = []
            trace_current_group_before: List[List[Tuple[int, int, int]]] = []
            if crack_step_callback is not None:
                trace_Hbond_before = self.Hbond.copy()
                trace_Vbond_before = self.Vbond.copy()
                trace_crack_groups_before = [
                    [list(path) for path in group]
                    for group in crack_groups
                ]
                trace_current_group_before = [list(path) for path in current_group]

            _rc_end, _complete, E, path_info = self.run_one_fracture(rng=rng, record_path=True)

            labels_new, cnt1_new, cnt2_new = self.analyze_fragments_compact()
            accepted_for_trace = bool(self.accept_all_cracks or cnt1_new.size > cnt1.size)

            if self.accept_all_cracks:
                energy_total += E
                if need_crack_paths:
                    current_group.append([(a, i, j) for (a, i, j, _old) in path_info])
                if not self.use_weighted_start:
                    touch: list[tuple[int, int]] = []
                    for (axis, ii, jj, _old_type) in path_info:
                        r1, c1, r2, c2 = self._bond_endpoints(self.meta.H, self.meta.W, (axis, ii, jj))
                        touch.append((r1, c1))
                        touch.append((r2, c2))
                    self._cache.recompute_at(self.Hbond, self.Vbond, touch)
                else:
                    self._cache.on_bonds_broken(path_info)
                if cnt1_new.size > cnt1.size:
                    if need_crack_paths and len(current_group) > 0:
                        crack_groups.append(current_group)
                        current_group = []
                    if plot_intermediate and self.plotter is not None:
                        self.plotter.plot_compact(
                            self.M,
                            self.Hbond,
                            self.Vbond,
                            labels=None,
                            crack_paths=(crack_groups if need_crack_paths else None),
                            title=f"Fragment #{cnt1_new.size} created",
                            mode='materials',
                        )
            else:
                if cnt1_new.size > cnt1.size:
                    energy_total += E
                    if need_crack_paths:
                        current_group.append([(a, i, j) for (a, i, j, _old) in path_info])
                        crack_groups.append(current_group)
                        current_group = []
                    if not self.use_weighted_start:
                        touch: list[tuple[int, int]] = []
                        for (axis, ii, jj, _old_type) in path_info:
                            r1, c1, r2, c2 = self._bond_endpoints(self.meta.H, self.meta.W, (axis, ii, jj))
                            touch.append((r1, c1))
                            touch.append((r2, c2))
                        self._cache.recompute_at(self.Hbond, self.Vbond, touch)
                    else:
                        self._cache.on_bonds_broken(path_info)
                    if plot_intermediate and self.plotter is not None:
                        self.plotter.plot_compact(
                            self.M,
                            self.Hbond,
                            self.Vbond,
                            labels=None,
                            crack_paths=(crack_groups if need_crack_paths else None),
                            title=f"Fragment #{cnt1_new.size} created",
                            mode='materials',
                        )
                else:
                    for (axis, ii, jj, old) in reversed(path_info):
                        if axis == 0:
                            self.Hbond[ii, jj] = int(old)
                        else:
                            self.Vbond[ii, jj] = int(old)
                    labels_new, cnt1_new, cnt2_new = labels, cnt1, cnt2

            if (
                crack_step_callback is not None
                and (accepted_for_trace or bool(trace_rejected_cracks))
                and trace_Hbond_before is not None
                and trace_Vbond_before is not None
            ):
                self._emit_crack_step_trace(
                    callback=crack_step_callback,
                    Hbond_before=trace_Hbond_before,
                    Vbond_before=trace_Vbond_before,
                    path_info=path_info,
                    crack_groups_before=trace_crack_groups_before,
                    current_group_before=trace_current_group_before,
                    crack_index=steps,
                    accepted=accepted_for_trace,
                    fragment_count_before=int(cnt1.size),
                    fragment_count_after=int(cnt1_new.size),
                    crack_energy=float(E),
                    trace_stride=int(trace_stride),
                    context=crack_step_context,
                )

            labels, cnt1, cnt2 = labels_new, cnt1_new, cnt2_new

            if not self.use_weighted_start:
                if ((self.Hbond != -1).sum() + (self.Vbond != -1).sum()) == 0:
                    break
            else:
                if self._cache.is_empty():
                    break

        if plot_final and self.plotter is not None:
            self.plotter.plot_compact(
                self.M,
                self.Hbond,
                self.Vbond,
                labels=labels,
                crack_paths=(crack_groups if need_crack_paths else None),
                title=f"Final: {cnt1.size} fragments",
                mode='fragments',
            )
        return labels, cnt1, cnt2, energy_total, (crack_groups if need_crack_paths else [])
    # ------------------------------
    # Lattice Monte Carlo (repeat)
    # ------------------------------
    def mc_breakage_repeat(self,
                           A: float, X1: float, X2: float,
                           *,
                           N_GRIDS: int = 10, N_FRACS: int = 10,
                           A0: float = 0.0,
                           aspect_ratio: float = 1.0,
                           int_bre: float = 0.0,
                           seed: int | None = None,
                           plot_each: bool = False) -> np.ndarray:
        """
        Repeat experiment over multiple random grids and multiple runs per grid.
        Returns F of shape (N_GRIDS*N_FRACS*NO_FRAG, 4) with columns:
          [ total_volume, volume_A, volume_B, fracture_energy ]
        """
        rng = np.random.default_rng(seed)
        total_rows = int(N_GRIDS) * int(N_FRACS) * int(self.NO_FRAG)
        F = np.zeros((total_rows, 4), dtype=float)
        row = 0

        for g in range(int(N_GRIDS)):
            grid_seed = int(rng.integers(0, 2**31 - 1))
            self.generate_grid(A=A, X1=X1, X2=X2, A0=A0,
                               aspect_ratio=aspect_ratio, int_bre=int_bre, seed=grid_seed)
            Hbond_ori = self.Hbond.copy()
            Vbond_ori = self.Vbond.copy()
            for f in range(int(N_FRACS)):
                sim_seed = int(rng.integers(0, 2**31 - 1))
                labels, c1, c2, E, paths = self.simulate_until_fragments(
                    NO_FRAG=self.NO_FRAG, seed=sim_seed,
                    plot_intermediate=False, plot_final=False,
                    track_crack_paths=bool(plot_each)
                )
                K = int(c1.size)
                if K > 0:
                    # distribute remainders proportionally (legacy-compatible)
                    units_area = self.meta.A0 * (c1 + c2).astype(float)
                    denom = max((A - (self.meta.R[0] + self.meta.R[1])), 1e-12)
                    share = units_area / denom
                    VA = self.meta.A0 * c1.astype(float) + share * self.meta.R[0]
                    VB = self.meta.A0 * c2.astype(float) + share * self.meta.R[1]
                    VT = VA + VB
                    energy = float(E * np.sqrt(max(self.meta.A0, 1e-12)))
                    n_write = min(K, int(self.NO_FRAG))
                    F[row:row+n_write, 0] = VT[:n_write]
                    F[row:row+n_write, 1] = VA[:n_write]
                    F[row:row+n_write, 2] = VB[:n_write]
                    F[row:row+n_write, 3] = energy
                    row += int(self.NO_FRAG)
                else:
                    row += int(self.NO_FRAG)
                self.Hbond = Hbond_ori.copy()
                self.Vbond = Vbond_ori.copy()
                if plot_each and self.plotter is not None:
                    self.plotter.plot_compact(self.M, self.Hbond, self.Vbond,
                                              labels=labels, crack_paths=paths,
                                              title=f"Grid {g+1}/{N_GRIDS}, run {f+1}/{N_FRACS}",
                                              mode='fragments')
        self._pool_debug_logger.maybe_log(self, tag='after_repeat_call')
        return F

    # ------------------------------
    # Lattice Monte Carlo (with user defined particle)
    # ------------------------------
    def mc_breakage_udp(self,
                        mat: np.ndarray,
                        *,
                        N_FRACS: int = 10,
                        a_code: int = 0,
                        b_code: int = 1,
                        empty_code: int = -1,
                        A0: float = 1.0,
                        int_bre: float = 0.0,
                        seed: int | None = None,
                        plot_each: bool = False,
                        plot_intermediate: bool = False,
                        crack_step_callback: Optional[Callable[..., None]] = None,
                        trace_rejected_cracks: bool = False,
                        trace_stride: int = 1) -> np.ndarray:
        """
        Monte Carlo breakage over user-provided material grids.
    
        Parameters
        ----------
        mat : np.ndarray
            2D or 3D array of material maps.
            Convention by default: -1=empty, 0=material A, 1=material B.
            If 3D, assumed shape (N_GRIDS, H, W). If 2D, treated as one grid.
        N_FRACS : int
            Repeats per grid.
        a_code, b_code, empty_code : int
            Codes used in `mat` for A/B/empty; will be mapped to internal {0=empty,1=A,2=B}.
        A0 : float
            Area per occupied cell.
        int_bre : float
            Initial break depth ratio; converted to `int_bre_len = ceil(max(H,W)*int_bre)` (<=0 -> 1).
        seed : int | None
            RNG seed for reproducibility.
        plot_each : bool
            If True, plot final fragments after each run.
        plot_intermediate : bool
            If True, plot when a new fragment is created during each run.
            This is separate from plot_each.
        crack_step_callback : callable, optional
            If provided, called for each traced crack-extension step with
            keyword arguments including M, Hbond, Vbond, crack_paths,
            step_index, crack_index, accepted, and context.
        trace_rejected_cracks : bool
            If True, also emit callback frames for cracks that are rolled back.
        trace_stride : int
            Emit one callback every trace_stride extension steps, always
            including the final step of each traced crack.
    
        Returns
        -------
        F : (N_GRIDS * N_FRACS * self.NO_FRAG, 4) float array
            Columns: [total_volume, volume_A, volume_B, fracture_energy]
        """
        # normalize mats list
        if mat.ndim == 2:
            mats = [mat]
        elif mat.ndim == 3:
            mats = [mat[i] for i in range(mat.shape[0])]
        else:
            raise ValueError("`mat` must be 2D or 3D (N,H,W).")
    
        N_GRIDS = len(mats)
        rng = np.random.default_rng(seed)
        total_rows = int(N_GRIDS) * int(N_FRACS) * int(self.NO_FRAG)
        F = np.zeros((total_rows, 4), dtype=float)
        row = 0
    
        for g in range(N_GRIDS):
            self.generate_grid_udp(mats[g], a_code=a_code, b_code=b_code, 
                                   empty_code=empty_code, A0=A0, int_bre=int_bre)
    
            # Keep a copy of the original bonds so each trial can be restored afterwards
            Hbond_ori = self.Hbond.copy()
            Vbond_ori = self.Vbond.copy()
    
            for f in range(int(N_FRACS)):
                sim_seed = int(rng.integers(0, 2**31 - 1))
                labels, c1, c2, E, paths = self.simulate_until_fragments(
                    NO_FRAG=self.NO_FRAG,
                    seed=sim_seed,
                    plot_intermediate=bool(plot_intermediate),
                    plot_final=False,
                    track_crack_paths=bool(plot_each or plot_intermediate or crack_step_callback is not None),
                    crack_step_callback=crack_step_callback,
                    trace_rejected_cracks=bool(trace_rejected_cracks),
                    trace_stride=int(trace_stride),
                    crack_step_context={
                        "source": "mc_breakage_udp",
                        "grid_index": int(g),
                        "frac_index": int(f),
                        "sim_seed": int(sim_seed),
                    },
                )
    
                K = int(c1.size)
                if K > 0:
                    # Convert fragment cell counts directly to volumes (no remainder redistribution)
                    VA = self.meta.A0 * c1.astype(float)
                    VB = self.meta.A0 * c2.astype(float)
                    VT = VA + VB
    
                    energy = float(E * np.sqrt(max(self.meta.A0, 1e-12)))
    
                    n_write = min(K, int(self.NO_FRAG))
                    F[row:row + n_write, 0] = VT[:n_write]
                    F[row:row + n_write, 1] = VA[:n_write]
                    F[row:row + n_write, 2] = VB[:n_write]
                    F[row:row + n_write, 3] = energy
                    row += int(self.NO_FRAG)
                else:
                    row += int(self.NO_FRAG)
    
                # Restore the sampled grid bonds before the next fracture repeat so repeated runs on one grid remain independent
                self.Hbond = Hbond_ori.copy()
                self.Vbond = Vbond_ori.copy()
    
                if plot_each and self.plotter is not None:
                    self.plotter.plot_fragments_simple(
                        self.M, 
                        # self.Hbond, self.Vbond,
                        labels=labels, 
                        # crack_paths=paths,
                        title="Four-Fragment Aggregate",
                        # mode='fragments'
                    )
    
        self._pool_debug_logger.maybe_log(self, tag='after_udp_call')
        return F

    def mc_breakage_from_pool(
        self,
        pool_dir: str,
        Df: float,
        MAS: float,
        A: float,
        X1: float,
        X2: float | None = None,
        *,
        N_GRIDS: int = 1,
        N_FRACS: int = 10,
        A0: float = 1.0,
        int_bre: float = 0.0,
        seed: int | None = None,
        plot_each: bool = False,
        interp: str = "knn",
        KNN: int = 4,
        sigma: float = 0.35,
        max_draws: int = 15,
        tau_A: float | None = None,
        tau_X: float | None = None,
        log_bilinear: bool = False,
    ) -> np.ndarray:
        """
        Run Monte Carlo breakage using grids sampled from the NPZ+SQLite pool.

        Each outer draw selects one source grid from the nearest pool groups in
        `(A_norm, X1)` space, then reuses that sampled grid for `N_FRACS`
        fracture simulations. Fragment masses are rescaled afterwards so the
        phase totals remain consistent with the requested `(A, X1, X2)`.
        """

        if X2 is None:
            X2 = 1.0 - X1

        rng = np.random.default_rng(seed)

        if self.agg_pool is None:
            self.agg_pool = AggPool(
                pool_dir, warn_out_of_bounds=self.warn_pool_out_of_bounds
            )
        else:
            if self.agg_pool.pool_dir != pool_dir:
                self.agg_pool.close_pool_cache()
                self.agg_pool.pool_dir = pool_dir
            self.agg_pool.warn_out_of_bounds = self.warn_pool_out_of_bounds

        A_norm = float(A) / float(A0) if A0 > 0 else float(A)

        total_rows = int(N_GRIDS) * int(N_FRACS) * int(self.NO_FRAG)
        F = np.zeros((total_rows, 4), dtype=float)
        row = 0

        # Phase remainders are derived from the requested input mass split, not from pool metadata
        R1 = (A * X1) % A0 if A0 > 0 else 0.0
        R2 = (A * X2) % A0 if A0 > 0 else 0.0

        # Target phase masses used for post-fracture rescaling
        M1_tar = float(A * X1)
        M2_tar = float(A * X2)
        mass_scale = M1_tar + M2_tar
        eps = 1e-12 * mass_scale

        for g in range(int(N_GRIDS)):
            # Sample one source grid from the pool
            M, Hbond, Vbond = self.agg_pool.sample_grid(
                Df, MAS, A_norm, X1, rng,
                interp=interp, KNN=KNN, sigma=sigma,
                max_draws=max_draws, tau_A=tau_A, tau_X=tau_X,
                log_bilinear=log_bilinear
            )

            # Rebuild grid metadata using the requested simulation inputs
            H, W = M.shape
            N1_pool = int((M == 1).sum())
            N2_pool = int((M == 2).sum())
            total_units = N1_pool + N2_pool
            aspect_ratio = float(W) / float(H) if H > 0 else 1.0

            if int_bre <= 0:
                int_bre_len = 1
            else:
                int_bre_len = int(np.ceil(max(H, W) * float(int_bre)))

            self.M = M
            self.Hbond = Hbond
            self.Vbond = Vbond
            self.meta = GridMeta(
                H=int(H), W=int(W),
                A0=float(A0),
                N1=int(N1_pool), N2=int(N2_pool),
                R=(float(R1), float(R2)),
                total_units=int(total_units),
                aspect_ratio=float(aspect_ratio),
                int_bre=float(int_bre),
                int_bre_len=int(int_bre_len),
            )
            self._cache = None

            Hbond_ori = self.Hbond.copy()
            Vbond_ori = self.Vbond.copy()

            # Repeat fracture simulation on the sampled grid
            for f in range(int(N_FRACS)):
                sim_seed = int(rng.integers(0, 2**31 - 1))
                labels, c1, c2, E, paths = self.simulate_until_fragments(
                    NO_FRAG=self.NO_FRAG,
                    seed=sim_seed,
                    plot_intermediate=False,
                    plot_final=False,
                    track_crack_paths=bool(plot_each),
                )

                Kfrag = int(c1.size)
                if Kfrag > 0:
                    # Compute fragment masses before enforcing the requested phase totals
                    units_area = self.meta.A0 * (c1 + c2).astype(float)
                    denom = max((A - (self.meta.R[0] + self.meta.R[1])), eps)
                    share = units_area / denom

                    VA_raw = self.meta.A0 * c1.astype(float) + share * self.meta.R[0]
                    VB_raw = self.meta.A0 * c2.astype(float) + share * self.meta.R[1]

                    # Compute per-phase scaling factors for exact mass conservation
                    M1_raw = float(VA_raw.sum())
                    M2_raw = float(VB_raw.sum())

                    # Handle degenerate cases where one phase is absent in the sampled result
                    if M1_raw <= eps:
                        # Keep the output numerically stable when the sampled grid contains no mass of this phase.
                        alpha1 = 0.0
                    else:
                        alpha1 = M1_tar / M1_raw

                    if M2_raw <= eps:
                        alpha2 = 0.0
                    else:
                        alpha2 = M2_tar / M2_raw

                    # Apply scaling and write mass-conservative fragment outputs
                    VA = alpha1 * VA_raw
                    VB = alpha2 * VB_raw
                    VT = VA + VB

                    energy = float(E * np.sqrt(max(self.meta.A0, eps)))
                    n_write = min(Kfrag, int(self.NO_FRAG))

                    F[row:row+n_write, 0] = VT[:n_write]
                    F[row:row+n_write, 1] = VA[:n_write]
                    F[row:row+n_write, 2] = VB[:n_write]
                    F[row:row+n_write, 3] = energy
                    row += int(self.NO_FRAG)
                else:
                    row += int(self.NO_FRAG)

                # Restore the sampled grid bonds before the next fracture repeat
                self.Hbond = Hbond_ori.copy()
                self.Vbond = Vbond_ori.copy()

                if plot_each and self.plotter is not None:
                    self.plotter.plot_compact(
                        self.M, self.Hbond, self.Vbond,
                        labels=labels, crack_paths=paths,
                        title=f"[POOL] Grid {g+1}/{N_GRIDS}, run {f+1}/{N_FRACS}",
                        mode='fragments'
                    )

        self._pool_debug_logger.maybe_log(self, tag='after_pool_call')
        return F










