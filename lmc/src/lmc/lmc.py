from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import os
import math
import gc
import tracemalloc
import h5py

from .grid import GridFactory
from .eligiblecache import (
    JunctUniSampler,
    JunctUniBoundarySampler,
    BondWeightedSampler,
    BondWeightedBoundarySampler,
)
from .meta import GridMeta
from .visualize import Plotter
from .agg_pool import AggPool

# numba kernels
from .func_jit import (
    run_one_fracture_kernel,  # fractureæŽ¨è¿›æ ¸
    uf_label_bool,            # äºŒå€¼è¿žé€šåŸŸæ ‡è®°ï¼ˆjunction/bondå±•å¼€åŽï¼‰
    compress_count,           # æ ‡ç­¾åŽ‹ç¼© + A/B è®¡æ•°
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
                 accept_all_cracks: bool = False,   # <<< æ–°å¢žï¼šæŽ¥å—â€œæ— æ•ˆâ€è£‚ç¼ï¼Œä¸å›žæ»š
                 use_weighted_start: bool = False,
                 plotter: Plotter | None = None,
                 pool_dir: str | None = None) -> None:
        self.STR = np.asarray(STR, dtype=float)  # [11,12,22]
        self.NO_FRAG = int(NO_FRAG)
        self.gamma = float(gamma)
        self.allow_loops = bool(allow_loops)
        self.accept_all_cracks = bool(accept_all_cracks)
        if self.allow_loops and accept_all_cracks:
            print("[Warning]")
        self.use_weighted_start = bool(use_weighted_start)

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
        self._pool_call_counter = 0
        
        self.agg_pool = AggPool(pool_dir) if pool_dir is not None else None

    def close(self) -> None:
        if self.agg_pool is not None:
            self.agg_pool.close_pool_cache()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _runtime_memory_stats(self, *, force_gc: bool = False) -> Dict[str, float | int]:
        if force_gc:
            gc.collect()

        stats: Dict[str, float | int] = {}
        if self.M is not None:
            stats["M_bytes"] = int(self.M.nbytes)
        if self.Hbond is not None:
            stats["Hbond_bytes"] = int(self.Hbond.nbytes)
        if self.Vbond is not None:
            stats["Vbond_bytes"] = int(self.Vbond.nbytes)

        if self.agg_pool is not None:
            for key, value in self.agg_pool.cache_stats().items():
                stats[f"aggpool_{key}"] = int(value)

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            stats["py_current_bytes"] = int(current)
            stats["py_peak_bytes"] = int(peak)

        try:
            import psutil  # type: ignore
            stats["rss_bytes"] = int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass

        return stats

    def _maybe_log_pool_memory(self, tag: str) -> None:
        enabled = os.environ.get("LMC_POOL_DEBUG_MEMORY", "").strip().lower()
        if enabled not in ("1", "true", "yes", "on"):
            return

        self._pool_call_counter += 1
        every = int(os.environ.get("LMC_POOL_DEBUG_EVERY", "100") or "100")
        every = max(1, every)
        if (self._pool_call_counter % every) != 0:
            return

        if not tracemalloc.is_tracing():
            tracemalloc.start(10)

        force_gc = os.environ.get("LMC_POOL_DEBUG_GC", "0").strip().lower() in ("1", "true", "yes", "on")
        stats = self._runtime_memory_stats(force_gc=force_gc)
        ordered = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"[LMC memory][{tag}] call={self._pool_call_counter}, {ordered}")
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
                                 track_crack_paths: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[List[Tuple[int, int, int]]]]:
        """
        Keep breaking bonds (with rollback if no new fragment formed) until we reach NO_FRAG.
        Returns:
          labels, cnt1, cnt2, total_energy, crack_paths(list of paths, each path: [(axis,i,j), ...])
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
        need_crack_paths = bool(track_crack_paths or plot_intermediate or plot_final)
        crack_groups: List[List[List[Tuple[int, int, int]]]] = []
        current_group: List[List[Tuple[int, int, int]]] = []

        if max_steps is None:
            n_bonds = int((self.Hbond != -1).sum() + (self.Vbond != -1).sum())
            max_steps = max(target * 4, int(1.1 * max(1, n_bonds)))

        steps = 0
        while (cnt1.size + 0) < target and steps < max_steps:
            steps += 1
            _rc_end, _complete, E, path_info = self.run_one_fracture(rng=rng, record_path=True)

            labels_new, cnt1_new, cnt2_new = self.analyze_fragments_compact()

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
        self._maybe_log_pool_memory(tag='after_repeat_call')
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
                        plot_each: bool = False) -> np.ndarray:
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
    
            # åŽŸå§‹é”®çŸ©é˜µçš„å¿«ç…§ï¼Œä¾›æ¯æ¬¡è¯•éªŒåŽæ¢å¤
            Hbond_ori = self.Hbond.copy()
            Vbond_ori = self.Vbond.copy()
    
            for f in range(int(N_FRACS)):
                sim_seed = int(rng.integers(0, 2**31 - 1))
                labels, c1, c2, E, paths = self.simulate_until_fragments(
                    NO_FRAG=self.NO_FRAG,
                    seed=sim_seed,
                    plot_intermediate=False,
                    plot_final=False,
                    track_crack_paths=bool(plot_each),
                )
    
                K = int(c1.size)
                if K > 0:
                    # ç›´æŽ¥ç”±è®¡æ•°æ¢ç®—ä½“ç§¯ï¼ˆæ— ä½™é‡åˆ†é…ï¼‰
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
    
                # æ¢å¤é”®çŸ©é˜µï¼Œç¡®ä¿åŒä¸€ç½‘æ ¼çš„å¤šæ¬¡æ¨¡æ‹Ÿç›¸äº’ç‹¬ç«‹
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
    
        self._maybe_log_pool_memory(tag='after_udp_call')
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
        interp: str = "knn",     # "knn" or "bilinear"
        KNN: int = 4,
        sigma: float = 0.35,
        # ---- æ–¹æ¡ˆA + log-bilinear æ–°å¢ž ----
        max_draws: int = 15,
        tau_A: float | None = None,   # e.g. 0.10
        tau_X: float | None = None,   # e.g. 0.05
        log_bilinear: bool = False,
    ) -> np.ndarray:
        """
        ä»Žç¦»çº¿ aggregate æ± ä¸­æŒ‰ (A_norm, X1) é€‰æ‹©å°æ± å­å¹¶éšæœºæŠ½æ ·ï¼Œå†åšæ–­è£‚æ¨¡æ‹Ÿã€‚
        æ­¤ç‰ˆæœ¬åŠ å…¥ä¸¥æ ¼è´¨é‡å®ˆæ’ä¿®æ­£ï¼š
            - æ€»è´¨é‡å®ˆæ’: sum(VT)=A
            - ä¸¤ç›¸åˆ†åˆ«å®ˆæ’: sum(VA)=A*X1, sum(VB)=A*X2
        """

        if X2 is None:
            X2 = 1.0 - X1

        rng = np.random.default_rng(seed)

        if self.agg_pool is None:
            self.agg_pool = AggPool(pool_dir)
        else:
            if self.agg_pool.pool_dir != pool_dir:
                self.agg_pool.close_pool_cache()
                self.agg_pool.pool_dir = pool_dir

        A_norm = float(A) / float(A0) if A0 > 0 else float(A)

        total_rows = int(N_GRIDS) * int(N_FRACS) * int(self.NO_FRAG)
        F = np.zeros((total_rows, 4), dtype=float)
        row = 0

        # remainder ç”±è¾“å…¥ A/X1/X2 + A0 å¾—åˆ°ï¼ˆä¸è¯»æ± å­ï¼‰
        R1 = (A * X1) % A0 if A0 > 0 else 0.0
        R2 = (A * X2) % A0 if A0 > 0 else 0.0

        # ä¸¤ç›¸ç›®æ ‡æ€»è´¨é‡ï¼ˆé¢ç§¯ï¼‰
        M1_tar = float(A * X1)
        M2_tar = float(A * X2)
        mass_scale = M1_tar + M2_tar
        eps = 1e-12 * mass_scale

        for g in range(int(N_GRIDS)):
            # ---- ä»Žæ± å­ä¸­æŠ½ä¸€ä¸ª gridï¼ˆbilinear æ¨¡å¼ä¸‹å¯ç”¨æ–¹æ¡ˆAï¼‰ ----
            M, Hbond, Vbond = self.agg_pool.sample_grid(
                Df, MAS, A_norm, X1, rng,
                interp=interp, KNN=KNN, sigma=sigma,
                max_draws=max_draws, tau_A=tau_A, tau_X=tau_X,
                log_bilinear=log_bilinear
            )

            # ---- meta ç”¨æœ¬æ¬¡è¾“å…¥é‡å»º ----
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

            # ---- fracture repeats ----
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
                    # --- 1) å…ˆæŒ‰åŽŸé€»è¾‘ç®— raw è´¨é‡ï¼ˆå« remainder åˆ†é…ï¼‰ ---
                    units_area = self.meta.A0 * (c1 + c2).astype(float)
                    denom = max((A - (self.meta.R[0] + self.meta.R[1])), eps)
                    share = units_area / denom

                    VA_raw = self.meta.A0 * c1.astype(float) + share * self.meta.R[0]
                    VB_raw = self.meta.A0 * c2.astype(float) + share * self.meta.R[1]

                    # --- 2) è®¡ç®—ä¸¤ç›¸ç¼©æ”¾å› å­ï¼Œä¿è¯åˆ†åˆ«å®ˆæ’ ---
                    M1_raw = float(VA_raw.sum())
                    M2_raw = float(VB_raw.sum())

                    # å•ç›¸/ç¼ºç›¸å¤„ç†ï¼š
                    if M1_raw <= eps:
                        # ç›®æ ‡æœ‰è¯¥ç›¸ï¼Œä½†æŠ½æ ·ç½‘æ ¼é‡Œæ²¡æœ‰ -> è¯´æ˜ŽæŠ½æ ·å¤ªè¿œ
                        # è¿™é‡Œä¸æ­»å¾ªçŽ¯ï¼Œç›´æŽ¥æŠŠè¯¥ç›¸è´¨é‡å‡åŒ€ç½®å…¥ä¼šç ´åææ–™å«é‡ï¼Œ
                        # æ‰€ä»¥é‡‡ç”¨ fallbackï¼šä¸ç¼©æ”¾ä½†ç»™å‡ºä¿æŠ¤ï¼ˆä»å®ˆæ’é  alpha2ï¼‰
                        alpha1 = 0.0
                    else:
                        alpha1 = M1_tar / M1_raw

                    if M2_raw <= eps:
                        alpha2 = 0.0
                    else:
                        alpha2 = M2_tar / M2_raw

                    # --- 3) åº”ç”¨ç¼©æ”¾ï¼Œå¾—åˆ°å®ˆæ’åŽçš„è¾“å‡ºè´¨é‡ ---
                    VA = alpha1 * VA_raw
                    VB = alpha2 * VB_raw
                    VT = VA + VB  # æ€»é‡ä¹Ÿä¼šä¸¥æ ¼ç­‰äºŽ A

                    energy = float(E * np.sqrt(max(self.meta.A0, eps)))
                    n_write = min(Kfrag, int(self.NO_FRAG))

                    F[row:row+n_write, 0] = VT[:n_write]
                    F[row:row+n_write, 1] = VA[:n_write]
                    F[row:row+n_write, 2] = VB[:n_write]
                    F[row:row+n_write, 3] = energy
                    row += int(self.NO_FRAG)
                else:
                    row += int(self.NO_FRAG)

                # æ¢å¤é”®çŸ©é˜µ
                self.Hbond = Hbond_ori.copy()
                self.Vbond = Vbond_ori.copy()

                if plot_each and self.plotter is not None:
                    self.plotter.plot_compact(
                        self.M, self.Hbond, self.Vbond,
                        labels=labels, crack_paths=paths,
                        title=f"[POOL] Grid {g+1}/{N_GRIDS}, run {f+1}/{N_FRACS}",
                        mode='fragments'
                    )

        self._maybe_log_pool_memory(tag='after_pool_call')
        return F

