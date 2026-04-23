# Core simulation framework: initialization, capacity buffers, main loop,
# time stepping, doubling control volume, basic column ops.
from __future__ import annotations

import math
import os
import time
import warnings
from typing import Optional, Sequence, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

import numpy as np

from pbe_core.base.base_solver import BaseSolver
from .fenwick import FenwickSampler


class MCPBEBase(BaseSolver):
    """Base layer for MC-PBE:
    - validates & initializes particle state with capacity buffers
    - maintains control volume & time book-keeping
    - provides main solve loop (delegates events to mixins)
    - capacity growth & control-volume doubling
    - common utilities (dt calculators, column ops)
    """

    # ---------------------------------------------------------------------
    # Construction / init
    # ---------------------------------------------------------------------
    def __init__(
        self,
        dim: int = 2,
        t_total: int = 601,
        t_write: int = 10,
        t_vec: Optional[np.ndarray] = None,
        verbose: bool = False,
        load_attr: bool = True,
        config_path: Optional[str] = None,
        init: bool = True,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        # Base solver state
        self._init_base_parameters(dim, t_total, t_write, t_vec)

        # Simulation parameters (keep names for compatibility)
        self.c = np.full(dim, 0.1e-2)
        self.x = np.full(dim, 1e-6)
        self.x2 = np.full(dim, 1e-6)
        self.a0 = 1e3
        self.CDF_method = "disc"
        self.VERBOSE = verbose

        # Initial distributions flags
        self.PGV = np.full(dim, "mono")
        self.SIG = np.full(dim, 0.1)

        # State containers
        self.V_flat: Optional[np.ndarray] = None

        # Load external configuration / physics
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir, "config", "MCPBE_config.py")
        if load_attr:
            self._load_attributes(config_path)

        # RNG (single point of instantiation)
        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        # cache for breakage CDF tables (keyed by dim, N, BREAKFVAL, pl_v, pl_q)
        self._bf_cache = {}
        
        self.mcpbe_debug = False
        # Initialize state
        if init:
            self._initialize_particles()
            self._init_lmc()
            self._initialize_samplers()
            self._bf_ready = False  # breakage CDFs (mix-in will build on demand)
                        
    def _init_lmc(self):

        self.use_lmc_pre_model = bool(getattr(self, "use_lmc_pre_model", False))
        self.lmc_pre_model = str(getattr(self, "lmc_pre_model", "table"))  # table|rank|copula|flow


        self.lmc_tables_path = getattr(self, "lmc_tables_path", None)
        self.lmc_rank_tables_path = getattr(self, "lmc_rank_tables_path", None)
        self.lmc_copula_path = getattr(self, "lmc_copula_path", None)
        self.lmc_flow_pure_path = getattr(self, "lmc_flow_pure_path", None)
        self.lmc_flow_mix_path = getattr(self, "lmc_flow_mix_path", None)


        self.lmc_A0_runtime = float(getattr(self, "lmc_A0_runtime", 1.0))
        self.lmc_interp = str(getattr(self, "lmc_interp", "bilinear"))
        self.lmc_tables_cache = bool(getattr(self, "lmc_tables_cache", False))
        self.lmc_small_particle_policy = str(
            getattr(self, "lmc_small_particle_policy", "fallback")
        )
        self.lmc_pool_dir = getattr(self, "lmc_pool_dir", "Pool_Path")


        self.lmc_STR = np.asarray(
            getattr(self, "lmc_STR", np.array([1.0, 1.0, 1.0], dtype=float)),
            dtype=float,
        )
        self.lmc_NO_FRAG = int(getattr(self, "lmc_NO_FRAG", 4))
        self.lmc_gamma = float(getattr(self, "lmc_gamma", 1.0))
        self.lmc_allow_loops = bool(getattr(self, "lmc_allow_loops", True))
        self.lmc_accept_all_cracks = bool(getattr(self, "lmc_accept_all_cracks", False))
        self.lmc_use_weighted_start = bool(getattr(self, "lmc_use_weighted_start", False))
        self.lmc_aspect_ratio = float(getattr(self, "lmc_aspect_ratio", 1.0))
        self.lmc_int_bre = float(getattr(self, "lmc_int_bre", 0.0))
        self.lmc_delta_cells = float(getattr(self, "lmc_delta_cells", 0.1))
        self.lmc_Df = float(getattr(self, "lmc_Df", 1.6))
        self.lmc_MAS = float(getattr(self, "lmc_MAS", 0.5))
        self.use_lmc_live = bool(getattr(self, "use_lmc_live", False))

        self.lmc_use_breakage_model = bool(
            getattr(self, "lmc_use_breakage_model", False)
        )
        self.lmc_breakage_model_path = getattr(
            self, "lmc_breakage_model_path", None
        )
        # E_in(V) = lambda_E * V^energy_exp
        self.lmc_lambda_E = float(getattr(self, "lmc_lambda_E", 1.0))
        self.lmc_energy_exp = float(getattr(self, "lmc_energy_exp", 1.0))

        self.lmc_rate_min = float(getattr(self, "lmc_rate_min", 0.0))
        self.lmc_rate_max = getattr(self, "lmc_rate_max", None)

        self.lmc_adapter = None        
        self.lmc_live = None             
        self.lmc_breakage_adapter = None 


        if self.use_lmc_pre_model:
            from .lmc_adapter import (
                LMCTableAdapter,
                LMCRankAdapter,
                LMCCopulaAdapter,
                LMCFlowAdapter,
            )

            pre = self.lmc_pre_model.lower().strip()
            valid = {"table", "rank", "copula", "flow"}
            if pre not in valid:
                raise ValueError(f"lmc_pre_model='{self.lmc_pre_model}' is not in {valid}")

            if pre == "table":
                if not self.lmc_tables_path:
                    raise ValueError("lmc_pre_model='table' but lmc_tables_path is not set.")
                self.lmc_adapter = LMCTableAdapter(
                    self.lmc_tables_path,
                    interp=self.lmc_interp,
                    A0_run=self.lmc_A0_runtime,
                    cache_enabled=self.lmc_tables_cache,
                )

            elif pre == "rank":
                if not self.lmc_rank_tables_path:
                    raise ValueError("lmc_pre_model='rank' but lmc_rank_tables_path is not set.")
                self.lmc_adapter = LMCRankAdapter(
                    self.lmc_rank_tables_path,
                    interp=self.lmc_interp,
                    A0_run=self.lmc_A0_runtime,
                    cache_enabled=self.lmc_tables_cache,
                )

            elif pre == "copula":
                if not self.lmc_copula_path:
                    raise ValueError("lmc_pre_model='copula' but lmc_copula_path is not set.")
                self.lmc_adapter = LMCCopulaAdapter(
                    self.lmc_copula_path,
                    interp=self.lmc_interp,
                    A0_run=self.lmc_A0_runtime,
                    cache_enabled=self.lmc_tables_cache,
                )

            elif pre == "flow":
                if not self.lmc_flow_pure_path or not self.lmc_flow_mix_path: 
                    raise ValueError("lmc_pre_model='flow' but lmc_flow_path is not set.") 
                self.lmc_adapter = LMCFlowAdapter( 
                        pure_model_path=self.lmc_flow_pure_path, 
                        mix_model_path=self.lmc_flow_mix_path, 
                        A0_run=self.lmc_A0_runtime, 
                        cache_enabled=self.lmc_tables_cache, 
                    )


            if self.lmc_adapter is not None and hasattr(self.lmc_adapter, "set_small_particle_policy"):
                self.lmc_adapter.set_small_particle_policy(
                    policy=self.lmc_small_particle_policy
                )


        if self.use_lmc_live:
            from .lmc_adapter import LMCLiveAdapter

            self.lmc_live = LMCLiveAdapter()
            self.lmc_live.configure_simulator(
                STR=self.lmc_STR,
                NO_FRAG=self.lmc_NO_FRAG,
                gamma=self.lmc_gamma,
                allow_loops=self.lmc_allow_loops,
                accept_all_cracks=self.lmc_accept_all_cracks,
                use_weighted_start=self.lmc_use_weighted_start,
                aspect_ratio=self.lmc_aspect_ratio,
                int_bre=self.lmc_int_bre,
                A0_run=self.lmc_A0_runtime,
                small_particle_policy=self.lmc_small_particle_policy,
                delta_cells=self.lmc_delta_cells,
                pool_dir=self.lmc_pool_dir,
                Df=self.lmc_Df,
                MAS=self.lmc_MAS,
                rebuild=True,
            )


        if self.lmc_use_breakage_model:
            if not self.lmc_breakage_model_path:
                raise ValueError(
                    "lmc_use_breakage_model=True but lmc_breakage_model_path is not set."
                )

            from .mlp_breakage_adapter import MLPBreakageRateAdapter

            self.lmc_breakage_adapter = MLPBreakageRateAdapter(
                model_path=self.lmc_breakage_model_path,
                lambda_E=self.lmc_lambda_E,
                energy_exp=self.lmc_energy_exp,
                gamma=self.lmc_gamma,
                NO_FRAG=self.lmc_NO_FRAG,
                int_bre=self.lmc_int_bre,
                Df=self.lmc_Df,
                MAS=self.lmc_MAS,
                rate_min=self.lmc_rate_min,
                rate_max=self.lmc_rate_max,
                A0_run=self.lmc_A0_runtime,
            )

    
    # ---------------------------------------------------------------------
    # Validation & helpers
    # ---------------------------------------------------------------------
    def _validate_input_arrays(self):
        dim = self.dim

        def _len(name: str) -> int:
            v = getattr(self, name, None)
            try:
                return len(v)
            except Exception:
                return -1

        for name in ("c", "x", "PGV", "SIG"):
            L = _len(name)
            if L != dim:
                raise ValueError(
                    f"`{name}` must be a 1D array (sequence) of length dim={dim}, got length {L}."
                )

    def _growth_factor(self) -> float:
        """Dynamic capacity growth factor in [1.1, 2.0], more aggressive early in time."""
        T = float(getattr(self, "t_total", 1.0))
        t = float(getattr(self, "_elapsed", 0.0))
        r = 1.0 - min(max(t / max(T, 1e-12), 0.0), 1.0)
        f = 1.1 + 0.9 * r  # in [1.1, 2.0]
        return float(min(2.0, max(1.1, f)))

    def _compute_frag_num(self):
        """Expected number of fragments per break event from BREAKFVAL & v."""
        v = float(getattr(self, "pl_v", 1.0))
        bf = int(getattr(self, "BREAKFVAL", 1))
        if self.dim == 1:
            if bf == 1:
                p = 4.0
            elif bf == 2:
                p = 2.0
            elif bf == 3:
                p = v
            elif bf == 4:
                p = (v + 1.0) / max(v, 1e-12)
            elif bf == 5:
                p = (v + 2.0) / max(v, 1e-12)
            else:
                raise ValueError(f"Unsupported BREAKFVAL={bf} for 1D.")
        else:
            if bf == 1:
                p = 4.0
            elif bf == 2:
                p = 2.0
            elif bf == 3:
                raise ValueError("BREAKFVAL=3 (product function) not implemented for 2D!")
            elif bf == 4:
                p = (v + 1.0) / max(v, 1e-12)
            elif bf == 5:
                p = (2.0 * v + 1.0) * (v + 2.0) / (2.0 * max(v, 1e-12) * (v + 1.0))
            else:
                raise ValueError(f"Unsupported BREAKFVAL={bf} for 2D.")
        if p <= 1.0:
            raise ValueError(
                f"Expected number of fragments p={p:.3f} must be > 1. Check BREAKFVAL/pl_v."
            )
        self.frag_num = float(p)

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    def _initialize_particles(self, init_Vc: bool = True, V_flat: Optional[np.ndarray] = None):
        """Build initial V_flat (capacity style), X, and save initial snapshots."""
        dim = self.dim
        self._validate_input_arrays()

        if init_Vc:
            self.c = np.asarray(self.c, dtype=float)
            self.x = np.asarray(self.x, dtype=float)
            self.PGV = np.asarray(self.PGV)
            self.SIG = np.asarray(self.SIG, dtype=float)
            self.v = (self.x ** 3) * math.pi / 6.0
            self.n = np.round(self.c / self.v)
            self.n0 = float(np.sum(self.n))
            if self.n0 <= 0:
                raise ValueError("Total primary particle count `n0` must be > 0 (check c and x).")
            self.Vc = self.a0 / self.n0
            self.a = np.round(self.n * self.Vc).astype(int)
            total_cols = int(np.sum(self.a))
            if total_cols <= 0:
                raise ValueError("No particles to initialize (sum(a) == 0). Check c/x/PGV/SIG.")

        if V_flat is None:
            V_init = np.zeros((dim + 1, total_cols), dtype=float)
            cnt = 0
            for i in range(dim):
                ai = int(self.a[i])
                if ai <= 0:
                    continue
                p = str(self.PGV[i])
                if p == "mono":
                    V_init[i, cnt : cnt + ai] = self.v[i]
                elif p == "norm":
                    mu = self.v[i]
                    sig = float(self.SIG[i]) * mu
                    V_init[i, cnt : cnt + ai] = self._rng.normal(mu, sig, ai)
                elif p == "weibull":
                    V_init[i, cnt : cnt + ai] = self._rng.weibull(2.0, ai) * (
                        self.SIG[i] * self.v[i]
                    )
                else:
                    raise ValueError(f"Unsupported PGV[{i}]='{p}'. Use 'mono' | 'norm' | 'weibull'.")
                cnt += ai
            # total volume row & filter invalid columns
            V_init[-1, :] = np.sum(V_init[:dim, :], axis=0)
            keep = V_init[-1, :] > 0.0
            V_init = V_init[:, keep]
        else:
            V_init = V_flat

        a0_eff = V_init.shape[1]
        if a0_eff <= 0:
            raise ValueError("No particles initialized after filtering non-positive volumes.")

        # Capacity buffers (>= active + ~10%)
        cap = max(a0_eff + max(8, a0_eff // 10), 16)
        self._cap = int(cap)
        self.V_flat = np.zeros((dim + 1, self._cap), dtype=float)
        self.V_flat[:, :a0_eff] = V_init
        self.a_tot = a0_eff

        self.X = np.zeros(self._cap, dtype=float)
        self.X[:a0_eff] = self._vol2diam(self.V_flat[-1, :a0_eff])

        # Time & saved snapshots
        self.t = [0.0]
        if self.t_vec is None:
            steps = max(1, int(self.t_total // max(1, self.t_write)))
            self.t_vec = np.linspace(0.0, float(self.t_total), steps + 1)

        # Expected fragment number for breakage
        self._compute_frag_num()

        # Save initial state (only active slice)
        self.V0 = self.V_flat[:, :self.a_tot].copy()
        self.X0 = self.X[:self.a_tot].copy()
        self.V0_save = [self.V0.copy()]
        self.V_save = [self.V_flat[:, :self.a_tot].copy()]
        self.Vc_save = [float(self.Vc)]
        self.step = 1
        # initialize left/right snapshot containers for post-processing
        # right snapshots remain in self.V_save / self.Vc_save as before
        self.V_save_left = [self.V_flat[:, :self.a_tot].copy()]
        self.t_left = [0.0]
        self.t_right = [0.0]

    def _initialize_samplers(self):
        """Build (or resize) samplers for agglomeration/breakage based on process_type."""
        pt = getattr(self, "process_type", "agglomeration")

        # Agglomeration
        if pt in ("agglomeration", "mix"):
            self._rebuild_all_propensities()  # from AgglomerationMixin
            if not hasattr(self, "_r_agg") or self._r_agg is None or self._r_agg.shape[0] < self._cap:
                buf = np.zeros(self._cap, dtype=float)
                if hasattr(self, "_r_agg") and self._r_agg is not None:
                    buf[:self.a_tot] = self._r_agg[:self.a_tot]
                self._r_agg = buf
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        else:
            self._r_agg = np.zeros(self._cap, dtype=float)
            self._agg_sampler = None

        # Breakage
        if pt in ("breakage", "mix"):
            self._calc_break_rates_full()  # from BreakageMixin
            if (
                not hasattr(self, "_break_rate")
                or self._break_rate is None
                or self._break_rate.shape[0] < self._cap
            ):
                br = np.zeros(self._cap, dtype=float)
                if hasattr(self, "_break_rate") and self._break_rate is not None:
                    br[:self.a_tot] = self._break_rate[:self.a_tot]
                self._break_rate = br
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
        else:
            self._break_rate = np.zeros(self._cap, dtype=float)
            self._break_sampler = None

    # ---------------------------------------------------------------------
    # Capacity management
    # ---------------------------------------------------------------------
    def _ensure_capacity_for(self, extra: int):
        """Ensure capacity for `a_tot + extra` active columns. Grow with factor in [1.1, 2.0]."""
        need = self.a_tot + int(extra)
        if self._cap >= need:
            return
        old_cap = self._cap
        factor = self._growth_factor()
        new_cap = int(max(math.ceil(old_cap * factor), need))

        V_new = np.zeros((self.dim + 1, new_cap), dtype=float)
        X_new = np.zeros(new_cap, dtype=float)
        V_new[:, :self.a_tot] = self.V_flat[:, :self.a_tot]
        X_new[:self.a_tot] = self.X[:self.a_tot]
        self.V_flat = V_new
        self.X = X_new
        self._cap = new_cap

        # Extend auxiliary arrays if present
        if hasattr(self, "_r_agg") and self._r_agg is not None:
            r_new = np.zeros(new_cap, dtype=float)
            r_new[:self.a_tot] = self._r_agg[:self.a_tot]
            self._r_agg = r_new
        if hasattr(self, "_break_rate") and self._break_rate is not None:
            b_new = np.zeros(new_cap, dtype=float)
            b_new[:self.a_tot] = self._break_rate[:self.a_tot]
            self._break_rate = b_new

        # Print expansion info
        if self.VERBOSE:   
            print(
                f"[MC-PBE] Capacity grown at t={getattr(self,'_elapsed',0.0):.6g} "
                f"after {getattr(self,'_iter_count',0)} events: cap {old_cap} -> {new_cap} "
                f"(x{new_cap/max(old_cap,1):.2f})"
                # f"[TEST] dt_break = {self.test_dt_break}"
            )

    def _maybe_double_control_volume(self, elapsed_time: float, iter_count: int):
        """Duplicate state to keep statistics when particle count drops (agglomeration dominates)."""
        if getattr(self, "process_type", "agglomeration") not in ("agglomeration", "mix"):
            return
        if getattr(self, "a0", 0) <= 0:
            return
        if self.a_tot > self.a0 / 2:
            return

        old_a = self.a_tot
        old_Vc = float(self.Vc)

        # Double control volume and duplicate active slice
        self.Vc *= 2.0
        V_active = self.V_flat[:, :self.a_tot]
        X_active = self.X[:self.a_tot]
        V_dup = np.concatenate((V_active, V_active), axis=1)
        X_dup = np.concatenate((X_active, X_active))
        self.a_tot = V_dup.shape[1]

        # Ensure capacity and write back
        if self._cap < self.a_tot:
            self._cap = int(self.a_tot * 1.2) + 8
            V_new = np.zeros((self.dim + 1, self._cap), dtype=float)
            X_new = np.zeros(self._cap, dtype=float)
            V_new[:, :self.a_tot] = V_dup
            X_new[:self.a_tot] = X_dup
            self.V_flat = V_new
            self.X = X_new
        else:
            self.V_flat[:, :self.a_tot] = V_dup
            self.X[:self.a_tot] = X_dup

        if hasattr(self, "V0") and isinstance(self.V0, np.ndarray):
            self.V0 = np.concatenate((self.V0, self.V0), axis=1)

        # Rebuild samplers from active slices
        if getattr(self, "process_type", "agglomeration") in ("agglomeration", "mix"):
            self._rebuild_all_propensities()
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        if getattr(self, "process_type", "agglomeration") in ("breakage", "mix"):
            self._calc_break_rates_full()
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
        if self.VERBOSE:    
            print(
                f"[MC-PBE] Control volume doubled at t={elapsed_time:.6g} after {iter_count} events: "
                f"a_tot {old_a} -> {self.a_tot}, Vc {old_Vc:.6g} -> {self.Vc:.6g}"
            )

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _vol2diam(self, V: np.ndarray) -> np.ndarray:
        return (6.0 * V / math.pi) ** (1.0 / 3.0)

    def _dt_agg(self) -> float:
        """Agglomeration time-step Î”t using current r_agg (active slice only)."""
        a = self.a_tot
        if a < 2:
            return float("inf")
        sum_r = float(np.sum(self._r_agg[:a]))
        if sum_r <= 0.0:
            return float("inf")
        return 2.0 * float(self.Vc) * (a - 1) / (a * sum_r)

    def _dt_break(self) -> float:
        """Breakage Î”t with mean break rate (active slice only)."""
        a = self.a_tot
        if a <= 0:
            return float("inf")
        s = float(np.mean(self._break_rate[:a])) if a > 0 else 0.0
        if s <= 0.0:
            return float("inf")
        self.test_dt_break = 1.0 / (a * s)
        return 1.0 / (a * s)

    # ---------------------------------------------------------------------
    # Main solve loop
    # ---------------------------------------------------------------------
    def solve(self, maxiter: int = int(1e8)):
        t0 = time.time()
        count = 0

        pt = getattr(self, "process_type", "agglomeration")
        timer_agg = 0.0
        timer_break = 0.0
        dtd_agg = self._dt_agg() if pt in ("agglomeration", "mix") else float("inf")
        dtd_break = self._dt_break() if pt in ("breakage", "mix") else float("inf")
        timer_agg += dtd_agg
        timer_break += dtd_break

        if self.VERBOSE:
            if np.isfinite(dtd_agg):
                print(f"Initial dt_agg = {dtd_agg:.3e} s")
            if np.isfinite(dtd_break):
                print(f"Initial dt_break = {dtd_break:.3e} s")
                
        if self.mcpbe_debug:
            self._check_state_before_solve()
            self._log_debug_config()

        next_save_idx = 1 if len(self.t_vec) > 1 else 0
        self._elapsed = 0.0
        self._iter_count = 0

        cancel_flag = getattr(self, "cancel_flag", None)
        while self.t[-1] <= float(self.t_vec[-1]) and count < maxiter:
            if cancel_flag is not None and cancel_flag.get("cancel", False):
                break
            # keep context for logging/expansion
            self._elapsed = self.t[-1]
            self._iter_count = count
            
            # cache "left" state: state after previous event
            t_prev = self.t[-1]
            V_prev_active = self.V_flat[:, :self.a_tot].copy()

            if pt == "agglomeration":
                self._do_one_agg()  # from AgglomerationMixin
                elapsed_time = timer_agg
                dtd_agg = self._dt_agg()
                timer_agg += dtd_agg
            elif pt == "breakage":
                self._do_one_break()  # from BreakageMixin
                elapsed_time = timer_break
                dtd_break = self._dt_break()
                timer_break += dtd_break
            else:  # mix
                if timer_agg <= timer_break:
                    self._do_one_agg()
                    elapsed_time = timer_agg
                    dtd_agg = self._dt_agg()
                    timer_agg += dtd_agg
                else:
                    self._do_one_break()
                    elapsed_time = timer_break
                    dtd_break = self._dt_break()
                    timer_break += dtd_break

            self.t.append(elapsed_time)
            
            # current "right" state after this event
            V_right_active = self.V_flat[:, :self.a_tot]

            # Save snapshots at requested times (active slice only)
            while next_save_idx < len(self.t_vec) and elapsed_time >= self.t_vec[next_save_idx]:
                # right snapshots: same behavior as original code
                self.V_save.append(V_right_active.copy())
                self.Vc_save.append(float(self.Vc))
                self.step += 1

                # left/right metadata for this time point
                self.V_save_left.append(V_prev_active.copy())
                self.t_left.append(t_prev)
                self.t_right.append(elapsed_time)

                next_save_idx += 1

            # agglomeration-dominated safety (duplicate CV)
            self._maybe_double_control_volume(self.t[-1], count)

            count += 1
            # if count%100 == 0: print([f"[Test] events = {count}"])
            if self.a_tot < 2 and pt in ("agglomeration", "mix"):
                break
        if self.use_lmc_live:
            self.lmc_live._sim.agg_pool.close_pool_cache()
        self.MACHINE_TIME = time.time() - t0
        if self.VERBOSE:
            print(f"[MC-PBE] The calculation took {getattr(self,'MACHINE_TIME',0.0):.4g}s after {count} events")
        return self
    def solve_repeats(
        self,
        N: int = 5,
        base_seed: int = 42,
        seeds: Optional[Sequence[int]] = None,
        maxiter: int = int(1e8),
        init_Vc: bool = True,
        Vc: float = None,
        V_flat: Optional[np.ndarray] = None,
        workers: int = 1,
        psd_enable: bool = False,
        psd_basis: str = "volume",                 # "volume" or "number"
        psd_x_grid: Optional[np.ndarray] = None,   # if given -> output Q(x)
        psd_Q_grid: Optional[np.ndarray] = None,   # if given -> output x(Q)
    ):
        """
        Run N Monte Carlo realizations (repeats).

        workers = 1  -> serial (original behavior + optional PSD computation)
        workers > 1  -> parallel with ProcessPoolExecutor (PSD currently unsupported)

        Returns
        -------
        If psd_enable == False:
            List[{"seed_info", "t_vec", "moments"}]

        If psd_enable == True and workers == 1:
            (results, psd_info)  # tuple

            results: list of dicts as above

            If psd_x_grid is used (Q(x) mode):
                psd_info = {
                    "mode": "Q_of_x",
                    "basis": "volume" or "number",
                    "t_vec": t_vec_reference,    # shape (T,)
                    "x_50": x_50_mean,           # shape (T,)
                    "x_grid": x_grid,            # shape (M,)
                    "Q_mean": Q_mean,            # shape (T, M)
                    "note": "...",
                }

            If psd_Q_grid is used (x(Q) mode):
                psd_info = {
                    "mode": "x_of_Q",
                    "basis": "volume" or "number",
                    "t_vec": t_vec_reference,    # shape (T,)
                    "x_50": x_50_mean,           # shape (T,)
                    "Q_grid": Q_grid,            # shape (M,)
                    "x_mean": x_mean,            # shape (T, M)
                    "note": "...",
                }
        """
        # ----- build seeds -----
        if seeds is None:
            master = np.random.SeedSequence(base_seed)
            seeds = master.spawn(N)
        if len(seeds) != N:
            raise ValueError("Length of seeds must equal N.")

        # warn if PSD grids are given but PSD is disabled
        if not psd_enable and (psd_x_grid is not None or psd_Q_grid is not None):
            warnings.warn(
                "psd_enable=False but psd_x_grid/psd_Q_grid are provided; PSD computation will be skipped.",
                RuntimeWarning,
            )

        if psd_enable and workers > 1:
            raise NotImplementedError(
                "psd_enable=True is currently only supported for workers=1 (serial mode). "
            )

        # ----- serial path (supports PSD) -----
        if workers == 1:
            results: list[dict[str, Any]] = []
            cancel_flag = getattr(self, "cancel_flag", None)

            # For PSD aggregation across repeats
            cdf_repeats: list[Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]] = []
            t_vec_ref: Optional[np.ndarray] = None

            for k in range(N):
                if cancel_flag is not None and cancel_flag.get("cancel", False):
                    break
                # Deep copy self and run a single realization
                m = copy.deepcopy(self)
                if cancel_flag is not None:
                    m.cancel_flag = cancel_flag
                sk = seeds[k]
                if isinstance(sk, np.random.SeedSequence):
                    rng = np.random.default_rng(sk)
                    seed_info = {"spawn_key": tuple(sk.spawn_key)}
                else:
                    rng = np.random.default_rng(int(sk))
                    seed_info = {"seed": int(sk)}
                m._rng = rng
                m.V_flat = None
                if not init_Vc and Vc is not None:
                    m.Vc = Vc
                m._initialize_particles(init_Vc=init_Vc, V_flat=V_flat)
                m._init_lmc()
                m._initialize_samplers()
                m.solve(maxiter=maxiter)
                mu, tv = m.calc_moments_over_time(normalize=True)
                results.append({"seed_info": seed_info, "t_vec": tv, "moments": mu})

                # PSD CDFs for this realization over all saved times
                if psd_enable:
                    cdf_list, t_vec_local = m.compute_psd_cdf_over_time(psd_basis=psd_basis,
                                                                        time_scheme="interp")
                    if t_vec_ref is None:
                        t_vec_ref = np.asarray(t_vec_local, dtype=float)
                    else:
                        if len(t_vec_ref) != len(t_vec_local) or not np.allclose(
                            t_vec_ref, t_vec_local, rtol=1e-6, atol=1e-12
                        ):
                            warnings.warn(
                                "t_vec differs between repeats. PSD averaging assumes identical t_vec; "
                                "results may be inconsistent.",
                                RuntimeWarning,
                            )
                    cdf_repeats.append(cdf_list)

            if not psd_enable:
                # original behavior: only moments
                return results, None

            # Aggregate PSD over repeats using post-processing utilities
            if t_vec_ref is None:
                # No PSD data collected
                psd_info = {
                    "basis": psd_basis,
                    "mode": None,
                    "t_vec": None,
                    "x_grid": None,
                    "Q_mean": None,
                    "Q_grid": None,
                    "x_mean": None,
                    "x_50":   None,
                    "note": "No PSD snapshots were available.",
                }
            else:
                # `self` is a MCPBESolver (MCPBEPost is in MRO), so we can call aggregate_psd_repeats
                psd_info = self.aggregate_psd_repeats(
                    cdf_repeats=cdf_repeats,
                    t_vec=t_vec_ref,
                    psd_basis=psd_basis,
                    psd_x_grid=psd_x_grid,
                    psd_Q_grid=psd_Q_grid,
                )

            return results, psd_info

        # ----- parallel path (PSD not implemented here) -----
        base_state = copy.deepcopy(self.__dict__)

        payloads = []
        for k in range(N):
            payloads.append(
                {
                    "cls": self.__class__,
                    "state": base_state,
                    "seed": seeds[k],
                    "maxiter": maxiter,
                    "init_Vc": init_Vc,
                    "Vc": Vc,
                    "V_flat": V_flat,
                }
            )

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_mcpbe_run_single_parallel, pl): i for i, pl in enumerate(payloads)}
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    raise RuntimeError(f"[parallel] worker {idx} failed: {e}")
                else:
                    results.append(res)

        if psd_enable:
            warnings.warn(
                "psd_enable=True but workers>1: PSD computation on parallel path is not implemented; "
                "only moments are returned.",
                RuntimeWarning,
            )

        return results, None


    # ---------------------------------------------------------------------
    # Column ops (capacity style)
    # ---------------------------------------------------------------------
    def _remove_particle_column(self, j: int):
        """Remove particle at index j using swap-with-last, maintain samplers incrementally (B2)."""
        a = self.a_tot
        if j < 0 or j >= a:
            raise IndexError("column index out of range")
    
        if a <= 1:
            self.a_tot = max(0, a - 1)
            if a == 1:
                self.V_flat[:, 0:1] = 0.0
                self.X[0:1] = 0.0
                if hasattr(self, "_r_agg") and self._r_agg is not None:
                    self._r_agg[0:1] = 0.0
                if hasattr(self, "_break_rate") and self._break_rate is not None:
                    self._break_rate[0:1] = 0.0
            if self._agg_sampler is not None:
                self._agg_sampler = FenwickSampler(np.zeros(0))
            if self._break_sampler is not None:
                self._break_sampler = FenwickSampler(np.zeros(0))
            return
    
        last = a - 1
    
        # swap particle state arrays so that the 'removed' particle moves to last
        if j != last:
            self.V_flat[:, [j, last]] = self.V_flat[:, [last, j]]
            self.X[j], self.X[last] = self.X[last], self.X[j]
    
            if hasattr(self, "_r_agg") and self._r_agg is not None:
                self._r_agg[j], self._r_agg[last] = self._r_agg[last], self._r_agg[j]
            if hasattr(self, "_break_rate") and self._break_rate is not None:
                self._break_rate[j], self._break_rate[last] = self._break_rate[last], self._break_rate[j]
    
        # sampler remove must happen while sampler still has size == a
        if self._agg_sampler is not None:
            self._agg_sampler.remove_swap_last(j)
        if self._break_sampler is not None:
            self._break_sampler.remove_swap_last(j)
    
        # shrink
        self.a_tot = last
    
        # zero freed slot
        self.V_flat[:, self.a_tot : self.a_tot + 1] = 0.0
        self.X[self.a_tot : self.a_tot + 1] = 0.0
        if hasattr(self, "_r_agg") and self._r_agg is not None:
            self._r_agg[self.a_tot : self.a_tot + 1] = 0.0
        if hasattr(self, "_break_rate") and self._break_rate is not None:
            self._break_rate[self.a_tot : self.a_tot + 1] = 0.0
    


    def _append_particle_column(self, frag_vols: np.ndarray):
        """Append one particle from per-component volumes; maintain samplers incrementally."""
        frag_vols = np.asarray(frag_vols, dtype=float)
        if frag_vols.shape != (self.dim,):
            raise ValueError("frag_vols must have shape (dim,)")
    
        self._ensure_capacity_for(1)
        Vnew = float(np.sum(frag_vols))
    
        idx = self.a_tot
        self.V_flat[: self.dim, idx] = frag_vols
        self.V_flat[-1, idx] = Vnew
        self.X[idx] = float(self._vol2diam(Vnew))
        self.a_tot += 1
    
        # Always initialize auxiliary arrays at idx to 0 to avoid stale values
        if hasattr(self, "_r_agg") and self._r_agg is not None:
            self._r_agg[idx] = 0.0
        if hasattr(self, "_break_rate") and self._break_rate is not None:
            self._break_rate[idx] = 0.0
    
        # Extend samplers with 0.0; caller will update() to the true value ASAP
        if self._agg_sampler is not None:
            self._agg_sampler.append(0.0)
        if self._break_sampler is not None:
            self._break_sampler.append(0.0)


    def _ensure_break_sampler(self):
        """(Re)build break sampler from active slice if needed."""
        if not hasattr(self, "_break_rate"):
            return
        if self._break_sampler is None:
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
    
    def _close(self, gc_clean=True):
        big_attrs = ("V_flat", "X", "V0", "X0",
             "V0_save", "V_save", "Vc_save",
             "_r_agg", "_break_rate",
             "_agg_sampler", "_break_sampler")
        for name in big_attrs:
            setattr(self, name, None)
        self._bf_cache.clear()
        self.lmc_live = None
        if gc_clean:
            import gc
            gc.collect()
            
    def _check_state_before_solve(self):
        """Light-weight sanity checks before entering the main solve loop.

        This is only called when `mcpbe_debug` is True. It is meant to catch
        obvious configuration/state issues early, with minimal overhead.
        """
        # --- dimension & basic attributes ---
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError(f"[MC-PBE][DEBUG] `dim` must be a positive integer, got {self.dim!r}.")

        # time grid checks
        if self.t_vec is None:
            raise ValueError("[MC-PBE][DEBUG] `t_vec` is None; time grid must be initialized.")
        tv = np.asarray(self.t_vec, dtype=float)
        if tv.ndim != 1 or tv.size == 0:
            raise ValueError("[MC-PBE][DEBUG] `t_vec` must be a non-empty 1D array.")
        if not np.all(np.diff(tv) > 0):
            raise ValueError("[MC-PBE][DEBUG] `t_vec` must be strictly increasing.")
        if abs(float(tv[-1]) - float(self.t_total)) > 1e-8:
            warnings.warn(
                f"[MC-PBE][DEBUG] t_vec[-1]={tv[-1]:.6g} differs from t_total={float(self.t_total):.6g}.",
                RuntimeWarning,
            )

        # validate input arrays (c, x, PGV, SIG) against dim
        try:
            self._validate_input_arrays()
        except Exception as exc:
            raise ValueError(f"[MC-PBE][DEBUG] Input arrays invalid: {exc}") from exc

        # process_type consistency
        pt = str(getattr(self, "process_type", "agglomeration")).lower()
        if pt not in ("agglomeration", "breakage", "mix"):
            raise ValueError(
                f"[MC-PBE][DEBUG] Unsupported process_type={pt!r}. "
                "Use 'agglomeration' | 'breakage' | 'mix'."
            )

        # state containers: V_flat, X, a_tot, capacity
        if self.V_flat is None or not isinstance(self.V_flat, np.ndarray):
            raise ValueError(
                "[MC-PBE][DEBUG] `V_flat` is not initialized. "
                "Make sure `_initialize_particles` has been called."
            )
        if self.V_flat.shape[0] != self.dim + 1:
            raise ValueError(
                f"[MC-PBE][DEBUG] `V_flat` must have shape (dim+1, cap); "
                f"got {self.V_flat.shape}, dim={self.dim}."
            )
        if not hasattr(self, "X") or self.X is None:
            raise ValueError(
                "[MC-PBE][DEBUG] `X` (diameter array) is not initialized. "
                "Make sure `_initialize_particles` has been called."
            )
        if not hasattr(self, "a_tot"):
            raise ValueError("[MC-PBE][DEBUG] `a_tot` is missing on solver instance.")
        if not hasattr(self, "_cap"):
            raise ValueError("[MC-PBE][DEBUG] `_cap` (capacity) is missing on solver instance.")
        if self.a_tot < 0 or self.a_tot > self._cap:
            raise ValueError(
                f"[MC-PBE][DEBUG] Inconsistent a_tot={self.a_tot}, cap={self._cap}."
            )

        # sampler presence (only sanity check; they may be rebuilt during solve)
        if pt in ("agglomeration", "mix"):
            if not hasattr(self, "_agg_sampler") or self._agg_sampler is None:
                warnings.warn(
                    "[MC-PBE][DEBUG] Agglomeration enabled but `_agg_sampler` is None. "
                    "It will be rebuilt, but this may indicate that `_initialize_samplers` "
                    "was not called explicitly.",
                    RuntimeWarning,
                )
        if pt in ("breakage", "mix"):
            if not hasattr(self, "_break_sampler") or self._break_sampler is None:
                warnings.warn(
                    "[MC-PBE][DEBUG] Breakage enabled but `_break_sampler` is None. "
                    "It will be rebuilt, but this may indicate that `_initialize_samplers` "
                    "was not called explicitly.",
                    RuntimeWarning,
                )

        # LMC configuration sanity
        use_lmc_pre = bool(getattr(self, "use_lmc_pre_model", False))
        if use_lmc_pre and getattr(self, "lmc_adapter", None) is None:
            warnings.warn(
                "[MC-PBE][DEBUG] use_lmc_pre_model=True but `lmc_adapter` is None. "
                "Check LMC table/rank/copula/flow paths in config.",
                RuntimeWarning,
            )

    
    def _log_debug_config(self):
        """Print a categorized snapshot of key MCPBE configuration parameters.

        Categories:
          - General parameters
          - Agglomeration parameters
          - Breakage & LMC parameters
        """
        print("\n[MC-PBE][DEBUG] Configuration snapshot")

        # -------------------------
        # General parameters
        # -------------------------
        print("  [General parameters]")
        print(f"    dim          = {getattr(self, 'dim', None)}")
        print(f"    t_total      = {getattr(self, 't_total', None)}")
        print(f"    t_write      = {getattr(self, 't_write', None)}")

        tv = np.asarray(getattr(self, "t_vec", []), dtype=float)
        if tv.size > 0:
            print(
                f"    t_vec        = len={tv.size}, "
                f"first={tv[0]:.6g}, last={tv[-1]:.6g}"
            )
        else:
            print("    t_vec        = <empty or None>")

        print(f"    a0           = {getattr(self, 'a0', None)}")
        print(f"    c            = {getattr(self, 'c', None)}")
        print(f"    x            = {getattr(self, 'x', None)}")
        print(f"    Vc           = {getattr(self, 'Vc', None)}")
        print(f"    PGV          = {getattr(self, 'PGV', None)}")
        print(f"    SIG          = {getattr(self, 'SIG', None)}")
        print(f"    VERBOSE      = {getattr(self, 'VERBOSE', None)}")
        print(f"    process_type = {getattr(self, 'process_type', None)}")
        print(f"    CDF_method   = {getattr(self, 'CDF_method', None)}")
        print(f"    USE_PSD      = {getattr(self, 'USE_PSD', None)}")
        print(f"    DIST1_path   = {getattr(self, 'DIST1_path', None)}")
        print(f"    DIST1_name   = {getattr(self, 'DIST1_name', None)}")
        print(f"    DIST3_path   = {getattr(self, 'DIST3_path', None)}")
        print(f"    DIST3_name   = {getattr(self, 'DIST3_name', None)}")

        # -------------------------
        # Agglomeration parameters
        # -------------------------
        print("  [Agglomeration parameters]")
        print(f"    COLEVAL      = {getattr(self, 'COLEVAL', None)}")
        print(f"    SIZEEVAL     = {getattr(self, 'SIZEEVAL', None)}")
        print(f"    CORR_BETA    = {getattr(self, 'CORR_BETA', None)}")
        print(f"    alpha_prim   = {getattr(self, 'alpha_prim', None)}")
        print(f"    G (shear)    = {getattr(self, 'G', None)}")

        # If current state already has propensities, log basic stats
        if hasattr(self, "_r_agg") and isinstance(self._r_agg, np.ndarray):
            r_active = self._r_agg[: getattr(self, "a_tot", 0)]
            if r_active.size > 0:
                print(
                    "    r_agg       = active size={}, min={:.3e}, max={:.3e}, mean={:.3e}".format(
                        r_active.size,
                        float(np.min(r_active)),
                        float(np.max(r_active)),
                        float(np.mean(r_active)),
                    )
                )
            else:
                print("    r_agg       = <no active entries>")
        else:
            print("    r_agg       = <not initialized>")

        # -------------------------
        # Breakage & LMC parameters
        # -------------------------
        print("  [Breakage & LMC parameters]")
        print(f"    BREAKRVAL    = {getattr(self, 'BREAKRVAL', None)}")
        print(f"    BREAKFVAL    = {getattr(self, 'BREAKFVAL', None)}")
        print(f"    pl_v         = {getattr(self, 'pl_v', None)}")
        print(f"    pl_P1        = {getattr(self, 'pl_P1', None)}")
        print(f"    pl_P2        = {getattr(self, 'pl_P2', None)}")
        print(f"    pl_P3        = {getattr(self, 'pl_P3', None)}")
        print(f"    pl_P4        = {getattr(self, 'pl_P4', None)}")
        if hasattr(self, "frag_num"):
            print(f"    frag_num     = {getattr(self, 'frag_num', None)}")

        if hasattr(self, "_break_rate") and isinstance(self._break_rate, np.ndarray):
            br_active = self._break_rate[: getattr(self, "a_tot", 0)]
            if br_active.size > 0:
                print(
                    "    break_rate  = active size={}, min={:.3e}, max={:.3e}, mean={:.3e}".format(
                        br_active.size,
                        float(np.min(br_active)),
                        float(np.max(br_active)),
                        float(np.mean(br_active)),
                    )
                )
            else:
                print("    break_rate  = <no active entries>")
        else:
            print("    break_rate  = <not initialized>")

        # LMC-related configuration
        print(f"    CDF_method   = {getattr(self, 'CDF_method', None)}")
        print(f"    use_lmc_pre_model  = {getattr(self, 'use_lmc_pre_model', None)}")
        print(f"    lmc_pre_model      = {getattr(self, 'lmc_pre_model', None)}")
        print(f"    lmc_tables_path    = {getattr(self, 'lmc_tables_path', None)}")
        print(f"    lmc_rank_tables_path = {getattr(self, 'lmc_rank_tables_path', None)}")
        print(f"    lmc_copula_path    = {getattr(self, 'lmc_copula_path', None)}")
        print(f"    lmc_flow_pure_path = {getattr(self, 'lmc_flow_pure_path', None)}")
        print(f"    lmc_flow_mix_path  = {getattr(self, 'lmc_flow_mix_path', None)}")
        print(f"    lmc_A0_runtime     = {getattr(self, 'lmc_A0_runtime', None)}")
        print(f"    lmc_interp         = {getattr(self, 'lmc_interp', None)}")
        print(f"    lmc_tables_cache   = {getattr(self, 'lmc_tables_cache', None)}")
        print(f"    use_lmc_live       = {getattr(self, 'use_lmc_live', None)}")
        print(f"    lmc_small_particle_policy = {getattr(self, 'lmc_small_particle_policy', None)}")
        print(f"    lmc_pool_dir       = {getattr(self, 'lmc_pool_dir', None)}")
        print(f"    lmc_Df             = {getattr(self, 'lmc_Df', None)}")
        print(f"    lmc_MAS            = {getattr(self, 'lmc_MAS', None)}")

        # LMC adapter/live presence
        adapter = getattr(self, "lmc_adapter", None)
        live = getattr(self, "lmc_live", None)
        print(f"    lmc_adapter        = {type(adapter).__name__ if adapter is not None else None}")
        print(f"    lmc_live           = {type(live).__name__ if live is not None else None}")
        print("[MC-PBE][DEBUG] End of configuration snapshot\n")


def _mcpbe_run_single_parallel(payload: dict):
    """
    Top-level worker for running a single MCPBE repeat in a subprocess.

    payload keys:
        - "cls": the class object (e.g. MCPBEBase)
        - "state": dict copied from solver.__dict__ (picklable)
        - "seed": np.random.SeedSequence or int
        - "maxiter", "init_Vc", "Vc", "V_flat"
    """
    cls = payload["cls"]
    state = payload["state"]
    seed_k = payload["seed"]
    maxiter = payload["maxiter"]
    init_Vc = payload["init_Vc"]
    Vc = payload["Vc"]
    V_flat = payload["V_flat"]

    # 1) rebuild solver skeleton
    # we create an instance with minimal init (init=False) and then restore dict
    dim = int(state.get("dim", 2))
    obj = cls(dim=dim, init=False)

    # 2) restore all attributes (what we saved in parent)
    # this is shallow here because it was deepcopied in parent
    obj.__dict__.update(state)

    # 3) re-seed RNG
    if isinstance(seed_k, np.random.SeedSequence):
        rng = np.random.default_rng(seed_k)
        seed_info = {"spawn_key": tuple(seed_k.spawn_key)}
    else:
        rng = np.random.default_rng(int(seed_k))
        seed_info = {"seed": int(seed_k)}
    obj._rng = rng

    # 4) re-init LMC adapter (some adapters open files, so do it in child)
    obj._init_lmc()

    # 5) re-init particles / samplers
    if not init_Vc and Vc is not None:
        obj.Vc = Vc
    obj._initialize_particles(init_Vc=init_Vc, V_flat=V_flat)
    obj._initialize_samplers()

    # 6) run solve
    obj.solve(maxiter=maxiter)

    # 7) collect moments
    mu, tv = obj.calc_moments_over_time(normalize=True)
    return {"seed_info": seed_info, "t_vec": tv, "moments": mu}
