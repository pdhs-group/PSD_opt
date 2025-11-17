# Core simulation framework: initialization, capacity buffers, main loop,
# time stepping, doubling control volume, basic column ops.
from __future__ import annotations

import math
import os
import time
import warnings
from typing import Optional, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

import numpy as np

from optframework.base.base_solver import BaseSolver
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

        # Initialize state
        if init:
            self._initialize_particles()
            self._initialize_samplers()
            self._bf_ready = False  # breakage CDFs (mix-in will build on demand)

        # cache for breakage CDF tables (keyed by dim, N, BREAKFVAL, pl_v, pl_q)
        self._bf_cache = {}

        self._init_lmc()
                        
    def _init_lmc(self):
        """
        初始化 MCPBE 用到的各类 LMC 适配器：
          - 预处理/离线类: table | rank | copula | flow
          - 在线 lmc: live
        规则：
          1) use_lmc_pre_model=False → 不加载任何离线 adapter
          2) use_lmc_pre_model=True → 看 lmc_pre_model 选择具体的 adapter
          3) live 若 small_particle_policy='fallback'，则必须有一个离线 adapter 可用
          4) 路径缺失时给出清晰报错；同时传进多个路径但实际没用到的给出警告
        """
        # -------------- 基本配置取值 --------------
        self.use_lmc_pre_model    = bool(getattr(self, "use_lmc_pre_model", False))
        self.lmc_pre_model        = str(getattr(self, "lmc_pre_model", "table"))  # table|rank|copula|flow
        self.lmc_tables_path      = getattr(self, "lmc_tables_path", None)
        self.lmc_rank_tables_path = getattr(self, "lmc_rank_tables_path", None)
        self.lmc_copula_path      = getattr(self, "lmc_copula_path", None)
        self.lmc_flow_pure_path        = getattr(self, "lmc_flow_pure_path", None)
        self.lmc_flow_mix_path        = getattr(self, "lmc_flow_mix_path", None)
        self.lmc_A0_runtime       = getattr(self, "lmc_A0_runtime", None)
        self.lmc_interp           = str(getattr(self, "lmc_interp", "bilinear"))
        self.lmc_tables_cache     = bool(getattr(self, "lmc_tables_cache", False))
        self.lmc_small_particle_policy = str(getattr(self, "lmc_small_particle_policy", "fallback"))
    
        # 这里保存最终实际创建出来的 adapter
        self.lmc_adapter = None
    
        # -------------- 离线/预处理模型的构建 --------------
        if self.use_lmc_pre_model:
            # 统一从同一个地方 import，避免上面那段老代码的循环 import
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
    
            # 根据选择的类型构建
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
    
            # 无论哪种 adapter，都让它知道小颗粒策略
            if self.lmc_adapter is not None and hasattr(self.lmc_adapter, "set_small_particle_policy"):
                self.lmc_adapter.set_small_particle_policy(policy=self.lmc_small_particle_policy)
    
        # -------------- Live LMC 的初始化 --------------
        self.use_lmc_live = bool(getattr(self, "use_lmc_live", False))
        self.lmc_live = None
        if self.use_lmc_live:
            from .lmc_adapter import LMCLiveAdapter
            self.lmc_live = LMCLiveAdapter()
            self.lmc_live.configure_simulator(
                STR=self.STR if hasattr(self, "STR") else np.array([1.0, 1.0, 1.0]),
                NO_FRAG=int(getattr(self, "NO_FRAG", 4)),
                gamma=float(getattr(self, "gamma", 1.0)),
                allow_loops=bool(getattr(self, "allow_loops", True)),
                accept_all_cracks=bool(getattr(self, "accept_all_cracks", False)),
                use_weighted_start=bool(getattr(self, "use_weighted_start", False)),
                aspect_ratio=float(getattr(self, "aspect_ratio", 1.0)),
                int_bre=float(getattr(self, "int_bre", 0.0)),
                A0_run=float(getattr(self, "lmc_A0_runtime", 1.0)),
                small_particle_policy=str(getattr(self, "lmc_small_particle_policy", "fallback")),
                delta_cells=float(getattr(self, "lmc_delta_cells", 0.1)),
                rebuild=True,
            )
    
            # live 的 fallback 依赖一个离线 adapter
            if self.lmc_live.small_particle_policy == "fallback" and self.lmc_adapter is None:
                raise ValueError(
                    "use_lmc_live=True with small_particle_policy='fallback' requires a pre LMC adapter "
                    "(set use_lmc_pre_model=True and choose one of table/rank/copula/flow)."
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
        """Agglomeration time-step Δt using current r_agg (active slice only)."""
        a = self.a_tot
        if a < 2:
            return float("inf")
        sum_r = float(np.sum(self._r_agg[:a]))
        if sum_r <= 0.0:
            return float("inf")
        return 2.0 * float(self.Vc) * (a - 1) / (a * sum_r)

    def _dt_break(self) -> float:
        """Breakage Δt with mean break rate (active slice only)."""
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

        next_save_idx = 1 if len(self.t_vec) > 1 else 0
        self._elapsed = 0.0
        self._iter_count = 0

        while self.t[-1] <= float(self.t_vec[-1]) and count < maxiter:
            # keep context for logging/expansion
            self._elapsed = self.t[-1]
            self._iter_count = count

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

            # Save snapshots at requested times (active slice only)
            while next_save_idx < len(self.t_vec) and elapsed_time >= self.t_vec[next_save_idx]:
                self.V_save.append(self.V_flat[:, :self.a_tot].copy())
                self.Vc_save.append(float(self.Vc))
                self.step += 1
                next_save_idx += 1

            # agglomeration-dominated safety (duplicate CV)
            self._maybe_double_control_volume(self.t[-1], count)

            count += 1
            # if count%100 == 0: print([f"[Test] events = {count}"])
            if self.a_tot < 2 and pt in ("agglomeration", "mix"):
                break

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
    ):
        """
        Run N Monte Carlo realizations (repeats).

        workers = 1  -> serial (original behavior)
        workers > 1  -> parallel with ProcessPoolExecutor

        We serialize the current solver state (self.__dict__) and send it
        to workers, where a fresh solver is rebuilt.
        """
        # ----- build seeds -----
        if seeds is None:
            master = np.random.SeedSequence(base_seed)
            seeds = master.spawn(N)
        if len(seeds) != N:
            raise ValueError("Length of seeds must equal N.")

        # ----- serial path (original behavior) -----
        if workers == 1:
            results = []
            for k in range(N):
                # local run, just deepcopy self and do what we did before
                m = copy.deepcopy(self)
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
                m._initialize_samplers()
                m.solve(maxiter=maxiter)
                mu, tv = m.calc_moments_over_time(normalize=True)
                results.append({"seed_info": seed_info, "t_vec": tv, "moments": mu})
            return results

        # ----- parallel path -----
        # 1) snapshot current solver state into a picklable dict
        #    deep copy to detach from parent
        base_state = copy.deepcopy(self.__dict__)

        # 2) build payloads for each worker
        payloads = []
        for k in range(N):
            payloads.append(
                {
                    "cls": self.__class__,   # MCPBEBase or subclass
                    "state": base_state,
                    "seed": seeds[k],
                    "maxiter": maxiter,
                    "init_Vc": init_Vc,
                    "Vc": Vc,
                    "V_flat": V_flat,
                }
            )

        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_mcpbe_run_single_parallel, pl): i for i, pl in enumerate(payloads)}
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    # 这里你可以选择 raise，也可以选择记录失败的那个
                    raise RuntimeError(f"[parallel] worker {idx} failed: {e}")
                else:
                    results.append(res)

        return results
    # ---------------------------------------------------------------------
    # Column ops (capacity style)
    # ---------------------------------------------------------------------
    def _remove_particle_column(self, j: int):
        """Swap j with last active, shrink a_tot by 1, zero freed slot, rebuild samplers."""
        a = self.a_tot
        if j < 0 or j >= a:
            raise IndexError("column index out of range")
        if a <= 1:
            # reset to empty active set
            self.a_tot = max(0, a - 1)
            # clear slot 0
            if a == 1:
                self.V_flat[:, 0:1] = 0.0
                self.X[0:1] = 0.0
                if hasattr(self, "_r_agg"):
                    self._r_agg[0:1] = 0.0
                if hasattr(self, "_break_rate"):
                    self._break_rate[0:1] = 0.0
            self._agg_sampler = FenwickSampler(np.zeros(0)) if self._agg_sampler is not None else None
            self._break_sampler = (
                FenwickSampler(np.zeros(0)) if self._break_sampler is not None else None
            )
            return

        last = a - 1
        if j != last:
            # swap active columns
            self.V_flat[:, [j, last]] = self.V_flat[:, [last, j]]
            self.X[j], self.X[last] = self.X[last], self.X[j]
            if hasattr(self, "_r_agg") and self._r_agg is not None and self._r_agg.shape[0] >= a:
                self._r_agg[j], self._r_agg[last] = self._r_agg[last], self._r_agg[j]
            if (
                hasattr(self, "_break_rate")
                and self._break_rate is not None
                and self._break_rate.shape[0] >= a
            ):
                self._break_rate[j], self._break_rate[last] = (
                    self._break_rate[last],
                    self._break_rate[j],
                )

        # logical shrink & zero freed slot
        self.a_tot = last
        self.V_flat[:, self.a_tot : self.a_tot + 1] = 0.0
        self.X[self.a_tot : self.a_tot + 1] = 0.0
        if hasattr(self, "_r_agg") and self._r_agg is not None and self._r_agg.shape[0] > self.a_tot:
            self._r_agg[self.a_tot : self.a_tot + 1] = 0.0
        if (
            hasattr(self, "_break_rate")
            and self._break_rate is not None
            and self._break_rate.shape[0] > self.a_tot
        ):
            self._break_rate[self.a_tot : self.a_tot + 1] = 0.0

        # rebuild samplers from active slices (simple & correct)
        if self._agg_sampler is not None:
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        if self._break_sampler is not None:
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])

    def _append_particle_column(self, frag_vols: np.ndarray):
        """Append one particle from its per-component volumes; capacity aware."""
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

        # rebuild samplers from active slices (simple baseline)
        if self._agg_sampler is not None and hasattr(self, "_r_agg"):
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        if self._break_sampler is not None and hasattr(self, "_break_rate"):
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])

    def _ensure_break_sampler(self):
        """(Re)build break sampler from active slice if needed."""
        if not hasattr(self, "_break_rate"):
            return
        if self._break_sampler is None:
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])

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