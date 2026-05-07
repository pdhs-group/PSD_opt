# Core simulation framework: initialization, capacity buffers, main loop,
# time stepping, doubling control volume, basic column ops.
from __future__ import annotations

import math
import os
import time
from typing import Optional, Sequence, Any, Tuple
import copy

import numpy as np
from pbe_core.base.base_solver import BaseSolver
from .fenwick import FenwickSampler
from .mcpbe_initialization import InitialParticleMixin
from .mcpbe_time_helper import MCPBETimeHelper


class MCPBEBase(InitialParticleMixin, MCPBETimeHelper, BaseSolver):
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
        self.exp_time_step = False
        self.sum_prop_pair = True
        self.maybe_double_control_volume=False

        # Initial distributions flags
        self.PGV = np.full(dim, "mono")
        self.SIG = np.full(dim, 0.1)

        # State containers
        self.V_flat: Optional[np.ndarray] = None
        self.V_eff_init = 0     # 0 -> no compression
        self.V_eff_mod = "Q3"   # "Q0" or "Q3"

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
            self._initialize_samplers()
            self._bf_ready = False  # breakage CDFs (mix-in will build on demand)

    # ---------------------------------------------------------------------
    # Validation & helpers
    # ---------------------------------------------------------------------
    def _validate_input_arrays(self):
        dim = self.dim
        lengths = {
            "c": len(self.c),
            "x": len(self.x),
            "PGV": len(self.PGV),
            "SIG": len(self.SIG),
        }
        for name, length in lengths.items():
            if length != dim:
                raise ValueError(
                    f"`{name}` must be a 1D array (sequence) of length dim={dim}, got length {length}."
                )

    def _growth_factor(self) -> float:
        """Dynamic capacity growth factor in [1.1, 2.0], more aggressive early in time."""
        T = float(self.t_total)
        t = float(self._elapsed)
        r = 1.0 - min(max(t / max(T, 1e-12), 0.0), 1.0)
        f = 1.1 + 0.9 * r  # in [1.1, 2.0]
        return float(min(2.0, max(1.1, f)))

    def _compute_frag_num(self):
        """Expected number of fragments per break event from BREAKFVAL & v."""
        v = float(self.pl_v)
        bf = int(self.BREAKFVAL)
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
    def _initialize_particles(
        self,
        init_Vc: bool = True,
        V_flat: Optional[np.ndarray] = None,
        W_init: Optional[np.ndarray] = None,
    ):
        """Initialize particle state from model settings or explicit particles."""
        total_cols = self._prepare_initial_counts(
            init_Vc=init_Vc,
            require_model_init=(V_flat is None),
        )
        payload = self._build_initial_particle_payload(
            total_cols=total_cols,
            V_flat=V_flat,
            W_init=W_init,
        )
        payload = self._normalize_initial_particle_payload(payload)
        payload = self._maybe_compress_initial_particle_payload(payload)
        self._commit_initial_particle_payload(payload)

    def _initialize_samplers(self):
        """Build (or resize) samplers for agglomeration/breakage based on process_type."""
        pt = self.process_type

        # Agglomeration
        if pt in ("agglomeration", "mix"):
            self._prepare_agg_delta_config()
            self._r_agg = np.zeros(self._cap, dtype=float)
            self._rebuild_all_propensities()  # from AgglomerationMixin
            if self._r_agg.shape[0] < self._cap:
                buf = np.zeros(self._cap, dtype=float)
                buf[:self.a_tot] = self._r_agg[:self.a_tot]
                self._r_agg = buf
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        else:
            self._r_agg = np.zeros(self._cap, dtype=float)
            self._delta_agg = np.zeros(self._cap, dtype=float)
            self._agg_sampler = None

        # Breakage
        if pt in ("breakage", "mix"):
            self._prepare_break_config()
            self._calc_break_rates_full()  # from BreakageMixin
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
        else:
            self._break_rate = np.zeros(self._cap, dtype=float)
            self._delta_break = np.zeros(self._cap, dtype=float)
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
    
        W_new = np.zeros(new_cap, dtype=float)
        W_new[:self.a_tot] = self.W[:self.a_tot]
        self.W = W_new

        r_new = np.zeros(new_cap, dtype=float)
        r_new[:self.a_tot] = self._r_agg[:self.a_tot]
        self._r_agg = r_new

        b_new = np.zeros(new_cap, dtype=float)
        b_new[:self.a_tot] = self._break_rate[:self.a_tot]
        self._break_rate = b_new

        d_new = np.zeros(new_cap, dtype=float)
        d_new[:self.a_tot] = self._delta_agg[:self.a_tot]
        self._delta_agg = d_new

        d_new = np.zeros(new_cap, dtype=float)
        d_new[:self.a_tot] = self._delta_break[:self.a_tot]
        self._delta_break = d_new
    
        # Print expansion info
        if self.VERBOSE:
            print(
                f"[MC-PBE] Capacity grown at t={self._elapsed:.6g} "
                f"after {self._iter_count} events: cap {old_cap} -> {new_cap} "
                f"(x{new_cap/max(old_cap,1):.2f})"
            )


    def _maybe_double_control_volume(self, elapsed_time: float, iter_count: int):
        """Duplicate state to keep statistics when particle count drops (agglomeration dominates)."""
        if self.process_type not in ("agglomeration", "mix"):
            return
        if self.a_tot <= 0:
            return

        # Use compressed-initial active count as baseline to avoid false trigger
        # when V_eff_init << a0.
        a_ref = int(self._cv_a_ref)
        if a_ref <= 0:
            a_ref = int(self.a_tot)
            self._cv_a_ref = a_ref

        if self.a_tot > a_ref / 2:
            return

        old_a = self.a_tot
        old_Vc = float(self.Vc)

        # Double control volume and duplicate active slice
        self.Vc *= 2.0
        V_active = self.V_flat[:, :self.a_tot]
        X_active = self.X[:self.a_tot]
        W_active = self.W[:self.a_tot]
        V_dup = np.concatenate((V_active, V_active), axis=1)
        X_dup = np.concatenate((X_active, X_active))
        W_dup = np.concatenate((W_active, W_active))
        self.a_tot = V_dup.shape[1]

        # Ensure capacity and write back
        if self._cap < self.a_tot:
            self._cap = int(self.a_tot * 1.2) + 8
            V_new = np.zeros((self.dim + 1, self._cap), dtype=float)
            X_new = np.zeros(self._cap, dtype=float)
            W_new = np.zeros(self._cap, dtype=float)
            V_new[:, :self.a_tot] = V_dup
            X_new[:self.a_tot] = X_dup
            W_new[:self.a_tot] = W_dup
            self.V_flat = V_new
            self.X = X_new
            self.W = W_new

            r_new = np.zeros(self._cap, dtype=float)
            r_new[:old_a] = self._r_agg[:old_a]
            self._r_agg = r_new

            b_new = np.zeros(self._cap, dtype=float)
            b_new[:old_a] = self._break_rate[:old_a]
            self._break_rate = b_new

            d_new = np.zeros(self._cap, dtype=float)
            d_new[:old_a] = self._delta_agg[:old_a]
            self._delta_agg = d_new

            d_new = np.zeros(self._cap, dtype=float)
            d_new[:old_a] = self._delta_break[:old_a]
            self._delta_break = d_new
        else:
            self.V_flat[:, :self.a_tot] = V_dup
            self.X[:self.a_tot] = X_dup
            self.W[:self.a_tot] = W_dup
            self._delta_agg[:self.a_tot] = np.concatenate((self._delta_agg[:old_a], self._delta_agg[:old_a]))
            self._delta_break[:self.a_tot] = np.concatenate((self._delta_break[:old_a], self._delta_break[:old_a]))

        # Keep the original initial support fixed; control-volume doubling
        # only changes the represented multiplicity of that support.
        self.W0 *= 2.0

        # Rebuild samplers from active slices
        if self.process_type in ("agglomeration", "mix"):
            self._rebuild_all_propensities()
            self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
        if self.process_type in ("breakage", "mix"):
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


    # ---------------------------------------------------------------------
    # Main solve loop
    # ---------------------------------------------------------------------
    def solve(self, maxiter: int = int(1e12)):
        t0 = time.time()
        count = 0
        current_time = 0.0
        self.real_agg_events = 0.0
        self.real_break_events = 0.0
        self.real_agg_events_save = [0.0]
        self.real_break_events_save = [0.0]

        pt = self.process_type
        agg_total_propensity = (
            (lambda: float(self._agg_sampler.total()))
            if self._agg_sampler is not None
            else (lambda: float(np.sum(self._r_agg[:self.a_tot])))
        )
        break_total_propensity = (
            (lambda: float(self._break_sampler.total()))
            if self._break_sampler is not None
            else (lambda: float(np.sum(self._break_rate[:self.a_tot])))
        )
        agg_initial_dt, agg_event_dt = self._build_agg_dt_strategy()
        break_initial_dt, break_event_dt = self._build_break_dt_strategy()
        mix_initial_dt, mix_event_dt = self._build_mix_dt_strategy()

        timer_agg = agg_initial_dt(agg_total_propensity()) if pt == "agglomeration" else float("inf")
        timer_break = break_initial_dt(break_total_propensity()) if pt == "breakage" else float("inf")
        if pt == "mix":
            agg_prop0 = agg_total_propensity()
            break_prop0 = break_total_propensity()
            timer_mix = mix_initial_dt(self._mix_event_rate_from_sum_prop(agg_prop0, break_prop0))
        else:
            timer_mix = float("inf")

        if self.VERBOSE:
            if np.isfinite(timer_agg):
                print(f"Initial dt_agg = {timer_agg:.3e} s")
            if np.isfinite(timer_break):
                print(f"Initial dt_break = {timer_break:.3e} s")
            if np.isfinite(timer_mix):
                print(f"Initial dt_mix = {timer_mix:.3e} s")

        if self.mcpbe_debug:
            self._check_state_before_solve()
            self._log_debug_config()

        next_save_idx = 1 if len(self.t_vec) > 1 else 0
        self._elapsed = 0.0
        self._iter_count = 0

        while current_time <= float(self.t_vec[-1]) and count < maxiter:
            # keep context for logging/expansion
            self._elapsed = current_time
            self._iter_count = count
            
            # cache "left" state: state after previous event
            t_prev = current_time
            V_prev_active = self.V_flat[:, :self.a_tot].copy()
            W_prev_active = self.W[:self.a_tot].copy()

            if pt == "agglomeration":
                sum_prop_before = agg_total_propensity()
                self._do_one_agg()  # from AgglomerationMixin
                self.real_agg_events += float(max(0.0, float(self._last_agg_dW)))
                sum_prop_after = agg_total_propensity()
                elapsed_time = timer_agg
                dtd_agg = agg_event_dt(sum_prop_before, sum_prop_after)
                timer_agg += dtd_agg
            elif pt == "breakage":
                # Total propensity before the event; dt uses event dW over pre-event propensity.
                sum_prop_before = break_total_propensity()
                self._do_one_break()  # sets self._last_break_dW for packeted events
                self.real_break_events += float(max(0.0, float(self._last_break_dW)))
                sum_prop_after = break_total_propensity()
                elapsed_time = timer_break
                dtd_break = break_event_dt(sum_prop_before, sum_prop_after)
                timer_break += dtd_break
            else:  # mix
                agg_prop_before = agg_total_propensity()
                break_prop_before = break_total_propensity()
                agg_rate_before = self._agg_event_rate_from_sum_prop(agg_prop_before)
                break_rate_before = self._break_event_rate_from_sum_prop(break_prop_before)
                total_rate_before = agg_rate_before + break_rate_before
                if total_rate_before <= 0.0:
                    break

                u_event = float(self._rng.random()) * total_rate_before
                if u_event < agg_rate_before:
                    self._do_one_agg()
                    self.real_agg_events += float(max(0.0, float(self._last_agg_dW)))
                else:
                    self._do_one_break()
                    self.real_break_events += float(max(0.0, float(self._last_break_dW)))

                agg_prop_after = agg_total_propensity()
                break_prop_after = break_total_propensity()
                total_rate_after = self._mix_event_rate_from_sum_prop(agg_prop_after, break_prop_after)
                elapsed_time = timer_mix
                dtd_mix = mix_event_dt(total_rate_before, total_rate_after)
                timer_mix += dtd_mix

            current_time = float(elapsed_time)
            self._elapsed = current_time
            
            # current "right" state after this event
            V_right_active = self.V_flat[:, :self.a_tot]
            W_right_active = self.W[:self.a_tot]

            # Save snapshots at requested times (active slice only)
            while next_save_idx < len(self.t_vec) and elapsed_time >= self.t_vec[next_save_idx]:
                # right snapshots: same behavior as original code
                self.V_save.append(V_right_active.copy())
                self.W_save.append(W_right_active.copy())
            
                self.Vc_save.append(float(self.Vc))
                self.step += 1
            
                # left/right metadata for this time point
                self.V_save_left.append(V_prev_active.copy())
                self.W_save_left.append(W_prev_active.copy())
            
                self.t_left.append(t_prev)
                self.t_right.append(elapsed_time)
                self.real_agg_events_save.append(float(self.real_agg_events))
                self.real_break_events_save.append(float(self.real_break_events))
            
                next_save_idx += 1
                if self.VERBOSE:    
                    print(
                        f"[MC-PBE] Calculate t={elapsed_time:.6g} after {self._iter_count} events "
                        f"(real agg={self.real_agg_events:.6g}, real break={self.real_break_events:.6g})"
                    )
            # agglomeration-dominated safety (duplicate CV)
            if self.maybe_double_control_volume:
                self._maybe_double_control_volume(current_time, count)
            self.maybe_reconstruct(iter_count=self._iter_count, reason=f"post_event_{pt}")

            count += 1
            # if count%100 == 0: print([f"[Test] events = {count}"])
            if self.a_tot < 2 and pt in ("agglomeration", "mix"):
                break
        self.MACHINE_TIME = time.time() - t0
        if self.VERBOSE:
            print(
                f"[MC-PBE] The calculation took {self.MACHINE_TIME:.4g}s "
                f"after {count} events "
                f"(real agg={self.real_agg_events:.6g}, real break={self.real_break_events:.6g})"
            )
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
        W_init: Optional[np.ndarray] = None,
        psd_enable: bool = False,
        psd_basis: str = "volume",                 # "volume" or "number"
        psd_x_grid: Optional[np.ndarray] = None,   # if given -> output Q(x)
        psd_Q_grid: Optional[np.ndarray] = None,   # if given -> output x(Q)
    ):
        """
        Run N Monte Carlo realizations (repeats).


        Returns
        -------
        If psd_enable == False:
            List[{"seed_info", "t_vec", "moments"}]

        If psd_enable == True:
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

        # PSD grids are meaningful only when PSD aggregation is enabled.
        if not psd_enable and (psd_x_grid is not None or psd_Q_grid is not None):
            raise ValueError("psd_x_grid/psd_Q_grid require psd_enable=True.")

        # ----- serial path (supports PSD) -----
        results: list[dict[str, Any]] = []

        # For PSD aggregation across repeats
        cdf_repeats: list[Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]] = []
        t_vec_ref: Optional[np.ndarray] = None

        for k in range(N):
            # Deep copy self and run a single realization
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
                # m.Vc = 1e-10
                # print("Controll volume : ", m.Vc)
            m._initialize_particles(init_Vc=init_Vc, V_flat=V_flat, W_init=W_init)
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
                        raise ValueError(
                            "t_vec differs between repeats. PSD averaging requires identical t_vec."
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

    # ---------------------------------------------------------------------
    # Column ops (capacity style)
    # ---------------------------------------------------------------------
    def _remove_particle_column(self, j: int):
        """Swap j with last active, shrink a_tot by 1, zero freed slot, rebuild samplers.
        Keeps W aligned with particle columns.
        """
        a = self.a_tot
        if j < 0 or j >= a:
            raise IndexError("column index out of range")

        last = a - 1
        if j != last:
            # swap active columns (V/X/W + propensities)
            tmp = self.V_flat[:, j].copy()
            self.V_flat[:, j] = self.V_flat[:, last]
            self.V_flat[:, last] = tmp
            self.X[j], self.X[last] = self.X[last], self.X[j]
            self.W[j], self.W[last] = self.W[last], self.W[j]

            self._r_agg[j], self._r_agg[last] = self._r_agg[last], self._r_agg[j]
            self._break_rate[j], self._break_rate[last] = self._break_rate[last], self._break_rate[j]
            self._delta_agg[j], self._delta_agg[last] = self._delta_agg[last], self._delta_agg[j]
            self._delta_break[j], self._delta_break[last] = self._delta_break[last], self._delta_break[j]

        # logical shrink & zero freed slot
        self.a_tot = last
        self.V_flat[:, self.a_tot] = 0.0
        self.X[self.a_tot] = 0.0
        self.W[self.a_tot] = 0.0

        self._r_agg[self.a_tot] = 0.0
        self._break_rate[self.a_tot] = 0.0
        self._delta_agg[self.a_tot] = 0.0
        self._delta_break[self.a_tot] = 0.0

        # local sampler remove (swap-with-last behavior kept consistent with array swap above)
        if self._agg_sampler is not None:
            self._agg_sampler.remove(j)
        if self._break_sampler is not None:
            self._break_sampler.remove(j)

    def _append_particle_column(self, frag_vols: np.ndarray):
        """Append one particle from per-component volumes; caller is responsible for sampler updates."""
        frag_vols = np.asarray(frag_vols, dtype=float)
        if frag_vols.shape != (self.dim,):
            raise ValueError("frag_vols must have shape (dim,)")
    
        # Ensure capacity for V/X (and W via _ensure_capacity_for)
        self._ensure_capacity_for(1)
    
        Vnew = float(np.sum(frag_vols))
        idx = self.a_tot
    
        self.V_flat[: self.dim, idx] = frag_vols
        self.V_flat[-1, idx] = Vnew
        self.X[idx] = float(self._vol2diam(Vnew))
    
        # Default weight for new particle (DSMC baseline).
        # Note: breakage/agglomeration code may overwrite this immediately.
        self.W[idx] = 1.0
        self._delta_agg[idx] = 0.0
        self._delta_break[idx] = 0.0
    
        self.a_tot += 1
    
        # local sampler append
        if self._agg_sampler is not None:
            self._agg_sampler.append(float(self._r_agg[idx]))
        if self._break_sampler is not None:
            self._break_sampler.append(float(self._break_rate[idx]))

    def _ensure_break_sampler(self):
        """Rebuild break sampler from the active slice."""
        self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
    
    def _close(self, gc_clean=True):
        big_attrs = ("V_flat", "X", "V0", "X0",
             "V0_save", "V_save", "Vc_save",
             "_r_agg", "_break_rate", "_delta_agg", "_delta_break",
             "_agg_sampler", "_break_sampler")
        for name in big_attrs:
            setattr(self, name, None)
        self._bf_cache.clear()
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
            raise ValueError(
                f"[MC-PBE][DEBUG] t_vec[-1]={tv[-1]:.6g} differs from t_total={float(self.t_total):.6g}.",
            )

        # validate input arrays (c, x, PGV, SIG) against dim
        self._validate_input_arrays()

        # process_type consistency
        pt = str(self.process_type).lower()
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
        if self.X is None:
            raise ValueError(
                "[MC-PBE][DEBUG] `X` (diameter array) is not initialized. "
                "Make sure `_initialize_particles` has been called."
            )
        if self.a_tot < 0 or self.a_tot > self._cap:
            raise ValueError(
                f"[MC-PBE][DEBUG] Inconsistent a_tot={self.a_tot}, cap={self._cap}."
            )

        # sampler presence (only sanity check; they may be rebuilt during solve)
        if pt in ("agglomeration", "mix"):
            if self._agg_sampler is None:
                raise ValueError(
                    "[MC-PBE][DEBUG] Agglomeration enabled but `_agg_sampler` is None. "
                    "Call `_initialize_samplers` before `solve`.",
                )
        if pt in ("breakage", "mix"):
            if self._break_sampler is None:
                raise ValueError(
                    "[MC-PBE][DEBUG] Breakage enabled but `_break_sampler` is None. "
                    "Call `_initialize_samplers` before `solve`.",
                )

    def _log_debug_config(self):
        """Print a categorized snapshot of key MCPBE configuration parameters.

        Categories:
          - General parameters
          - Agglomeration parameters
          - Breakage parameters
        """
        print("\n[MC-PBE][DEBUG] Configuration snapshot")

        # -------------------------
        # General parameters
        # -------------------------
        print("  [General parameters]")
        print(f"    dim          = {self.dim}")
        print(f"    t_total      = {self.t_total}")
        print(f"    t_write      = {self.t_write}")

        tv = np.asarray(self.t_vec, dtype=float)
        if tv.size > 0:
            print(
                f"    t_vec        = len={tv.size}, "
                f"first={tv[0]:.6g}, last={tv[-1]:.6g}"
            )
        else:
            print("    t_vec        = <empty or None>")

        print(f"    a0           = {self.a0}")
        print(f"    c            = {self.c}")
        print(f"    x            = {self.x}")
        print(f"    Vc           = {self.Vc}")
        print(f"    PGV          = {self.PGV}")
        print(f"    SIG          = {self.SIG}")
        print(f"    VERBOSE      = {self.VERBOSE}")
        print(f"    process_type = {self.process_type}")
        print(f"    CDF_method   = {self.CDF_method}")
        print(f"    USE_PSD      = {self.USE_PSD}")
        print(f"    DIST1_path   = {self.DIST1_path}")
        print(f"    DIST1_name   = {self.DIST1_name}")
        print(f"    DIST3_path   = {self.DIST3_path}")
        print(f"    DIST3_name   = {self.DIST3_name}")

        # -------------------------
        # Agglomeration parameters
        # -------------------------
        print("  [Agglomeration parameters]")
        print(f"    COLEVAL      = {self.COLEVAL}")
        print(f"    SIZEEVAL     = {self.SIZEEVAL}")
        print(f"    CORR_BETA    = {self.CORR_BETA}")
        print(f"    alpha_prim   = {self.alpha_prim}")
        print(f"    G (shear)    = {self.G}")

        # If current state already has propensities, log basic stats
        r_active = self._r_agg[: self.a_tot]
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

        # -------------------------
        # Breakage parameters
        # -------------------------
        print("  [Breakage parameters]")
        print(f"    BREAKRVAL    = {self.BREAKRVAL}")
        print(f"    BREAKFVAL    = {self.BREAKFVAL}")
        print(f"    pl_v         = {self.pl_v}")
        print(f"    pl_P1        = {self.pl_P1}")
        print(f"    pl_P2        = {self.pl_P2}")
        print(f"    pl_P3        = {self.pl_P3}")
        print(f"    pl_P4        = {self.pl_P4}")
        print(f"    frag_num     = {self.frag_num}")

        br_active = self._break_rate[: self.a_tot]
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
        print("[MC-PBE][DEBUG] End of configuration snapshot\n")


