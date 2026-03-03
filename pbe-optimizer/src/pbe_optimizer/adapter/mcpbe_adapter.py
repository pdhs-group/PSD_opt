# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:54:06 2025

@author: px2030
"""
import os
from typing import Any, Literal
import numpy as np
import h5py
import threading
from .adapters_api_basics import WriteThroughAdapter
from wmcpbe import MCPBESolver

class MCPBEAdapter(WriteThroughAdapter):
    """
    Adapter for DPBESolver with role-aware handling.

    Roles:
      - "main": 2D main solver; alpha_prim [a0, a1, a2] -> [a0, a1, a1, a2]
      - "NM"  : 1D auxiliary solver; alpha_prim takes a0
      - "M"   : 1D auxiliary solver; alpha_prim takes a2
                For dim==2:
                  * pl_P3 -> impl.pl_P1 (written immediately)
                  * pl_P4 -> impl.pl_P2 (written immediately)
    """

    def __init__(self, *, opt, role: Literal["main","NM","M"]="main", **kw: Any):
        kw.pop("init", None)
        impl = MCPBESolver(init=False, **kw)
        super().__init__(impl)
        
        # Optional: name mappings
        # self._map.update({
        #     "grid_x": "V1",
        #     "grid_y": "V3",
        # })
        
        # Optional: Adapter-only field (won't write-through)
        self._skip.update({"role", "opt", "NC", "MC_seed", "init_Vc", "Vc_init", 
                           "V_flat_init", "_psd_basis", "_psd_x_grid", "_psd_Q_grid",
                           "_init_cdf_payload", "use_exp_cdf_init", "data_mod"})
        self.role = role
        self.opt = opt
        self.use_exp_cdf_init = bool(getattr(opt, "use_exp_cdf_init", False))
        self._init_cdf_payload = None
        
        # p.calc_status will be checked during the optimization process and must exist
        self.calc_status = True

        # ---------- alpha_prim  ----------
        def set_alpha_prim(impl, value):
            arr = np.asarray(value)
            dim = impl.dim  
            if dim is None:
                raise ValueError("MCPBEAdapter: 'dim' must be set on impl before alpha_prim.")
        
            r = self.role  
            if dim == 1:
                if arr.ndim == 0:
                    impl.alpha_prim = float(arr)
                else:
                    flat = arr.ravel()
                    if flat.size < 1:
                        raise ValueError("alpha_prim must contain at least one value for dim=1.")
                    if r == "NM" or r == "main":
                        impl.alpha_prim = float(flat[0])
                    elif r == "M":
                        if flat.size < 3:
                            raise ValueError("alpha_prim must contain at least a2 (index 2) for M in 2D.")
                        impl.alpha_prim = float(flat[-1])
                    else:
                        raise ValueError(f"Unknown role '{r}'.")
            elif dim == 2:
                flat = arr.ravel()
                if r == "main":
                    if flat.size == 3:
                        a0, a1, a2 = map(float, flat.tolist())
                        impl.alpha_prim = np.array([a0, a1, a1, a2], dtype=float)
                    elif flat.size == 4:
                        impl.alpha_prim = np.array(flat, dtype=float).reshape(4,)
                    else:
                        raise ValueError(
                            f"alpha_prim for main 2D should have length 3 (a0,a1,a2)"
                            f" or 4 (a0,a1,a1,a2); got {flat.size}."
                        )
            else:
                raise ValueError(f"Unsupported dim={dim} for MCPBEAdapter alpha_prim handling.")
        
        self._setters["alpha_prim"] = set_alpha_prim
        # ---------- c and x and PGV  ----------
        if self.role == "NM":
            def set_c(impl, value):
                impl.c = np.array([value[0]])
            def set_x(impl, value):
                impl.x = np.array([value[0]])
            def set_PGV(impl, value):
                impl.PGV = np.array([value[0]])
            self._setters["c"] = set_c
            self._setters["x"] = set_x
            self._setters["PGV"] = set_PGV
        if self.role == "M":
            def set_c(impl, value):
                impl.c = np.array([value[1]])
            def set_x(impl, value):
                impl.x = np.array([value[1]])
            def set_PGV(impl, value):
                impl.PGV = np.array([value[1]])
            self._setters["c"] = set_c
            self._setters["x"] = set_x
            self._setters["PGV"] = set_PGV
        # ---------- P1 and P2 in p_M  ----------    
        if self.role == "M":
            def set_pl_P3(impl, value):
                impl.pl_P1 = value
        
            def set_pl_P4(impl, value):
                impl.pl_P2 = value
        
            self._setters["pl_P3"] = set_pl_P3
            self._setters["pl_P4"] = set_pl_P4


    # %% ESSENTIAL METHOD INTERFACE
    def set_comp_para(self, data_path: str) -> None:
        opt = self.opt
        flag = getattr(opt, "data_flag", "Q0")
        flag = str(flag).upper()
    
        allowed = {"Q0", "Q3", "Q0_X_50", "Q3_X_50"}
        if flag not in allowed:
            raise ValueError(
                f"opt.data_flag must be one of {sorted(allowed)} for MCPBEAdapter, "
                f"got {flag!r}."
            )
    
        # Map flag -> PSD basis used by MCPBE
        if flag.startswith("Q0"):
            self._psd_basis = "number"
        else:  # Q3 / Q3_X_50
            self._psd_basis = "volume"
    
        self.init_Vc = False            # tell solver to use provided Vc
        self._psd_Q_grid = None         # we only use Q(x), not x(Q), here
        self.opt.set_comp_para_flag = True
    
        self.impl.lmc_pool_dir = data_path
        self.impl.lmc_breakage_model_path = os.path.join(data_path, "mlp_model.pkl")
        return None

    def reset_params(self) -> None:
        self.impl._reset_params()
    
    def calc_init_from_data(self, exp_data_paths, init_flag) -> None:
        return None
            
    def calc_matrix(self, init_N) -> None:
        return None

    # def solve(self, t_vec):
    #     if not np.allclose(np.asarray(t_vec), np.asarray(self.opt.t_vec)):
    #         raise ValueError("Adapter.solve: provided t_vec differs from opt.t_vec.")
    
    #     self.calc_status = True
    #     max_time = float(getattr(self.opt, "max_iter_time", 0.0) or 0.0)
    #     dump_results = getattr(self.impl, "dump_results", False)
    
    #     # 共享的取消标志：所有拷贝都应该指向它
    #     shared_flag = {"cancel": False}
    #     self.impl.cancel_flag = shared_flag
    
    #     result_container = {}
    
    #     def _worker():
    #         try:
    #             r, p = self.impl.solve_repeats(
    #                 N=self.NC,
    #                 base_seed=self.MC_seed,
    #                 init_Vc=self.init_Vc,
    #                 Vc=self.opt.Vc_init,
    #                 V_flat=self.opt.V_flat_init,
    #                 workers=1,
    #                 psd_enable=True,
    #                 psd_basis=self._psd_basis,
    #                 psd_x_grid=self.opt._psd_x_grid,
    #                 psd_Q_grid=self._psd_Q_grid,
    #                 dump_results=dump_results,
    #             )
    #             result_container["result"] = (r, p)
    #         except Exception as e:
    #             result_container["error"] = e
    
    #     # --- start worker thread ---
    #     th = threading.Thread(target=_worker)
    #     th.daemon = True
    #     th.start()
    
    #     # --- wait with timeout ---
    #     th.join(timeout=max_time if max_time > 0 else None)
    
    #     # --- check timeout ---
    #     if th.is_alive():
    #         # Timeout: request cancellation
    #         shared_flag["cancel"] = True

    #         # Give worker a short grace period to observe cancel_flag and release resources
    #         th.join(timeout=5.0)

    #         self.calc_status = False
    #         self.data_mod = None
    #         self.close()
    #         return
    
    #     # --- thread finished normally ---
    #     if "error" in result_container:
    #         self.calc_status = False
    #         raise result_container["error"]
    
    #     results, psd_info = result_container["result"]
    
    #     if "Q_mean" not in psd_info:
    #         self.calc_status = False
    #         raise KeyError("psd_info missing Q_mean")
    
    #     self.data_mod = psd_info["Q_mean"]
    #     self.x_50_mod = psd_info["x_50"]
    #     self.calc_status = True
        
    def solve(self, t_vec):
        if not np.allclose(np.asarray(t_vec), np.asarray(self.opt.t_vec)):
            raise ValueError("Adapter.solve: provided t_vec differs from opt.t_vec.")
    
        self.calc_status = True
        dump_results = getattr(self.impl, "dump_results", False)
        # 共享的取消标志：所有拷贝都应该指向它
        shared_flag = {"cancel": False}
        self.impl.cancel_flag = shared_flag
        results, psd_info = self.impl.solve_repeats(
            N=self.NC,
            base_seed=self.MC_seed,
            init_Vc=self.init_Vc,
            Vc=self.opt.Vc_init,
            V_flat=self.opt.V_flat_init,
            init_cdf_payload=self._init_cdf_payload,
            workers=1,
            psd_enable=True,
            psd_basis=self._psd_basis,
            psd_x_grid=self.opt._psd_x_grid,
            psd_Q_grid=self._psd_Q_grid,
            dump_results=dump_results,
        )
    
        if "Q_mean" not in psd_info:
            self.calc_status = False
            raise KeyError("psd_info missing Q_mean")
    
        self.data_mod = psd_info["Q_mean"]
        self.x_50_mod = psd_info["x_50"]
        self.calc_status = True
        
        
    def get_all_data(self, exp_data_path) -> tuple[np.ndarray, np.ndarray]:
        """
        Load experimental PSD data from an HDF5 file and prepare it for optimization.

        The HDF5 file is assumed to contain multiple groups (labels), each
        corresponding to a measurement time. Under each group, the following
        datasets and attributes are expected (as produced by Import_PSD_CPS_h5):

            Datasets:
                - d_agg       : aggregate diameters used for MC initialization
                - x_dis       : log-spaced diameter grid (for Q0)
                - q0_sum_log  : cumulative number-based PSD Q0(x_dis)
                - d3_cent     : original linear diameter grid (for Q3)
                - q3_sum_agg  : cumulative volume-based PSD Q3(d3_cent)

            Attributes:
                - exp_t   : experimental sampling time (float, in seconds)
                - Vc      : control volume used for this measurement (float)
                - N_bins, N_aggs_target, N_aggs_eff, cell_size, phi_s, V_mean, ...

        This method:
          1) Scans all groups in the HDF5 file and reads their exp_t.
          2) For each time in self.opt.t_vec, selects the group whose exp_t
             matches that time (within a small tolerance).
          3) According to self.opt.data_flag ('Q0' or 'Q3'), collects the
             cumulative distribution at these times into a 2D array data_exp
             with shape (Nx, Nt), where Nx is the number of x points and
             Nt = len(self.opt.t_vec).
          4) Uses the first time point's group to initialize MCPBE:
             - x_uni is set from x_dis (Q0) or d3_cent (Q3).
             - self.Vc_init is set from that group's Vc.
             - self.V_flat_init is built from that group's d_agg
               (diameters → volumes, dim=1).
             - self.init_Vc is set to False so that MCPBE uses the provided Vc.
             - self._psd_basis is set to "number" (Q0) or "volume" (Q3).
             - self._psd_x_grid is set to x_uni so MCPBE PSD is computed on
               the same x-grid as the experimental data.

        Parameters
        ----------
        exp_data_path : str
            Full path to the HDF5 file containing experimental PSD data.

        Returns
        -------
        x_uni : ndarray, shape (Nx,)
            Diameter grid on which the experimental PSD CDF is defined.
        data_exp : ndarray, shape (Nx, Nt)
            Experimental cumulative PSD data (Q0 or Q3) at the requested
            time points, ordered according to self.opt.t_vec.
        """
        opt = self.opt
        t_vec = np.asarray(opt.t_vec, dtype=float)

        # MCPBE here is assumed to be 1D
        if getattr(self.impl, "dim", None) != 1:
            raise ValueError(
                f"MCPBEAdapter.get_all_data currently assumes dim=1, "
                f"but impl.dim={getattr(self.impl, 'dim', None)!r}."
            )

        # Decide which experimental quantity to read: cumulative Q0 or Q3.
        flag = getattr(opt, "data_flag", "Q0")
        flag = str(flag).upper()
        
        if flag.startswith("Q0"):
            grid_key = "x_dis"
            data_key = "q0_sum_log"
            x_50_key = "x50_Q0"
        else:  # Q3*
            grid_key = "d3_cent"
            data_key = "q3_sum_agg"
            x_50_key = "x50_Q3"

        with h5py.File(exp_data_path, mode="r") as h5f:
            group_names = [name for name in h5f.keys()]
            if not group_names:
                raise ValueError(
                    f"No groups found in HDF5 file {exp_data_path!r}."
                )

            # ------------------------------------------------------------------
            # 1) Scan all groups and collect their exp_t
            # ------------------------------------------------------------------
            exp_t_list = []
            for name in group_names:
                grp = h5f[name]
                if "exp_t" not in grp.attrs:
                    raise KeyError(
                        f"Group {name!r} in {exp_data_path!r} has no 'exp_t' attribute."
                    )
                exp_t_list.append(float(grp.attrs["exp_t"]))
            exp_t_arr = np.asarray(exp_t_list, dtype=float)

            # ------------------------------------------------------------------
            # 2) For each time in t_vec, find the matching group by exp_t
            # ------------------------------------------------------------------
            tol = 1e-8
            group_idx_for_t: list[int] = []
            for t in t_vec:
                idx = np.where(np.isclose(exp_t_arr, t, rtol=0.0, atol=tol))[0]
                if idx.size == 0:
                    raise ValueError(
                        f"No group with exp_t matching t={t} found in {exp_data_path!r}."
                    )
                if idx.size > 1:
                    raise ValueError(
                        f"Multiple groups with exp_t ~ {t} found in {exp_data_path!r}; "
                        f"exp_t values: {exp_t_arr[idx]}."
                    )
                group_idx_for_t.append(int(idx[0]))

            # ------------------------------------------------------------------
            # 3) Use the first time point's group to define x_uni and init state
            # ------------------------------------------------------------------
            first_grp_name = group_names[group_idx_for_t[0]]
            first_grp = h5f[first_grp_name]

            # Diameter grid: x_dis (for Q0) or d3_cent (for Q3)
            if grid_key not in first_grp:
                raise KeyError(
                    f"Group {first_grp_name!r} has no dataset {grid_key!r}."
                )
            x_uni = first_grp[grid_key][...].astype(float)

            # Read control volume Vc and aggregate diameters d_agg for initialization
            if "Vc" not in first_grp.attrs:
                raise KeyError(
                    f"Group {first_grp_name!r} has no attribute 'Vc'."
                )
            Vc = float(first_grp.attrs["Vc"])

            if "d_agg" not in first_grp:
                raise KeyError(
                    f"Group {first_grp_name!r} has no dataset 'd_agg'."
                )
            d_agg_init = first_grp["d_agg"][...].astype(float)

            # Build V_flat_init for 1D MCPBE: volume from diameters
            # V = (pi/6) * d^3
            v_init = (np.pi / 6.0) * d_agg_init**3
            v_init = np.asarray(v_init, dtype=float).ravel()
            V_flat_init = np.vstack([v_init, v_init])

            # Store into optimizer for later use in solve()
            opt.Vc_init = Vc
            opt.V_flat_init = V_flat_init
            opt._psd_x_grid = x_uni        # MCPBE PSD grid = experimental grid

            # ------------------------------------------------------------------
            # 4) Collect cumulative PSD data for all requested times into data_exp
            # ------------------------------------------------------------------
            Nx = x_uni.size
            Nt = t_vec.size
            data_exp = np.zeros((Nx, Nt), dtype=float)
            V_mean_exp = np.zeros(Nt, dtype=float)
            x_50_exp = np.zeros(Nt, dtype=float)

            for it, gidx in enumerate(group_idx_for_t):
                gname = group_names[gidx]
                grp = h5f[gname]

                if data_key not in grp:
                    raise KeyError(
                        f"Group {gname!r} has no dataset {data_key!r} "
                        f"required for flag={flag!r}."
                    )
                y = grp[data_key][...].astype(float)

                if y.size != Nx:
                    raise ValueError(
                        f"Dataset size mismatch in group {gname!r}: "
                        f"expected {Nx} points (same as {grid_key}), got {y.size}."
                    )

                data_exp[:, it] = y
                
                # V_mean (experimental mean volume) from group attributes
                if "V_mean" in grp.attrs:
                    V_mean_exp[it] = float(grp.attrs["V_mean"])
                else:
                    # If missing, mark as NaN to avoid silently using 0
                    V_mean_exp[it] = np.nan
                if x_50_key in grp.attrs:
                    x_50_exp[it] = float(grp.attrs[x_50_key])
                else:
                    # If missing, mark as NaN to avoid silently using 0
                    x_50_exp[it] = np.nan
                    
        # Store V_mean_exp for later plotting
        self.V_mean_exp = V_mean_exp
        self.x_50_exp = x_50_exp

        if self.use_exp_cdf_init:
            cdf0 = np.asarray(data_exp[:, 0], dtype=float)
            total_vol_ref = float(np.sum(V_flat_init[1, :]))
            self._init_cdf_payload = {
                "x_grid": np.asarray(x_uni, dtype=float),
                "cdf": cdf0,
                "basis": self._psd_basis,
                "n_ref": int(V_flat_init.shape[1]),
                "total_vol_ref": total_vol_ref,
                "target_n": int(getattr(self.impl, "V_eff_init", 0) or 0),
            }
        else:
            self._init_cdf_payload = None

        return x_uni, data_exp
    
    def calc_delta_pop(self, x_uni_exp, data_exp) -> float:
        """Compute the mismatch between experimental PSD and MCPBE result.

        Parameters
        ----------
        x_uni_exp : ndarray or list of ndarray
            Experimental x-grid(s). For the current implementation
            (sample_num == 1), this is a single 1D array and is not used
            explicitly here, because the MCPBE PSD has already been
            computed on the same x-grid via `psd_x_grid`.
        data_exp : ndarray or list of ndarray
            Experimental PSD data. For the current implementation
            (sample_num == 1), this is a single 2D array with shape
            (Nx, Nt), where Nx is the number of x points and Nt is the
            number of time points in t_vec.

        Returns
        -------
        delta : float
            Scalar cost value computed by opt.cost_fun.

        Notes
        -----
        - The current implementation only supports the case where the
          experimental PSD is provided as a single dataset (sample_num == 1).
        - For sample_num > 1 (multiple experimental repeats), the intended
          design is that x_uni_exp and data_exp become lists of arrays,
          one per experiment. Since get_all_data has not yet been extended
          to read multiple repeats from HDF5, this branch is not implemented
          and returns 0.0 with a warning.
        """
        opt = self.opt
        sample_num = getattr(opt, "sample_num", 1)

        # Multi-repeat experimental data is not supported yet.
        if sample_num != 1:
            import warnings

            warnings.warn(
                "MCPBEAdapter.calc_delta_pop: sample_num > 1 is not implemented yet. "
                "For now, this branch returns 0.0 without using the data.",
                RuntimeWarning,
            )
            return 0.0

        # For sample_num == 1, we expect data_exp to be a single 2D array
        # with the same shape as self.data_mod: (Nx, Nt).
        if np.shape(data_exp) != np.shape(self.data_mod):
            raise ValueError(
                f"MCPBEAdapter.calc_delta_pop: shape mismatch between experimental "
                f"data {np.shape(data_exp)} and model data {np.shape(self.data_mod)}."
            )
        
        if opt.data_flag in ("Q0_X_50", "Q3_X_50"):
            delta = opt.cost_fun(self.x_50_exp, self.x_50_mod, opt.cost_flag, opt.data_flag)
        else:
            delta = opt.cost_fun(data_exp, self.data_mod, opt.cost_flag, opt.data_flag)
        return float(delta)
        
    def close(self) -> None:
        self.impl._close()
            
    # %% OPTIONAL METHOD INTERFACE      
            
    # %% INTERNAL METHOD INTERFACE  
