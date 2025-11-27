# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:54:06 2025

@author: px2030
"""
from typing import Any, Literal
import numpy as np
from .adapters_api_basics import WriteThroughAdapter
from optframework.mcpbe import MCPBESolver

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
        self._skip.update({"role", "opt"})
        self.role = role
        self.opt = opt
        
        # Write-through attributes
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
                raise ValueError(f"Unsupported dim={dim} for DPBEAdapter alpha_prim handling.")
        
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
        return None
        
    def reset_params(self) -> None:
        self.impl._reset_params()
    
    def calc_init_from_data(self, exp_data_paths, init_flag) -> None:
        return None
            
    def calc_matrix(self, init_N) -> None:
        return None
            
    def solve(self, t_vec) -> None:
        self.init_Vc = True
        self.Vc_init = None
        self.V_flat_init = None
        results, psd_info = self.impl.solve_repeats(N=self.NC, base_seed=self.MC_seed, init_Vc=self.init_Vc, 
                                                Vc=self.Vc_init, V_flat=self.V_flat_init,
                                                workers=1, psd_enable=False,) 
                                                # psd_basis=self.psd_basis,
                                                # psd_x_grid=self.psd_x_grid, psd_Q_grid=self.psd_Q_grid)
        mu_tmp = []
        for l in range(self.NC):
            # Moments provide statistical characterization of the particle distribution
            mu_tmp.append(results[l]['moments'])
        mu_mc = np.mean(mu_tmp,axis=0)
        self.test_out_m20 = np.mean(mu_mc[2,0,:])
        
    def get_all_data(self, exp_data_path) -> tuple[np.ndarray, np.ndarray]:
        return None, None
    
    def calc_delta_pop(self, x_uni_exp, data_exp) -> float:
        return self.test_out_m20
        
    def close(self) -> None:
        self.impl._close()
            
    # %% OPTIONAL METHOD INTERFACE      
            
    # %% INTERNAL METHOD INTERFACE  
