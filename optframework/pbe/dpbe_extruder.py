# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:01:34 2024

@author: px2030
"""
import os ,sys
from pathlib import Path
import numpy as np
import math
import scipy.integrate as integrate
from optframework.pbe import DPBESolver
import optframework.utils.func.jit_extruder as jit_rhs
import optframework.utils.func.func_math as func_math

class ExtruderPBESolver():
    def __init__(self, dim, NC, t_total=601, t_write=100, t_vec=None, 
                 load_attr=True, disc='geo', **attr):
        self.work_dir = Path(os.getcwd()).resolve()
        # Geometric parameters for extruder
        self.NC = NC
        self.screwspeed = 2                         # Screwspeed [1/s]
        self.geom_diameter = 0.011                  # Diameter of the screw [m]
        self.geom_depth = 0.002385                  # Depth of the screw [m]
        self.Vdot = 1.47e-7                         # Axial volume flow rate [m³/s]
        # self.FILL = [1,1,1]                         # Fill ratio
        # self.geom_length = [0.022, 0.02, 0.022]     # Measurements for calibration
        self.cs_area = 8.75e-5                      # Cross sectional area the extruder
        self.int_method = 'Radau'                   # Integration method used in solve_extruder
        
        self.dim = dim
        self.disc = disc
        self.p = DPBESolver(dim, t_total, t_write, t_vec, False, None, disc, **attr)
        ## 1d-dPBE parameter/attribute names required for calculating Extruder
        if dim == 1:
            self.extruder_attrs = [
            "N", "V", "V_e", "F_M", "B_R",
            "int_B_F", "intx_B_F", "process_type", "aggl_crit_id"
            ]
        elif dim == 2:
            self.extruder_attrs = [
            "N", "V", "V_e1", "V_e3", "F_M", "B_R",
            "int_B_F", "intx_B_F", "inty_B_F", "process_type", "aggl_crit_id"
            ]
        # Reset the time vector if t_vec is None
        if t_vec is None:
            self.t_vec = np.arange(0, t_total, t_write, dtype=float)
        else: self.t_vec = t_vec
            
        # Set the number of time steps based on the time vector
        if self.t_vec is not None:
            self.t_num = len(self.t_vec)
            
    def set_comp_geom_params(self, fill_rate, geom_length):
        if len(fill_rate) != self.NC or len(geom_length) != self.NC:
            raise ValueError("The length of fill_rate or geom_length does not match the number of components.")
        
        self.fill_rate = fill_rate    # Fill ratio
        self.geom_length = geom_length   # Measurements for calibration
        self.VC = self.geom_length*self.cs_area
        self.V_flow = self.Vdot/(self.fill_rate*self.geom_length)
        
    def get_all_comp_params(self, config_paths=None, same_pbe=False, N_feed=None):
        ## For N in Extruder, the order of dimension is like [NC, NS..., t]     
        if config_paths is None:
            config_paths = [
                os.path.join(self.work_dir, "config", f"Extruder{i}_config.py")
                for i in range(self.NC)
            ]

        if len(config_paths) != self.NC:
            raise ValueError("The number of config data for dPBEs does not match the number of components.")
        
        for i, config_path in enumerate(config_paths):
            if same_pbe and i > 0:
                pass
            else:
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Warning: Config file not found at: {config_path}.")
                print(f"The dPBE-Extruder simulation is using config file at : {config_path}")
                config_path = config_paths[0] if same_pbe else config_paths[i]
                self.p.load_attributes(config_path)
                self.p.full_init(calc_alpha=False)
            if i == 0:
                ## Initialize the first dPBE to get the array dimensions
                self.init_comp_params()
                if N_feed is None:
                    self.N[0, ...] = self.p.N
                else:
                    self.N[0, ..., 0] = N_feed
            self.get_one_comp_params(i)
                
    def init_comp_params(self):
        self.NS = self.p.NS
        for attr in self.extruder_attrs:
            value = getattr(self.p, attr)
            ## For attributes with value type
            if isinstance(value, np.ndarray):
                shape = value.shape
                new_shape = (self.NC, *shape) if attr != "N" else (self.NC+1, *shape)
                dtype = value.dtype
                setattr(self, attr, np.zeros(new_shape, dtype=dtype))
            ## For attributes with other type(bool, string...)
            else:
                setattr(self, attr, [None] * (self.NC + 1 if attr == "N" else self.NC))
            
    def get_one_comp_params(self, comp_idx):
        if self.p.NS != self.NS:
            raise ValueError("NS in each dPBE must be same.")
        for attr in self.extruder_attrs:
            target_attr = getattr(self, attr)
            source_attr = getattr(self.p, attr)
            if isinstance(source_attr, np.ndarray):
                if attr == "N":
                    target_attr[comp_idx + 1, ...] = source_attr
                else:
                    target_attr[comp_idx, ...] = source_attr
            else:
                target_attr[comp_idx] = source_attr
            if attr == "aggl_crit_id": 
                target_attr = func_math.ensure_integer_array(target_attr)
            
    def solve_extruder(self, t_vec=None):
        if t_vec is None:
            t_vec = self.t_vec
            t_max = self.t_vec[-1]
        else:
            t_max = t_vec[-1]
            
        if self.dim == 1:
            # Define right-hand-side function depending on discretization
            if self.disc == 'geo':
                rhs = jit_rhs.get_dNdt_1d_geo_extruder
                args=(self.NS,self.V,self.V_e,self.F_M,self.B_R,self.int_B_F,
                      self.intx_B_F,self.process_type,self.aggl_crit_id, self.NC, self.V_flow)
            with np.errstate(divide='raise', over='raise',invalid='raise'):
                try:
                    self.RES = integrate.solve_ivp(rhs, 
                                                    [0, t_max], 
                                                    np.reshape(self.N[:,:,0],-1), t_eval=t_vec,
                                                    args=args,
                                                    ## If `rtol` is set too small, it may cause the results to diverge, 
                                                    ## leading to the termination of the calculation.
                                                    method=self.int_method,first_step=0.1,rtol=1e-1)
                    
                    # Reshape and save result to N and t_vec
                    t_vec = self.RES.t
                    y_evaluated = self.RES.y.reshape((self.NC+1,self.NS,len(t_vec)))
                    status = True if self.RES.status == 0 else False
                except (FloatingPointError, ValueError) as e:
                    print(f"Exception encountered: {e}")
                    y_evaluated = -np.ones((self.NC+1,self.NS,len(t_vec)))
                    status = False  
                    
        if self.dim == 2:
            # Define right-hand-side function depending on discretization
            if self.disc == 'geo':
                rhs = jit_rhs.get_dNdt_2d_geo_extruder
                args=(self.NS,self.V,self.V_e1,self.V_e3,self.F_M,self.B_R,self.int_B_F,
                      self.intx_B_F,self.inty_B_F,self.process_type,self.aggl_crit_id, self.NC, self.V_flow)
            with np.errstate(divide='raise', over='raise',invalid='raise'):
                try:
                    self.RES = integrate.solve_ivp(rhs, 
                                                    [0, t_max], 
                                                    np.reshape(self.N[:,:,:,0],-1), t_eval=t_vec,
                                                    args=args,
                                                    ## If `rtol` is set too small, it may cause the results to diverge, 
                                                    ## leading to the termination of the calculation.
                                                    method=self.int_method,first_step=0.1,rtol=1e-1)
                    
                    # Reshape and save result to N and t_vec
                    t_vec = self.RES.t
                    y_evaluated = self.RES.y.reshape((self.NC+1,self.NS,self.NS,len(t_vec)))
                    status = True if self.RES.status == 0 else False
                except (FloatingPointError, ValueError) as e:
                    print(f"Exception encountered: {e}")
                    y_evaluated = -np.ones((self.NC+1,self.NS,self.NS,len(t_vec)))
                    status = False  
                    
        # Monitor whether integration are completed  
        self.t_vec = t_vec 
        # self.N = y_evaluated / eva_N_scale
        self.N = y_evaluated
        self.calc_status = status   
        if not self.calc_status:
            print('Warning: The integral failed to converge!')