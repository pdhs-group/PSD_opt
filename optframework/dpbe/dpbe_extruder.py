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
from optframework.dpbe import DPBESolver
import optframework.utils.func.jit_extruder_rhs as jit_rhs
import optframework.utils.func.func_math as func_math

class ExtruderPBESolver():
    def __init__(self, dim, NC, t_total=601, t_write=100, t_vec=None, 
                 load_attr=True, disc='geo', **attr):
        self.work_dir = Path(os.getcwd()).resolve()
        # Geometric parameters for extruder
        self.NC = NC
        # self.screwspeed = 2                         # Screwspeed [1/s]
        # self.geom_diameter = 0.011                  # Diameter of the screw [m]
        # self.geom_depth = 0.002385                  # Depth of the screw [m]
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
        self.V_flow = self.Vdot/(self.fill_rate*self.VC)
        
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
                self.p.V_unit = self.fill_rate[i]*self.VC[i]
                ## After V_unit is modified, only the initial N and agglomeration rate F_M need to be recalculated
                self.p.init_N()
                self.p.calc_F_M()
            else:
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Warning: Config file not found at: {config_path}.")
                print(f"The dPBE-Extruder simulation is using config file at : {config_path}")
                config_path = config_paths[i]
                self.p.load_attributes(config_path)
                ## V_unit must be the same as the geometric volume/capacity of the corresponding compartment.
                self.p.V_unit = self.fill_rate[i]*self.VC[i]
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
                target_attr = func_math.ensure_integer_array(source_attr)

    def get_all_comp_params_from_dict(self, params_dict=None, same_pbe=False, N_feed=None,
                                      new_geo=None, new_global=None, new_local=None):
        """
        Load component parameters from a dictionary instead of config files.
        
        The input dictionary should contain two sub-dictionaries:
          - "geom_params": contains geometric parameters (e.g., fill_rate, geom_length)
          - "pbe_params": contains PBE parameters, with a sub-dictionary "global" for parameters common to all 
               components and additional sub-dictionaries "local_0", "local_1", ... for each component.
        
        Additionally, three extra input variables can be provided:
          - new_geo: if provided (a dictionary), updates the "geom_params" entries.
          - new_global: if provided (a dictionary), updates the "global" sub-dictionary in "pbe_params".
          - new_local: if provided (a dictionary), it should contain keys (e.g., "local_0", "local_1", etc.)
                       whose corresponding values are dictionaries to update the existing local parameters.
        
        The method performs the following:
          1. Sets geometric parameters (similar to set_comp_geom_params) using the (possibly updated) geom_params.
          2. For each component, it updates the PBE parameters:
             - If same_pbe is True and i > 0, then only update V_unit and recalculate minimal parameters.
             - Otherwise, load the local parameters from the corresponding "local_i" dictionary (after updating with new_local if provided) into self.p.
               The global parameters (updated by new_global if provided) are applied first.
          3. For the first component (i == 0), it calls init_comp_params to initialize arrays.
          4. Then, for each component, get_one_comp_params(i) is called to extract that component's parameters.
        
        Parameters
        ----------
        params_dict : dict
            Dictionary with keys "geom_params" and "pbe_params". "pbe_params" must contain a sub-dictionary
            "global" and local sub-dictionaries named "local_0", "local_1", ..., "local_{NC-1}".
        same_pbe : bool, optional
            If True, the same PBE is used for all components except that V_unit is updated; default is False.
        N_feed : array_like, optional
            If provided, the feed number is set in the first component's N.
        new_geo : dict, optional
            A dictionary of updates for geometric parameters. If provided, updates the contents of params_dict["geom_params"].
        new_global : dict, optional
            A dictionary of updates for global PBE parameters. If provided, updates the contents of params_dict["pbe_params"]["global"].
        new_local : dict, optional
            A dictionary of updates for local PBE parameters. Expected keys are like "local_0", "local_1", etc.
            For each key present, the corresponding dictionary will update the local parameters stored in params_dict["pbe_params"].
        
        Raises
        ------
        ValueError
            If the lengths of fill_rate or geom_length do not match the number of components (NC), or if any required key is missing.
        Exception
            If the length of 'alpha_prim' (if provided) does not match self.dim**2.
        
        Returns
        -------
        None
            Updates the class attributes (especially the large matrices containing component parameters) in-place.
        """
        if params_dict is None:
            params_dict = self.params_dict
        else:
            self.params_dict = params_dict.copy()
        # First, process the geometric parameters (similar to set_comp_geom_params)
        if "geom_params" not in params_dict:
            raise ValueError("Dictionary must contain key 'geom_params'.")
        
        geom_params = params_dict["geom_params"]
        if new_geo is not None:
            # Merge new_geo dictionary: new values override existing ones.
            geom_params.update(new_geo)
        fill_rate = geom_params["fill_rate"]
        geom_length = geom_params["geom_length"]
        if len(fill_rate) != self.NC or len(geom_length) != self.NC:
            raise ValueError("The length of fill_rate or geom_length does not match the number of components.")
        
        # Set geometric parameters and compute VC and V_flow as in set_comp_geom_params.
        self.fill_rate = fill_rate         # Fill ratio
        self.geom_length = geom_length     # Calibration measurements
        self.VC = self.geom_length * self.cs_area
        self.V_flow = self.Vdot / (self.fill_rate * self.VC)
        
        # Next, process PBE parameters from the dictionary
        if "pbe_params" not in params_dict:
            raise ValueError("Dictionary must contain key 'pbe_params'.")
        
        pbe_params = params_dict["pbe_params"]
        
        # First, apply global parameters to self.p.
        if "global" not in pbe_params:
            raise ValueError("The 'pbe_params' dictionary must contain a 'global' sub-dictionary.")
        global_params = pbe_params["global"]
        if new_global is not None:
            global_params.update(new_global)
        for key, value in global_params.items():
            if value is not None:
                # For key 'alpha_prim', check that its length matches dim**2.
                if key == "alpha_prim" and len(value) != self.dim**2:
                    raise Exception(f"The length of the array alpha_prim must be {self.dim**2}.")
                setattr(self.p, key, value)
        
        # Process each component's parameters.
        # Expect local parameters in keys "local_0", "local_1", ... , "local_{NC-1}"
        num_locals = sum(1 for key in pbe_params if key.startswith("local_"))
        if num_locals != self.NC:
            raise ValueError(f"Expected {self.NC} local entries, but found {num_locals}.")
        for i in range(self.NC):
            local_key = f"local_{i}"
            # If new_local is provided and contains local_key, update the local dictionary.
            if new_local is not None and local_key in new_local:
                if local_key in pbe_params:
                    pbe_params[local_key].update(new_local[local_key])
                else:
                    print(f"[Warning] Local key '{local_key}' not found in pbe_params. Skipping update for component {i}.")
            if same_pbe and i > 0:
                # If using the same pbe for all components, only update V_unit and recalc minimal parameters.
                self.p.V_unit = self.fill_rate[i] * self.VC[i]
                # self.p.init_N()
                self.p.calc_F_M()
            else:
                if local_key not in pbe_params:
                    raise ValueError(f"Missing local parameters for component {i} (expected key '{local_key}').")
                local_params = pbe_params[local_key]
                # Print a message indicating that parameters for this component are being used.
                # print(f"[Info] Using provided PBE parameters for component {i} from key '{local_key}'.")
                for key, value in local_params.items():
                    if value is not None:
                        if key == "alpha_prim" and len(value) != self.dim**2:
                            raise Exception(f"The length of the array alpha_prim must be {self.dim**2} for component {i}.")
                        setattr(self.p, key, value)
                self.p.V_unit = self.fill_rate[i] * self.VC[i]
                if N_feed is None:
                    self.p.full_init(calc_alpha=False)
                else:
                    self.p.full_init(calc_alpha=False, init_N=False)
            # For the first component, initialize the global arrays.
            if i == 0:
                self.init_comp_params()
                if N_feed is None:
                    self.N[0, ...] = self.p.N
                else:
                    self.N[0, ..., 0] = N_feed
            # Finally, extract and store the component parameters.
            self.get_one_comp_params(i)

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
            
    # def return_distribution(self, comp='all', t=0, N=None, flag='all', rel_q=False, q_type='q3'):
    #     if N is None:
    #         N = self.N[-1, ...]
    #     return self.p.return_distribution(comp, t, N, flag, rel_q, q_type)
    