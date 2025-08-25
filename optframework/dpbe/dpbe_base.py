# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:53:09 2024

@author: Administrator
"""

import os
from pathlib import Path
import runpy
import math
import numpy as np
from optframework.base.base_solver import BaseSolver
from optframework.dpbe.dpbe_core import DPBECore
from optframework.dpbe.dpbe_visualization import DPBEVisual
from optframework.dpbe.dpbe_post import DPBEPost

        
class DPBESolver(BaseSolver):
    """
    A discrete population balance equation (dPBE) solver for 1D and 2D particle systems.
    
    This class is responsible for initializing ,solving and post-processing population balance equations (PBE) 
    in 1D or 2D, depending on the specified dimensionality. It integrates core PBE functionality 
    with visualization, post-processing, and magnetic separation capabilities by dynamically 
    binding methods from external modules.
    
    Note
    ----
    This class uses the `bind_methods_from_module` function to dynamically bind methods from external
    modules. Some methods in this class are not explicitly defined here, but instead are imported from
    other files. To fully understand or modify those methods, please refer to the corresponding external
    file.
    
    Parameters
    ----------
    dim : int
        The dimensionality of the PBE problem (1 for 1D, 2 for 2D).
    t_total : int, optional
        The total process time in second. Defaults to 601.
    t_write : int, optional
        The frequency (per second) for writing output data. Defaults to 100.
    t_vec : array-like, optional
        A time vector directly specifying output time points for the simulation.
    load_attr : bool, optional
        If True, loads attributes from a configuration file. Defaults to True.
    config_path : str, optional
        The file path to the configuration file. If None, the default config path is used.
    disc : str, optional
        The discretization scheme to use for the PBE. Defaults to 'geo'.
    attr : dict, optional
        Additional attributes for PBE initialization.
    """
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, load_attr=True, config_path=None, disc='geo', **attr):
        # Check if given dimension and discretization is valid
        if not (dim in [1,2,3] and disc in ['geo','uni']):
            print('Given dimension and/or discretization are not valid. Exiting..')
            
        self._init_base_parameters(dim, t_total, t_write, t_vec)
        
        self.disc = disc                      # 'geo': geometric grid, 'uni': uniform grid
        self.NS = 12                          # Grid parameter [-]
        self.S = 2                            # Geometric grid ratio (V_e[i] = S*V_e[i-1])    
                                              # In a geometric grid, volumes are calculated as midpoints between volume edges (V_e).
                                              # Therefore, when using a geo-grid, the specified value here corresponds to the radius
                                              # of the left edge of the grid, V_e[1]. The actual primary particle size is given by 
                                              # R[1] = ((1+S)/2)**(1/3)*R01  
        self.solve_algo = "ivp"                   # "ivp": use integrate.solve_ivp
                                              # "radau": use RK.radau_ii_a, only for debug, not recommended  
        self.aggl_crit = 1e3                  # The upper volume limit for agglomeration (grid-based). Compared to SIZEEVAL, 
        ## MATERIAL parameters:
        # NOTE: component 3 is defined as the magnetic component (both in 2D and 3D case)
        self.R01 = 2.9e-7                     # Basic radius component 1 [m] - NM1 ()
        self.R02 = 2.9e-7                     # Basic radius component 2 [m] - NM2
        self.R03 = 2.9e-7                     # Basic radius component 3 [m] - M3 
        
        ## GENERAL constants
        self.KT = 1.38*(10**-23)*293          # k*T in [J]
        self.MU0 = 4*math.pi*10**-7           # Permeability constant vacuum [N/A²]
        self.EPS0 = 8.854*10**-12             # Permettivity constant vacuum [F/m]
        self.EPSR = 80                        # Permettivity material factor [-]
        self.E = 1.602*10**-19                # Electron charge [C]    
        self.NA = 6.022*10**23                # Avogadro number [1/mol]
        self.MU_W = 10**-3                    # Viscosity water [Pa*s]
        self.EPS = self.EPSR*self.EPS0
        
        ## EXPERIMENTAL / PROCESS parameters:
        self.I = 1e-3*1e3                     # Ionic strength [mol/m³] - CARE: Standard unit is mol/L

        self.JIT_FM = True                    # Define wheter or not the FM calculation should be precompiled
        self.JIT_BF = True                    # Define wheter or not the BF calculation should be precompiled
        # Initialize **attr
        for key, value in attr.items():
            setattr(self, key, value)
            
            
        # Initialize PBE core, visualization, post-processing
        self.core = DPBECore(self)
        self.post = DPBEPost(self)
        self.visualization = DPBEVisual(self)
        
        # Load the configuration file, if available
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir,"config","PBE_config.py")

        if load_attr:
            self.load_attributes(config_path)
        self._check_params()
        self._reset_params()
