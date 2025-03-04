# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:53:09 2024

@author: Administrator
"""

import os
import runpy
# import importlib.util
import optframework.dpbe.dpbe_core as dpbe_core
import optframework.dpbe.dpbe_visualization as dpbe_visualization
import optframework.dpbe.dpbe_post as dpbe_post
import optframework.dpbe.dpbe_mag_sep as dpbe_mag_sep
from optframework.utils.func.bind_methods import bind_methods_from_module
        
class DPBESolver():
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
    **attr : dict, optional
        Additional attributes for PBE initialization.
    """
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, load_attr=True, config_path=None, disc='geo', **attr):
        # Initialize PBE core, visualization, post-processing, and magnetic separation parameters
        dpbe_core.init_pbe_params(self, dim, t_total, t_write, t_vec, disc, **attr)
        dpbe_visualization.init_visual_params(self)
        dpbe_post.init_post_params(self)
        dpbe_mag_sep.init_mag_sep_params(self)
        
        # Load the configuration file, if available
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir,"config","PBE_config.py")

        if load_attr:
            self.load_attributes(config_path)
        self.check_params()
        
    def check_params(self):
        """
        Check the validity of dPBE parameters.
        """
        pass

    def load_attributes(self, config_path):
        """
        Load attributes dynamically from a configuration file.
        
        This method dynamically loads attributes from a specified Python configuration file 
        and assigns them to the DPBESolver instance. It checks for certain key attributes like 
        `alpha_prim` to ensure they match the PBE's dimensionality.
        
        Parameters
        ----------
        config_name : str
            The name of the configuration file (without the extension).
        config_path : str
            The file path to the configuration file.
        
        Raises
        ------
        Exception
            If the length of `alpha_prim` does not match the expected dimensionality.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Warning: Config file not found at: {config_path}.")
        print(f"The dPBE is using config file at : {config_path}." )
        # Dynamically load the configuration file
        conf = runpy.run_path(config_path)
        config = conf['config']
        
        # Assign attributes from the configuration file to the DPBESolver instance
        for key, value in config.items():
            if value is not None:
                if key == "alpha_prim" and len(value) != self.dim**2:
                    raise Exception(f"The length of the array alpha_prim needs to be {self.dim**2}.")
                setattr(self, key, value)
                
        # Reset parameters, including time-related attributes
        reset_t = self.t_vec is None
        self.reset_params(reset_t=reset_t)
            
# Bind methods from different PBE-related modules to DPBESolver            
bind_methods_from_module(DPBESolver, 'optframework.dpbe.dpbe_core')
bind_methods_from_module(DPBESolver, 'optframework.dpbe.dpbe_visualization')
bind_methods_from_module(DPBESolver, 'optframework.dpbe.dpbe_post')
bind_methods_from_module(DPBESolver, 'optframework.dpbe.dpbe_mag_sep')
