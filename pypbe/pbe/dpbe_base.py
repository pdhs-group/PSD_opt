# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:53:09 2024

@author: Administrator
"""

import os ,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
import importlib
import inspect
# import importlib.util
import pypbe.pbe.dpbe_core as dpbe_core
import pypbe.pbe.dpbe_visualization as dpbe_visualization
import pypbe.pbe.dpbe_post as dpbe_post
import pypbe.pbe.dpbe_mag_sep as dpbe_mag_sep
        
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
            config_path = os.path.join(self.pth, "..","..","config","PBE_config.py")

        if load_attr:
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            self.load_attributes(config_name, config_path)

    def load_attributes(self, config_name, config_path):
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
        # Dynamically load the configuration file
        spec = importlib.util.spec_from_file_location(config_name, config_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)
        config = conf.config
        
        # Assign attributes from the configuration file to the DPBESolver instance
        for key, value in config.items():
            if value is not None:
                if key == "alpha_prim" and len(value) != self.dim**2:
                    raise Exception(f"The length of the array alpha_prim needs to be {self.dim**2}.")
                setattr(self, key, value)
                
        # Reset parameters, including time-related attributes
        reset_t = self.t_vec is None
        self.reset_params(reset_t=reset_t)
    
# Function to bind all methods from a module to a class    
def bind_methods_from_module(cls, module_name):
    """
    Bind all functions from a specified module to the given class.

    Parameters
    ----------
    cls : type
        The class to which the methods will be bound.
    module_name : str
        The name of the module from which the methods will be imported.
    """
    # Import the specified module
    module = importlib.import_module(module_name)

    # # Iterate over all functions in the module and bind them to the class
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Bind functions statically to classes, with function names as method names
        setattr(cls, name, func)

# Function to bind selected methods from a module to a class        
def bind_selected_methods_from_module(cls, module_name, methods_to_bind):
    """
    Bind selected functions from a specified module to the given class.
    
    Parameters
    ----------
    cls : type
        The class to which the methods will be bound.
    module_name : str
        The name of the module from which the methods will be imported.
    methods_to_bind : list of str
        A list of method names to bind to the class.
    """
    # Import the specified module
    module = importlib.import_module(module_name)
    
    # Bind only the selected methods to the class
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name in methods_to_bind:
            setattr(cls, name, func)
    
# Function to unbind methods from a class
def unbind_methods_from_class(cls, methods_to_remove):
    """
    Unbind specific methods from the given class.
    
    Parameters
    ----------
    cls : type
        The class from which the methods will be unbound.
    methods_to_remove : list of str
        A list of method names to remove from the class.
    """
    for method_name in methods_to_remove:
        if hasattr(cls, method_name):
            delattr(cls, method_name)  
            
# Bind methods from different PBE-related modules to DPBESolver            
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_core')
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_visualization')
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_post')
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_mag_sep')
