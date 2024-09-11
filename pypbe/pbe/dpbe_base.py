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
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, load_attr=True, config_path=None, disc='geo', **attr):
        dpbe_core.init_pbe_params(self, dim, t_total, t_write, t_vec, disc, **attr)
        dpbe_visualization.init_visual_params(self)
        dpbe_post.init_post_params(self)
        dpbe_mag_sep.init_mag_sep_params(self)
        
        if config_path is None:
            config_path = os.path.join(self.pth, "..","..","config","PBE_config.py")
            config_name = "PBE_config"
        if not os.path.exists(config_path):
            print (f"Warning: Config file not found at: {config_path}.")
        elif load_attr:
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            self.load_attributes(config_name, config_path)

    def load_attributes(self, config_name, config_path):
        ## Dynamically loading config.py
        spec = importlib.util.spec_from_file_location(config_name, config_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)
        config = conf.config
        
        for key, value in config.items():
            if value is not None:
                if key == "alpha_prim" and len(value) != self.dim**2:
                    raise Exception(f"The length of the array alpha_prim needs to be {self.pbe.dim**2}.")
                setattr(self, key, value)
        self.reset_params(reset_t=True)
    
def bind_methods_from_module(cls, module_name):
    module = importlib.import_module(module_name)

    # Iterate over all functions in the module
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Bind functions statically to classes, with function names as method names
        setattr(cls, name, func)
        
def bind_selected_methods_from_module(cls, module_name, methods_to_bind):
    module = importlib.import_module(module_name)

    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name in methods_to_bind:
            setattr(cls, name, func)
            
def unbind_methods_from_class(cls, methods_to_remove):
    for method_name in methods_to_remove:
        if hasattr(cls, method_name):
            delattr(cls, method_name)  
            
            
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_core')
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_visualization')
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_post')
bind_methods_from_module(DPBESolver, 'pypbe.pbe.dpbe_mag_sep')
