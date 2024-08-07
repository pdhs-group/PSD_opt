# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:53:09 2024

@author: Administrator
"""

import os
import importlib.util
from .dpbe_core import Population
from .pbe_visualization import PBEVisualization
from .dpbe_post import PBEPost
from .dpbe_mag_sep import MagSep

class DPBESolver():
    def __init__(self, dim, t_total=601, t_write=100, config_path=None, disc='geo', **attr):
        self.pbe = Population(dim, t_total, t_write, disc, **attr)
        self.post = PBEPost(self.pbe)
        self.visual = PBEVisualization(self.pbe, self.post)
        self.mag_sep = MagSep(self.pbe)
        
        if config_path is None:
            config_path = os.path.join(self.pbe.pth, "..","..","config","PBE_config.py")
            config_name = "PBE_config"
        if not os.path.exists(config_path):
            print (f"Warning: Config file not found at: {config_path}.")
        else:
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
                if key == "alpha_prim" and len(value) != self.pbe.dim**2:
                    raise Exception(f"The length of the array alpha_prim needs to be {self.pbe.dim**2}.")
                setattr(self.pbe, key, value)
        self.pbe.reset_params()