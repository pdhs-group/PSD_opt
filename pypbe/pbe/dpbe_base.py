# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:53:09 2024

@author: Administrator
"""

import os
import json
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
            config_path = os.path.join(self.pbe.pth, "..","..","config","PBE_config.json")
        
        if not os.path.exists(config_path):
            print (f"Warning: Config file not found at: {config_path}.")
        else:
            self.load_attributes(config_path)

    def load_attributes(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for key, value in config.items():
            if value is not None:
                setattr(self.pbe, key, value)
        self.pbe.reset_params()