# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:15:15 2024

@author: JI
"""

import sys, os
import numpy as np
from optframework.pbe import ExtruderPBESolver

if __name__ == "__main__":
    dim = 1
    NC = 3
    t_total = 601
    t_write = 100
    FILL = np.array([1,1,1]) 
    geom_length = np.array([0.022, 0.02, 0.022])
    # t_vec = np.arange(0, 601, 100, dtype=float)
    p_ex = ExtruderPBESolver(dim, NC, t_total, t_write)
    
    p_ex.set_comp_geom_params(FILL, geom_length)
    
    p_ex.get_all_comp_params(same_pbe=True)
    
    p_ex.solve_extruder()
    
    N = p_ex.N