# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:42:20 2024

@author: px2030
"""
import numpy as np
## Config for Optimization

config = {
    'dim': 2,
    't_vec' : np.concatenate(([0.0, 0.1, 0.3, 0.6, 0.9], np.arange(1, 602, 60, dtype=float))),
    'add_noise': True,
    'smoothing': True,
    'noise_type': 'Mul',
    'noise_strength': 0.1,
    'sample_num': 5,
    'method': 'BO',
    'delta_flag': 1,
    
    'multi_flag': True,
    'n_iter': 800,
    'calc_init_N': True,
    'R_NM': 2.9e-7,
    'R_M': 2.9e-7,
    'delta_flag': 1,
    'cost_func_type': 'KL',
    'weight_2d': 1,
}