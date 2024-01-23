# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:42:20 2024

@author: px2030
"""
import numpy as np
## Config for Optimization

config = {
    'dim': 2,
    't_init' : np.array([0.1, 0.3, 0.6, 0.9]),
    't_vec' : np.arange(0, 601, 60, dtype=float),
    'add_noise': False,
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
    
    'dist_scale_1': "PSD_x50_2.0E-6_v50_4.2E-18_RelSigmaV_1.5E-1.npy",
    'dist_scale_5': "PSD_x50_1.0E-5_v50_5.2E-16_RelSigmaV_1.5E-1.npy",
    'dist_scale_10': "PSD_x50_2.0E-5_v50_4.2E-15_RelSigmaV_1.5E-1.npy",
}