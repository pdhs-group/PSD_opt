# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:42:20 2024

@author: px2030
"""
import numpy as np
## Config for Optimization

config = {
    ## Use only 2D Data or 1D+2D
    'multi_flag': True,
    
    'algo_params': {
        'dim': 2,
        't_init' : np.array([3, 15, 22, 40]),
        't_vec' : np.arange(0, 11, 1, dtype=float),
        'add_noise': True,
        'smoothing': True,
        'noise_type': 'Mul',
        'noise_strength': 0.1,
        'sample_num': 5,
        'method': 'BO',
        'n_iter': 400,
        'calc_init_N': True,
        ## delta_flag = q3: use q3
        ## delta_flag = Q3: use Q3
        ## delta_flag = x_10: use x_10
        ## delta_flag = x_50: use x_50
        ## delta_flag = x_90: use x_90
        'delta_flag': 'q3',
        ##   'MSE': Mean Squared Error
        ##   'RMSE': Root Mean Squared Error
        ##   'MAE': Mean Absolute Error
        ##   'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
        'cost_func_type': 'MSE',
        },
    
    ## PBE parameters
    'pop_params': {
        'NS' : 15,
        'S' : 4,
        'BREAKRVAL' : 3,
        'BREAKFVAL' : 5,
        'aggl_crit' : 1e20,
        'process_type' : "breakage",
        'pl_v' : 2,
        'pl_q' : 1,
        'pl_P1' : 1e-4,
        'pl_P2' : 0.5,
        'COLEVAL' : 2,
        'EFFEVAL' : 1,
        'SIZEEVAL' : 1,
        'alpha_prim': np.array([1, 0.5, 1]),
        'CORR_BETA' : 150,
        },
      
    ## The diameter ratio of the primary particles can also be used as a variable
    'R_NM': 8.68e-7,
    'R_M': 8.68e-7*1,
    
    ## The error of 2d pop may be more important, so weight needs to be added
    'weight_2d': 1,
    
    'dist_scale_1': "PSD_x50_2.0E-6_v50_4.2E-18_RelSigmaV_1.5E-1.npy",
    'dist_scale_5': "PSD_x50_1.0E-5_v50_5.2E-16_RelSigmaV_1.5E-1.npy",
    'dist_scale_10': "PSD_x50_2.0E-5_v50_4.2E-15_RelSigmaV_1.5E-1.npy",
}