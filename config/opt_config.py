# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:42:20 2024

@author: px2030
"""
import numpy as np
import os
## Config for Optimization

config = {
    ## Use only 2D Data or 1D+2D
    'multi_flag': True,
    
    'algo_params': {
        'dim': 2,
        't_init' : np.array([0, 1, 3, 5, 9]),
        't_vec' : np.arange(0, 151, 15, dtype=float),
        ## Sometimes there is a gap between the initial conditions calculated based on experimental data 
        ## and the real values, resulting in unavoidable errors in the first few time steps. 
        ## These errors will gradually disappear as the calculation time becomes longer. 
        ## Therefore, skipping the data of the first few time steps during optimization 
        ## may yield better results.
        'delta_t_start_step' : 1,
        'add_noise': True,
        'smoothing': True,
        'noise_type': 'Mul',
        'noise_strength': 0.1,
        'sample_num': 5,
        ## method = basinhopping
        ## method = BO: use Bayesian Optimization
        'method': 'BO',

        'n_iter': 100,
        'calc_init_N': True,
        ## delta_flag = q3: use q3
        ## delta_flag = Q3: use Q3
        ## delta_flag = x_10: use x_10
        ## delta_flag = x_50: use x_50
        ## delta_flag = x_90: use x_90
        ## 'MSE': Mean Squared Error
        ## 'RMSE': Root Mean Squared Error
        ## 'MAE': Mean Absolute Error
        ## 'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 

        'delta_flag': [#('q3','KL'), 
                       #('Q3','MSE'), 
                       ('x_50','MSE')
                       ],
        },
    
    ## PBE parameters
    'pop_params': {
        'NS' : 15,
        'S' : 4,
        'BREAKRVAL' : 4,
        'BREAKFVAL' : 5,
        ## aggl_crit: The sequence number of the particle that allows further agglomeration
        'aggl_crit' : 100,
        'process_type' : "breakage",
        'pl_v' : 1,
        'pl_P1' : 5e-4,
        'pl_P2' : 0.6,
        'pl_P3' : 2e-3,
        'pl_P4' : 0.4,
        # 'pl_P5' : 3e-4,
        # 'pl_P6' : 0.3,
        'COLEVAL' : 2,
        'EFFEVAL' : 1,
        'SIZEEVAL' : 1,
        'alpha_prim': np.array([0.5, 0.5, 0.5]),
        # 'alpha_prim': 0.5,
        'CORR_BETA' : 100,
        ## Reduce particle number desity concentration to improve calculation stability
        ## Default value = 1e14 
        'N_scale': 1e-18,
        ## When True, use distribution data simulated using MC-bond-break methods
        'USE_MC_BOND' : False,
        },
    
    ## Parameters which should be optimized
    'opt_params' : {
        'corr_agg_0': {'bounds': (-3.0, 3.0), 'log_scale': True},
        'corr_agg_1': {'bounds': (-3.0, 3.0), 'log_scale': True},
        'corr_agg_2': {'bounds': (-3.0, 3.0), 'log_scale': True},
        'pl_v': {'bounds': (0.1, 2), 'log_scale': False},
        'pl_P1': {'bounds': (-6, -2), 'log_scale': True},
        'pl_P2': {'bounds': (0.1, 0.6), 'log_scale': False},
        'pl_P3': {'bounds': (-6, -2), 'log_scale': True},
        'pl_P4': {'bounds': (0.1, 0.6), 'log_scale': False},
        # 'pl_P5': {'bounds': (-6, -1), 'log_scale': True},
        # 'pl_P6': {'bounds': (-3, 0), 'log_scale': True},

    },
    ## The diameter ratio of the primary particles can also be used as a variable
    'R_NM': 8.677468940430804e-07,
    'R_M': 8.677468940430804e-07*1,
    ## Adjust the coordinate of PBE(optional)
    'R01_0_scl': 1e-1,
    'R03_0_scl': 1e-1,
    
    ## The error of 2d pop may be more important, so weight needs to be added
    'weight_2d': 1,
    
    'dist_scale_1': "PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy",
    'dist_scale_5': "PSD_x50_1.0E-5_RelSigmaV_1.5E-1.npy",
    'dist_scale_10': "PSD_x50_2.0E-5_RelSigmaV_1.5E-1.npy",
}