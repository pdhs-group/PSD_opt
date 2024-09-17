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
        't_init' : np.array([0, 0]),
        't_vec' : np.arange(0, 3601, 100, dtype=float),
        # 't_vec' : np.array([0, 300, 600, 900, 1200, 1500, 2100, 2700]),
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
        'exp_data' : False, 
        'sheet_name' : None, 
        ## method = HEBO: Heteroscedastic Evolutionary Bayesian Optimization
        ## method = GP: Sampler using Gaussian process-based Bayesian optimization.
        ## method = TPE: Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
        ## method = Cmaes: A sampler using cmaes as the backend.
        ## method = NSGA: Multi-objective sampler using the NSGA-III(Nondominated Sorting Genetic Algorithm III) algorithm.
        ## method = QMC: A Quasi Monte Carlo Sampler that generates low-discrepancy sequences.    
        'method': 'HEBO',
        'n_iter': 100,
        ## Initialize PBE using psd data(False) or 
        ## with the help of first few time points of experimental data(True)
        'calc_init_N': False,
        ## Setting the basic parameters of the smallest particles, 
        ## whether they are read from PSD data or not
        'USE_PSD' : True,
        'R01_0' : 'r0_001',
        'R03_0' : 'r0_001',
        ## The diameter ratio of the primary particles can also be used as a variable
        'R_01': 8.677468940430804e-07,
        'R_03': 8.677468940430804e-07*1,
        ## Adjust the coordinate of PBE(optional)
        'R01_0_scl': 1,
        'R03_0_scl': 1,
        'PSD_R01' : 'PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy',
        'PSD_R03' : 'PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy',
        ## The error of 2d pop may be more important, so weight needs to be added
        'weight_2d': 1,
        ## delta_flag = q3: use q3
        ## delta_flag = Q3: use Q3
        ## delta_flag = x_10: use x_10
        ## delta_flag = x_50: use x_50
        ## delta_flag = x_90: use x_90
        ## 'MSE': Mean Squared Error
        ## 'RMSE': Root Mean Squared Error
        ## 'MAE': Mean Absolute Error
        ## 'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 

        'delta_flag': [('q3','MSE'), 
                       # ('Q3','KL'), 
                       #('x_50','MSE')
                       ],
        'tune_storage_path': r'C:\Users\px2030\Code\Ray_Tune',
        ## When use_bundles is True, multiple Tune will be run locally at the same time. 
        ## The psd-data must also be multiple! Multiple here means data with different conditions 
        ## rather than the same data in different dimensions.
        'multi_jobs': True,
        'num_jobs': 3,
        'cpus_per_trail': 4,
        'max_concurrent': 8
        },
    
    ## PBE parameters
    'pop_params': {
        'NS' : 8,
        'S' : 4,
        'BREAKRVAL' : 4,
        'BREAKFVAL' : 5,
        ## aggl_crit: The sequence number of the particle that allows further agglomeration
        'aggl_crit' : 100,
        'process_type' : "mix",
        'pl_v' : 2,
        'pl_P1' : 1e1,
        'pl_P2' : 2,
        # 'pl_P3' : 1e1,
        'pl_P4' : 2,
        'COLEVAL' : 2,
        'EFFEVAL' : 1,
        'SIZEEVAL' : 1,
        'alpha_prim': np.array([1, 1, 1]),
        # 'alpha_prim': 0.5,
        'CORR_BETA' : 100,
        ## The "average volume" of the two elemental particles in the system.
        ## Used to scale the particle volume in calculation of the breakage rate.
        'V1_mean' : 1e-15,
        'V3_mean' : 1e-15,
        ## Reduce particle number desity concentration to improve calculation stability
        ## Default value = 1e14 
        'V_unit': 1e-15,
        ## When True, use distribution data simulated using MC-bond-break methods
        'USE_MC_BOND' : False,
        'solver' : "ivp",
        },
    
    ## Parameters which should be optimized
    'opt_params' : {
        'corr_agg_0': {'bounds': (-4.0, -1.0), 'log_scale': True},
        'corr_agg_1': {'bounds': (-4.0, -1.0), 'log_scale': True},
        'corr_agg_2': {'bounds': (-4.0, -1.0), 'log_scale': True},
        'pl_v': {'bounds': (0.5, 2.0), 'log_scale': False},
        'pl_P1': {'bounds': (-4.0, 0.0), 'log_scale': True},
        'pl_P2': {'bounds': (0.5, 3.0), 'log_scale': False},
        'pl_P3': {'bounds': (-4.0, 0.0), 'log_scale': True},
        'pl_P4': {'bounds': (0.5, 3.0), 'log_scale': False},
    },

}