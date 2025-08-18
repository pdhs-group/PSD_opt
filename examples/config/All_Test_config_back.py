"""
Created on Thu Jan 18 15:42:20 2024

@author: px2030
"""
import numpy as np
import os
_config_opt_path = os.path.dirname(__file__)
config = {
    ## Use only 2D Data or 1D+2D
    'multi_flag': True, 
    'single_case': True, 
    'algo_params': {
        'dim': 2, 
        't_init': np.array([0, 0]), 
        't_vec': np.array([0, 5, 10, 15, 20, 25, 30]) * 60, 
        'delta_t_start_step': 1, 'add_noise': True, 
        'smoothing': True, 
        'noise_type': 'Mul', 
        'noise_strength': 0.1, 
        'sample_num': 1, 
        'exp_data': False, 
        'sheet_name': None, 
        'method': 'Cmaes', 
        'random_seed': 1, 
        'n_iter': 20, 
        'calc_init_N': False, 
        'USE_PSD': True, 
        'USE_PSD_R': False, 
        'R01_0': 'r0_001', 
        'R03_0': 'r0_001', 
        'R_01': 2.9e-07, 
        'R_03': 2.9e-07, 
        'R01_0_scl': 1, 
        'R03_0_scl': 1, 
        'PSD_R01': 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy', 
        'PSD_R03': 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy', 
        'weight_2d': 1, 
        'dist_type': 'q3', 
        'delta_flag': [('qx', 'MSE')], 
        'tune_storage_path': os.path.join(_config_opt_path, 'Ray_Tune'), 
        'verbose': 0, 
        'multi_jobs': False, 
        'num_jobs': 3, 
        'cpus_per_trail': 2, 
        'max_concurrent': 2
        }, 
    'pop_params': {
        'NS': 10, 
        'S': 4, 
        'SIZEEVAL': 1, 
        'COLEVAL': 1, 
        'EFFEVAL': 1, 
        'BREAKRVAL': 4, 
        'BREAKFVAL': 5, 
        'aggl_crit': 100, 
        'process_type': 'mix', 
        'V_unit': 1e-12, 
        'USE_MC_BOND': False, 
        'solver': 'ivp', 
        'CORR_BETA': 1, 
        'alpha_prim': np.array([0.01, 0.01, 0.01]), 
        'pl_v': 2, 
        'pl_P1': 0.01, 
        'pl_P2': 1, 
        'pl_P3': 0.01, 
        'pl_P4': 1, 
        'G': 80
        }, 
    'opt_params': {
        'corr_agg_0': {'bounds': (-4.0, 0.0), 'log_scale': True}, 
        'corr_agg_1': {'bounds': (-4.0, 0.0), 'log_scale': True}, 
        'corr_agg_2': {'bounds': (-4.0, 0.0), 'log_scale': True}, 
        'pl_v': {'bounds': (0.5, 2.0), 'log_scale': False}, 
        'pl_P1': {'bounds': (-5.0, -1.0), 'log_scale': True}, 
        'pl_P2': {'bounds': (0.3, 3.0), 'log_scale': False}, 
        'pl_P3': {'bounds': (-5.0, -1.0), 'log_scale': True}, 
        'pl_P4': {'bounds': (0.3, 3.0), 'log_scale': False}
        }
    }