# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:42:20 2024

@author: px2030
"""
import numpy as np
import os
## Config for Optimization
# _config_opt_path = os.environ.get('TMP_PATH')
_config_opt_path = os.path.dirname(__file__)
config = {
    # Use only 2D Data or 1D+2D
    'multi_flag': True, 
    # Input only one/one set of PSD data
    'single_case': True, 
    ## Core parameters for optimization
    'algo_params': {
        'adapter': "mcpbe",
        # The dimensionality of the PBE
        'dim': 1, 
        # Initial time points for simulation. 
        # These values are used to initialize N in dPBE wenn calc_init_N is True.
        # Note: The first value in t_init must be zero.
        't_init': np.array([0, 0]), 
        # Time vector for the entire simulation, specifying the time points at which 
        # calculations are performed.
        't_vec' : np.arange(0, 21, 5, dtype=float),
        # Specifies the number of initial time steps to skip during optimization, 
        # often useful to avoid the impact of initialization errors.
        'delta_t_start_step': 1, 
        # Whether to add noise to the generated data.
        'add_noise': True, 
        # Whether to apply smoothing to the simulated data, usually performed using 
        # kernel density estimation (KDE).
        # Note: When NS is large, KDE may tend to underestimate the distribution function.
        'smoothing': True, 
        # Type of noise to add to the data. Options include:
        # - 'Gaus': Gaussian noise
        # - 'Uni': Uniform noise
        # - 'Po': Poisson noise
        # - 'Mul': Multiplicative noise
        'noise_type': 'Mul', 
        # The strength of the added noise.
        'noise_strength': 0.1, 
        # Number of experimental or synthetic data samples used during the optimization process.
        'sample_num': 1, 
        # Whether to use experimental data (True) or synthetic data (False) during optimization.
        'exp_data': False, 
        # Name of the sheet in the experimental data file (if applicable).
        'sheet_name': None, 
        # Optimization method to use. Options include:
        # - 'GP': Gaussian Process-based Bayesian Optimization
        # - 'TPE': Tree-structured Parzen Estimator
        # - 'Cmaes': Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        # - 'NSGA': Nondominated Sorting Genetic Algorithm (NSGA-III)
        # - 'QMC': Quasi Monte Carlo sampling
        'method': 'Cmaes', 
        # Seed for reproducible results, int or None. 
        # This value will change global random states for numpy and torch on 
        # initalization and loading from checkpoint.
        'random_seed': 1, 
        # Number of iterations for the optimization process.
        'n_iter': 20, 
        # Whether to initialize the PBE using the first few time points of experimental data (True).
        'calc_init_N': False, 
        # Whether to use R01_0 and R03_0 below to get the particle size in the PSD data as 
        # the starting coordinates for PBE. If False, the values ​​of R_01 and R_03 are used.
        'USE_PSD_R': False,
        # Radius of primary particles corresponding to the x% position (Q3) in the PSD data.
        # - 'r0_001': Corresponding to the 1% position
        # - 'r0_005': Corresponding to the 5% position
        # - 'r0_01': Corresponding to the 10% position
        'R01_0': 'r0_001', 
        'R03_0': 'r0_001',
        # Directly define the radius of the primary particles, 
        # the choice of R01_0 or R_01 depending on the value of USE_PSD and USE_PSD_R.
        'R_01': 2.9e-07, 
        'R_03': 2.9e-07, 
        # Scaling factor for the primary particle radius.
        'R01_0_scl': 1, 
        'R03_0_scl': 1, 
        # The filename of the PSD file used for initialization. 
        # Note: it must not be moved into the pop_params sub-dictionary, otherwise it may cause calculation errors.
        'DIST1_name': 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy', 
        'DIST3_name': 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy', 
        # Weight applied to the error (delta) of 2D particle populations, giving it 
        # more importance during optimization.
        'weight_2d': 1, 
        # Determine on what the PSD returned after the PBE calculation is based.
        # - 'q0': Number-based PSD (weight = N, i.e., V^0 × N)
        # - 'q3': Volume-based PSD (weight = V * N, i.e., V^1 × N)
        # - 'q6': Square-volume PSD (weight = V^2 * N)
        'dist_type': 'qx', 
        # Specify the objectives and calculation method in the optimization process.
        # - 'qx': density distribution
        # - 'Qx': Cumulative distribution
        # - 'x_10': Particle size at 10% of cumulative distribution
        # - 'x_50': Particle size at 50% of cumulative distribution
        # - 'x_90': Particle size at 90% of cumulative distribution
        # Cost function options include:
        # - 'MSE': Mean Squared Error
        # - 'RMSE': Root Mean Squared Error
        # - 'MAE': Mean Absolute Error
        # - 'KL': Kullback-Leibler divergence (only compatible with q3 and Q3)
        'delta_flag': [# ('qx','MSE'), 
                       ('Qx','MSE'), 
                       # ('x_50','MSE'),
                       # ('y_weibull','MSE'),
                       ],
        # Path to store Ray Tune optimization infomation.
        'tune_storage_path': os.path.join(_config_opt_path, 'Ray_Tune'), 
        # Whether to print information during the Ray Tune run.
        # - 0: Silent mode.
        # - 1': Only main information.
        # - 2: Detailed information.
        'verbose': 1, 
        # Whether to run multiple optimization tasks (Tune jobs) concurrently. 
        # If True, multiple PSD datasets should be provided.
        'multi_jobs': False, 
        # Number of parallel optimization jobs to run.
        'num_jobs': 3, 
        # Number of CPU cores allocated to each optimization trial.
        'cpus_per_trail': 2, 
        # Maximum number of trials that can be run concurrently.
        'max_concurrent': 2
        }, 
    
    ## PBE parameters
    ## For a detailed explanation of the PBE parameters, please refer to the `PBE_config.py` file.
    'pop_params': {
        "NC": 2,
        "MC_seed": 42,
        
        "a0": 1000,
        "c": np.array([1.0]), 
        "x": np.array([1e-2]),
        "PGV": np.array(['mono']),
        "VERBOSE": True,
        'USE_PSD': True, 
        'SIZEEVAL': 1, 
        'COLEVAL': 1, 
        'BREAKRVAL': 4, 
        'BREAKFVAL': 5, 
        'process_type': 'breakage', 
        'CORR_BETA': 1e-2, 
        'alpha_prim': np.array([1, 1, 1]), 
        'pl_v': 2, 
        'pl_P1': 1e5, 
        'pl_P2': 1, 
        'pl_P3': 1e5, 
        'pl_P4': 1, 
        'G': 1,
        "CDF_method": "disc", 
        "use_lmc_pre_model": True,
        "lmc_pre_model": "table",     # "table" | "rank" | "copula" | "flow"
        "lmc_A0_runtime": 1e-10,
        "lmc_tables_path": "lmc_tables_grid.npz",
        "lmc_rank_tables_path": "lmc_rank_tables_grid.npz",
        "lmc_copula_path": "lmc_copula_grid.npz",
        "lmc_flow_pure_path": "lmc_cond_flow_pure.pt",
        "lmc_flow_mix_path": "lmc_cond_flow_mix.pt",
        "lmc_interp ": "bilinear",
        "lmc_tables_cache": True,
        
        "use_lmc_live": True,
        "lmc_small_particle_policy": "disable", # 'fallback' | 'disable'
        "lmc_pool_dir": r"C:\Users\px2030\Code\LMC_ANN\agggenerator",
        "lmc_Df": 1.8,
        "lmc_MAS": 0.4,
        },
    
    ## Optimized parameters and their search ranges.
    # Except for corr_agg, the names of the optimized parameters should be consistent with 
    # their actual names in the PBE.
    'opt_params': {
        'corr_agg_0': {'bounds': (-4.0, 0.0), 'log_scale': True}, 
        'corr_agg_1': {'bounds': (-4.0, 0.0), 'log_scale': True}, 
        'corr_agg_2': {'bounds': (-4.0, 0.0), 'log_scale': True}, 
        'pl_v': {'bounds': (0.5, 2.0), 'log_scale': False}, 
        'pl_P1': {'bounds': (10.0, 15.0), 'log_scale': True}, 
        'pl_P2': {'bounds': (0.3, 3.0), 'log_scale': False}, 
        'pl_P3': {'bounds': (10.0, 15.0), 'log_scale': True}, 
        'pl_P4': {'bounds': (0.3, 3.0), 'log_scale': False},
        
        ## Fixed parameters passed to the internal optimizer Actor.
        # Whether to wait after each calculation is completed.
        'actor_wait': {"fixed": True},
        # Waiting time.
        'wait_time': {"fixed": 1},
        # The maximum number of times a single Actor can be reused. 
        # After this number is exceeded, the Actor will be reset and resources released to prevent memory overflow.
        'max_reuse': {"fixed": 10}
        }
    }
