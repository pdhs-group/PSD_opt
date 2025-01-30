# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:42:20 2024

@author: px2030
"""
import numpy as np
import os
## Config for Optimization
_config_opt_path = os.path.dirname(__file__)
config = {
    ## Use only 2D Data or 1D+2D
    'multi_flag': False,
    ## Input only one/one set of PSD data
    'single_case': False,
    
    'algo_params': {
        'dim': 1,
        # The dimensionality of the PBE
        
        't_init' : np.array([0, 0]),
        # Initial time points for simulation. 
        # These values are used to initialize N in dPBE wenn calc_init_N is True.
        # Note: The first value in t_init must be zero.
        
        # 't_vec' : np.array([0, 0]),
        't_vec' : np.array([0, 5, 10, 15, 20, 25, 30])*60,
        # Time vector for the entire simulation, specifying the time points at which 
        # calculations are performed.
        
        'delta_t_start_step' : 1,
        # Specifies the number of initial time steps to skip during optimization, 
        # often useful to avoid the impact of initialization errors.
        
        'add_noise': True,
        # Whether to add noise to the generated data.
        
        'smoothing': True,
        # Whether to apply smoothing to the simulated data, usually performed using 
        # kernel density estimation (KDE).
        
        'noise_type': 'Mul',
        # Type of noise to add to the data. Options include:
        # - 'Gaus': Gaussian noise
        # - 'Uni': Uniform noise
        # - 'Po': Poisson noise
        # - 'Mul': Multiplicative noise
        
        'noise_strength': 0.1,
        # The strength of the added noise.
        
        'sample_num': 1,
        # Number of experimental or synthetic data samples used during the optimization process.
        
        'exp_data' : False, 
        # Whether to use experimental data (True) or synthetic data (False) during optimization.
        
        'sheet_name' : None, 
        # Name of the sheet in the experimental data file (if applicable).
         
        'method': 'Cmaes',
        # Optimization method to use. Options include:
        # - 'GP': Gaussian Process-based Bayesian Optimization
        # - 'TPE': Tree-structured Parzen Estimator
        # - 'Cmaes': Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        # - 'NSGA': Nondominated Sorting Genetic Algorithm (NSGA-III)
        # - 'QMC': Quasi Monte Carlo sampling
        
        'random_seed': 1,
        # Seed for reproducible results, int or None. 
        # This value will change global random states for numpy and torch on 
        # initalization and loading from checkpoint.
        
        'n_iter': 10,
        # Number of iterations for the optimization process.

        'calc_init_N': False,
        # Whether to initialize the PBE using the first few time points of experimental data (True).
    
        'USE_PSD' : True,
        # Whether to use PSD (particle size distribution) data for setting the initial conditions 
        # of N.
        
        'USE_PSD_R': False,
        # Whether to use R01_0 and R03_0 below to get the particle size in the PSD data as 
        # the starting coordinates for PBE. If False, the values ​​of R_01 and R_03 are used.
        
        'R01_0' : 'r0_001',
        # Radius of NM1 primary particles corresponding to the 1% position (Q3) in the PSD data.
        
        'R03_0' : 'r0_001',
        # Radius of M primary particles corresponding to the 1% position (Q3) in the PSD data.

        'R_01': 2.9e-7,
        'R_03': 2.9e-7,
        'R01_0_scl': 1,
        # Scaling factor for the NM1 primary particle radius.
        
        'R03_0_scl': 1,
        # Scaling factor for the M primary particle radius.
        
        'PSD_R01' : 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy',
        # File name for the PSD data for NM1 particles.
        
        'PSD_R03': 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy',  
        # File name for the PSD data for M particles.
    
        'weight_2d': 1,  
        # Weight applied to the error (delta) of 2D particle populations, giving it 
        # more importance during optimization.
    
        'delta_flag': [('q3','MSE'), 
                       # ('Q3','RMSE'), 
                       #('x_50','MSE')
                       ],
        # Specifies which particle size distribution (PSD) and cost function to use 
        # during optimization. Options for PSD include:
        # - 'q3': Number-based distribution
        # - 'Q3': Cumulative distribution
        # - 'x_10': Particle size at 10% of cumulative distribution
        # - 'x_50': Particle size at 50% of cumulative distribution
        # - 'x_90': Particle size at 90% of cumulative distribution
        # Cost function options include:
        # - 'MSE': Mean Squared Error
        # - 'RMSE': Root Mean Squared Error
        # - 'MAE': Mean Absolute Error
        # - 'KL': Kullback-Leibler divergence (only compatible with q3 and Q3)
        # It is allowed to use combinations of different PSDs and cost functions as optimization targets.
        # In such cases, the objective function is the sum of the individual errors.
        'tune_storage_path': os.path.join(_config_opt_path, "Ray_Tune"),   
        # Path to store Ray Tune optimization infomation.
        
        'verbose': 1,
    
        'multi_jobs': False,  
        # Whether to run multiple optimization tasks (Tune jobs) concurrently. 
        # If True, multiple PSD datasets should be provided.
    
        'num_jobs': 3,  
        # Number of parallel optimization jobs to run.
    
        'cpus_per_trail': 3,  
        # Number of CPU cores allocated to each optimization trial.
    
        'max_concurrent': 4,  
        # Maximum number of trials that can be run concurrently.
        },
    
    ## PBE parameters
    ## For a detailed explanation of the PBE parameters, please refer to the `PBE_config.py` file.
    'pop_params': {
        'NS' : 10,
        'S' : 4,
        "SIZEEVAL": 1,
        "COLEVAL": 1,
        "EFFEVAL": 1,
        'BREAKRVAL' : 4,
        'BREAKFVAL' : 5,
        ## aggl_crit: The sequence number of the particle that allows further agglomeration
        'aggl_crit' : 100,
        'process_type' : "mix",
        ## The "average volume" of the two elemental particles in the system.
        ## Used to scale the particle volume in calculation of the breakage rate.
        'V1_mean' : 1e-15,
        'V3_mean' : 1e-15,
        ## Reduce particle number desity concentration to improve calculation stability
        ## Default value = 1e14 
        'V_unit': 1e-12,
        ## When True, use distribution data simulated using MC-bond-break methods
        'USE_MC_BOND' : False,
        'solver' : "ivp",
        
        "CORR_BETA": 1,
        'alpha_prim': np.array([1e-2, 1e-2, 1e-2]),
        # 'alpha_prim': np.array([1]),
        "pl_v": 2,
        "pl_P1": 1e-2,
        "pl_P2": 1,
        "pl_P3": 1e-2,
        "pl_P4": 1,
        "G": 80,
        },
    
    ## Parameters which should be optimized
    'opt_params' : {
        'corr_agg_0': {'bounds': (-4.0, 0.0), 'log_scale': True},
        'corr_agg_1': {'bounds': (-4.0, 0.0), 'log_scale': True},
        'corr_agg_2': {'bounds': (-4.0, 0.0), 'log_scale': True},
        'pl_v': {'bounds': (0.5, 2.0), 'log_scale': False},
        'pl_P1': {'bounds': (-5.0, -1.0), 'log_scale': True},
        'pl_P2': {'bounds': (0.3, 3.0), 'log_scale': False},
        'pl_P3': {'bounds': (-5.0, -1.0), 'log_scale': True},
        'pl_P4': {'bounds': (0.3, 3.0), 'log_scale': False},
    },

}