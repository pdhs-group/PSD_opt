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
    'multi_flag': False,
    
    'algo_params': {
        'dim': 1,
        # The dimensionality of the PBE
        
        't_init' : np.array([0, 0]),
        # Initial time points for simulation. 
        # These values are used to initialize N in dPBE wenn calc_init_N is True.
        # Note: The first value in t_init must be zero.
        
        # 't_vec' : np.array([0, 0]),
        't_vec' : np.array([0, 5, 10, 15, 20, 25, 35, 45, 60, 80, 100, 120])*60,
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
        
        'exp_data' : True, 
        # Whether to use experimental data (True) or synthetic data (False) during optimization.
        
        'sheet_name' : "x(Q)_mean", 
        # Name of the sheet in the experimental data file (if applicable).
         
        'method': 'Cmaes',
        # Optimization method to use. Options include:
        # - 'HEBO': Heteroscedastic Evolutionary Bayesian Optimization
        # - 'GP': Gaussian Process-based Bayesian Optimization
        # - 'TPE': Tree-structured Parzen Estimator
        # - 'Cmaes': Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        # - 'NSGA': Nondominated Sorting Genetic Algorithm (NSGA-III)
        # - 'QMC': Quasi Monte Carlo sampling
        
        'random_seed': 1,
        # Seed for reproducible results, int or None. 
        # This value will change global random states for numpy and torch on 
        # initalization and loading from checkpoint.
        
        'n_iter': 2000,
        # Number of iterations for the optimization process.

        'calc_init_N': True,
        # Whether to initialize the PBE using the first few time points of experimental data (True).
    
        'USE_PSD' : False,
        # Whether to use PSD (particle size distribution) data for setting the initial conditions 
        # of N.
        
        'R01_0' : 'r0_001',
        # Radius of NM1 primary particles corresponding to the 1% position (Q3) in the PSD data.
        
        'R03_0' : 'r0_001',
        # Radius of M primary particles corresponding to the 1% position (Q3) in the PSD data.

        'R_01': 1.9e-8,
        'R_03': 1.9e-8,
        'R01_0_scl': 1,
        # Scaling factor for the NM1 primary particle radius.
        
        'R03_0_scl': 1,
        # Scaling factor for the M primary particle radius.
        
        # 'PSD_R01' : 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy',
        # File name for the PSD data for NM1 particles.
        
        # 'PSD_R03': 'PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy',  
        # File name for the PSD data for M particles.
    
        'weight_2d': 1,  
        # Weight applied to the error (delta) of 2D particle populations, giving it 
        # more importance during optimization.
    
        'delta_flag': [#('q3','MSE'), 
                       ('Q3','RMSE'), 
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
        'tune_storage_path': r'C:\Users\px2030\Code\Ray_Tune',  
        # Path to store Ray Tune optimization infomation.
    
        'multi_jobs': False,  
        # Whether to run multiple optimization tasks (Tune jobs) concurrently. 
        # If True, multiple PSD datasets should be provided.
    
        'num_jobs': 3,  
        # Number of parallel optimization jobs to run.
    
        'cpus_per_trail': 2,  
        # Number of CPU cores allocated to each optimization trial.
    
        'max_concurrent': 2,  
        # Maximum number of trials that can be run concurrently.
        },
    
    ## PBE parameters
    'pop_params': {
        'NS' : 56,
        'S' : 1.5,
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
        'alpha_prim': np.array([0.00068498, 0.00086928, 0.00011673]),
        # 'alpha_prim': np.array([1]),
        "pl_v": 1.6372233629226685,
        "pl_P1": 0.0106435922581415,
        "pl_P2": 0.49002260278233983,
        "pl_P3": 0.00020077684158093307,
        "pl_P4": 1.6331881284713745,
        "G": 87.2642, # n=600rpm(Mean_Integral)
        },
    
    ## Parameters which should be optimized
    'opt_params' : {
        'corr_agg_0': {'bounds': (-10.0, 10.0), 'log_scale': True},
        # 'corr_agg_1': {'bounds': (-4.0, 0.0), 'log_scale': True},
        # 'corr_agg_2': {'bounds': (-4.0, 0.0), 'log_scale': True},
        'pl_v': {'bounds': (0.01, 2.0), 'log_scale': False},
        'pl_P1': {'bounds': (-10.0, 10.0), 'log_scale': True},
        'pl_P2': {'bounds': (0.1, 10.0), 'log_scale': False},
        # 'pl_P3': {'bounds': (-5.0, -1.0), 'log_scale': True},
        # 'pl_P4': {'bounds': (0.3, 3.0), 'log_scale': False},
    },

}