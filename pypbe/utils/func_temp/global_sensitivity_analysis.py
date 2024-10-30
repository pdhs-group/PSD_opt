# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
import time
import numpy as np
import multiprocessing
from SALib.sample import saltelli
from SALib.analyze import sobol
import copy
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
from pypbe.kernel_opt.opt_base import OptBase

def transform_parameters(X):
    """
    Transform parameters from sampled values to actual values, considering log scale.
    """
    X_transformed = np.zeros_like(X)
    for i, name in enumerate(problem['names']):
        bounds = problem['bounds'][i]
        log_scale = log_scale_params[name]
        if log_scale:
            # Log scale: power of 10
            X_transformed[:, i] = 10 ** X[:, i]
        else:
            # Linear scale
            X_transformed[:, i] = X[:, i]
    return X_transformed

def transform_parameters_to_dict(X):
    """
    Convert parameter array to a list of parameter dictionaries.
    """
    params_list = []
    X_transformed = transform_parameters(X)
    for params in X_transformed:
        params_dict = {name: value for name, value in zip(problem['names'], params)}
        params_list.append(params_dict)
    return params_list

def evaluate_model(params):
    """
    Run the PBE model for a given set of parameters and return Moment M.
    """
    opt = OptBase(multi_flag=False)
    params_checked = opt.core.check_corr_agg(params)
    opt.core.calc_all_pop(opt.core.p, params_checked, opt.core.t_vec)
    N = opt.core.p.N
    
    return N

if __name__ == '__main__':
    # Define parameter names and ranges
    param_names = ['corr_agg_0', 'corr_agg_1', 'corr_agg_2', 'pl_v', 'pl_P1', 'pl_P2', 'pl_P3', 'pl_P4']
    
    # Define parameter bounds
    problem = {
        'num_vars': 8,
        'names': param_names,
        'bounds': [
            [-4.0, 0.0],    # corr_agg_0
            [-4.0, 0.0],    # corr_agg_1
            [-4.0, 0.0],    # corr_agg_2
            [0.5, 2.0],     # pl_v
            [-5.0, -1.0],   # pl_P1
            [0.3, 3.0],     # pl_P2
            [-5.0, -1.0],   # pl_P3
            [0.3, 3.0],     # pl_P4
        ]
    }
    
    # Define which parameters are on a logarithmic scale
    log_scale_params = {
        'corr_agg_0': True,
        'corr_agg_1': True,
        'corr_agg_2': True,
        'pl_v': False,
        'pl_P1': True,
        'pl_P2': False,
        'pl_P3': True,
        'pl_P4': False,
    }
    
    # Set the number of sampling points
    N = 1024  # Adjust this number based on available computational resources
    
    # Generate sampling points
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    # Transform parameters to get a list of parameter dictionaries
    params_list = transform_parameters_to_dict(param_values)
    
    pool = multiprocessing.Pool(processes=8)
    try:
        results = pool.map(evaluate_model, params_list)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    finally:          
        pool.close()
        pool.join()                        
        results_arr = np.array(results)  

    # Convert the results to an array format
    Y = np.array(results)

    # Calculate Sobol' indices
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=True)           