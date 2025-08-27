# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
import numpy as np
import multiprocessing
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
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

def evaluate_model(params, moment_flag):
    """
    Run the PBE model for a given set of parameters and return Moment M.
    """
    opt = OptBase(multi_flag=False)
    params_trans = params.copy()
    params_trans = opt.core.array_dict_transform(params_trans)
    opt.core.calc_pop(opt.core.p, params_trans, opt.core.t_vec)
    ## if PBE failed to converge
    if not opt.core.p.calc_status:
        return np.nan
    moment = opt.core.p.calc_mom_t()
    ## Rate of change of particle number
    M00 = (moment[0,0,-1] - moment[0,0,0]) / moment[0,0,0]
    ## How evenly the materials are distributed in a particle, 
    ## scaled by the combined volume of the two materials
    M11 = moment[1,1,-1] / (moment[1,0,0] * moment[0,1,0])
    if moment_flag == "m00":
        return M00
    elif moment_flag == "m11":
        return M11
    elif moment_flag == "m_wight":
        return M00 + 1000*M11
    
def save_to_csv(Si, param_names, N, moment_flag):
    # 步骤 1：保存一阶和总效应敏感性指数
    df_Si = pd.DataFrame({
        'Parameter': param_names,
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf'],
    })
    
    # 保存为 CSV 文件
    df_Si.fillna(0, inplace=True)
    df_Si.to_csv(f'sensitivity_indices_{moment_flag}_{N}.csv', index=False, float_format='%.6f')
    
    # 步骤 2：保存二阶敏感性指数
    S2 = Si['S2']
    S2_conf = Si['S2_conf']
    
    param_pair = []
    S2_values = []
    S2_conf_values = []
    
    num_params = len(param_names)

    for i in range(num_params):
        for j in range(i+1, num_params):
            param_pair.append(f"{param_names[i]} & {param_names[j]}")
            S2_values.append(S2[i, j])
            S2_conf_values.append(S2_conf[i, j])
    
    df_S2 = pd.DataFrame({
        'Parameter Pair': param_pair,
        'S2': S2_values,
        'S2_conf': S2_conf_values,
    })

    # 保存为 CSV 文件
    df_S2.fillna(0, inplace=True)
    df_S2.to_csv(f'second_order_sensitivity_indices_{moment_flag}_{N}.csv', index=False, float_format='%.6f')

if __name__ == '__main__':
    # moment_flag = "m00"
    moment_flag = "m11"
    # moment_flag = "m_wight"
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
    ## Note: For parameters that use logarithmic values, sampling is uniform on a logarithmic scale, 
    ## so the result actually reflects the effect of a linear change in the exponential 
    ## of the corresponding parameter. This is consistent with the logic of the optimizer 
    ## (as it should be!).
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
    
    ## only for test
    # mu = evaluate_model(params_list[0], moment_flag)
    
    moment_flag_list = [moment_flag] * len(params_list)
    N_list = [N] * len(params_list)
    moment_flag_list = [moment_flag] * len(params_list)
    args_list = list(zip(params_list, moment_flag_list, N_list))
    pool = multiprocessing.Pool(processes=8)
    try:
        results = pool.starmap(evaluate_model, args_list, moment_flag_list)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    finally:          
        pool.close()
        pool.join()                        
        results_arr = np.array(results)  

    # Convert the results to an array format
    Y = np.array(results)
    valid_indices = ~np.isnan(Y)
    Y_valid = Y[valid_indices]

    # Calculate Sobol' indices
    Si = sobol.analyze(problem, Y_valid, calc_second_order=True, print_to_console=True)  
    
    # save the results
    save_to_csv(Si, param_names)
             