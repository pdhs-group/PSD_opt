# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import numpy as np
import time
import pandas as pd
import opt_method as opt
# from generate_psd import full_psd
# from pop import population
## For plots
# import matplotlib.pyplot as plt
# import plotter.plotter as pt   

if __name__ == '__main__':
    ## Input for Opt
    dim = 2
    t_vec = np.arange(1, 602, 60, dtype=float)
    delta_flag = 1
    add_noise = True
    smoothing = True
    noise_type='Mul'
    noise_strength = 0.1
    sample_num = 5
    multi_flag = True
    
    ## Instantiate Opt
    Opt = opt.opt_method(add_noise, smoothing, dim, delta_flag, noise_type, 
                         noise_strength, t_vec, multi_flag)
    
    # Optimize method: 
    #   'BO': Bayesian Optimization with package BayesianOptimization
    #   'gp_minimize': Bayesian Optimization with package skopt.gp_minimize
    Opt.algo='BO'
    
    # Type of cost function to use
    #   'MSE': Mean Squared Error
    #   'RMSE': Root Mean Squared Error
    #   'MAE': Mean Absolute Error
    #   'KL': Kullback–Leibler divergence
    Opt.k.cost_func_type = 'KL'
    
    # Iteration steps for optimierer
    Opt.k.n_iter = 1000
    
    # The error of 2d pop may be more important, so weights need to be added
    Opt.k.weight_2d = 1
    
    ## define the range of corr_beta
    var_corr_beta = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    
    ## define the range of alpha_prim 64x3
    values = [0, 0.33, 0.67, 1]
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))

    ## define the range of particle size scale
    size_scale = [1, 5, 10]

    ## get the initial condition for particle size number distribution
    # Opt.get_init_N()
    
    ## Traverse all combinations of physical parameters
    data_size = np.array([len(size_scale), len(var_corr_beta), len(var_alpha_prim)])
    corr_beta_opt = np.zeros(data_size)
    # The asterisk (*) is used in a function call to indicate an "unpacking" operation, 
    # which means that it expands the elements of 'data_size' into individual arguments
    alpha_prim_opt = np.zeros((*data_size, 3))
    para_diff = np.zeros(data_size)
    delta_opt = np.zeros(data_size)
    elapsed_time = np.zeros(data_size)
    for k, scale in enumerate(size_scale):
        for i, corr_beta in enumerate(var_corr_beta):
            for j, alpha_prim in enumerate(var_alpha_prim):
                Opt.k.corr_beta = corr_beta
                Opt.k.alpha_prim = alpha_prim
                data_name = f"Sim_{noise_type}_{noise_strength}_para_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_{scale}.xlsx"
                start_time = time.time()
                corr_beta_opt[k,i,j], alpha_prim_opt[k,i,j,:], para_diff[k,i,j], delta_opt[k,i,j] = 1,1,1,1\
                    # Opt.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
                end_time = time.time()
                elapsed_time[k,i,j] = end_time - start_time
                
    ## save the results in excel            
    # with pd.ExcelWriter('result.xlsx', engine='openpyxl') as writer:
    #     pd.DataFrame(corr_beta_opt.reshape(-1, corr_beta_opt.shape[-1])).to_excel(writer, sheet_name='corr_beta_opt')
    #     pd.DataFrame(alpha_prim_opt.reshape(-1, alpha_prim_opt.shape[-1])).to_excel(writer, sheet_name='alpha_prim_opt')
    #     pd.DataFrame(para_diff.reshape(-1, para_diff.shape[-1])).to_excel(writer, sheet_name='para_diff')
    #     pd.DataFrame(delta_opt.reshape(-1, delta_opt.shape[-1])).to_excel(writer, sheet_name='delta_opt')
    #     pd.DataFrame(elapsed_time.reshape(-1, elapsed_time.shape[-1])).to_excel(writer, sheet_name='elapsed_time')
    
    ## save the results in npz
    np.savez('result.npz', 
         corr_beta_opt=corr_beta_opt, 
         alpha_prim_opt=alpha_prim_opt, 
         para_diff=para_diff, 
         delta_opt=delta_opt, 
         elapsed_time=elapsed_time)
    
    
    
