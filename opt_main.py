# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import numpy as np
import time
import opt_find as opt
import multiprocessing
import opt_config as conf

def optimization_process(args):
    i, j, corr_beta, alpha_prim, data_name = args
    
    #%%  Input for Opt
    dim = conf.config['dim']
    t_vec = conf.config['t_vec']
    add_noise = conf.config['add_noise']
    smoothing = conf.config['smoothing']
    noise_type= conf.config['noise_type']
    noise_strength = conf.config['noise_strength']
    sample_num = conf.config['sample_num']
    
    ## Instantiate find and algo.
    ## The find class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    find = opt.opt_find()
     
    #%% Variable parameters
    ## Set the R0 particle radius and 
    ## whether to calculate the initial conditions from experimental data
    ## 0. Use only 2D Data or 1D+2D
    find.multi_flag = conf.config['multi_flag']
    find.init_opt_algo(dim, t_vec, add_noise, noise_type, noise_strength, smoothing)
    ## Iteration steps for optimierer
    find.algo.n_iter = conf.config['n_iter']
    
    ## 1. The diameter ratio of the primary particles can also be used as a variable
    find.algo.calc_init_N = conf.config['calc_init_N']
    find.algo.set_comp_para(R_NM=2.9e-7, R_M=2.9e-7)
    
    ## 2. Criteria of optimization target
    ## delta_flag = q3: use q3
    ## delta_flag = Q3: use Q3
    ## delta_flag = x_10: use x_10
    ## delta_flag = x_50: use x_50
    ## delta_flag = x_90: use x_90
    find.algo.delta_flag = conf.config['multi_flag']
    
    ## 3. Optimize method: 
    ##   'BO': Bayesian Optimization with package BayesianOptimization
    find.method = conf.config['method']
    
    ## 4. Type of cost function to use
    ##   'MSE': Mean Squared Error
    ##   'RMSE': Root Mean Squared Error
    ##   'MAE': Mean Absolute Error
    ##   'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
    find.algo.cost_func_type = conf.config['cost_func_type']
    
    ## 5. Weight of 2D data
    ## The error of 2d pop may be more important, so weight needs to be added
    find.algo.weight_2d = conf.config['weight_2d']
    
    ## 6. Method how to use the datasets, kernels or delta
    ## kernels: Find the kernel for each set of data, and then average these kernels.
    ## delta: Read all input directly and use all data to find the kernel once
    ## wait to write hier 
    
    #%% Perform optimization
    find.algo.corr_beta = corr_beta
    find.algo.alpha_prim = alpha_prim

    start_time = time.time()
    results = \
        find.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    end_time = time.time()
    elapsed_time[i,j] = end_time - start_time
    
    return i, j, results, elapsed_time

if __name__ == '__main__':
    noise_type = conf.config['noise_type']
    noise_strength = conf.config['noise_strength']
    multi_flag = conf.config['multi_flag']
    delta_flag = conf.config['delta_flag']
    method = conf.config['method']
    cost_func_type = conf.config['cost_func_type']
    weight_2d = conf.config['weight_2d']

    #%% Prepare test data set
    ## define the range of corr_beta
    var_corr_beta = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    
    ## define the range of alpha_prim 27x3
    values = [0, 0.5, 1]
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    ## remove element [0, 0, 0]
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]
    
    ## For cases where R01 and R03 have the same size, the elements of alpha_prim mirror symmetry 
    ## are equivalent and can be removed to simplify the calculation.
    unique_alpha_prim = []
    for comp in var_alpha_prim:
        comp_reversed = comp[::-1]  
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_alpha_prim):
            unique_alpha_prim.append(comp)
            
    var_alpha_prim = np.array(unique_alpha_prim)
    ## The case of all zero α is meaningless, that means no Agglomeration occurs
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]
    
    pool = multiprocessing.Pool(processes=4)
    tasks = []
    for i, corr_beta in enumerate(var_corr_beta):
        for j, alpha_prim in enumerate(var_alpha_prim):
            data_name = f"Sim_{noise_type}_{noise_strength}_para_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_1.xlsx"
            tasks.append((i, j, corr_beta, alpha_prim, data_name))
    
    results = pool.map(optimization_process, tasks)
    
    data_size = np.array([len(var_corr_beta), len(var_alpha_prim)])
    corr_beta_opt = np.zeros(data_size)
    # The asterisk (*) is used in a function call to indicate an "unpacking" operation, 
    # which means that it expands the elements of 'data_size' into individual arguments
    alpha_prim_opt = np.zeros((*data_size, 3))
    para_diff = np.zeros(data_size)
    delta_opt = np.zeros(data_size)
    elapsed_time = np.zeros(data_size)
    corr_agg = np.zeros((*data_size, 3))
    corr_agg_opt = np.zeros((*data_size, 3))
    corr_agg_diff = np.zeros((*data_size, 3))
    
    for result in results:
        i, j, (corr_beta_opt_res, alpha_prim_opt_res, para_diff_res, delta_opt_res, \
               corr_agg_res, corr_agg_opt_res, corr_agg_diff_res), elapsed = result
        corr_beta_opt[i,j] = corr_beta_opt_res
        alpha_prim_opt[i,j,:] = alpha_prim_opt_res
        para_diff[i,j] = para_diff_res
        delta_opt[i,j] = delta_opt_res
        elapsed_time[i,j] = elapsed
        corr_agg[i,j,:] = corr_agg_res
        corr_agg_opt[i,j,:] = corr_agg_opt_res
        corr_agg_diff[i,j,:] = corr_agg_diff_res
            
    ## save the results in npz
    if multi_flag:
        result_name = f'multi_{delta_flag}_{method}_{cost_func_type}_wight_{weight_2d}'
    else:
        result_name =  f'{delta_flag}_{method}_{cost_func_type}_wight_{weight_2d}'
        
    np.savez(f'{result_name}.npz', 
         corr_beta_opt=corr_beta_opt, 
         alpha_prim_opt=alpha_prim_opt, 
         para_diff=para_diff, 
         delta_opt=delta_opt, 
         corr_agg = corr_agg,
         corr_agg_opt = corr_agg_opt,
         corr_agg_diff = corr_agg_diff,
         elapsed_time=elapsed_time)
    
    
    
