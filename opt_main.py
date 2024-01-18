# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import numpy as np
import time
import opt_find as opt

if __name__ == '__main__':
    #%%  Input for Opt
    dim = 2
    t_vec = np.concatenate(([0.0, 0.1, 0.3, 0.6, 0.9], np.arange(1, 602, 60, dtype=float)))
    add_noise = True
    smoothing = True
    noise_type='Mul'
    noise_strength = 0.1
    sample_num = 5
    
    ## Instantiate find and algo.
    ## The find class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    find = opt.opt_find()
    find.init_opt_algo(dim, t_vec, add_noise, noise_type, noise_strength, smoothing)
     
    
    # Iteration steps for optimierer
    find.algo.n_iter = 800
    
    #%% Variable parameters
    ## Set the R0 particle radius and 
    ## whether to calculate the initial conditions from experimental data
    ## 0. The diameter ratio of the primary particles can also be used as a variable
    find.algo.calc_init_N = True
    find.algo.set_comp_para(R_NM=2.9e-7, R_M=2.9e-7)
    
    ## 1. Use only 2D Data or 1D+2D
    multi_flag = True
    
    ## 2. Criteria of optimization target
    ## delta_flag = 1: use q3
    ## delta_flag = 2: use Q3
    ## delta_flag = 3: use x_10
    ## delta_flag = 4: use x_50
    ## delta_flag = 5: use x_90
    find.algo.delta_flag = 1
    delta_flag_target = ['','q3','Q3','x_10','x_50','x_90']
    
    ## 3. Optimize method: 
    ##   'BO': Bayesian Optimization with package BayesianOptimization
    find.method='BO'
    
    ## 4. Type of cost function to use
    ##   'MSE': Mean Squared Error
    ##   'RMSE': Root Mean Squared Error
    ##   'MAE': Mean Absolute Error
    ##   'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
    find.algo.cost_func_type = 'KL'
    
    ## 5. Weight of 2D data
    ## The error of 2d pop may be more important, so weight needs to be added
    find.algo.weight_2d = 1
    
    ## 6. Method how to use the datasets, kernels or delta
    ## kernels: Find the kernel for each set of data, and then average these kernels.
    ## delta: Read all input directly and use all data to find the kernel once
    ## wait to write hier 
    
    #%% Perform calculations on a data set
    ## define the range of corr_beta
    var_corr_beta = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    
    ## define the range of alpha_prim 27x3
    values = [0, 0.5, 1]
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    
    ## For cases where R01 and R03 have the same size, the elements of alpha_prim mirror symmetry 
    ## are equivalent and can be removed to simplify the calculation.
    unique_alpha_prim = []
    for comp in var_alpha_prim:
        comp_reversed = comp[::-1]  
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_alpha_prim):
            unique_alpha_prim.append(comp)
            
    var_alpha_prim = np.array(unique_alpha_prim)
    
    ## Traverse all combinations of physical parameters
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
    for i, corr_beta in enumerate(var_corr_beta):
        for j, alpha_prim in enumerate(var_alpha_prim):
            find.algo.corr_beta = corr_beta
            find.algo.alpha_prim = alpha_prim
            data_name = f"Sim_{noise_type}_{noise_strength}_para_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_1.xlsx"
            print(f"optimize for data sets {data_name}")
            start_time = time.time()
            corr_beta_opt[i,j], alpha_prim_opt[i,j,:], para_diff[i,j], delta_opt[i,j], \
                corr_agg[i,j,:], corr_agg_opt[i,j,:], corr_agg_diff[i,j,:] = \
                find.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
            end_time = time.time()
            elapsed_time[i,j] = end_time - start_time
                
    ## save the results in excel            
    # with pd.ExcelWriter('result.xlsx', engine='openpyxl') as writer:
    #     pd.DataFrame(corr_beta_opt.reshape(-1, corr_beta_opt.shape[-1])).to_excel(writer, sheet_name='corr_beta_opt')
    #     pd.DataFrame(alpha_prim_opt.reshape(-1, alpha_prim_opt.shape[-1])).to_excel(writer, sheet_name='alpha_prim_opt')
    #     pd.DataFrame(para_diff.reshape(-1, para_diff.shape[-1])).to_excel(writer, sheet_name='para_diff')
    #     pd.DataFrame(delta_opt.reshape(-1, delta_opt.shape[-1])).to_excel(writer, sheet_name='delta_opt')
    #     pd.DataFrame(elapsed_time.reshape(-1, elapsed_time.shape[-1])).to_excel(writer, sheet_name='elapsed_time')
    
    ## save the results in npz
    if multi_flag:
        result = f'multi_{delta_flag_target[find.algo.delta_flag]}_{find.method}_{find.algo.cost_func_type}_wight_{find.algo.weight_2d}'
    else:
        result =  f'{delta_flag_target[find.algo.delta_flag]}_{find.method}_{find.algo.cost_func_type}_wight_{find.algo.weight_2d}'
    np.savez(f'{result}.npz', 
         corr_beta_opt=corr_beta_opt, 
         alpha_prim_opt=alpha_prim_opt, 
         para_diff=para_diff, 
         delta_opt=delta_opt, 
         corr_agg = corr_agg,
         corr_agg_opt = corr_agg_opt,
         corr_agg_diff = corr_agg_diff,
         elapsed_time=elapsed_time)
    
    
    
