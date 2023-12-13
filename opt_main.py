# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import opt_method as opt
import numpy as np
import time

if __name__ == '__main__':
    dim = 2
    # The search range for corr_beta is [0, 50], see optimierer() in kern_opt.py
    corr_beta = 25
    #alpha_prim = 0.5
    alpha_prim = np.array([0.8, 0.5, 0.2])
    # t=0 is initial conditions which should be excluded
    t_vec = np.arange(1, 602, 60, dtype=float)
    
    # delta_flag = 1: use q3
    # delta_flag = 2: use Q3
    # delta_flag = 3: use x_10
    # delta_flag = 4: use x_50
    # delta_flag = 5: use x_90
    delta_flag = 1
    # noise_type: Gaussian, Uniform, Poisson, Multiplicative
    add_noise = True
    smoothing = True
    noise_type='Multiplicative'
    noise_strength = 0.1
    
    sample_num = 5
    multi_flag = True
    
    Opt = opt.opt_method(add_noise, smoothing, corr_beta, alpha_prim, dim,
                        delta_flag, noise_type, noise_strength, t_vec, multi_flag)

    # parameter for particle component NM and M
    Opt.k.set_comp_para(R_NM=2.9e-7, R_M=2.9e-7)
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
    Opt.k.n_iter = 200
    
    # The error of 2d pop may be more important, so weights need to be added
    Opt.k.weight_2d = 5
    
    # If other data are used, you need to specify the file name without a numerical label.
    # For example: 'CED_focus_Sim_Gaussian_0.01.xlsx' instead of CED_focus_Sim_Gaussian_0.01_0.xlsx
    if add_noise:
        data_name = f"CED_focus_Sim_{noise_type}_{noise_strength}.xlsx"
    else:
        data_name = "CED_focus_Sim.xlsx"
    
    # use real experimental data, currently unavailable, missing underlying data
    # data_name = "CED_focus.xlsx"
    
    # Save the results of each test
    test_num = 3
    corr_beta_opt = np.zeros(test_num)
    alpha_prim_opt = np.zeros((test_num, 3))
    para_opt = np.zeros(test_num)
    para_diff = np.zeros((test_num, 4))
    delta_opt = np.zeros(test_num)
    elapsed_time = np.zeros(test_num)
    
    # Using to observe the results
    # corr_beta_opt=25
    # alpha_prim_opt=([1, 0.5, 0.2])
    # Opt.visualize_distribution(Opt.k.p, corr_beta, alpha_prim, 
    #                             corr_beta_opt, alpha_prim_opt, exp_data_path=None)
    # Opt.visualize_distribution(Opt.k.p_NM, corr_beta, alpha_prim[0], 
    #                             corr_beta_opt, alpha_prim_opt[0], exp_data_path=None)
    # Opt.visualize_distribution(Opt.k.p_M, corr_beta, alpha_prim[2], 
    #                             corr_beta_opt, alpha_prim_opt[2], exp_data_path=None)
    
    for i in range(test_num):
        # Generate synthetic Data
        # The file name is automatically generated from the content specified when initializing Opt_method
        # in form "CED_focus_Sim_{noise_type}_{noise_strength}_{label}.xlsx". 
        # PS: There is no label when sample_num = 1.
        Opt.generate_synth_data(sample_num=sample_num)
        
        start_time = time.time()
        # method = 'kernels': Using the t_vec time step of each dataset,
        #                     optimize each data independently and then average across all kernels
        # method = 'delta': Using the t_vec time step of each dataset,
        #                   use the average error(delta) with all data as the optimization goal to find the optimal kernels
        corr_beta_opt[i], alpha_prim_opt[i, :], para_opt[i], para_diff[i, :], delta_opt[i] = \
            Opt.mean_kernels(sample_num=sample_num, method='delta', data_name=data_name)
        
        end_time = time.time()
        elapsed_time[i] = end_time - start_time
        print(f"The execution of optimierer takes：{elapsed_time} seconds")
        Opt.visualize_distribution(Opt.k.p, corr_beta, alpha_prim, 
                                    corr_beta_opt[i], alpha_prim_opt[i, :], exp_data_path=None)
        Opt.visualize_distribution(Opt.k.p_NM, corr_beta, alpha_prim[0], 
                                    corr_beta_opt[i], alpha_prim_opt[i, 0], exp_data_path=None)
        Opt.visualize_distribution(Opt.k.p_M, corr_beta, alpha_prim[2], 
                                    corr_beta_opt[i], alpha_prim_opt[i, 2], exp_data_path=None)
    mean_diff=para_diff.mean(axis=1)