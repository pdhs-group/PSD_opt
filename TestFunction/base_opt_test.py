# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import opt_method as opt
import numpy as np
import time

def repeat_tests(Opt, sample_num, data_name, gruppe_num, test_num=3):
    # Save the results of each test
    corr_beta_opt = np.zeros(test_num)
    alpha_prim_opt = np.zeros((test_num, 3))
    para_opt = np.zeros(test_num)
    para_diff = np.zeros((test_num, 4))
    delta_opt = np.zeros(test_num)
    elapsed_time = np.zeros(test_num)
    
    for i in range(test_num):  
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
        fig_mix = Opt.visualize_distribution(Opt.k.p, corr_beta, alpha_prim, 
                                    corr_beta_opt[i], alpha_prim_opt[i, :], exp_data_path=None)
        fig_NM = Opt.visualize_distribution(Opt.k.p_NM, corr_beta, alpha_prim[0], 
                                    corr_beta_opt[i], alpha_prim_opt[i, 0], exp_data_path=None)
        fig_M = Opt.visualize_distribution(Opt.k.p_M, corr_beta, alpha_prim[2], 
                                    corr_beta_opt[i], alpha_prim_opt[i, 2], exp_data_path=None)
        Opt.save_as_png(fig_mix, f"Gruppe{gruppe_num}-{i+1}")
        Opt.save_as_png(fig_NM, f"Gruppe{gruppe_num}-NM{i+1}")
        Opt.save_as_png(fig_M, f"Gruppe{gruppe_num}-M{i+1}")
        
        Opt.generate_synth_data(sample_num=sample_num)
    mean_diff=para_diff.mean(axis=1)
    return corr_beta_opt, alpha_prim_opt, para_opt, para_diff, delta_opt, elapsed_time, mean_diff

def normal_test(Opt, sample_num, data_name):
    start_time = time.time()

    corr_beta_opt, alpha_prim_opt, para_diff, delta_opt = \
        Opt.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")
    
    fig_mix = Opt.visualize_distribution(Opt.k.p, Opt.k.corr_beta, Opt.k.alpha_prim, 
                                corr_beta_opt, alpha_prim_opt, exp_data_path=None)
    fig_NM = Opt.visualize_distribution(Opt.k.p_NM, Opt.k.corr_beta, Opt.k.alpha_prim[0], 
                                corr_beta_opt, alpha_prim_opt[0], exp_data_path=None)
    fig_M = Opt.visualize_distribution(Opt.k.p_M, Opt.k.corr_beta, Opt.k.alpha_prim[2], 
                                corr_beta_opt, alpha_prim_opt[2], exp_data_path=None)
    Opt.save_as_png(fig_mix, "PSD")
    Opt.save_as_png(fig_NM, "PSD-NM")
    Opt.save_as_png(fig_M, "PSD-M")
    
    return corr_beta_opt, alpha_prim_opt, para_diff, delta_opt, elapsed_time


if __name__ == '__main__':
    dim = 2
    # The search range for corr_beta is [0, 50], see optimierer() in kern_opt.py
    # corr_beta = 0.1
    #alpha_prim = 0.5
    # alpha_prim = np.array([0.8, 0.5, 0.2])
    # t=0 is initial conditions which should be excluded
    t_vec = np.concatenate(([0.0, 0.1, 0.3, 0.6, 0.9], np.arange(1, 602, 60, dtype=float)))
    
    # delta_flag = 1: use q3
    # delta_flag = 2: use Q3
    # delta_flag = 3: use x_10
    # delta_flag = 4: use x_50
    # delta_flag = 5: use x_90
    delta_flag = 1
    # noise_type: Gaussian, Uniform, Poisson, Multiplicative
    add_noise = True
    smoothing = True
    noise_type='Mul'
    noise_strength = 0.1
    
    sample_num = 5
    multi_flag = True
    
    Opt = opt.opt_method(add_noise, smoothing, dim,
                        delta_flag, noise_type, noise_strength, t_vec, multi_flag)

    # parameter for particle component NM and M
    # pth = os.path.dirname( __file__ )
    # dist_path_1 = os.path.join(pth, "..", "data\\PSD_data\\")+'PSD_x50_1.0E-6_v50_5.2E-19_RelSigmaV_1.0E+0.npy'
    # R01_0 = 'r0_005'
    # R03_0 = 'r0_005'
    # Opt.k.set_comp_para(R01_0, R03_0, dist_path_1, dist_path_1)
    ## Set initial conditions to be calculated from experimental data, or use PSD data
    Opt.k.calc_init_N = True
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
    Opt.k.n_iter = 100
    
    # The error of 2d pop may be more important, so weights need to be added
    Opt.k.weight_2d = 1
    
    # Generate synthetic Data
    # The file name is automatically generated from the content specified when initializing Opt_method
    # in form "CED_focus_Sim_{noise_type}_{noise_strength}_{label}.xlsx". 
    # PS: There is no label when sample_num = 1.
    # Opt.generate_synth_data(sample_num=sample_num)
    
    # If other data are used, you need to specify the file name without a numerical label.
    # For example: 'CED_focus_Sim_Gaussian_0.01.xlsx' instead of CED_focus_Sim_Gaussian_0.01_0.xlsx
    if add_noise:
        data_name = f"Sim_{noise_type}_{noise_strength}_para_0.1_1.0_1.0_0.5_1.xlsx"
    else:
        data_name = "Sim_para_0.1_1.0_1.0_0.5_1.xlsx"
    
    Opt.k.corr_beta = 0.1
    Opt.k.alpha_prim = np.array([1.0, 1.0, 0.5])
    
    # use real experimental data, currently unavailable, missing underlying data
    # data_name = "CED_focus.xlsx"
    
    # Using to observe the results
    # corr_beta_opt=25
    # alpha_prim_opt=([1, 0.5, 0.2])
    # Opt.visualize_distribution(Opt.k.p, corr_beta, alpha_prim, 
    #                             corr_beta_opt, alpha_prim_opt, exp_data_path=None)
    # Opt.visualize_distribution(Opt.k.p_NM, corr_beta, alpha_prim[0], 
    #                             corr_beta_opt, alpha_prim_opt[0], exp_data_path=None)
    # Opt.visualize_distribution(Opt.k.p_M, corr_beta, alpha_prim[2], 
    #                             corr_beta_opt, alpha_prim_opt[2], exp_data_path=None)
    
    # Run the test with the same number of iterations each time but regenerate the experimental data
    # corr_beta_opt, alpha_prim_opt, para_opt, para_diff, delta_opt, elapsed_time, mean_diff = \
    # repeat_tests(Opt, sample_num, data_name, gruppe_num=100, test_num=3)
    
    # Run the test with same experimental data and increased iteration number
    # n_iter = np.array([800])
    # iter_num = len(n_iter)
    # corr_beta_opt = np.zeros(iter_num)
    # alpha_prim_opt = np.zeros((iter_num, 3))
    # para_opt = np.zeros(iter_num)
    # para_diff = np.zeros((iter_num, 4))
    # delta_opt = np.zeros(iter_num)
    # elapsed_time = np.zeros(iter_num)
    # for i, num in enumerate(n_iter):
    #     Opt.k.n_iter = num
    #     gruppe_num = 66 + i
    #     corr_beta_opt[i], alpha_prim_opt[i, :], para_opt[i], para_diff[i, :], \
    #         delta_opt[i], elapsed_time[i]= normal_test(
    #             Opt, sample_num, data_name, gruppe_num)
    # mean_diff=para_diff.mean(axis=1)
    
    
    corr_beta_opt, alpha_prim_opt, para_diff, \
        delta_opt, elapsed_time= normal_test(
            Opt, sample_num, data_name)
        
    # mean_diff=para_diff.mean()
