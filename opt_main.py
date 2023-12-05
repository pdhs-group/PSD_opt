# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import opt_method as opt
import numpy as np

if __name__ == '__main__':
    corr_beta = 25
    alpha_prim = np.array([0.8, 0.5, 0.2])
    t_vec = np.arange(0, 601, 60, dtype=float)
    
    # delta_flag = 1: use q3
    # delta_flag = 2: use Q3
    # delta_flag = 3: use x_50
    delta_flag = 1
    # noise_type: Gaussian, Uniform, Poisson, Multiplicative
    add_noise = True
    smoothing = True
    noise_type='Gaussian'
    noise_strength = 0.005
    
    sample_num = 10
    
    Opt = opt.opt_method(add_noise, smoothing, corr_beta, alpha_prim, t_vec=t_vec, noise_type=noise_type, noise_strength=noise_strength)
    Opt.dim = 2
    # Optimize method: 
    #   'BO': Bayesian Optimization with package BayesianOptimization
    #   'gp_minimize': Bayesian Optimization with package skopt.gp_minimize
    Opt.algo='BO'
    
    # Generate synthetic Data
    # The file name is automatically generated from the content specified when initializing Opt_method.
    Opt.generate_synth_data(sample_num=sample_num)
    
    # If you use other data, you need to specify the file name without a numerical label.
    # For example: 'CED_focus_Sim_Gaussian_0.01.xlsx' instead of CED_focus_Sim_Gaussian_0.01_0.xlsx
    if add_noise:
        data_name = f"CED_focus_Sim_{noise_type}_{noise_strength}.xlsx"
    else:
        data_name = "CED_focus_Sim.xlsx"
    
    # method = 'kernels': Using the max(t_vec) time step of each dataset,
    #                     optimize each data independently and then average across all kernels
    # method = 'delta': Using the max(t_vec) time step of each dataset,
    #                   use the average error(delta) with all data as the optimization goal to find the optimal kernels
    # method = 'time_kernels': Using time step in t_vec for one dataset,
    #                          optimize data in each time step independently and then average across all kernels 
    corr_beta_opt, alpha_prim_opt, para_opt, para_diff, delta_opt = \
        Opt.mean_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    