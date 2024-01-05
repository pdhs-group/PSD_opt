# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import opt_method as opt
import numpy as np
from generate_psd import full_psd

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
    
    generate_new_psd = True
    
    if generate_new_psd:
        ## Input for generating psd-data
        x50 = 1   # /mm
        resigma = 1
        minscale = 1e-3
        maxscale = 1e3
        dist_path_1 = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        dist_path_10 = full_psd(x50*10, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
    else:
        pth = os.path.dirname( __file__ )
        dist_path_1 = os.path.join(pth, "..", "\\data\\PSD_data\\")+'PSD_x50_1.0E-6_v50_5.2E-19_RelSigmaV_1.0E+0.npy'
        dist_path_10 = os.path.join(pth, "..", "\\data\\PSD_data\\")+'PSD_x50_1.0E-5_v50_5.2E-16_RelSigmaV_1.0E+0.npy'
    
    ## define the range of corr_beta
    corr_beta = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    
    ## define the range of alpha_prim 64x3
    values = [0, 0.33, 0.67, 1]
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))

    ## Instantiate Opt
    Opt = opt.opt_method(add_noise, smoothing, dim, delta_flag, noise_type, 
                         noise_strength, t_vec, multi_flag)

    ## Reinitialization of pop equations using psd data  
    dist_path_NM = dist_path_1
    dist_path_M = dist_path_1
    R01_0 = 'r0_005'
    R03_0 = 'r0_005'
    Opt.k.set_comp_para(R01_0, R03_0, dist_path_NM, dist_path_M)
    
    ## Set α and β_corr
    Opt.k.corr_beta = corr_beta[2]
    Opt.k.alpha_prim = alpha_prim[31, :]
    
    add_info = f"_para_{Opt.k.corr_beta}_{Opt.k.alpha_prim[0]}_{Opt.k.alpha_prim[1]}_{Opt.k.alpha_prim[2]}_1"
    # Generate synthetic Data
    Opt.generate_synth_data(sample_num=sample_num, add_info=add_info)
    
    
    
    
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
    Opt.k.n_iter = 800
    
    # The error of 2d pop may be more important, so weights need to be added
    Opt.k.weight_2d = 1