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

if __name__ == '__main__':
    dim = 2
    
    corr_beta = 0.1
    
    alpha_prim = np.array([0.8, 0.5, 0.2])
    
    t_vec = np.arange(1, 602, 60, dtype=float)
       
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
    Opt.k.n_iter = 800
    
    # The error of 2d pop may be more important, so weights need to be added
    Opt.k.weight_2d = 1
    
    # Generate synthetic Data
    # The file name is automatically generated from the content specified when initializing Opt_method
    # in form "CED_focus_Sim_{noise_type}_{noise_strength}_{label}.xlsx". 
    # PS: There is no label when sample_num = 1.
    Opt.generate_synth_data(sample_num=sample_num)