# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:34:14 2023

@author: px2030
"""
import sys
import os
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from kerns_opt import kernel_opt

def noise_test():
    k=kernel_opt()
    # k.cal_delta(beta_corr=25, alpha_prim=0.5, first_cal=True)
    delta = k.cal_delta(beta_corr=25, alpha_prim=0.5)
    k.visualize_distribution()
    
    return delta

def init_points_test(add_noise=True):
    init_points_opt = 0
    init_points_max = 10
    
    target = np.zeros((3, init_points_max-1))
    target_diff = np.zeros((3, init_points_max-1))
    target_opt = 1e10
    
    k=kernel_opt(first_cal=True)
    # Generate artificial data
    k.add_noise = add_noise
    k.cal_delta(corr_beta=25, alpha_prim=0.5)
    
    k.first_cal = False
    for i in range(2, init_points_max+1):
        target[0, i-2], target[1, i-2], target[2, i-2] = k.optimierer(algo='gp_minimize', init_points=i)
        target_diff[0,i-2] = abs(target[0,i-2] - k.corr_beta) / k.corr_beta
        target_diff[1,i-2] = abs(target[1,i-2] - k.alpha_prim) / k.alpha_prim
        target_diff[2,i-2] = target[2,i-2]
        
        if target_opt > target[2, i-2]:
            target_opt = target[2, i-2]
            init_points_opt = i - 2
            
    return target, target_opt, init_points_opt, target_diff

def opt_test(algo='BO', add_noise=True):
    k=kernel_opt(first_cal=True)
    # Generate artificial data
    k.add_noise = add_noise
    k.cal_delta(corr_beta=25, alpha_prim=0.5)
    
    k.first_cal = False
    corr_beta_opt, alpha_prim_opt, delta_opt = k.optimierer(algo='gp_minimize')
    
    corr_beta_diff = abs(corr_beta_opt - k.corr_beta) / k.corr_beta
    alpha_prim_diff = abs(alpha_prim_opt - k.alpha_prim) / k.alpha_prim
    
    return corr_beta_opt, alpha_prim_opt, delta_opt, corr_beta_diff, alpha_prim_diff    

if __name__ == '__main__':
    
    test = 2
    
    
    if test == 1:
        delta = noise_test()
        
    elif test == 2:
        target, target_opt, init_points_opt, target_diff = init_points_test(add_noise=False)
        
    elif test == 3:
        corr_beta_opt, alpha_prim_opt, delta_opt, corr_beta_diff, alpha_prim_diff = opt_test(algo='gp_minimize', add_noise=False)

    