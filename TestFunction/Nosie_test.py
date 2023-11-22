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

def init_points_test():
    init_points_opt = 0
    init_points_max = 10
    
    target = np.zeros((3, init_points_max-1))
    target_max = -1e-90
    
    k=kernel_opt(first_cal=True)
    # Generate artificial data
    k.cal_delta(corr_beta=25, alpha_prim=0.5)
    
    k.first_cal = False
    for i in range(2, init_points_max+1):
        target[0, i-2], target[1, i-2], target[2, i-2] = k.optimierer(init_points=i)
        if target_max > target[2, i-2]:
            target_max = target[2, i-2]
            init_points_opt = i
            
    return target, target_max, init_points_opt

if __name__ == '__main__':
    
    delta = noise_test()
    
    target, target_max, init_points_opt = init_points_test()
    