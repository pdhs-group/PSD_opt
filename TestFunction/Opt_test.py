# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:34:14 2023

@author: px2030
"""
import sys
import os
import numpy as np
import time
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from kerns_opt import kernel_opt

class Opt_test():
    def __init__(self, generate_data=True):
        self.dim = 2
        self.algo='BO'
        self.delta_flag = 1
        self.add_noise = True
        self.corr_beta = 25
        self.alpha_prim = np.array([0.5, 0.3, 0.1])
        
        self.k = kernel_opt(dim=self.dim, generate_data=generate_data)
        self.k.delta_flag = self.delta_flag
        if generate_data:
            self.k.add_noise = self.add_noise
            self.k.cal_delta(corr_beta=self.corr_beta, alpha_prim=self.alpha_prim)
            self.k.generate_data = False     
        
    def noise_test(self):
        # k.cal_delta(beta_corr=25, alpha_prim=0.5, generate_data=True)
        delta = self.k.cal_delta(beta_corr=self.corr_beta, alpha_prim=self.alpha_prim)
        self.k.visualize_distribution()
        
        return delta
    
    def init_points_test(self, Init_points_max=10):
        init_points_opt = 0
        init_points_max = Init_points_max
        
        target = np.zeros((3, init_points_max-1))
        target_diff = np.zeros((3, init_points_max-1))
        target_opt = 1e10

        for i in range(2, init_points_max+1):
            start_time = time.time()
            
            target[0, i-2], target[1, i-2] = self.k.optimierer(algo=self.algo, init_points=i)
            target_diff[0,i-2] = \
                abs(target[0,i-2] - self.k.corr_beta * self.k.alpha_prim) / (self.k.corr_beta * self.k.alpha_prim)
            target_diff[1,i-2] = target[2,i-2]
            
            end_time = time.time()
            target_diff[2,i-2]  = target[3, i-2] = end_time - start_time
            
            if target_opt > target[1, i-2]:
                target_opt = target[1, i-2]
                init_points_opt = i - 2
                
        return target, target_opt, init_points_opt, target_diff
    
    def mean_kernel(self, sample_num=10, same_time=True):
        para_opt_sample = np.zeros(sample_num)
        corr_beta_sample = np.zeros(sample_num)
        if self.dim == 1:
            alpha_prim_sample = np.zeros(sample_num)
            
        if self.dim == 2:
            alpha_prim_sample = np.zeros((4, sample_num))
            
        if same_time:
            for i in range(sample_num):
                para_opt_sample[i], _ = self.k.optimierer(algo=self.algo)
                corr_beta_sample[i] = self.k.corr_beta_opt
                alpha_prim_sample[:, i] = self.k.alpha_prim_opt
            
        para_opt = np.mean(para_opt_sample)
        corr_beta = np.mean(corr_beta_sample)
        alpha_prim = np.mean(alpha_prim_sample, axis=1)
        
        para_diff = abs(para_opt - self.k.corr_beta * np.sum(self.k.alpha_prim)) / (self.k.corr_beta * np.sum(self.k.alpha_prim))
        
        self.k.corr_beta_opt = corr_beta
        self.k.alpha_prim_opt = alpha_prim
        self.k.visualize_distribution(new_cal=True)
        
        return para_opt, para_diff
        
    def opt_test(self):

        para_opt, delta_opt = self.k.optimierer(algo=self.algo)
        
        para_diff = abs(para_opt - self.k.corr_beta * np.sum(self.k.alpha_prim)) / (self.k.corr_beta * np.sum(self.k.alpha_prim))
        
        self.k.visualize_distribution(new_cal=True)
        
        return para_opt, delta_opt, para_diff    
        
if __name__ == '__main__':
    
    Opt = Opt_test()
   
    Opt.dim = 2
    Opt.algo='BO'
    Opt.add_noise = True
    Opt.corr_beta = 25
    Opt.alpha_prim = np.array([0.5, 0.3, 0.1])
    
    test = 4
    
    if test == 1:
        delta = Opt.noise_test()
        
    elif test == 2:
        target, target_opt, init_points_opt, target_diff = Opt.init_points_test(Init_points_max=20)
        
    elif test == 3:
        para_opt, delta_opt, para_diff = Opt.opt_test()
        
    elif test == 4:
        para_opt, para_diff = Opt.mean_kernel(sample_num=10, same_time=True)

    