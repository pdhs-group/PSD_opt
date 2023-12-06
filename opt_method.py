# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:22:00 2023

@author: px2030
"""

import numpy as np
import time
import os
import warnings
import pandas as pd
from kernel_opt import kernel_opt
from PSD_Exp import write_read_exp

import matplotlib.pyplot as plt
import plotter.plotter as pt   

class opt_method():
    def __init__(self, add_noise, smoothing, corr_beta, alpha_prim, delta_flag=1, noise_type='Gaussian', noise_strength=0.01, t_vec=None):
        self.dim = 2
        self.algo='BO'
        
        self.k = kernel_opt(add_noise=add_noise, smoothing=smoothing, dim=self.dim, t_vec=t_vec, noise_type=noise_type, noise_strength=noise_strength)
        self.k.delta_flag = delta_flag
        self.k.corr_beta = corr_beta
        self.k.alpha_prim = alpha_prim

    def generate_synth_data(self, sample_num=1):
        self.k.cal_pop(self.k.corr_beta, self.k.alpha_prim)

        for i in range(0, sample_num):
            if i ==0:
                self.k.exp_data_path = self.k.exp_data_path.replace(".xlsx", f"_{i}.xlsx")
            else:
                self.k.exp_data_path = self.k.exp_data_path.replace(f"_{i-1}.xlsx", f"_{i}.xlsx")
            # print(self.k.exp_data_path)
            self.k.write_new_data()
        
    def mean_kernels(self, sample_num, method='kernels', data_name=None):
        if data_name == None:
            warnings.warn("Please specify the name of the training data without labels!")
        else:
            self.k.exp_data_path = os.path.join(self.k.base_path, data_name)
            
            if method == 'kernels':
                para_opt_sample = np.zeros(sample_num)
                delta_opt_sample = np.zeros(sample_num)
                corr_beta_sample = np.zeros(sample_num)
                if self.dim == 1:
                    alpha_prim_sample = np.zeros(sample_num)
                    
                if self.dim == 2:
                    alpha_prim_sample = np.zeros((3, sample_num))
                
                if sample_num == 1:
                    para_opt, delta_opt = self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo)
                    corr_beta = self.k.corr_beta_opt
                    alpha_prim = self.k.alpha_prim_opt
                else:
                    for i in range(0, sample_num):
                        if i ==0:
                            self.k.exp_data_path = self.k.exp_data_path.replace(".xlsx", f"_{i}.xlsx")
                        else:
                            self.k.exp_data_path = \
                                self.k.exp_data_path.replace(f"_{i-1}.xlsx", f"_{i}.xlsx")
                        para_opt_sample[i], delta_opt_sample[i] = \
                            self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo)
                        corr_beta_sample[i] = self.k.corr_beta_opt
                        alpha_prim_sample[:, i] = self.k.alpha_prim_opt
    

                if not sample_num == 1:
                    para_opt = np.mean(para_opt_sample)
                    delta_opt = np.mean(delta_opt_sample)
                    corr_beta = np.mean(corr_beta_sample)
                    alpha_prim = np.mean(alpha_prim_sample, axis=1)
                    self.k.corr_beta_opt = corr_beta
                    self.k.alpha_prim_opt = alpha_prim
                
            elif method == 'delta':
                para_opt, delta_opt = \
                    self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo, sample_num=sample_num)
                
            elif method == 'time_kernels':
                sample_num=len(self.k.t_vec)
                para_opt_sample = np.zeros(sample_num)
                delta_opt_sample = np.zeros(sample_num)
                corr_beta_sample = np.zeros(sample_num)
                if self.dim == 1:
                    alpha_prim_sample = np.zeros(sample_num)
                    
                if self.dim == 2:
                    alpha_prim_sample = np.zeros((3, sample_num))
                
                # t_step=0 is initial conditions which should be excluded
                for i in range(1, sample_num):
                    para_opt_sample[i], delta_opt_sample[i] = self.k.optimierer(t_step=i, algo=self.algo)
                    corr_beta_sample[i] = self.k.corr_beta_opt
                    alpha_prim_sample[:, i] = self.k.alpha_prim_opt
                para_opt = np.mean(para_opt_sample)
                delta_opt = np.mean(delta_opt_sample)
                corr_beta = np.mean(corr_beta_sample)
                alpha_prim = np.mean(alpha_prim_sample, axis=1)
                self.k.corr_beta_opt = corr_beta
                self.k.alpha_prim_opt = alpha_prim
                
                
            if self.dim == 1:
                para_diff = abs(para_opt - self.k.corr_beta * np.sum(self.k.alpha_prim)) / (self.k.corr_beta * np.sum(self.k.alpha_prim))
            elif self.dim == 2:
                para_diff = np.zeros(4)
                para_diff[0] = abs(self.k.corr_beta_opt- self.k.corr_beta) / self.k.corr_beta
                para_diff[1:] = abs(self.k.alpha_prim_opt - self.k.alpha_prim) / self.k.alpha_prim
            
            return self.k.corr_beta_opt, self.k.alpha_prim_opt, para_opt, para_diff, delta_opt
        
        