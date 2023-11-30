# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:34:14 2023

@author: px2030
"""
import sys
import os
import numpy as np
import time
import pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from kerns_opt import kernel_opt
from PSD_Exp import write_read_exp

import matplotlib.pyplot as plt
import plotter.plotter as pt   

class Opt_test():
    def __init__(self, add_noise, corr_beta, alpha_prim, delta_flag=1, noise_type='Gaussian', noise_strength=0.01, t_vec=None, generate_data=True):
        self.dim = 2
        self.algo='BO'
        
        self.k = kernel_opt(add_noise=add_noise, dim=self.dim, t_vec=t_vec, noise_type=noise_type, noise_strength=noise_strength)
        self.k.delta_flag = delta_flag
        if generate_data:
            self.k.cal_pop(corr_beta, alpha_prim)
            self.k.generate_new_data()
    
    # Test for BO with different number of init_points
    # only 1D
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
    # 
    def mean_kernel(self, sample_num=10, method=1):
        if method ==1:
            sample_num=len(self.k.t_vec)
            para_opt_sample = np.zeros(sample_num)
            corr_beta_sample = np.zeros(sample_num)
            if self.dim == 1:
                alpha_prim_sample = np.zeros(sample_num)
                
            if self.dim == 2:
                alpha_prim_sample = np.zeros((4, sample_num))
            
            # t_step=0 is initial conditions which should be excluded
            for i in range(1, sample_num):
                para_opt_sample[i], _ = self.k.optimierer(t_step=i, algo=self.algo)
                corr_beta_sample[i] = self.k.corr_beta_opt
                alpha_prim_sample[:, i] = self.k.alpha_prim_opt
            para_opt = np.mean(para_opt_sample)
            corr_beta = np.mean(corr_beta_sample)
            alpha_prim = np.mean(alpha_prim_sample, axis=1)
            self.k.corr_beta_opt = corr_beta
            self.k.alpha_prim_opt = alpha_prim
        
        if method ==2:
            para_opt_sample = np.zeros(sample_num)
            corr_beta_sample = np.zeros(sample_num)
            if self.dim == 1:
                alpha_prim_sample = np.zeros(sample_num)
                
            if self.dim == 2:
                alpha_prim_sample = np.zeros((4, sample_num))
                
            for i in range(0, sample_num):
                para_opt_sample[i], _ = self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo)
                corr_beta_sample[i] = self.k.corr_beta_opt
                alpha_prim_sample[:, i] = self.k.alpha_prim_opt
                self.k.generate_new_data()

            para_opt = np.mean(para_opt_sample)
            corr_beta = np.mean(corr_beta_sample)
            alpha_prim = np.mean(alpha_prim_sample, axis=1)
            self.k.corr_beta_opt = corr_beta
            self.k.alpha_prim_opt = alpha_prim
            
        if method ==3:
            x_uni, _, _, _, _, _ = self.k.p.return_num_distribution(t=len(self.k.t_vec)-1)
            x_uni *= 1e6 
            q3_exp = np.zeros((sample_num, len(x_uni)))
             
            for i in range(0, sample_num):
                if i ==0:
                    self.k.exp_data_path = self.k.exp_data_path.replace(".xlsx", f"_{i}.xlsx")
                else:
                    self.k.exp_data_path = self.k.exp_data_path.replace(f"_{i-1}.xlsx", f"_{i}.xlsx")
                # print(self.k.exp_data_path)
                self.k.generate_new_data()
                '''
                # Read the data from the last time step and calculate average
                t = self.k.t_vec[len(self.k.t_vec)-1]   
                _, q3_exp[i,:], _, _, _, _ = self.k.read_exp(x_uni, t)   
                
            q3_exp_mean = np.mean(q3_exp, 0)
            Q3_exp_mean = q3_exp_mean.cumsum()
            x_50_exp_mean = np.interp(0.5, Q3_exp_mean, x_uni)
            para_opt, _ = self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo, Q3_exp=Q3_exp_mean, x_50_exp=x_50_exp_mean)
            '''
            df = pd.DataFrame(index=x_uni)
            df.index.name = 'Circular Equivalent Diameter'
            formatted_times = write_read_exp.convert_seconds_to_time(self.k.t_vec)

            for idt in range(0, len(self.k.t_vec)):
                t = self.k.t_vec[idt]
                for i in range(0, sample_num):
                    if i ==0:
                        self.k.exp_data_path = self.k.exp_data_path.replace(f"_{sample_num-1}.xlsx", f"_{i}.xlsx")
                    else:
                        self.k.exp_data_path = self.k.exp_data_path.replace(f"_{i-1}.xlsx", f"_{i}.xlsx")
                        
                    _, q3_exp[i,:], _, _, _, _ = self.k.read_exp(x_uni, t)   
                    
                q3_exp_mean = np.mean(q3_exp, 0)
                df[formatted_times[idt]] = q3_exp_mean

            # save DataFrame as Excel file
            self.k.exp_data_path = self.k.exp_data_path.replace(f"_{sample_num-1}.xlsx", ".xlsx")
            df.to_excel(self.k.exp_data_path)
            para_opt, delta_opt = self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo)
        
        if method ==4:
            x_uni, _, _, _, _, _ = self.k.p.return_num_distribution(t=len(self.k.t_vec)-1)
            x_uni *= 1e6 
            q3_exp = np.zeros((sample_num, len(x_uni)))
             
            for i in range(0, sample_num):
                if i ==0:
                    self.k.exp_data_path = self.k.exp_data_path.replace(".xlsx", f"_{i}.xlsx")
                else:
                    self.k.exp_data_path = self.k.exp_data_path.replace(f"_{i-1}.xlsx", f"_{i}.xlsx")
                # print(self.k.exp_data_path)
                self.k.generate_new_data()

            para_opt, delta_opt = self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo, sample_num=sample_num)
        
        if self.dim == 1:
            para_diff = abs(para_opt - self.k.corr_beta * np.sum(self.k.alpha_prim)) / (self.k.corr_beta * np.sum(self.k.alpha_prim))
        elif self.dim == 2:
            para_diff = np.zeros(5)
            para_diff[0] = abs(self.k.corr_beta_opt- self.k.corr_beta) / self.k.corr_beta
            para_diff[1:] = abs(self.k.alpha_prim_opt - self.k.alpha_prim) / self.k.alpha_prim
            
        if self.dim == 1:
            para_diff = abs(para_opt - self.k.corr_beta * np.sum(self.k.alpha_prim)) / (self.k.corr_beta * np.sum(self.k.alpha_prim))
        elif self.dim == 2:
            para_diff = np.zeros(5)
            para_diff[0] = abs(self.k.corr_beta_opt- self.k.corr_beta) / self.k.corr_beta
            para_diff[1:] = abs(self.k.alpha_prim_opt - self.k.alpha_prim) / self.k.alpha_prim
        
        self.k.visualize_distribution()
        
        return self.k.corr_beta_opt, self.k.alpha_prim_opt, para_opt, para_diff, delta_opt
        
    def opt_test(self):

        para_opt, delta_opt = self.k.optimierer(t_step=len(self.k.t_vec)-1, algo=self.algo)
        
        para_diff = abs(para_opt - self.k.corr_beta * np.sum(self.k.alpha_prim)) / (self.k.corr_beta * np.sum(self.k.alpha_prim))
        
        self.k.visualize_distribution()
        
        return para_opt, delta_opt, para_diff    
        
    def smoothing_test(self):
        x_uni_ori, q3_ori, Q3_ori, x_10_ori, x_50_ori, x_90_ori = self.k.p.return_num_distribution(t=len(self.k.t_vec)-1)
        # Conversion unit
        x_uni_ori *= 1e6    
        x_10_ori *= 1e6   
        x_50_ori *= 1e6   
        x_90_ori *= 1e6  
        
        x_uni, q3, _, _, _, _ = self.k.p.return_num_distribution(t=len(self.k.t_vec)-1)
        
        # bandwidth = None: use Bootstrapping method to estimate bandwidth
        # kernel_func = 'Gaussian': use Gaussian kernel
        # kernel_func = 'tri': use triangle kernel
        # kernel_func = 'epa':use Epanechnikov kernel
        sumN_uni = self.k.KDE_smoothing(x_uni, q3, bandwidth=None, kernel_func='uni')
        sumN = np.sum(sumN_uni)

        Q3 = np.cumsum(sumN_uni)/sumN
        q3 = sumN_uni/np.sum(sumN_uni)
    
        
        
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)

        fig=plt.figure()    
        axq3=fig.add_subplot(1,2,1)   
        axQ3=fig.add_subplot(1,2,2)   
            
        
        axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $q3$ / $-$',
                               lbl='q3_smooth',clr='b',mrk='o')
        axq3, fig = pt.plot_data(x_uni_ori, q3_ori, fig=fig, ax=axq3,
                               lbl='q3_ori',clr='r',mrk='v')
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $Q3$ / $-$',
                               lbl='Q3_smooth',clr='b',mrk='o')
        axQ3, fig = pt.plot_data(x_uni_ori, Q3_ori, fig=fig, ax=axQ3,
                               lbl='Q3_ori',clr='r',mrk='v')

        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return axq3, axQ3, fig
        
if __name__ == '__main__':
    corr_beta = 25
    alpha_prim = np.array([0.5, 0.3, 0.1])
    t_vec = np.arange(0, 601, 60, dtype=float)
    
    # delta_flag = 1: use q3
    # delta_flag = 2: use Q3
    # delta_flag = 3: use x_50
    delta_flag = 1
    # noise_type: Gaussian, Uniform, Poisson, Multiplicative
    add_noise = False
    noise_type='Gaussian'
    noise_strength = 0.01
    
    Opt = Opt_test(add_noise, corr_beta, alpha_prim, t_vec=t_vec, noise_type=noise_type, noise_strength=noise_strength, generate_data=True)
    Opt.dim = 2
    Opt.algo='BO'
    
    test = 4
        
    if test == 1:
        para_opt, delta_opt, para_diff = Opt.opt_test()
        
    elif test == 2:
        target, target_opt, init_points_opt, target_diff = Opt.init_points_test(Init_points_max=20)
        
    elif test == 3:
        # method=1: Using different time points in a dataset
        # method=2: Using different datasets at same time points, mean kernels
        # method=3: Using different datasets at same time points, mean datasets
        # method=4: Using different datasets at same time points, mean delta
        corr_beta_opt, alpha_prim_opt, para_opt, para_diff, delta_opt = Opt.mean_kernel(sample_num=100, method=4)
    elif test == 4:
        Opt.smoothing_test()
    else:
        print('Current test not available')
    