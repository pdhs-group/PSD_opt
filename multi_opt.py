# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:05:42 2023

@author: px2030
"""

import os
import warnings
import numpy as np
import pandas as pd
from pop import population
from bayes_opt import BayesianOptimization
from skopt import gp_minimize
from skopt.space import Real
# from scipy.stats import gaussian_kde
# import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
# from functools import partial
import ast
from PSD_Exp import write_read_exp
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt          
# from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

class kernel_opt():
    def __init__(self, add_noise=True, smoothing=True, dim=1, disc='geo', 
                 noise_type='Gaussian', noise_strength=0.01, t_vec=None):
        # delta_flag = 1: use q3
        # delta_flag = 2: use Q3
        # delta_flag = 3: use x_50
        self.delta_flag = 1         
        self.add_noise = add_noise
        self.smoothing = smoothing
        self.noise_type = noise_type    # Gaussian, Uniform, Poisson, Multiplicative
        self.noise_strength = noise_strength
        self.t_vec = t_vec
        self.num_t_steps = len(t_vec)
        self.Multi_Opt = False
        if not self.Multi_Opt:
            self.p = population(dim=dim, disc=disc)
        else:
            self.create_all_pop(disc)
        
        # Set the base path for exp_data_path
        self.base_path = os.path.join(self.p.pth, "data\\")

        # Check if noise should be added
        if self.add_noise:
            # Modify the file name to include noise type and strength
            filename = f"CED_focus_Sim_{self.noise_type}_{self.noise_strength}.xlsx"
        else:
            # Use the default file name
            filename = "CED_focus_Sim.xlsx"

        # Combine the base path with the modified file name
        self.exp_data_path = os.path.join(self.base_path, filename)
        
        self.filename_kernels = os.path.join(self.base_path, "kernels.txt")
        
    def cal_delta(self, corr_beta=None, alpha_prim=None, scale=1, Q3_exp=None, x_50_exp=None, sample_num=1):
        # read the kernels of original data
        if not os.path.exists(self.filename_kernels):
            warnings.warn("file does not exist: {}".format(self.filename_kernels))
            
        else:
            with open(self.filename_kernels, 'r') as file:
                lines = file.readlines()  
                
                # self.corr_beta = None
                # self.alpha_prim = None
                for line in lines:
                    if 'CORR_BETA:' in line:
                        self.corr_beta = float(line.split(':')[1].strip())
                    elif 'alpha_prim:' in line:
                        array_str = line.split(':')[1].strip()
                        # array_str = array_str.replace(" ", ", ")
                        self.alpha_prim = np.array(ast.literal_eval(array_str))
        
        # x_uni = self.cal_x_uni(self.p)
        self.cal_pop(corr_beta, alpha_prim)

        if not self.smoothing:
            '''
            data = self.p.return_num_distribution_fixed(t=t_step)
            # Conversion unit
            indexes = [0, 3, 4, 5]
            x_uni, x_10, x_50, x_90 = [data[i] for i in indexes]

            x_uni *= 1e6    
            x_10 *= 1e6   
            x_50 *= 1e6   
            x_90 *= 1e6

            # read and calculate the experimental data
            t = self.p.t_vec[t_step]
            delta_sum = 0
            
            if sample_num == 1:
                data_exp = self.read_exp(x_uni, t) 
                delta = self.cost_fun(data_exp[self.delta_flag], data[self.delta_flag])
                
                return (delta * scale)
            else:
                for i in range (0, sample_num):
                    if i ==0:
                        self.exp_data_path = self.exp_data_path.replace(".xlsx", f"_{i}.xlsx")
                    else:
                        self.exp_data_path = self.exp_data_path.replace(f"_{i-1}.xlsx", f"_{i}.xlsx")
                    data_exp = self.read_exp(x_uni, t) 
                    # Calculate the error between experimental data and simulation results
                    delta = self.cost_fun(data_exp[self.delta_flag], data[self.delta_flag])
                    delta_sum +=delta
                # Restore the original name of the file to prepare for the next step of training
                self.exp_data_path = self.exp_data_path.replace(f"_{sample_num-1}.xlsx", ".xlsx")
                delta_sum /= sample_num
            '''
            return
        else:
            kde_list = []
            x_uni = self.cal_x_uni(self.p)
            for idt in range(self.num_t_steps):
                _, q3, Q3, x_10, x_50, x_90 = self.p.return_num_distribution_fixed(t=idt)
                
                kde = self.KDE_fit(x_uni, q3)
                kde_list.append(kde)
            
            if sample_num == 1:
                data_exp = self.read_exp() 
                sumN_uni = np.zeros((len(data_exp[0]), self.num_t_steps))
                for idt in range(self.num_t_steps):
                    sumN_uni_tem = self.KDE_score(kde_list[idt], data_exp[0])
                    sumN_uni[:, idt] = sumN_uni_tem
                data_mod = self.re_cal_distribution(data_exp[0], sumN_uni)
                # Calculate the error between experimental data and simulation results
                delta = self.cost_fun(data_exp[self.delta_flag], data_mod[self.delta_flag])
                
                return (delta * scale)
            else:
                delta_sum = 0           
                for i in range (0, sample_num):
                    self.traverse_path(i)
                    data_exp = self.read_exp() 
                    sumN_uni = np.zeros((len(data_exp[0]), self.num_t_steps))
                    
                    for idt in range(self.num_t_steps):
                        sumN_uni_tem = self.KDE_score(kde_list[idt], data_exp[0])
                        sumN_uni[:, idt] = sumN_uni_tem
                        
                    data_mod = self.re_cal_distribution(data_exp[0], sumN_uni)
                    # Calculate the error between experimental data and simulation results
                    delta = self.cost_fun(data_exp[self.delta_flag], data_mod[self.delta_flag])
                    delta_sum +=delta
                # Restore the original name of the file to prepare for the next step of training
                self.exp_data_path = self.exp_data_path.replace(f"_{sample_num-1}.xlsx", ".xlsx")
                delta_sum /= sample_num
                    
                return (delta_sum * scale)
        

        
    def write_new_data(self):
        # save the kernels
        if self.p.dim == 1:
            with open(self.filename_kernels, 'w') as file:
                file.write('CORR_BETA: {}\n'.format(self.corr_beta))
                file.write('alpha_prim: {}\n'.format(self.alpha_prim))
        else:
            alpha_prim_str = ', '.join([str(x) for x in self.alpha_prim])
            with open(self.filename_kernels, 'w') as file:
                file.write('CORR_BETA: {}\n'.format(self.corr_beta))
                file.write('alpha_prim: {}\n'.format(alpha_prim_str))
        
        # save the calculation result in experimental data form
        x_uni = self.cal_x_uni(self.p)
        formatted_times = write_read_exp.convert_seconds_to_time(self.p.t_vec)
        sumN_uni = np.zeros((len(x_uni), self.num_t_steps))
        
        for idt in range(self.num_t_steps):
            _, q3, _, _, _, _ = self.p.return_num_distribution_fixed(t=idt)
            kde = self.KDE_fit(x_uni, q3)
            if self.smoothing:
                kde = self.KDE_fit(x_uni, q3)
                sumN_uni[:, idt] = self.KDE_score(kde, x_uni)
                
        _, q3, _, _, _,_ = self.re_cal_distribution(x_uni, sumN_uni)
        # add noise to the original data
        if self.add_noise:
            q3 = self.function_noise(q3)
        df = pd.DataFrame(data=q3, index=x_uni, columns=formatted_times)
        df.index.name = 'Circular Equivalent Diameter'
        # save DataFrame as Excel file
        df.to_excel(self.exp_data_path)
        
        return 
        
    def cal_pop(self, corr_beta, alpha_prim):
        if not self.Multi_Opt:
            self.p.COLEVAL = 2
            self.p.EFFEVAL = 2
            self.p.CORR_BETA = corr_beta
            if self.p.dim == 1:
                alpha_prim_temp = alpha_prim
            elif self.p.dim == 2:
                alpha_prim_temp = np.zeros(4)
                alpha_prim_temp[0] = alpha_prim[0]
                alpha_prim_temp[1] = alpha_prim_temp[2] = alpha_prim[1]
                alpha_prim_temp[3] = alpha_prim[2]
            self.p.alpha_prim = alpha_prim_temp
            self.p.full_init(calc_alpha=False)
            self.p.solve_PBE(t_vec=self.t_vec)
        
        else:
            self.p_N.COLEVAL = 2
            self.p_N.EFFEVAL = 2
            self.p_N.CORR_BETA = corr_beta
            alpha_prim_temp = alpha_prim[0]
            self.p_N.alpha_prim = alpha_prim_temp
            self.p_N.full_init(calc_alpha=False)
            self.p_N.solve_PBE(t_vec=self.t_vec)
            
            self.p_M.COLEVAL = 2
            self.p_M.EFFEVAL = 2
            self.p_M.CORR_BETA = corr_beta
            alpha_prim_temp = alpha_prim[2]
            self.p_M.alpha_prim = alpha_prim_temp
            self.p_M.full_init(calc_alpha=False)
            self.p_M.solve_PBE(t_vec=self.t_vec)
            
            self.p_mix.COLEVAL = 2
            self.p_mix.EFFEVAL = 2
            self.p_mix.CORR_BETA = corr_beta
            alpha_prim_temp = np.zeros(4)
            alpha_prim_temp[0] = alpha_prim[0]
            alpha_prim_temp[1] = alpha_prim_temp[2] = alpha_prim[1]
            alpha_prim_temp[3] = alpha_prim[2]
            self.p_mix.alpha_prim = alpha_prim_temp
            self.p_mix.full_init(calc_alpha=False)
            self.p_mix.solve_PBE(t_vec=self.t_vec)
            
            
    
    # Read the experimental data and re-interpolate the particle distribution 
    # of the experimental data according to the simulation results.
    def read_exp(self, x_uni=None):
        
        compare = write_read_exp(self.exp_data_path, read=True)
        df = compare.get_exp_data(self.t_vec)
        x_uni_exp = df.index.to_numpy()
        q3_exp = df.to_numpy()
        if not self.smoothing:
            # If the experimental data has a different particle size scale than the simulations, 
            # interpolation is required
            q3_exp_interpolated = np.interp(x_uni, x_uni_exp, q3_exp)
            q3_exp_interpolated_clipped = np.clip(q3_exp_interpolated, 0, 1)
            q3_exp_interpolated = q3_exp_interpolated_clipped / np.sum(q3_exp_interpolated_clipped)
            q3_exp = pd.Series(q3_exp_interpolated, index=x_uni)
            Q3_exp = q3_exp.cumsum()
            x_10_exp = np.interp(0.1, Q3_exp, x_uni)
            x_50_exp = np.interp(0.5, Q3_exp, x_uni)
            x_90_exp = np.interp(0.9, Q3_exp, x_uni)
        
            # q3_exp has been redistributed according to x_uni, so actually x_uni rather than x_uni_exp is returned
            return x_uni, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp
        
        # Smoothing can return the value of the simulation result at any x, 
        # so there is no need to process the experimental data.
        else:
            return self.re_cal_distribution(x_uni_exp, q3_exp)
    
    def cost_fun(self, data_exp, data_mod):
        return ((data_mod*100-data_exp*100)**2).sum()
    
    def optimierer(self, algo='BO', init_points=4, Q3_exp=None, x_50_exp=None, sample_num=1, hyperparameter=None):
        if algo == 'BO':
            if self.p.dim == 1:
                pbounds = {'corr_beta': (0, 50), 'alpha_prim': (0, 1)}
                objective = lambda corr_beta, alpha_prim: self.cal_delta(
                    corr_beta=corr_beta, alpha_prim=np.array([alpha_prim]),
                    scale=-1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, sample_num=sample_num)
                
            elif self.p.dim == 2:
                pbounds = {'corr_beta': (0, 50), 'alpha_prim_0': (0, 1), 'alpha_prim_1': (0, 1), 'alpha_prim_2': (0, 1)}
                objective = lambda corr_beta, alpha_prim_0, alpha_prim_1, alpha_prim_2: self.cal_delta(
                    corr_beta=corr_beta, 
                    alpha_prim=np.array([alpha_prim_0, alpha_prim_1, alpha_prim_2]), 
                    scale=-1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, sample_num=sample_num)
                
            opt = BayesianOptimization(
                f=objective, 
                pbounds=pbounds,
                random_state=1,
            )
            
            opt.maximize(
                init_points=init_points,
                n_iter=100,
            )   
            if self.p.dim == 1:
                para_opt = opt.max['params']['corr_beta'] * opt.max['params']['alpha_prim']
                self.corr_beta_opt = opt.max['params']['corr_beta']
                self.alpha_prim_opt = opt.max['params']['alpha_prim']
                
            elif self.p.dim == 2:
                self.alpha_prim_opt = np.zeros(3)
                para_opt = opt.max['params']['corr_beta'] *\
                (opt.max['params']['alpha_prim_0'] + opt.max['params']['alpha_prim_1'] + opt.max['params']['alpha_prim_2'])
                self.corr_beta_opt = opt.max['params']['corr_beta']
                self.alpha_prim_opt[0] = opt.max['params']['alpha_prim_0']
                self.alpha_prim_opt[1] = opt.max['params']['alpha_prim_1']
                self.alpha_prim_opt[2] = opt.max['params']['alpha_prim_2']
            
            delta_opt = -opt.max['target']
            
        if algo == 'gp_minimize':
            if self.p.dim == 1:
                space = [Real(0, 100), Real(0, 1)]
                objective = lambda params: self.cal_delta(corr_beta=params[0], alpha_prim=np.array([params[1]]), 
                                                          scale=1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, 
                                                          sample_num=sample_num)
            elif self.p.dim == 2:
                space = [Real(0, 100), Real(0, 1), Real(0, 1), Real(0, 1)]
                objective = lambda params: self.cal_delta(
                    corr_beta=params[0], 
                    alpha_prim=np.array([params[1], params[2], params[3]]), 
                    scale=1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, sample_num=sample_num)

            opt = gp_minimize(
                objective,
                space,
                n_calls=100,
                n_initial_points=init_points,
                random_state=0
            )

            if self.p.dim == 1:
                para_opt = opt.x[0] * opt.x[1]
                self.corr_beta_opt = opt.x[0]
                self.alpha_prim_opt = opt.x[1]
            elif self.p.dim == 2:
                self.alpha_prim_opt = np.zeros(3)
                para_opt = opt.x[0] * sum(opt.x[1:])
                self.corr_beta_opt = opt.x[0]
                self.alpha_prim_opt[0] = opt.x[1]
                self.alpha_prim_opt[1] = opt.x[2]
                self.alpha_prim_opt[2] = opt.x[3]

            delta_opt = opt.fun
            
        return para_opt, delta_opt
    
    # Visualize only the last time step of the specified time vector and the last used experimental data
    def visualize_distribution(self, exp_data_name=None,ax=None,fig=None,
                               close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    # Recalculate PSD using original parameter
        self.cal_pop(corr_beta=self.corr_beta, alpha_prim=self.alpha_prim)
        '''if self.p.dim == 1:       
            self.p.CORR_BETA = self.corr_beta
            self.p.alpha_prim=self.corr_beta
   
        elif self.p.dim == 2:
            self.p.CORR_BETA = self.corr_beta 
            self.p.alpha_prim[:]=[self.corr_beta[0], self.corr_beta[1], self.corr_beta[2], self.corr_beta[3]]

        self.p.full_init(calc_alpha=False)
        self.p.solve_PBE(t_vec=self.t_vec)'''
        x_uni_ori, q3_ori, Q3_ori, x_10_ori, x_50_ori, x_90_ori = self.p.return_num_distribution_fixed(t=len(self.p.t_vec)-1)
        # Conversion unit
        x_uni_ori *= 1e6    
        x_10_ori *= 1e6   
        x_50_ori *= 1e6   
        x_90_ori *= 1e6  
        if self.smoothing:
            kde = self.KDE_fit(x_uni_ori, q3_ori)
            sumN_uni = self.KDE_score(kde, x_uni_ori)
            _, q3_ori, Q3_ori, _, _,_ = self.re_cal_distribution(x_uni_ori, sumN_uni)

        # Recalculate PSD using optimization results
        if hasattr(self, 'corr_beta_opt') and hasattr(self, 'alpha_prim_opt'):
            self.cal_pop(self.corr_beta_opt, self.alpha_prim_opt)
        else:
            print("Need to run the optimization process at least once！")    
            
        x_uni, q3, Q3, x_10, x_50, x_90 = self.p.return_num_distribution_fixed(t=len(self.p.t_vec)-1)
        # Conversion unit
        x_uni *= 1e6    
        x_10 *= 1e6   
        x_50 *= 1e6   
        x_90 *= 1e6  
        if self.smoothing:
            kde = self.KDE_fit(x_uni, q3)
            sumN_uni = self.KDE_score(kde, x_uni)
            _, q3, Q3, _, _,_ = self.re_cal_distribution(x_uni, sumN_uni)
        
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig=plt.figure()    
            axq3=fig.add_subplot(1,2,1)   
            axQ3=fig.add_subplot(1,2,2)   
            
        
        axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $q3$ / $-$',
                               lbl='q3_mod',clr='b',mrk='o')
        

        axq3, fig = pt.plot_data(x_uni_ori, q3_ori, fig=fig, ax=axq3,
                               lbl='q3_ori',clr='r',mrk='v')
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $Q3$ / $-$',
                               lbl='Q3_mod',clr='b',mrk='o')

        axQ3, fig = pt.plot_data(x_uni_ori, Q3_ori, fig=fig, ax=axQ3,
                               lbl='Q3_ori',clr='r',mrk='v')

        if not exp_data_name == None:
            # read and calculate the experimental data
            t = max(self.p.t_vec)
    
            x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp = self.read_exp(x_uni, t)
            
            axq3, fig = pt.plot_data(x_uni_exp, q3_exp, fig=fig, ax=axq3,
                                   lbl='q3_exp',clr='g',mrk='^')
            axQ3, fig = pt.plot_data(x_uni_exp, Q3_exp, fig=fig, ax=axQ3,
                                   lbl='Q3_exp',clr='g',mrk='^')
        
        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return axq3, axQ3, fig
    
    
    def function_noise(self, ori_data):
        rows, cols = ori_data.shape
        noise = np.zeros((rows, cols))
        if self.noise_type == 'Gaussian':
            # The first parameter 0 represents the mean value of the noise, 
            # the second parameter is the standard deviation of the noise,
            for i in range(cols):
                noise[:, i] = np.random.normal(0, self.noise_strength, rows)              
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Uniform':
            # Noises are uniformly distributed over the half-open interval [low, high)
            for i in range(cols):
                noise[:, i] = np.random.uniform(low=-self.noise_strength/2, high=self.noise_strength/2, size=rows)
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Poisson':
            for i in range(cols):
                noise[:, i] = np.random.poisson(self.noise_strength, rows)
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Multiplicative':
            for i in range(cols):
                noise[:, i] = np.random.normal(1, self.noise_strength, rows)
            noised_data = ori_data * noise
        # Cliping the data out of range and rescale the data    
        noised_data_clipped = np.clip(noised_data, 0, 1)
        cols_sums = np.sum(noised_data_clipped, axis=0, keepdims=True)
        noised_data = noised_data_clipped /cols_sums
        
        return noised_data

    ## Kernel density estimation
    ## data_ori must be a quantity rather than a relative value!
    def KDE_fit(self, x_uni_ori, data_ori, bandwidth='scott', kernel_func='gaussian'):
        '''
        # estimate the value of bandwidth
        # Bootstrapping method is used 
        # Because the original data may not conform to the normal distribution
        if bandwidth == None:
            n_bootstrap = 100
            estimated_bandwidth = np.zeros(n_bootstrap)
            
            for i in range(n_bootstrap):
                sample_indices = np.random.choice(len(x_uni_ori), size=len(x_uni_ori), p=data_ori)
                sample = x_uni_ori[sample_indices]
                kde = gaussian_kde(sample)
                estimated_bandwidth[i] = kde.factor * np.std(sample, ddof=1)
            # bandwidth = np.median(estimated_bandwidth)
            bandwidth = np.mean(estimated_bandwidth)
        '''
        
        # KernelDensity requires input to be a column vector
        # So x_uni_re must be reshaped
        x_uni_ori_re = x_uni_ori.reshape(-1, 1)
        # Avoid divide-by-zero warnings when calculating KDE
        data_ori_adjested = np.where(data_ori == 0, 1e-20, data_ori)
        
        kde = KernelDensity(kernel=kernel_func, bandwidth=bandwidth)
        kde.fit(x_uni_ori_re, sample_weight=data_ori_adjested)
        
        return kde
    
    def KDE_score(self, kde, x_uni_new):
        
        x_uni_new_re = x_uni_new.reshape(-1, 1) 
        data_smoothing = np.exp(kde.score_samples(x_uni_new_re))
        
        # Flatten a column vector into a one-dimensional array
        data_smoothing = data_smoothing.ravel()
        
        return data_smoothing
            
    def re_cal_distribution(self, x_uni, sumN_uni):
        if sumN_uni.ndim == 1:
            sumN = np.sum(sumN_uni)
            
            Q3 = np.cumsum(sumN_uni)/sumN
            q3 = sumN_uni/sumN

            x_10 = np.interp(0.1, Q3, x_uni)
            x_50 = np.interp(0.5, Q3, x_uni)
            x_90 = np.interp(0.9, Q3, x_uni)
        else:
            sumN = np.sum(sumN_uni, axis=0)
            
            Q3 = np.cumsum(sumN_uni, axis=0)/sumN
            q3 = sumN_uni/sumN
            
            x_10 = np.zeros(self.num_t_steps)
            x_50 = np.zeros(self.num_t_steps)
            x_90 = np.zeros(self.num_t_steps)
            for idt in range(self.num_t_steps):
                x_10[idt] = np.interp(0.1, Q3[:, idt], x_uni)
                x_50[idt] = np.interp(0.5, Q3[:, idt], x_uni)
                x_90[idt] = np.interp(0.9, Q3[:, idt], x_uni)
        
        return x_uni, q3, Q3, x_10, x_50, x_90
    
    def cal_x_uni(self, population):
        v_uni = np.setdiff1d(population.V, [-1, 0])
        # Because the length unit in the experimental data is millimeters 
        # and in the simulation it is meters, so it needs to be converted 
        # before use.
        return (6*v_uni/np.pi)**(1/3)*1e6

    def traverse_path(self, label):
        if label ==0:
            self.exp_data_path = self.exp_data_path.replace(".xlsx", f"_{label}.xlsx")
        else:
            self.exp_data_path = self.exp_data_path.replace(f"_{label-1}.xlsx", f"_{label}.xlsx")
        
    def create_all_pop(self, disc):
        self.p_N = population(1,disc=disc)
        self.p_M = population(1,disc=disc)
        self.p_mix = population(2,disc=disc)
        
        # parameter for particle component 1 - NM
        self.p_N.R01 = 2.9e-7 
        self.p_N.DIST1 = os.path.join(self.p_N.pth,"data\\PSD_data\\")+'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        
        # parameter for particle component 2 - M
        self.p_M.R01 = 2.9e-7 
        self.p_M.DIST1 = os.path.join(self.p_M.pth,"data\\PSD_data\\")+'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        # Pass parameters from 1D simulations to 2D
        self.p_mix.R01 = self.p_N.R01
        self.p_mix.R03 = self.p_M.R01
        self.p_mix.DIST1 = self.p_N.DIST1
        self.p_mix.DIST3 = self.p_M.DIST1