# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:38:56 2023

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
    def __init__(self, add_noise=True, smoothing=True, dim=1, disc='geo', noise_type='Gaussian', noise_strength=0.01, t_vec=None):
        # kernel = 1: optimazition with constant beta and alpha_prim, but too slow, don't use
        # kernel = 2: optimazition with constant corr_beta and alpha_prim
        # kernel = 3: optimazition with constant corr_beta and calculated alpha_prim
        self.kernel = 2
        # delta_flag = 1: use q3
        # delta_flag = 2: use Q3
        # delta_flag = 3: use x_50
        self.delta_flag = 1         
        self.add_noise = add_noise
        self.smoothing = smoothing
        self.noise_type = noise_type    # Gaussian, Uniform, Poisson, Multiplicative
        self.noise_strength = noise_strength
        self.t_vec = t_vec
        self.p = population(dim=dim, disc=disc)
        
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
        
    def cal_delta(self, corr_beta=None, alpha_prim=None, scale=1, t_step=0, Q3_exp=None, x_50_exp=None, sample_num=1):
        
        self.cal_pop(corr_beta, alpha_prim)
            
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
                        
        if t_step >= len(self.p.t_vec):
            raise ValueError("Current time step is out of the range of the data table.")
            
        else:
            x_uni, q3, Q3, x_10, x_50, x_90 = self.p.return_num_distribution_fixed(t=t_step)
            # Conversion unit
            x_uni *= 1e6    
            x_10 *= 1e6   
            x_50 *= 1e6   
            x_90 *= 1e6
            if self.smoothing:
                sumN_uni = self.KDE_smoothing(x_uni, q3, bandwidth='scott', kernel_func='gaussian')
                sumN = np.sum(sumN_uni)

                Q3 = np.cumsum(sumN_uni)/sumN
                q3 = sumN_uni/np.sum(sumN_uni)
                
                x_10 = np.interp(0.1, Q3, x_uni)
                x_50 = np.interp(0.5, Q3, x_uni)
                x_90 = np.interp(0.9, Q3, x_uni)

            # read and calculate the experimental data
            t = self.p.t_vec[t_step]
            delta_sum = 0
            
            if Q3_exp != None and x_50_exp != None:
                x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp = self.read_exp(x_uni, t) 
                delta = self.cost_fun(q3_exp, q3, Q3_exp, Q3, x_50_exp, x_50)
            else:
                if sample_num == 1:
                    x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp = self.read_exp(x_uni, t) 
                else:
                    for i in range (0, sample_num):
                        if i ==0:
                            self.exp_data_path = self.exp_data_path.replace(".xlsx", f"_{i}.xlsx")
                        else:
                            self.exp_data_path = self.exp_data_path.replace(f"_{i-1}.xlsx", f"_{i}.xlsx")
                        x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp = self.read_exp(x_uni, t) 
                        # Calculate the error between experimental data and simulation results
                        delta = self.cost_fun(q3_exp, q3, Q3_exp, Q3, x_50_exp, x_50)
                        delta_sum +=delta
                
            delta_sum /= sample_num

        
            return (delta_sum * scale)
        
    def write_new_data(self):
        # save the kernels
        alpha_prim_str = ', '.join([str(x) for x in self.alpha_prim])
        with open(self.filename_kernels, 'w') as file:
            file.write('CORR_BETA: {}\n'.format(self.corr_beta))
            file.write('alpha_prim: {}\n'.format(alpha_prim_str))
        
        # save the calculation result in experimental data form
        x_uni, _, _, _, _, _ = self.p.return_num_distribution_fixed(t=len(self.p.t_vec)-1)
        x_uni *= 1e6 
        df = pd.DataFrame(index=x_uni)
        df.index.name = 'Circular Equivalent Diameter'
        formatted_times = write_read_exp.convert_seconds_to_time(self.p.t_vec)

        for idt in range(0, len(self.p.t_vec)):
            _, q3, _, _, _, _ = self.p.return_num_distribution_fixed(t=idt)
            if self.smoothing:
                sumN_uni = self.KDE_smoothing(x_uni, q3, bandwidth='scott', kernel_func='gaussian')
                q3 = sumN_uni/np.sum(sumN_uni)
                # add noise to the original data
                if self.add_noise:
                    q3 = self.function_noise(q3, noise_type=self.noise_type)

            if len(q3) < len(x_uni):
                # Pad all arrays to the same length
                q3 = np.pad(q3, (0, len(x_uni) - len(q3)), 'constant')
            
            df[formatted_times[idt]] = q3

        # save DataFrame as Excel file
        df.to_excel(self.exp_data_path)
        
        return 
        
    def cal_pop(self, corr_beta, alpha_prim):
        # Case(1): optimazition with constant beta and alpha_prim, but too slow, don't use
        # Case(2): optimazition with constant corr_beta and alpha_prim
        # Case(3): optimazition with constant corr_beta and calculated alpha_prim
        
        if self.kernel == 1:
            self.p.COLEVAL = 3
            self.p.EFFEVAL = 2
            self.p.CORR_BETA = corr_beta
            self.p.alpha_prim = alpha_prim
            self.p.full_init(calc_alpha=False)
            self.p.solve_PBE(t_vec=self.t_vec)
            
        elif self.kernel == 2:
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
            
        elif self.kernel == 3:
            self.p.COLEVAL = 2
            self.p.EFFEVAL = 2
            self.p.CORR_BETA = corr_beta
            self.p.full_init(calc_alpha=True)
            self.p.solve_PBE(t_vec=self.t_vec)
    
    # Read the experimental data and re-interpolate the particle distribution 
    # of the experimental data according to the simulation results.
    def read_exp(self, x_uni, t):
        
        compare = write_read_exp(self.exp_data_path, read=True)
        q3_exp = compare.get_exp_data(t)
        x_uni_exp = q3_exp.index.to_numpy()
        if not np.array_equal(x_uni_exp, x_uni):
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
    
    def cost_fun(self, q3_exp, q3, Q3_exp, Q3, x_50_exp, x_50):
        # delta_flag == 1: use q3
        # delta_flag == 2: use Q3
        # delta_flag == 3: use x_50
        if self.delta_flag == 1:
            delta = ((q3*100-q3_exp*100)**2).sum()
        elif self.delta_flag == 2:
            delta = ((Q3*100-Q3_exp*100)**2).sum()
        else:
            delta = (x_50 - x_50_exp)**2
        
        return delta
    
    def optimierer(self, algo='BO', t_step=0, init_points=4, Q3_exp=None, x_50_exp=None, sample_num=1, hyperparameter=None):
        if algo == 'BO':
            if self.p.dim == 1:
                pbounds = {'corr_beta': (0, 50), 'alpha_prim': (0, 1)}
                objective = lambda corr_beta, alpha_prim: self.cal_delta(
                    corr_beta=corr_beta, alpha_prim=np.array([alpha_prim]), t_step=t_step, 
                    scale=-1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, sample_num=sample_num)
                
            elif self.p.dim == 2:
                pbounds = {'corr_beta': (0, 50), 'alpha_prim_0': (0, 1), 'alpha_prim_1': (0, 1), 'alpha_prim_2': (0, 1)}
                objective = lambda corr_beta, alpha_prim_0, alpha_prim_1, alpha_prim_2: self.cal_delta(
                    corr_beta=corr_beta, 
                    alpha_prim=np.array([alpha_prim_0, alpha_prim_1, alpha_prim_2]), 
                    t_step=t_step, scale=-1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, sample_num=sample_num)
                
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
                                                          scale=1, t_step=t_step, Q3_exp=Q3_exp, x_50_exp=x_50_exp, 
                                                          sample_num=sample_num)
            elif self.p.dim == 2:
                space = [Real(0, 100), Real(0, 1), Real(0, 1), Real(0, 1)]
                objective = lambda params: self.cal_delta(
                    corr_beta=params[0], 
                    alpha_prim=np.array([params[1], params[2], params[3]]), 
                    scale=1, t_step=t_step, Q3_exp=Q3_exp, x_50_exp=x_50_exp, sample_num=sample_num)

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
    def visualize_distribution(self, ax=None,fig=None,close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
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
            sumN_uni_ori = self.KDE_smoothing(x_uni_ori, q3_ori, bandwidth='scott', kernel_func='gaussian')
            sumN_ori = np.sum(sumN_uni_ori)
    
            Q3_ori = np.cumsum(sumN_uni_ori)/sumN_ori
            q3_ori = sumN_uni_ori/np.sum(sumN_uni_ori)

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
            sumN_uni = self.KDE_smoothing(x_uni, q3, bandwidth='scott', kernel_func='gaussian')
            sumN = np.sum(sumN_uni)

            Q3 = np.cumsum(sumN_uni)/sumN
            q3 = sumN_uni/np.sum(sumN_uni)
 
        # read and calculate the experimental data
        t = max(self.p.t_vec)

        x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp = self.read_exp(x_uni, t)     
        
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
        
        axq3, fig = pt.plot_data(x_uni_exp, q3_exp, fig=fig, ax=axq3,
                               lbl='q3_exp',clr='g',mrk='^')
        axq3, fig = pt.plot_data(x_uni_ori, q3_ori, fig=fig, ax=axq3,
                               lbl='q3_ori',clr='r',mrk='v')
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $Q3$ / $-$',
                               lbl='Q3_mod',clr='b',mrk='o')
        axQ3, fig = pt.plot_data(x_uni_exp, Q3_exp, fig=fig, ax=axQ3,
                               lbl='Q3_exp',clr='g',mrk='^')
        axQ3, fig = pt.plot_data(x_uni_ori, Q3_ori, fig=fig, ax=axQ3,
                               lbl='Q3_ori',clr='r',mrk='v')

        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return axq3, axQ3, fig
    
    
    def function_noise(self, ori_data, noise_type='Gaussian'):
        
        if noise_type == 'Gaussian':
            # The first parameter 0 represents the mean value of the noise, 
            # the second parameter is the standard deviation of the noise,
            noise = np.random.normal(0, self.noise_strength, ori_data.shape)
            noised_data = ori_data + noise
            
        elif noise_type == 'Uniform':
            # Noises are uniformly distributed over the half-open interval [low, high)
            noise = np.random.uniform(low=-self.noise_strength/2, high=self.noise_strength/2, size=ori_data)
            noised_data = ori_data + noise
            
        elif noise_type == 'Poisson':
            noise = np.random.poisson(self.noise_strength, ori_data)
            noised_data = ori_data + noise
            
        elif noise_type == 'Multiplicative':
            noise = np.random.normal(1, self.noise_strength, ori_data.shape)
            noised_data = ori_data * noise
        # Cliping the data out of range and rescale the data    
        noised_data_clipped = np.clip(noised_data, 0, 1)
        noised_data = noised_data_clipped /np.sum(noised_data_clipped)
        
        return noised_data

    ## Kernel density estimation
    ## data_ori must be a quantity rather than a relative value!
    def KDE_smoothing(self, x_uni_ori, data_ori, bandwidth='scott', kernel_func='gaussian'):
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
        x_uni_re = x_uni_ori.reshape(-1, 1)
        # Avoid divide-by-zero warnings when calculating KDE
        data_ori_adjested = np.where(data_ori == 0, 1e-20, data_ori)
        
        kde = KernelDensity(kernel=kernel_func, bandwidth=bandwidth)
        kde.fit(x_uni_re, sample_weight=data_ori_adjested)
        data_smoothing = np.exp(kde.score_samples(x_uni_re))
        
        # Flatten a column vector into a one-dimensional array
        data_smoothing = data_smoothing.ravel()
        
        return data_smoothing
            


        