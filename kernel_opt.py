# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:38:56 2023

@author: px2030
"""
import numpy as np
import pandas as pd
from pop import population
from bayes_opt import BayesianOptimization
from skopt import gp_minimize
from skopt.space import Real
from scipy.stats import entropy
# import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from functools import partial
from PSD_Exp import write_read_exp


class kernel_opt():
    def __init__(self, add_noise=True, smoothing=True, dim=1, delta_flag=1, 
                 noise_type='Gaussian', noise_strength=0.01, t_vec=None):
        # delta_flag = 1: use q3
        # delta_flag = 2: use Q3
        # delta_flag = 3: use x_50
        self.delta_flag = delta_flag     
        self.cost_func_type = 'MSE'
        self.n_iter = 100
        self.add_noise = add_noise
        self.smoothing = smoothing
        self.noise_type = noise_type    # Gaussian, Uniform, Poisson, Multiplicative
        self.noise_strength = noise_strength
        self.t_vec = t_vec
        self.num_t_steps = len(t_vec)
        self.p = population(dim=dim, disc='geo')
        
    def cal_delta(self, corr_beta=None, alpha_prim=None, scale=1, Q3_exp=None,
                  x_50_exp=None, sample_num=1, exp_data_path=None):
        
        # x_uni = self.cal_x_uni(self.p)
        self.cal_pop(self.p, corr_beta, alpha_prim)

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
            
        else:
            return self.cal_delta_tem(sample_num, exp_data_path, scale, self.p)

    def cal_delta_tem(self, sample_num, exp_data_path, scale, pop):
        kde_list = []
        x_uni = self.cal_x_uni(pop)
        for idt in range(self.num_t_steps):
            _, q3, Q3, x_10, x_50, x_90 = pop.return_num_distribution_fixed(t=idt)
            
            kde = self.KDE_fit(x_uni, q3)
            kde_list.append(kde)
        
        if sample_num == 1:
            data_exp = self.read_exp(x_uni, exp_data_path) 
            sumN_uni = np.zeros((len(data_exp[0]), self.num_t_steps))
            for idt in range(self.num_t_steps):
                sumN_uni_tem = self.KDE_score(kde_list[idt], data_exp[0])
                sumN_uni[:, idt] = sumN_uni_tem
            data_mod = self.re_cal_distribution(data_exp[0], sumN_uni)
            # Calculate the error between experimental data and simulation results
            delta = self.cost_fun(data_exp[self.delta_flag], data_mod[self.delta_flag])
            
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(data_exp[0])
            return (delta * scale) / x_uni_num
        else:
            delta_sum = 0           
            for i in range (0, sample_num):
                exp_data_path = self.traverse_path(i, exp_data_path)
                data_exp = self.read_exp(x_uni, exp_data_path) 
                sumN_uni = np.zeros((len(data_exp[0]), self.num_t_steps))
                
                for idt in range(self.num_t_steps):
                    sumN_uni_tem = self.KDE_score(kde_list[idt], data_exp[0])
                    sumN_uni[:, idt] = sumN_uni_tem
                    
                data_mod = self.re_cal_distribution(data_exp[0], sumN_uni)
                # Calculate the error between experimental data and simulation results
                delta = self.cost_fun(data_exp[self.delta_flag], data_mod[self.delta_flag])
                delta_sum +=delta
            # Restore the original name of the file to prepare for the next step of training
            delta_sum /= sample_num
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(data_exp[0])    
            return (delta_sum * scale) / x_uni_num
        
    def cal_pop(self, pop, corr_beta, alpha_prim):
        pop.COLEVAL = 2
        pop.EFFEVAL = 2
        pop.CORR_BETA = corr_beta
        if pop.dim == 1:
            alpha_prim_temp = alpha_prim
        elif pop.dim == 2:
            alpha_prim_temp = np.zeros(4)
            alpha_prim_temp[0] = alpha_prim[0]
            alpha_prim_temp[1] = alpha_prim_temp[2] = alpha_prim[1]
            alpha_prim_temp[3] = alpha_prim[2]
        pop.alpha_prim = alpha_prim_temp
        pop.full_init(calc_alpha=False)
        pop.solve_PBE(t_vec=self.t_vec)                
    
    # Read the experimental data and re-interpolate the particle distribution 
    # of the experimental data according to the simulation results.
    def read_exp(self, x_uni, exp_data_path):      
        compare = write_read_exp(exp_data_path, read=True)
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
        if self.cost_func_type == 'MSE':
            return mean_squared_error(data_mod, data_exp)
        elif self.cost_func_type == 'RMSE':
            mse = mean_squared_error(data_mod, data_exp)
            return np.sqrt(mse)
        elif self.cost_func_type == 'MAE':
            return mean_absolute_error(data_mod, data_exp)
        elif (self.delta_flag == 1 or self.delta_flag == 2) and self.cost_func_type == 'KL':
            return entropy(data_mod, data_exp).mean()
        else:
            raise Exception("Current cost function type is not supported")
    
    def optimierer(self, algo='BO', init_points=4, Q3_exp=None, x_50_exp=None,
                   sample_num=1, hyperparameter=None, exp_data_path=None):
        if algo == 'BO':
            if self.p.dim == 1:
                pbounds = {'corr_beta': (0, 50), 'alpha_prim': (0, 1)}
                objective = lambda corr_beta, alpha_prim: self.cal_delta(
                    corr_beta=corr_beta, alpha_prim=np.array([alpha_prim]),
                    scale=-1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, 
                    sample_num=sample_num, exp_data_path=exp_data_path)
                
            elif self.p.dim == 2:
                pbounds = {'corr_beta': (0, 50), 'alpha_prim_0': (0, 1), 'alpha_prim_1': (0, 1), 'alpha_prim_2': (0, 1)}
                objective = lambda corr_beta, alpha_prim_0, alpha_prim_1, alpha_prim_2: self.cal_delta(
                    corr_beta=corr_beta, 
                    alpha_prim=np.array([alpha_prim_0, alpha_prim_1, alpha_prim_2]), 
                    scale=-1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, 
                    sample_num=sample_num, exp_data_path=exp_data_path)
                
            opt = BayesianOptimization(
                f=objective, 
                pbounds=pbounds,
                random_state=1,
            )
            
            opt.maximize(
                init_points=init_points,
                n_iter=self.n_iter,
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
                space = [Real(0, 50), Real(0, 1)]
                objective = lambda params: self.cal_delta(corr_beta=params[0], alpha_prim=np.array([params[1]]), 
                                                          scale=1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, 
                                                          sample_num=sample_num, exp_data_path=exp_data_path)
            elif self.p.dim == 2:
                space = [Real(0, 50), Real(0, 1), Real(0, 1), Real(0, 1)]
                objective = lambda params: self.cal_delta(
                    corr_beta=params[0], 
                    alpha_prim=np.array([params[1], params[2], params[3]]), 
                    scale=1, Q3_exp=Q3_exp, x_50_exp=x_50_exp, sample_num=sample_num,
                    exp_data_path=exp_data_path)

            opt = gp_minimize(
                objective,
                space,
                n_calls=self.n_iter,
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
    
    def cal_x_uni(self, pop):
        v_uni = np.setdiff1d(pop.V, [-1, 0])
        # Because the length unit in the experimental data is millimeters 
        # and in the simulation it is meters, so it needs to be converted 
        # before use.
        return (6*v_uni/np.pi)**(1/3)*1e6

    def traverse_path(self, label, path_ori):
        def update_path(path, label):
            if label == 0:
                return path.replace(".xlsx", f"_{label}.xlsx")
            else:
                return path.replace(f"_{label-1}.xlsx", f"_{label}.xlsx")
    
        if isinstance(path_ori, list):
            return [update_path(path, label) for path in path_ori]
        else:
            return update_path(path_ori, label)
    
    def set_comp_para(self, R_NM=None, R_M=None):
        if R_NM!=None and R_M!=None:
            self.p.R01 = R_NM
            self.p.R03 = R_M