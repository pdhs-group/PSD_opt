# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:38:56 2023

@author: px2030
"""
import numpy as np
import math
from pop import population
from bayes_opt import BayesianOptimization
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from func.func_read_exp import write_read_exp
from scipy.interpolate import interp1d

class opt_algo():
    def __init__(self):
        self.n_iter = 100
        ## delta_flag = 1: use q3
        ## delta_flag = 2: use Q3
        ## delta_flag = 3: use x_10
        ## delta_flag = 4: use x_50
        ## delta_flag = 5: use x_90
        self.delta_flag = 1     
        ## 'MSE': Mean Squared Error
        ## 'RMSE': Root Mean Squared Error
        ## 'MAE': Mean Absolute Error
        ## 'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
        self.cost_func_type = 'MSE'

        self.calc_init_N = False
    #%%  Optimierer    
    def calc_delta(self, corr_beta=None, alpha_prim=None, scale=1, sample_num=1, exp_data_path=None):
        self.calc_pop(self.p, corr_beta, alpha_prim, self.t_vec)

        return self.calc_delta_tem(sample_num, exp_data_path, scale, self.p)
    
    def calc_delta_agg(self, corr_agg=None, scale=1, sample_num=1, exp_data_path=None):
        corr_beta = self.return_syth_beta(corr_agg)
        alpha_prim = corr_agg / corr_beta

        self.calc_pop(self.p, corr_beta, alpha_prim, self.t_vec)

        return self.calc_delta_tem(sample_num, exp_data_path, scale, self.p)

    def calc_delta_tem(self, sample_num, exp_data_path, scale, pop):
        kde_list = []
        x_uni = self.calc_x_uni(pop)
        for idt in range(self.num_t_steps):
            sumvol_uni = pop.return_distribution(t=idt, flag='sumvol_uni')[0]
            kde = self.KDE_fit(x_uni, sumvol_uni)
            kde_list.append(kde)
        
        if sample_num == 1:
            x_uni_exp, sumN_uni_exp = self.read_exp(exp_data_path) 
            sumN_uni_exp = sumN_uni_exp[:, self.idt_vec]
            vol_uni = np.tile((1/6)*np.pi*x_uni_exp**3, (len(self.idt_vec), 1)).T
            sumvol_uni_exp = sumN_uni_exp * vol_uni
            q3_mod = np.zeros((len(x_uni_exp), self.num_t_steps))
            for idt in range(self.num_t_steps):
                q3_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp)
                q3_mod[:, idt] = q3_mod_tem
            data_mod = self.re_calc_distribution(x_uni_exp, q3=q3_mod, flag=self.delta_flag)[0]
            data_exp = self.re_calc_distribution(x_uni_exp, sum_uni=sumvol_uni_exp, flag=self.delta_flag)[0]
            # Calculate the error between experimental data and simulation results
            delta = self.cost_fun(data_exp, data_mod)
            
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(x_uni_exp)
            return (delta * scale) / x_uni_num
        else:
            delta_sum = 0           
            for i in range (0, sample_num):
                exp_data_path = self.traverse_path(i, exp_data_path)
                x_uni_exp, sumN_uni_exp = self.read_exp(exp_data_path) 
                sumN_uni_exp = sumN_uni_exp[:, self.idt_vec]
                vol_uni = np.tile((1/6)*np.pi*x_uni_exp**3, (len(self.idt_vec), 1)).T
                sumvol_uni_exp = sumN_uni_exp * vol_uni
                q3_mod = np.zeros((len(x_uni_exp), self.num_t_steps))
                
                for idt in range(self.num_t_steps):
                    q3_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp)
                    q3_mod[:, idt] = q3_mod_tem
                    
                data_mod = self.re_calc_distribution(x_uni_exp, q3=q3_mod, flag=self.delta_flag)[0]
                data_exp = self.re_calc_distribution(x_uni_exp, sum_uni=sumvol_uni_exp, flag=self.delta_flag)[0]
                # Calculate the error between experimental data and simulation results
                delta = self.cost_fun(data_exp, data_mod)
                delta_sum +=delta
            # Restore the original name of the file to prepare for the next step of training
            delta_sum /= sample_num
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(x_uni_exp)  
            return (delta_sum * scale) / x_uni_num
    
    def optimierer(self, init_points=4, sample_num=1, hyperparameter=None, exp_data_path=None):
        if self.method == 'BO':
            if self.p.dim == 1:
                pbounds = {'corr_beta_log': (-3, 3), 'alpha_prim': (0, 1)}
                objective = lambda corr_beta_log, alpha_prim: self.calc_delta(
                    corr_beta=10**corr_beta_log, alpha_prim=np.array([alpha_prim]),
                    scale=-1, sample_num=sample_num, exp_data_path=exp_data_path)
                
            elif self.p.dim == 2:
                pbounds = {'corr_beta_log': (-3, 3), 'alpha_prim_0': (0, 1), 'alpha_prim_1': (0, 1), 'alpha_prim_2': (0, 1)}
                objective = lambda corr_beta_log, alpha_prim_0, alpha_prim_1, alpha_prim_2: self.calc_delta(
                    corr_beta=10**corr_beta_log, 
                    alpha_prim=np.array([alpha_prim_0, alpha_prim_1, alpha_prim_2]), 
                    scale=-1, sample_num=sample_num, exp_data_path=exp_data_path)
                
            opt = BayesianOptimization(
                f=objective, 
                pbounds=pbounds,
                random_state=1,
                allow_duplicate_points=True
            )
            
            opt.maximize(
                init_points=init_points,
                n_iter=self.n_iter,
            )   
            if self.p.dim == 1:
                self.corr_beta_opt = 10**opt.max['params']['corr_beta_log']
                self.alpha_prim_opt = opt.max['params']['alpha_prim']
                
            elif self.p.dim == 2:
                self.alpha_prim_opt = np.zeros(3)
                self.corr_beta_opt = 10**opt.max['params']['corr_beta_log']
                self.alpha_prim_opt[0] = opt.max['params']['alpha_prim_0']
                self.alpha_prim_opt[1] = opt.max['params']['alpha_prim_1']
                self.alpha_prim_opt[2] = opt.max['params']['alpha_prim_2']
            
            delta_opt = -opt.max['target']           
            
        return delta_opt  
    
    def optimierer_agg(self, method='BO', init_points=4, sample_num=1, hyperparameter=None, exp_data_path=None):
        if method == 'BO':
            if self.p.dim == 1:
                pbounds = {'corr_agg_log': (-3, 3)}
                objective = lambda corr_agg_log: self.calc_delta_agg(
                    corr_agg=10**corr_agg_log, scale=-1, sample_num=sample_num, 
                    exp_data_path=exp_data_path)
                
            elif self.p.dim == 2:
                pbounds = {'corr_agg_log_0': (-3, 3), 'corr_agg_log_1': (-3, 3), 'corr_agg_log_2': (-3, 3)}
                objective = lambda corr_agg_log_0, corr_agg_log_1, corr_agg_log_2: self.calc_delta_agg(
                    corr_agg=10**np.array([corr_agg_log_0, corr_agg_log_1, corr_agg_log_2]), 
                    scale=-1, sample_num=sample_num, exp_data_path=exp_data_path)
                
            opt = BayesianOptimization(
                f=objective, 
                pbounds=pbounds,
                random_state=1,
                allow_duplicate_points=True
            )
            
            opt.maximize(
                init_points=init_points,
                n_iter=self.n_iter,
            )
            
            if self.p.dim == 1:
                corr_agg_opt = 10**opt.max['params']['corr_agg_log']
                self.corr_beta_opt = self.return_syth_beta(corr_agg_opt)
                self.alpha_prim_opt = corr_agg_opt / self.corr_beta_opt
                
            elif self.p.dim == 2:
                corr_agg_opt = np.zeros(3)
                corr_agg_opt[0] = 10**opt.max['params']['corr_agg_log_0']
                corr_agg_opt[1] = 10**opt.max['params']['corr_agg_log_1']
                corr_agg_opt[2] = 10**opt.max['params']['corr_agg_log_2']
                self.corr_beta_opt = self.return_syth_beta(corr_agg_opt)
                self.alpha_prim_opt = corr_agg_opt / self.corr_beta_opt
            
            delta_opt = -opt.max['target']           
            
        return delta_opt  
    
    def return_syth_beta(self,corr_agg):
        max_val = max(corr_agg)
        power = np.log10(max_val)
        power = np.ceil(power)
        return 10**power
    
    def cost_fun(self, data_exp, data_mod):
        if self.cost_func_type == 'MSE':
            return mean_squared_error(data_mod, data_exp)
        elif self.cost_func_type == 'RMSE':
            mse = mean_squared_error(data_mod, data_exp)
            return np.sqrt(mse)
        elif self.cost_func_type == 'MAE':
            return mean_absolute_error(data_mod, data_exp)
        elif (self.delta_flag == 'q3' or self.delta_flag == 'Q3') and self.cost_func_type == 'KL':
            data_mod = np.where(data_mod <= 10e-20, 10e-20, data_mod)
            data_exp = np.where(data_exp <= 10e-20, 10e-20, data_exp)
            return entropy(data_mod, data_exp).mean()
        else:
            raise Exception("Current cost function type is not supported")
    #%% Data Process  
    ## Read the experimental data and re-interpolate the particle distribution 
    ## of the experimental data according to the simulation results.
    def read_exp(self, exp_data_path):      
        exp_data = write_read_exp(exp_data_path, read=True)
        df = exp_data.get_exp_data(self.t_all)
        x_uni_exp = df.index.to_numpy()
        sumN_uni_exp = df.to_numpy()
        return x_uni_exp, sumN_uni_exp
    
    def function_noise(self, ori_data):
        rows, cols = ori_data.shape
        noise = np.zeros((rows, cols))
        if self.noise_type == 'Gaus':
            # The first parameter 0 represents the mean value of the noise, 
            # the second parameter is the standard deviation of the noise,
            for i in range(cols):
                noise[:, i] = np.random.normal(0, self.noise_strength, rows)              
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Uni':
            # Noises are uniformly distributed over the half-open interval [low, high)
            for i in range(cols):
                noise[:, i] = np.random.uniform(low=-self.noise_strength/2, high=self.noise_strength/2, size=rows)
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Po':
            for i in range(cols):
                noise[:, i] = np.random.poisson(self.noise_strength, rows)
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Mul':
            for i in range(cols):
                noise[:, i] = np.random.normal(1, self.noise_strength, rows)
            noised_data = ori_data * noise
        # Cliping the data out of range  
        noised_data = np.clip(noised_data, 0, np.inf)
        return noised_data

    ## Kernel density estimation
    ## data_ori must be a quantity rather than a relative value!
    def KDE_fit(self, x_uni_ori, data_ori, bandwidth='scott', kernel_func='epanechnikov'):
            
        # KernelDensity requires input to be a column vector
        # So x_uni_re must be reshaped
        x_uni_ori_re = x_uni_ori.reshape(-1, 1)
        # Avoid divide-by-zero warnings when calculating KDE
        data_ori_adjested = np.where(data_ori <= 0, 1e-20, data_ori)      
        kde = KernelDensity(kernel=kernel_func, bandwidth=bandwidth)
        kde.fit(x_uni_ori_re, sample_weight=data_ori_adjested)  
        return kde
    
    def KDE_score(self, kde, x_uni_new):
        x_uni_new_re = x_uni_new.reshape(-1, 1) 
        data_smoothing = np.exp(kde.score_samples(x_uni_new_re))
        
        # Flatten a column vector into a one-dimensional array
        data_smoothing = data_smoothing.ravel()
        data_smoothing = data_smoothing/np.trapz(data_smoothing,x_uni_new)
        return data_smoothing
    
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
    #%% PBE    
    def create_1d_pop(self, disc='geo'):
        
        self.p_NM = population(dim=1,disc=disc)
        self.p_M = population(dim=1,disc=disc)
            
    def calc_pop(self, pop, corr_beta, alpha_prim, t_vec=None):
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
        if not self.calc_init_N:
            pop.full_init(calc_alpha=False)
        else:
            pop.calc_F_M()
        if pop.dim == 1: pop.calc_B_M()
        
        if t_vec is None: pop.solve_PBE(t_vec=self.t_vec)      
        else: pop.solve_PBE(t_vec=t_vec)  
        
    def set_comp_para(self, R01_0='r0_005', R03_0='r0_005', dist_path_NM=None, dist_path_M=None,
                      R_NM=2.9e-7, R_M=2.9e-7):
        if (not self.calc_init_N) and (dist_path_NM is not None and dist_path_M is not None):
            self.p.USE_PSD = True
            psd_dict_NM = np.load(dist_path_NM,allow_pickle=True).item()
            psd_dict_M = np.load(dist_path_M,allow_pickle=True).item()
            self.p.DIST1 = dist_path_NM
            self.p.DIST3 = dist_path_M
            self.p.R01 = psd_dict_NM[R01_0]
            self.p.R03 = psd_dict_M[R03_0]
        else:
            self.p.USE_PSD = False
            self.p.R01 = R_NM
            self.p.R03 = R_M
        ## Set particle parameter for 1D PBE
        self.p_NM.USE_PSD = self.p_M.USE_PSD = self.p.USE_PSD
        # parameter for particle component 1 - NM
        self.p_NM.R01 = self.p.R01
        self.p_NM.DIST1 = self.p.DIST1
        
        # parameter for particle component 2 - M
        self.p_M.R01 = self.p.R03
        self.p_M.DIST1 = self.p.DIST3
        
    def calc_all_R(self):
        self.p.calc_R()
        self.p_NM.calc_R()
        self.p_M.calc_R()
    
    ## only for 1D-pop, 
    def set_init_N(self, sample_num, exp_data_paths, init_flag):
        self.calc_all_R()
        self.set_init_N_1D(self.p_NM, sample_num, exp_data_paths[1], init_flag)
        self.set_init_N_1D(self.p_M, sample_num, exp_data_paths[2], init_flag)
        self.p.N = np.zeros((self.p.NS+3, self.p.NS+3, len(self.p.t_vec)))
        self.p.N[2:, 1, 0] = self.p_NM.N[2:, 0]
        self.p.N[1, 2:, 0] = self.p_M.N[2:, 0]
    
    def set_init_N_1D(self, pop, sample_num, exp_data_path, init_flag):
        x_uni = self.calc_x_uni(pop)
        if sample_num == 1:
            x_uni_exp, sumN_uni_init_sets = self.read_exp(exp_data_path)
        else:
            exp_data_path=self.traverse_path(0, exp_data_path)
            x_uni_exp, sumN_uni_tem = self.read_exp(exp_data_path)
            sumN_uni_all_samples = np.zeros((len(x_uni_exp), len(self.t_all), sample_num))
            sumN_uni_all_samples[:, :, 0] = sumN_uni_tem
            for i in range(1, sample_num):
                exp_data_path=self.traverse_path(i, exp_data_path)
                _, sumN_uni_tem = self.read_exp(exp_data_path)
                sumN_uni_all_samples[:, :, i] = sumN_uni_tem
            sumN_uni_all = sumN_uni_all_samples.mean(axis=2)
            
        sumN_uni_init_sets = sumN_uni_all[:, self.idt_init]
        sumN_uni_init = np.zeros(len(x_uni))
            
        if init_flag == 'int':
            for idx in range(len(x_uni_exp)):
                interp_time = interp1d(self.t_init, sumN_uni_init_sets[idx, :], kind='linear', fill_value="extrapolate")
                sumN_uni_init[idx] = interp_time(0.0)

        elif init_flag == 'mean':
            sumN_uni_init = sumN_uni_init_sets.mean(axis=1)
                
        ## Remap q3 corresponding to the x value of the experimental data to x of the PBE
        # kde = self.KDE_fit(x_uni_exp, q3_init)
        # sumV_uni = self.KDE_score(kde, x_uni)
        # q3_init = sumV_uni / sumV_uni.sum()
        inter_grid = interp1d(x_uni_exp, sumN_uni_init, kind='linear', fill_value="extrapolate")
        sumN_uni_init = inter_grid(x_uni)
                
        pop.N = np.zeros((pop.NS+3, len(pop.t_vec)))
        ## Because sumN_uni_init[0] = 0
        pop.N[2:, 0]= sumN_uni_init[1:]
        thr = 1e-5
        pop.N[pop.N < (thr * pop.N[2:, 0].max())]=0     
        
    def calc_v_uni(self, pop):
        return np.setdiff1d(pop.V, [-1, 0])*1e18
    
    def calc_x_uni(self, pop):
        v_uni = self.calc_v_uni(pop)
        # Because the length unit in the experimental data is millimeters 
        # and in the simulation it is meters, so it needs to be converted 
        # before use.
        x_uni = np.zeros(len(v_uni)+1)
        x_uni[1:]=(6*v_uni/np.pi)**(1/3)
        return x_uni
        
    def re_calc_distribution(self, x_uni, q3=None, sum_uni=None, flag='all'):
        if q3 is not None:
            q3_new = q3
            Q3_new = np.apply_along_axis(lambda q3_slice: 
                                     self.calc_Q3(x_uni, q3=q3_slice), 0, q3)

        else:
            Q3_new = np.apply_along_axis(lambda sum_uni_slice: 
                                     self.calc_Q3(x_uni, sum_uni=sum_uni_slice), 0, sum_uni)
            q3_new = np.apply_along_axis(lambda Q3_slice: 
                                          self.calc_q3(Q3_slice, x_uni), 0, Q3_new)

        dim = q3_new.shape[1]
        x_10_new = np.zeros(dim)
        x_50_new = np.zeros(dim)
        x_90_new = np.zeros(dim)
        for idx in range(dim):
            x_10_new[idx] = np.interp(0.1, Q3_new[:, idx], x_uni)
            x_50_new[idx] = np.interp(0.5, Q3_new[:, idx], x_uni)
            x_90_new[idx] = np.interp(0.9, Q3_new[:, idx], x_uni)
        outputs = {
        'q3': q3_new,
        'Q3': Q3_new,
        'x_10': x_10_new,
        'x_50': x_50_new,
        'x_90': x_90_new,
        }
        
        if flag == 'all':
            return outputs.values()
        else:
            flags = flag.split(',')
            return tuple(outputs[f.strip()] for f in flags if f.strip() in outputs)
        
    def calc_Q3(self, x_uni, q3=None, sum_uni=None):
        Q3 = np.zeros_like(q3) if q3 is not None else np.zeros_like(sum_uni)
        if q3 is None:
            Q3 = np.cumsum(sum_uni)/sum_uni.sum()
        else:
            for i in range(1, len(Q3)):
                    Q3[i] = np.trapz(q3[:i+1], x_uni[:i+1])
        return Q3
    def calc_sum_uni(self, Q3, sum_total):
        sum_uni = np.zeros_like(Q3)
        for i in range(1, len(Q3)):
            sum_uni[i] = sum_total * max((Q3[i] -Q3[i-1] ), 0)
        return sum_uni
    def calc_q3(self, Q3, x_uni):
        q3 = np.zeros_like(Q3)
        for i in range(1,len(x_uni)):
            q3[i] = (Q3[i] - Q3[i-1]) / (x_uni[i]-x_uni[i-1])
        return q3