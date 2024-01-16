# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:38:56 2023

@author: px2030
"""
import numpy as np
from pop import population
from bayes_opt import BayesianOptimization
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from func.func_read_exp import write_read_exp

class opt_algo():
    def __init__(self):
        self.smoothing = False
        self.add_noise = False
        self.noise_type = 'Gaus'    # Gaus, Uni, Po, Mul
        self.noise_strength = 0.01
        
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

        self.t_vec = np.arange(0, 601, 20, dtype=float)
        self.num_t_steps = len(self.t_vec)
        self.calc_init_N = False
    #%%  Optimierer    
    def cal_delta(self, corr_beta=None, alpha_prim=None, scale=1, sample_num=1, exp_data_path=None):
        self.cal_pop(self.p, corr_beta, alpha_prim)

        return self.cal_delta_tem(sample_num, exp_data_path, scale, self.p)

    def cal_delta_tem(self, sample_num, exp_data_path, scale, pop):
        kde_list = []
        x_uni = self.cal_x_uni(pop)
        for idt in range(self.num_t_steps):
            _, q3, Q3, x_10, x_50, x_90 = pop.return_distribution(t=idt, flag='all')
            kde = self.KDE_fit(x_uni, q3)
            kde_list.append(kde)
        
        if sample_num == 1:
            data_exp = self.read_exp(exp_data_path) 
            sumV_uni = np.zeros((len(data_exp[0]), self.num_t_steps))
            for idt in range(self.num_t_steps):
                sumV_uni_tem = self.KDE_score(kde_list[idt], data_exp[0])
                sumV_uni[:, idt] = sumV_uni_tem
            data_mod = self.re_cal_distribution(data_exp[0], sumV_uni)
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
                data_exp = self.read_exp(exp_data_path) 
                sumV_uni = np.zeros((len(data_exp[0]), self.num_t_steps))
                
                for idt in range(self.num_t_steps):
                    sumV_uni_tem = self.KDE_score(kde_list[idt], data_exp[0])
                    sumV_uni[:, idt] = sumV_uni_tem
                    
                data_mod = self.re_cal_distribution(data_exp[0], sumV_uni)
                # Calculate the error between experimental data and simulation results
                delta = self.cost_fun(data_exp[self.delta_flag], data_mod[self.delta_flag])
                delta_sum +=delta
            # Restore the original name of the file to prepare for the next step of training
            delta_sum /= sample_num
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(data_exp[0])    
            return (delta_sum * scale) / x_uni_num
    
    def optimierer(self, method='BO', init_points=4, sample_num=1, hyperparameter=None, exp_data_path=None):
        if method == 'BO':
            if self.p.dim == 1:
                pbounds = {'corr_beta_log': (-3, 3), 'alpha_prim': (0, 1)}
                objective = lambda corr_beta_log, alpha_prim: self.cal_delta(
                    corr_beta=10**corr_beta_log, alpha_prim=np.array([alpha_prim]),
                    scale=-1, sample_num=sample_num, exp_data_path=exp_data_path)
                
            elif self.p.dim == 2:
                pbounds = {'corr_beta_log': (-3, 3), 'alpha_prim_0': (0, 1), 'alpha_prim_1': (0, 1), 'alpha_prim_2': (0, 1)}
                objective = lambda corr_beta_log, alpha_prim_0, alpha_prim_1, alpha_prim_2: self.cal_delta(
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
    #%% Data Process  
    ## Read the experimental data and re-interpolate the particle distribution 
    ## of the experimental data according to the simulation results.
    def read_exp(self, exp_data_path):      
        exp_data = write_read_exp(exp_data_path, read=True)
        df = exp_data.get_exp_data(self.t_vec)
        x_uni_exp = df.index.to_numpy()
        q3_exp = df.to_numpy()
        
        # The experimental data is the number distribution of particles,
        # which needs to be converted into a volume distribution to compare 
        # with the simulation results.
        q3_exp = self.convert_dist_num_to_vol(x_uni_exp, q3_exp)
        return self.re_cal_distribution(x_uni_exp, q3_exp)
    
    def convert_dist_num_to_vol(self, x_uni, q3_num):
        v_uni = np.pi*x_uni**3/6
        sumvol_uni = v_uni[:, np.newaxis] * q3_num
        q3 = sumvol_uni / np.sum(sumvol_uni, axis=0)
        q3 = q3 / np.sum(q3)
        return q3
    
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
        # Cliping the data out of range and rescale the data    
        noised_data_clipped = np.clip(noised_data, 0, 1)
        cols_sums = np.sum(noised_data_clipped, axis=0, keepdims=True)
        noised_data = noised_data_clipped /cols_sums      
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
        if not self.calc_init_N:
            pop.full_init(calc_alpha=False)
        else:
            pop.calc_F_M()
        if pop.dim == 1: pop.calc_B_M()
        pop.solve_PBE(t_vec=self.t_vec)                
        
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
    
    def set_init_N_1D(self, pop, sample_num, exp_data_path, init_flag, datasets=3):
        x_uni = self.cal_x_uni(pop)
        if sample_num == 1:
            init_q3_sets = np.zeros((len(x_uni), datasets))
            x_uni_exp, q3_exp, _, _ ,_, _ = self.read_exp(exp_data_path)
            q3_exp = self.convert_dist_num_to_vol(x_uni_exp, q3_exp)
            # calculate with the first three time point
            init_q3_raw = q3_exp[:,:datasets]
            for i in range(datasets):
                kde = self.KDE_fit(x_uni_exp, init_q3_raw[:, i])
                init_q3_sets[:, i] = self.KDE_score(kde, x_uni)
                
            if init_flag == 'mean':
                init_q3 = init_q3_sets.mean(axis=1)
        else:
            init_q3_sets = np.zeros((len(x_uni), datasets, sample_num))
            for i in range(0, sample_num):
                exp_data_path=self.traverse_path(i, exp_data_path)
                x_uni_exp, q3_exp, _, _ ,_, _ = self.read_exp(exp_data_path)
                q3_exp = self.convert_dist_num_to_vol(x_uni_exp, q3_exp)
                init_q3_raw = q3_exp[:,:datasets]
                for j in range(datasets):
                    kde = self.KDE_fit(x_uni_exp, init_q3_raw[:, j])
                    init_q3_sets[:, j, i] = self.KDE_score(kde, x_uni)
            if init_flag == 'mean':
                init_q3 = init_q3_sets.mean(axis=1).mean(axis=1)
                   
        pop.N = np.zeros((pop.NS+3, len(pop.t_vec)))
        pop.N[2:, 0]=init_q3 * pop.N01
            
    def cal_v_uni(self, pop):
        return np.setdiff1d(pop.V, [-1, 0])
    
    def cal_x_uni(self, pop):
        v_uni = self.cal_v_uni(pop)
        # Because the length unit in the experimental data is millimeters 
        # and in the simulation it is meters, so it needs to be converted 
        # before use.
        return (6*v_uni/np.pi)**(1/3)*1e6
        
    def re_cal_distribution(self, x_uni, sumV_uni):
        if sumV_uni.ndim == 1:
            sumV = np.sum(sumV_uni)
            Q3 = np.cumsum(sumV_uni)/sumV
            q3 = sumV_uni/sumV

            x_10 = np.interp(0.1, Q3, x_uni)
            x_50 = np.interp(0.5, Q3, x_uni)
            x_90 = np.interp(0.9, Q3, x_uni)
        else:
            sumV = np.sum(sumV_uni, axis=0) 
            Q3 = np.cumsum(sumV_uni, axis=0)/sumV
            q3 = sumV_uni/sumV
            
            x_10 = np.zeros(self.num_t_steps)
            x_50 = np.zeros(self.num_t_steps)
            x_90 = np.zeros(self.num_t_steps)
            for idt in range(self.num_t_steps):
                x_10[idt] = np.interp(0.1, Q3[:, idt], x_uni)
                x_50[idt] = np.interp(0.5, Q3[:, idt], x_uni)
                x_90[idt] = np.interp(0.9, Q3[:, idt], x_uni)
        
        return x_uni, q3, Q3, x_10, x_50, x_90