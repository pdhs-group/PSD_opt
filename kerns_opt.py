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
# from functools import partial
import ast
from PSD_Exp import write_read_exp
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt          
# from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

class kernel_opt():
    def __init__(self, dim=1, disc='geo', generate_data=False):
        # kernel = 1: optimazition with constant beta and alpha_prim, but too slow, don't use
        # kernel = 2: optimazition with constant corr_beta and alpha_prim
        # kernel = 3: optimazition with constant corr_beta and calculated alpha_prim
        self.kernel = 2
        # delta_flag = 1: use Q3
        # delta_flag = 2: use x_50
        self.delta_flag = 1         
        self.add_noise = True
        self.noise_type = 'Gaussian'     # Gaussian, Uniform, Poisson, Multiplicative
        self.noise_strength = 0.01
        self.generate_data = generate_data
        self.p = population(dim=dim, disc=disc)
        
        
        self.exp_data_path = os.path.join(self.p.pth,"data\\")+'CED_focus_Sim.xlsx'
        self.filename_kernels = "kernels.txt"
        
    def cal_delta(self, corr_beta=None, alpha_prim=None, scale=1):
        
        self.cal_pop(corr_beta, alpha_prim)
        
        if self.generate_data:
            
            # save the kernels
            with open(self.filename_kernels, 'w') as file:
                file.write('CORR_BETA: {}\n'.format(self.p.CORR_BETA))
                file.write('alpha_prim: {}\n'.format(self.p.alpha_prim))
            
            # save the calculation result in experimental data form
            x_uni, _, _, _, _, _ = self.p.return_num_distribution(t=len(self.p.t_vec)-1)
            df = pd.DataFrame(index=x_uni*1e6)
            df.index.name = 'Circular Equivalent Diameter'
            formatted_times = write_read_exp.convert_seconds_to_time(self.p.t_vec)

            for idt in range(0, len(self.p.t_vec)):
                _, q3, _, _, _, _ = self.p.return_num_distribution(t=idt)
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
            
        else:     
            # read the kernels of original data
            if not os.path.exists(self.filename_kernels):
                warnings.warn("file does not exist: {}".format(self.filename_kernels))
                
            else:
                with open(self.filename_kernels, 'r') as file:
                    lines = file.readlines()  
                    
                    self.corr_beta = None
                    self.alpha_prim = None
                    for line in lines:
                        if 'CORR_BETA:' in line:
                            self.corr_beta = float(line.split(':')[1].strip())
                        elif 'alpha_prim:' in line:
                            array_str = line.split(':')[1].strip()
                            array_str = array_str.replace(" ", ", ")
                            self.alpha_prim = ast.literal_eval(array_str)
            
            x_uni, q3, Q3, x_10, x_50, x_90 = self.p.return_num_distribution(t=len(self.p.t_vec)-1)
            # Conversion unit
            x_uni *= 1e6    
            x_10 *= 1e6   
            x_50 *= 1e6   
            x_90 *= 1e6   
            # read and calculate the experimental data
            t = max(self.p.t_vec)

            x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp = self.read_exp(x_uni, t)    

            # Calculate the error between experimental data and simulation results
            delta = self.cost_fun(x_uni_exp, x_uni, Q3_exp, Q3, x_50_exp, x_50)
        
            return (delta * scale)
        
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
            self.p.solve_PBE()
            
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
            self.p.solve_PBE()
            
        elif self.kernel == 3:
            self.p.COLEVAL = 2
            self.p.EFFEVAL = 2
            self.p.CORR_BETA = corr_beta
            self.p.full_init(calc_alpha=True)
            self.p.solve_PBE()
    
    def read_exp(self, x_uni, t):
        
        compare = write_read_exp(self.exp_data_path, read=True)
        q3_exp = compare.get_exp_data(t)
        x_uni_exp = q3_exp.index.to_numpy()
        if not np.array_equal(x_uni_exp, x_uni):
            # If the experimental data has a different particle size scale than the simulations, 
            # interpolation is required
            q3_exp_interpolated = np.interp(x_uni, x_uni_exp, q3_exp)
            q3_exp_interpolated_clipped = np.clip(q3_exp_interpolated, 0, 1)
            q3_exp = pd.Series(q3_exp_interpolated_clipped, index=x_uni)
        Q3_exp = q3_exp.cumsum()
        x_10_exp = np.interp(0.1, Q3_exp, x_uni)
        x_50_exp = np.interp(0.5, Q3_exp, x_uni)
        x_90_exp = np.interp(0.9, Q3_exp, x_uni)
        
        return x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp
    
    def cost_fun(self, x_uni_exp, x_uni, Q3_exp, Q3, x_50_exp, x_50):
        
        # delta_flag == 1: use Q3
        # delta_flag == 2: use x_50
        if self.delta_flag == 1:
            delta = ((Q3*100-Q3_exp*100)**2).sum()
        else:
            delta = (x_50 - x_50_exp)**2
        
        return delta
    
    def optimierer(self, algo='BO', init_points=4, hyperparameter=None):
        if algo == 'BO':
            if self.p.dim == 1:
                pbounds = {'corr_beta': (0, 100), 'alpha_prim': (0, 0.5)}
                objective = lambda corr_beta, alpha_prim: self.cal_delta(
                    corr_beta=corr_beta, alpha_prim=np.array([alpha_prim]), scale=-1)
                
            elif self.p.dim == 2:
                pbounds = {'corr_beta': (0, 100), 'alpha_prim_0': (0, 0.5), 'alpha_prim_1': (0, 0.5), 'alpha_prim_2': (0, 0.5)}
                objective = lambda corr_beta, alpha_prim_0, alpha_prim_1, alpha_prim_2: self.cal_delta(
                    corr_beta=corr_beta, 
                    alpha_prim=np.array([alpha_prim_0, alpha_prim_1, alpha_prim_1, alpha_prim_2]), 
                    scale=-1)
                
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
                self.alpha_prim_opt = np.zeros(4)
                para_opt = opt.max['params']['corr_beta'] *\
                (opt.max['params']['alpha_prim_0'] + 2 * opt.max['params']['alpha_prim_1'] + opt.max['params']['alpha_prim_2'])
                self.corr_beta_opt = opt.max['params']['corr_beta']
                self.alpha_prim_opt[0] = opt.max['params']['alpha_prim_0']
                self.alpha_prim_opt[1] = self.alpha_prim_opt[2] = opt.max['params']['alpha_prim_1']
                self.alpha_prim_opt[3] = opt.max['params']['alpha_prim_2']
            
            delta_opt = -opt.max['target']
            
        if algo == 'gp_minimize':
            if self.p.dim == 1:
                space = [Real(0, 100), Real(0, 0.5)]
                objective = lambda params: self.cal_delta(corr_beta=params[0], alpha_prim=np.array([params[1]]), scale=1)
            elif self.p.dim == 2:
                space = [Real(0, 100), Real(0, 0.5), Real(0, 0.5), Real(0, 0.5)]
                objective = lambda params: self.cal_delta(
                    corr_beta=params[0], 
                    alpha_prim=np.array([params[1], params[2], params[2], params[3]]), 
                    scale=1)

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
                self.alpha_prim_opt = np.zeros(4)
                para_opt = opt.x[0] * (opt.x[1] + 2 * opt.x[2] + opt.x[3])
                self.corr_beta_opt = opt.x[0]
                self.alpha_prim_opt[0] = opt.x[1]
                self.alpha_prim_opt[1] = self.alpha_prim_opt[2] = opt.x[2]
                self.alpha_prim_opt[3] = opt.x[3]

            delta_opt = opt.fun
            
        return para_opt, delta_opt
    
    def visualize_distribution(self, new_cal=False, ax=None,fig=None,close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
        
        # Recalculate PSD using optimization results
        if new_cal and hasattr(self, 'corr_beta_opt') and hasattr(self, 'alpha_prim_opt'):
            self.cal_pop(self.corr_beta_opt, self.alpha_prim_opt)
        else:
            print("Need to run the optimization process at least once！")    
            
        x_uni, q3, Q3, x_10, x_50, x_90 = self.p.return_num_distribution(t=len(self.p.t_vec)-1)
        # Conversion unit
        x_uni *= 1e6    
        x_10 *= 1e6   
        x_50 *= 1e6   
        x_90 *= 1e6   
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
                               lbl='q3_mod',clr='#1f77b4',mrk='o')
        
        axq3, fig = pt.plot_data(x_uni, q3_exp, fig=fig, ax=axq3,
                               lbl='q3_exp',clr='#2ca02c',mrk='^')
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $Q3$ / $-$',
                               lbl='Q3_mod',clr='#1f77b4',mrk='o')
        axQ3, fig = pt.plot_data(x_uni, Q3_exp, fig=fig, ax=axQ3,
                               lbl='Q3_exp',clr='#2ca02c',mrk='^')

        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return axq3, axQ3, fig
    
    
    def function_noise(self, ori_data, noise_type='Gaussian'):
        
        if noise_type == 'Gaussian':
            # The first parameter 0 represents the mean value of the noise, 
            # the second parameter is the standard deviation of the noise,
            noise = np.random.normal(0, self.noise_strength, ori_data.shape)
        elif noise_type == 'Uniform':
            # Noises are uniformly distributed over the half-open interval [low, high)
            noise = np.random.uniform(low=-self.noise_strength/2, high=self.noise_strength/2, size=ori_data)
        elif noise_type == 'Poisson':
            noise = np.random.poisson(self.noise_strength, ori_data)
        elif noise_type == 'Multiplicative':
            noise = np.random.normal(1, self.noise_strength, ori_data.shape) * ori_data - ori_data

        return (ori_data + noise)





        