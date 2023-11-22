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
from functools import partial
from PSD_Exp import write_read_exp
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt          
from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

class kernel_opt():
    def __init__(self, dim=1, disc='geo', first_cal=False):
        self.kernel = 2
        self.delta_flag = 1
        self.noise_strength = 0.01
        self.first_cal = first_cal
        self.p = population(dim=dim, disc=disc)
        
        
        self.exp_data_path = os.path.join(self.p.pth,"data\\")+'CED_focus_Sim.xlsx'
        self.filename_kernels = "kernels.txt"
        # check if the file exists
        if self.first_cal == True:
            if not os.path.exists(self.filename_kernels):
                warnings.warn("file does not exist: {}".format(self.filename_kernels))
                
            else:
                with open(self.filename_kernels, 'r') as file:
                    lines = file.readlines()  
                    
                self.corr_beta = None
                self.alpha_prime = None
                for line in lines:
                    if 'CORR_BETA:' in line:
                        self.corr_beta = float(line.split(':')[1].strip())
                    elif 'alpha_prim:' in line:
                        self.alpha_prime = float(line.split(':')[1].strip())
        
    def cal_delta(self, corr_beta=None, alpha_prim=None):
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
            self.p.alpha_prim = alpha_prim
            self.p.full_init(calc_alpha=False)
            self.p.solve_PBE()
            
        elif self.kernel == 3:
            self.p.COLEVAL = 2
            self.p.EFFEVAL = 2
            self.p.CORR_BETA = corr_beta
            self.p.full_init(calc_alpha=True)
            self.p.solve_PBE()
        
        if self.first_cal == True:
            
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
                q3 = self.function_noise(q3)

                if len(q3) < len(x_uni):
                    # Pad all arrays to the same length
                    q3 = np.pad(q3, (0, len(x_uni) - len(q3)), 'constant')
                
                df[formatted_times[idt]] = q3

            # save DataFrame as Excel file
            df.to_excel(self.exp_data_path)
            
            return 
            
        else:     
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
        
            return -delta
    
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
    
    def optimierer(self, algo='BO', init_points=2, hyperparameter=None):
        if algo == 'BO':
            pbounds = {'corr_beta': (0, 100), 'alpha_prim': (0, 0.5)}
            objective = partial(self.cal_delta)
            opt = BayesianOptimization(
                f=objective, 
                pbounds=pbounds,
                random_state=1,
            )
            
            opt.maximize(
                init_points=init_points,
                n_iter=100,
            )   
            
        return opt.max['params']['corr_beta'], opt.max['params']['alpha_prim'], opt.max['target']
    
    def visualize_distribution(self, ax=None,fig=None,close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
        
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
                               lbl='q3_mod',clr=clr,mrk='o')
        
        axq3, fig = pt.plot_data(x_uni, q3_exp, fig=fig, ax=axq3,
                               lbl='q3_exp',clr=clr,mrk='^')
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $Q3$ / $-$',
                               lbl='Q3_mod',clr=clr,mrk='o')
        axQ3, fig = pt.plot_data(x_uni, Q3_exp, fig=fig, ax=axQ3,
                               lbl='Q3_exp',clr=clr,mrk='^')

        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return axq3, axQ3, fig
    def function_noise(self, ori_data, noise_type='Gaussian'):
        
        if noise_type == 'Gaussian':
            # The first parameter 0 represents the mean value of the noise, 
            # the second parameter is the standard deviation of the noise,
            noise = np.random.normal(0, self.noise_strength, ori_data.shape)

        return (ori_data + noise)





        