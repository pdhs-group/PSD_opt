# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:22:00 2023

@author: px2030
"""

import numpy as np
import os
import warnings
import ast
import pandas as pd
from kernel_opt import kernel_opt 
from multi_opt import multi_opt
from PSD_Exp import write_read_exp
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt          
# from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

class opt_method():
    def __init__(self, add_noise, smoothing, corr_beta, alpha_prim, dim=1,
                 delta_flag=1, noise_type='Gaussian', noise_strength=0.01, 
                 t_vec=None, multi_flag=False):
        self.algo='BO'  
        self.multi_flag=multi_flag
        if not self.multi_flag:
            self.k = kernel_opt(add_noise, smoothing, 
                                dim, delta_flag, noise_type, 
                                noise_strength, t_vec)
        else:
            self.k = multi_opt(add_noise, smoothing, 
                                dim, delta_flag, noise_type, 
                                noise_strength, t_vec)
        self.k.corr_beta = corr_beta
        self.k.alpha_prim = alpha_prim   
        
        # Set the base path for exp_data_path
        self.base_path = os.path.join(self.k.p.pth, "data\\")
        self.filename_kernels = os.path.join(self.base_path, "kernels.txt")
        
    def generate_synth_data(self, sample_num=1):
        self.write_kernels()
        # Check if noise should be added
        if self.k.add_noise:
            # Modify the file name to include noise type and strength
            filename = f"CED_focus_Sim_{self.k.noise_type}_{self.k.noise_strength}.xlsx"
        else:
            # Use the default file name
            filename = "CED_focus_Sim.xlsx"

        # Combine the base path with the modified file name
        exp_data_path = os.path.join(self.base_path, filename)
        
        if not self.multi_flag:
            self.k.cal_pop(self.k.p, self.k.corr_beta, self.k.alpha_prim)
            
            for i in range(0, sample_num):
                exp_data_path=self.k.traverse_path(i, exp_data_path)
                # print(self.k.exp_data_path)
                self.write_new_data(self.k.p, exp_data_path)
        else:
            self.k.cal_all_pop(self.k.corr_beta, self.k.alpha_prim)
            exp_data_paths = [
                exp_data_path,
                exp_data_path.replace(".xlsx", "_NM.xlsx"),
                exp_data_path.replace(".xlsx", "_M.xlsx")
            ]
            
            for i in range(0, sample_num):
                for j, path in enumerate(exp_data_paths):
                    exp_data_paths[j] = self.k.traverse_path(i, path)
                    self.write_new_data(self.k.p, exp_data_paths[j])
            
    def mean_kernels(self, sample_num, method='kernels', data_name=None):
        if data_name == None:
            warnings.warn("Please specify the name of the training data without labels!")
        else:
            exp_data_path = os.path.join(self.base_path, data_name)
            if self.multi_flag:
                exp_data_paths = [
                    exp_data_path,
                    exp_data_path.replace(".xlsx", "_NM.xlsx"),
                    exp_data_path.replace(".xlsx", "_M.xlsx")
                ]
                # Rename just to make it easy for subsequent code
                exp_data_path = exp_data_paths
            
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
            
            if method == 'kernels':
                para_opt_sample = np.zeros(sample_num)
                delta_opt_sample = np.zeros(sample_num)
                corr_beta_sample = np.zeros(sample_num)
                if self.k.p.dim == 1:
                    alpha_prim_sample = np.zeros(sample_num)
                    
                elif self.k.p.dim == 2:
                    alpha_prim_sample = np.zeros((3, sample_num))
                
                if sample_num == 1:
                    para_opt, delta_opt = self.k.optimierer(algo=self.algo,
                                                            exp_data_path=exp_data_path)
                    corr_beta = self.k.corr_beta_opt
                    alpha_prim = self.k.alpha_prim_opt
                    
                else:
                    for i in range(0, sample_num):
                        exp_data_path=self.k.traverse_path(i, exp_data_path)
                        para_opt_sample[i], delta_opt_sample[i] = \
                            self.k.optimierer(algo=self.algo, 
                                              exp_data_path=exp_data_path)
                            
                        corr_beta_sample[i] = self.k.corr_beta_opt
                        if self.k.p.dim == 1:
                            alpha_prim_sample[i] = self.k.alpha_prim_opt
                            
                        elif self.k.p.dim == 2:
                            alpha_prim_sample[:, i] = self.k.alpha_prim_opt

                if not sample_num == 1:
                    para_opt = np.mean(para_opt_sample)
                    delta_opt = np.mean(delta_opt_sample)
                    corr_beta = np.mean(corr_beta_sample)
                    alpha_prim = np.mean(alpha_prim_sample, axis=self.k.p.dim-1)
                    self.k.corr_beta_opt = corr_beta
                    self.k.alpha_prim_opt = alpha_prim
                
            elif method == 'delta':
                para_opt, delta_opt = \
                    self.k.optimierer(algo=self.algo, sample_num=sample_num, 
                                      exp_data_path=exp_data_path)
                
                
            if self.k.p.dim == 1:
                para_diff = abs(para_opt - self.k.corr_beta * np.sum(self.k.alpha_prim)) / (self.k.corr_beta * np.sum(self.k.alpha_prim))
            elif self.k.p.dim == 2:
                para_diff = np.zeros(4)
                para_diff[0] = abs(self.k.corr_beta_opt- self.k.corr_beta) / self.k.corr_beta
                para_diff[1:] = abs(self.k.alpha_prim_opt - self.k.alpha_prim) / self.k.alpha_prim
            
            return self.k.corr_beta_opt, self.k.alpha_prim_opt, para_opt, para_diff, delta_opt
        
                
    def write_kernels(self):
        # save the kernels
        if self.k.p.dim == 1:
            with open(self.filename_kernels, 'w') as file:
                file.write('CORR_BETA: {}\n'.format(self.k.corr_beta))
                file.write('alpha_prim: {}\n'.format(self.k.alpha_prim))
        else:
            alpha_prim_str = ', '.join([str(x) for x in self.k.alpha_prim])
            with open(self.filename_kernels, 'w') as file:
                file.write('CORR_BETA: {}\n'.format(self.k.corr_beta))
                file.write('alpha_prim: {}\n'.format(alpha_prim_str))
        
    def write_new_data(self, pop, exp_data_path):
        # save the calculation result in experimental data form
        x_uni = self.k.cal_x_uni(pop)
        formatted_times = write_read_exp.convert_seconds_to_time(pop.t_vec)
        sumN_uni = np.zeros((len(x_uni), self.k.num_t_steps))
        
        for idt in range(self.k.num_t_steps):
            _, q3, _, _, _, _ = pop.return_num_distribution_fixed(t=idt)
            kde = self.k.KDE_fit(x_uni, q3)
            if self.k.smoothing:
                kde = self.k.KDE_fit(x_uni, q3)
                sumN_uni[:, idt] = self.k.KDE_score(kde, x_uni)
                
        _, q3, _, _, _,_ = self.k.re_cal_distribution(x_uni, sumN_uni)
        # add noise to the original data
        if self.k.add_noise:
            q3 = self.k.function_noise(q3)
        df = pd.DataFrame(data=q3, index=x_uni, columns=formatted_times)
        df.index.name = 'Circular Equivalent Diameter'
        # save DataFrame as Excel file
        df.to_excel(exp_data_path)
        
        return 
    
    # Visualize only the last time step of the specified time vector and the last used experimental data
    def visualize_distribution(self, exp_data_path=None,ax=None,fig=None,
                               close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    # Recalculate PSD using original parameter
        self.k.cal_pop(self.k.p, corr_beta=self.corr_beta, alpha_prim=self.alpha_prim)
        '''if self.p.dim == 1:       
            self.p.CORR_BETA = self.corr_beta
            self.p.alpha_prim=self.corr_beta
   
        elif self.p.dim == 2:
            self.p.CORR_BETA = self.corr_beta 
            self.p.alpha_prim[:]=[self.corr_beta[0], self.corr_beta[1], self.corr_beta[2], self.corr_beta[3]]

        self.p.full_init(calc_alpha=False)
        self.p.solve_PBE(t_vec=self.t_vec)'''
        x_uni_ori, q3_ori, Q3_ori, x_10_ori, x_50_ori, x_90_ori = self.k.p.return_num_distribution_fixed(t=len(self.k.p.t_vec)-1)
        # Conversion unit
        x_uni_ori *= 1e6    
        x_10_ori *= 1e6   
        x_50_ori *= 1e6   
        x_90_ori *= 1e6  
        if self.k.smoothing:
            kde = self.k.KDE_fit(x_uni_ori, q3_ori)
            sumN_uni = self.k.KDE_score(kde, x_uni_ori)
            _, q3_ori, Q3_ori, _, _,_ = self.k.re_cal_distribution(x_uni_ori, sumN_uni)

        # Recalculate PSD using optimization results
        if hasattr(self.k, 'corr_beta_opt') and hasattr(self.k, 'alpha_prim_opt'):
            self.k.cal_pop(self.k.p, self.k.corr_beta_opt, self.k.alpha_prim_opt)
        else:
            print("Need to run the optimization process at least once！")    
            
        x_uni, q3, Q3, x_10, x_50, x_90 = self.k.p.return_num_distribution_fixed(t=len(self.k.p.t_vec)-1)
        # Conversion unit
        x_uni *= 1e6    
        x_10 *= 1e6   
        x_50 *= 1e6   
        x_90 *= 1e6  
        if self.k.smoothing:
            kde = self.k.KDE_fit(x_uni, q3)
            sumN_uni = self.k.KDE_score(kde, x_uni)
            _, q3, Q3, _, _,_ = self.k.re_cal_distribution(x_uni, sumN_uni)
        
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

        if not exp_data_path == None:
            # read and calculate the experimental data
            t = max(self.p.t_vec)
    
            x_uni_exp, q3_exp, Q3_exp, x_10_exp, x_50_exp, x_90_exp = self.k.read_exp(x_uni, exp_data_path)
            
            axq3, fig = pt.plot_data(x_uni_exp, q3_exp, fig=fig, ax=axq3,
                                   lbl='q3_exp',clr='g',mrk='^')
            axQ3, fig = pt.plot_data(x_uni_exp, Q3_exp, fig=fig, ax=axQ3,
                                   lbl='Q3_exp',clr='g',mrk='^')
        
        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return axq3, axQ3, fig