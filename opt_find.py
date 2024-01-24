# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:22:00 2023

@author: px2030
"""

import numpy as np
import os
import warnings
import pandas as pd
from pop import population
from opt_algo import opt_algo 
from opt_algo_multi import opt_algo_multi
from func.func_read_exp import write_read_exp
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt          

class opt_find():
    def __init__(self):
        self.multi_flag=True
        
    def init_opt_algo(self, dim=1, t_init= None, t_vec=None, add_noise=False, noise_type='Gaus',
                      noise_strength=0.01,smoothing=False):
        if not self.multi_flag:
            self.algo = opt_algo()
        else:
            if dim == 1:
                warnings.warn("The multi algorithm does not support 1-D pop!")
            self.algo = opt_algo_multi()  
        self.algo.add_noise = add_noise
        self.algo.smoothing = smoothing
        self.algo.dim = dim
        self.algo.noise_type = noise_type
        self.algo.noise_strength = noise_strength
        self.algo.t_init = t_init
        self.algo.num_t_init = len(t_init)
        self.algo.t_vec = t_vec
        self.algo.num_t_steps = len(t_vec)
        ## Get the complete simulation time and get the indices corresponding 
        ## to the vec and init time vectors
        self.algo.t_all = np.sort(np.concatenate((self.algo.t_init, self.algo.t_vec)))
        self.algo.idt_vec = [np.where(self.algo.t_all == t_time)[0][0] for t_time in self.algo.t_vec]
        self.algo.idt_init = [np.where(self.algo.t_all == t_time)[0][0] for t_time in self.algo.t_init]
        
        self.algo.method='BO'
        
        self.algo.p = population(dim=dim, disc='geo')
        ## The 1D-pop data is also used when calculating the initial N of 2/3D-pop.
        if dim >= 2:
            self.algo.create_1d_pop(disc='geo')
        # Set the base path for exp_data_path
        self.base_path = os.path.join(self.algo.p.pth, "data")
        
    def generate_data(self, sample_num=1, add_info=""):
        if self.algo.add_noise:
            # Modify the file name to include noise type and strength
            filename = f"Sim_{self.algo.noise_type}_{self.algo.noise_strength}"+add_info+".xlsx"
        else:
            # Use the default file name
            filename = "Sim"+add_info+".xlsx"

        # Combine the base path with the modified file name
        exp_data_path = os.path.join(self.base_path, filename)
        
        if not self.multi_flag:
            self.algo.cal_pop(self.algo.p, self.algo.corr_beta, self.algo.alpha_prim, self.algo.t_all)
            
            for i in range(0, sample_num):
                if sample_num != 1:
                    exp_data_path=self.algo.traverse_path(i, exp_data_path)
                # print(self.algo.exp_data_path)
                self.write_new_data(self.algo.p, exp_data_path)
        else:
            exp_data_paths = [
                exp_data_path,
                exp_data_path.replace(".xlsx", "_NM.xlsx"),
                exp_data_path.replace(".xlsx", "_M.xlsx")
            ]
            self.algo.cal_all_pop(self.algo.corr_beta, self.algo.alpha_prim, self.algo.t_all)
            
            for i in range(0, sample_num):
                if sample_num != 1:
                    exp_data_paths = self.algo.traverse_path(i, exp_data_paths)
                    self.write_new_data(self.algo.p, exp_data_paths[0])
                    self.write_new_data(self.algo.p_NM, exp_data_paths[1])
                    self.write_new_data(self.algo.p_M, exp_data_paths[2])
            
    def find_opt_kernels(self, sample_num, method='kernels', data_name=None):
        if data_name == None:
            warnings.warn("Please specify the name of the training data without labels!")
        else:
            exp_data_path = os.path.join(self.base_path, data_name)

            exp_data_paths = [
                exp_data_path,
                exp_data_path.replace(".xlsx", "_NM.xlsx"),
                exp_data_path.replace(".xlsx", "_M.xlsx")
            ]
            
            if self.multi_flag:
                # Because manual initialization N always requires 1D data(use exp_data_paths), 
                # and only multi-optimization process requires 1D data.
                exp_data_path = exp_data_paths
            
            if self.algo.calc_init_N:
                self.algo.set_init_N(sample_num, exp_data_paths, init_flag='mean')
                
            if method == 'kernels':
                delta_opt_sample = np.zeros(sample_num)
                corr_beta_sample = np.zeros(sample_num)
                if self.algo.p.dim == 1:
                    alpha_prim_sample = np.zeros(sample_num)
                    
                elif self.algo.p.dim == 2:
                    alpha_prim_sample = np.zeros((3, sample_num))
                
                if sample_num == 1:
                    delta_opt = self.algo.optimierer_agg(exp_data_path=exp_data_path)
                    corr_beta = self.algo.corr_beta_opt
                    alpha_prim = self.algo.alpha_prim_opt
                    
                else:
                    for i in range(0, sample_num):
                        exp_data_path=self.algo.traverse_path(i, exp_data_path)
                        delta_opt_sample[i] = \
                            self.algo.optimierer_agg(exp_data_path=exp_data_path)
                            
                        corr_beta_sample[i] = self.algo.corr_beta_opt
                        if self.algo.p.dim == 1:
                            alpha_prim_sample[i] = self.algo.alpha_prim_opt
                            
                        elif self.algo.p.dim == 2:
                            alpha_prim_sample[:, i] = self.algo.alpha_prim_opt

                if not sample_num == 1:
                    delta_opt = np.mean(delta_opt_sample)
                    corr_beta = np.mean(corr_beta_sample)
                    alpha_prim = np.mean(alpha_prim_sample, axis=self.algo.p.dim-1)
                    self.algo.corr_beta_opt = corr_beta
                    self.algo.alpha_prim_opt = alpha_prim
                
            elif method == 'delta':
                delta_opt = self.algo.optimierer_agg(sample_num=sample_num, 
                                      exp_data_path=exp_data_path)
                # delta_opt = self.algo.optimierer(sample_num=sample_num, 
                #                       exp_data_path=exp_data_path)
                
                
            if self.algo.p.dim == 1:
                para_diff_i = np.zeros(2)
                para_diff_i[0] = abs(self.algo.corr_beta_opt- self.algo.corr_beta) / self.algo.corr_beta
                para_diff_i[1] = abs(self.algo.alpha_prim_opt - self.algo.alpha_prim)
                
            elif self.algo.p.dim == 2:
                para_diff_i = np.zeros(4)
                para_diff_i[0] = abs(self.algo.corr_beta_opt- self.algo.corr_beta) / self.algo.corr_beta
                para_diff_i[1:] = abs(self.algo.alpha_prim_opt - self.algo.alpha_prim)
            
            corr_agg = self.algo.corr_beta * self.algo.alpha_prim
            corr_agg_opt = self.algo.corr_beta_opt * self.algo.alpha_prim_opt
            corr_agg_diff = abs(corr_agg_opt - corr_agg) / corr_agg
            para_diff=para_diff_i.mean()
            return self.algo.corr_beta_opt, self.algo.alpha_prim_opt, para_diff, delta_opt, \
                corr_agg, corr_agg_opt, corr_agg_diff
                
            # return self.algo.corr_beta_opt, self.algo.alpha_prim_opt, para_diff, delta_opt
        
        
    def write_new_data(self, pop, exp_data_path):
        # save the calculation result in experimental data form
        x_uni = self.algo.cal_x_uni(pop)
        formatted_times = write_read_exp.convert_seconds_to_time(self.algo.t_all)
        q3 = np.zeros((len(x_uni), len(self.algo.t_all)))
        
        for idt in self.algo.idt_vec:
            q3_tem = pop.return_num_distribution(t=idt, flag='q3')[0]
            kde = self.algo.KDE_fit(x_uni, q3_tem)
            sumV_uni = self.algo.KDE_score(kde, x_uni)
            _, q3_tem, _, _, _,_ = self.algo.re_cal_distribution(x_uni, sumV_uni)
            q3[:, idt] = q3_tem
        ## Data used for initialization should not be smoothed
        for idt in self.algo.idt_init:
            q3[:, idt] = pop.return_num_distribution(t=idt, flag='q3')[0]
        
        if self.algo.add_noise:
            q3 = self.algo.function_noise(q3)

        df = pd.DataFrame(data=q3, index=x_uni, columns=formatted_times)
        df.index.name = 'Circular Equivalent Diameter'
        # save DataFrame as Excel file
        df.to_excel(exp_data_path)
        
        return 
    
    # Visualize only the last time step of the specified time vector and the last used experimental data
    def visualize_distribution(self, pop, corr_beta_ori, alpha_prim_ori, corr_beta_opt, 
                               alpha_prim_opt, exp_data_path=None,ax=None,fig=None,
                               close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    # Recalculate PSD using original parameter
        self.algo.cal_pop(pop, corr_beta=corr_beta_ori, alpha_prim=alpha_prim_ori)

        x_uni_ori, q3_ori, Q3_ori, x_10_ori, x_50_ori, x_90_ori = pop.return_distribution(t=-1, flag='all')
        # Conversion unit
        x_uni_ori *= 1e6    
        x_10_ori *= 1e6   
        x_50_ori *= 1e6   
        x_90_ori *= 1e6  
        if self.algo.smoothing:
            kde = self.algo.KDE_fit(x_uni_ori, q3_ori)
            sumV_uni = self.algo.KDE_score(kde, x_uni_ori)
            _, q3_ori, Q3_ori, _, _,_ = self.algo.re_cal_distribution(x_uni_ori, sumV_uni)

        self.algo.cal_pop(pop, corr_beta_opt, alpha_prim_opt)  
            
        x_uni, q3, Q3, x_10, x_50, x_90 = pop.return_distribution(t=-1, flag='all')
        # Conversion unit
        x_uni *= 1e6    
        x_10 *= 1e6   
        x_50 *= 1e6   
        x_90 *= 1e6  
        if self.algo.smoothing:
            kde = self.algo.KDE_fit(x_uni, q3)
            sumV_uni = self.algo.KDE_score(kde, x_uni)
            _, q3, Q3, _, _,_ = self.algo.re_cal_distribution(x_uni, sumV_uni)
        
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig=plt.figure()    
            axq3=fig.add_subplot(1,2,1)   
            axQ3=fig.add_subplot(1,2,2)   
        
        axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='number distribution of agglomerates $q3$ / $-$',
                               lbl='q3_mod',clr='b',mrk='o')
        

        axq3, fig = pt.plot_data(x_uni_ori, q3_ori, fig=fig, ax=axq3,
                               lbl='q3_ori',clr='r',mrk='v')
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='accumulated number distribution of agglomerates $Q3$ / $-$',
                               lbl='Q3_mod',clr='b',mrk='o')

        axQ3, fig = pt.plot_data(x_uni_ori, Q3_ori, fig=fig, ax=axQ3,
                               lbl='Q3_ori',clr='r',mrk='v')
        
        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return fig
    
    def save_as_png(self, fig, file_name):
        file_path = os.path.join(self.base_path, file_name)
        fig.savefig(file_path, dpi=150)
        return 0
    
        
