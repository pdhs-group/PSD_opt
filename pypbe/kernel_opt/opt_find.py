# -*- coding: utf-8 -*-
"""
This script is to instantiate the :class:`opt_algo` or :class:`opt_algo_multi()` classes, pass parameters to them, 
and return optimization results. It includes additional functionalities for generating synthetic data and 
visualizing results.
"""

import numpy as np
import os
import warnings
import pandas as pd
from ..dpbe import population
from .opt_algo import opt_algo 
from .opt_algo_multi import opt_algo_multi
from ..utils.func.func_read_exp import write_read_exp
## For plots
import matplotlib.pyplot as plt
from ..utils.plotter import plotter as pt        

class opt_find():
    """
    A class to manage the optimization process for finding the kernel of PBE.
    
    Attributes
    ----------
    multi_flag : bool
        Flag to choose between single dimension optimization (False) and multi-dimensional optimization (True).
    
    Methods
    -------

    """
    def __init__(self):
        self.multi_flag=True
        
    def init_opt_algo(self, multi_flag, algo_params, opt_params):
        """
        Initializes the optimization algorithm with specified parameters and configurations.
        
        Parameters
        ----------
        dim : `int`, optional
            Dimensionality of the population to optimize. Default is 1.
        t_init : `array`, optional
            Time points used to calculate the initial conditions for the simulation. Default is None.
        t_vec : `array`, optional
            Time vector over which to perform the optimization. Default is None.
        add_noise : `bool`, optional
            Flag to determine whether to add noise to the data. Default is False.
        noise_type : `str`, optional
            Type of noise to add: Gaussian ('Gaus'), Uniform ('Uni'), 
            Poisson ('Po'), and Multiplicative ('Mul'). Default is 'Gaus'.
        noise_strength : `float`, optional
            Strength of the noise to add. Default is 0.01.
        smoothing : `bool`, optional
            Flag to determine whether to apply smoothing(KDE) to the data. Default is False.
        """
        dim = algo_params.get('dim', None)
        self.opt_params = opt_params
        self.multi_flag = multi_flag
        if not self.multi_flag:
            self.algo = opt_algo()
        else:
            if dim == 1:
                warnings.warn("The multi algorithm does not support 1-D pop!")
            self.algo = opt_algo_multi()  
        for key, value in algo_params.items():
            setattr(self.algo, key, value)

        self.algo.num_t_init = len(self.algo.t_init)
        self.algo.num_t_steps = len(self.algo.t_vec)
        if self.algo.delta_t_start_step < 1 or self.algo.delta_t_start_step >= self.algo.num_t_steps:
            raise Exception("The value of delta_t_start_step must be within the indices range of t_vec! and >0")
        ## Get the complete simulation time and get the indices corresponding 
        ## to the vec and init time vectors
        self.algo.t_all = np.concatenate((self.algo.t_init, self.algo.t_vec))
        self.algo.t_all = np.unique(self.algo.t_all)
        self.algo.idt_init = [np.where(self.algo.t_all == t_time)[0][0] for t_time in self.algo.t_init]
        self.idt_vec = [np.where(self.algo.t_all == t_time)[0][0] for t_time in self.algo.t_vec]
        
        self.algo.p = population(dim=dim, disc='geo')
        ## The 1D-pop data is also used when calculating the initial N of 2/3D-pop.
        if dim >= 2:
            self.algo.create_1d_pop(disc='geo')
        # Set the base path for exp_data_path
        self.base_path = os.path.join(self.algo.p.pth, "data")
        
    def generate_data(self, pop_params=None, sample_num=1, add_info=""):
        """
        Generates synthetic data based on simulation results, optionally adding noise.
        
        Parameters
        ----------
        sample_num : `int`, optional
            Number of synthetic data samples to generate. Default is 1.
        add_info : `str`, optional
            Additional information to append to the file name. Default is an empty string.
        """
        if self.algo.add_noise:
            # Modify the file name to include noise type and strength
            filename = f"Sim_{self.algo.noise_type}_{self.algo.noise_strength}"+add_info+".xlsx"
        else:
            # Use the default file name
            filename = "Sim"+add_info+".xlsx"

        # Combine the base path with the modified file name
        exp_data_path = os.path.join(self.base_path, filename)
        
        if not self.multi_flag:
            self.algo.calc_pop(self.algo.p, pop_params, self.algo.t_all)
            
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
            self.algo.calc_all_pop(pop_params, self.algo.t_all)
            
            for i in range(0, sample_num):
                if sample_num != 1:
                    exp_data_paths = self.algo.traverse_path(i, exp_data_paths)
                    self.write_new_data(self.algo.p, exp_data_paths[0])
                    self.write_new_data(self.algo.p_NM, exp_data_paths[1])
                    self.write_new_data(self.algo.p_M, exp_data_paths[2])
            
    def find_opt_kernels(self, sample_num, method='kernels', data_name=None):
        """
        Finds optimal kernels for the PBE model by minimizing the difference between 
        simulation results and experimental data.
        
        Parameters
        ----------
        sample_num : `int`
            Number of samples to use in the optimization process.
        method : `str`, optional
            Optimization method to use.
                - 'kernels': Optimizes kernel parameters for each data set individually and then computes the average of the resulting kernel parameters across all data sets.
                - 'delta' : Averages the delta values across data sets before optimization, leading to a single kernel that optimizes the average delta.
        data_name : `str`, optional
            Name of the experimental data file (without labels).
        
        Returns
        -------
        `tuple`
            A tuple containing optimized kernels in PBE and their difference to original kernels(if given).
        """
        if self.algo.set_comp_para_flag is False:
            warnings.warn('Component parameters have not been set')
        if self.algo.set_init_pop_para_flag is False:
            warnings.warn('Initial PBE parameters have not been set')
            
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
                CORR_BETA_sample = np.zeros(sample_num)
                if self.algo.p.dim == 1:
                    alpha_prim_sample = np.zeros(sample_num)
                    
                elif self.algo.p.dim == 2:
                    alpha_prim_sample = np.zeros((3, sample_num))
                
                if sample_num == 1:
                    delta_opt = self.algo.optimierer_agg(self.opt_params, exp_data_path=exp_data_path)
                    CORR_BETA = self.algo.CORR_BETA_opt
                    alpha_prim = self.algo.alpha_prim_opt
                    
                else:
                    for i in range(0, sample_num):
                        exp_data_path=self.algo.traverse_path(i, exp_data_path)
                        delta_opt_sample[i] = \
                            self.algo.optimierer_agg(self.opt_params, exp_data_path=exp_data_path)
                            
                        CORR_BETA_sample[i] = self.algo.CORR_BETA_opt
                        if self.algo.p.dim == 1:
                            alpha_prim_sample[i] = self.algo.alpha_prim_opt
                            
                        elif self.algo.p.dim == 2:
                            alpha_prim_sample[:, i] = self.algo.alpha_prim_opt

                if not sample_num == 1:
                    delta_opt = np.mean(delta_opt_sample)
                    CORR_BETA = np.mean(CORR_BETA_sample)
                    alpha_prim = np.mean(alpha_prim_sample, axis=self.algo.p.dim-1)
                    self.algo.CORR_BETA_opt = CORR_BETA
                    self.algo.alpha_prim_opt = alpha_prim
                
            elif method == 'delta':
                delta_opt, opt_values = self.algo.optimierer_agg(self.opt_params, sample_num=sample_num,
                                                     exp_data_path=exp_data_path)
                # delta_opt = self.algo.optimierer(sample_num=sample_num, 
                #                       exp_data_path=exp_data_path)
                
                
            # if self.algo.p.dim == 1:
            #     para_diff_i = np.zeros(2)
            #     para_diff_i[0] = abs(self.algo.CORR_BETA_opt- self.algo.CORR_BETA) / self.algo.CORR_BETA
            #     para_diff_i[1] = abs(self.algo.alpha_prim_opt - self.algo.alpha_prim)
                
            # elif self.algo.p.dim == 2:
            #     para_diff_i = np.zeros(4)
            #     para_diff_i[0] = abs(self.algo.CORR_BETA_opt- self.algo.CORR_BETA) / self.algo.CORR_BETA
            #     para_diff_i[1:] = abs(self.algo.alpha_prim_opt - self.algo.alpha_prim)
            
            # corr_agg = self.algo.CORR_BETA * self.algo.alpha_prim
            # corr_agg_opt = self.algo.CORR_BETA_opt * self.algo.alpha_prim_opt
            # corr_agg_diff = abs(corr_agg_opt - corr_agg) / np.where(corr_agg == 0, 1, corr_agg)
            # para_diff=para_diff_i.mean()
            return delta_opt, opt_values
                
            # return self.algo.CORR_BETA_opt, self.algo.alpha_prim_opt, para_diff, delta_opt
        
        
    def write_new_data(self, pop, exp_data_path):
        """
        Saves the calculation results in the format of experimental data.
    
        Parameters
        ----------
        pop : :class:`pop.population`
            The population instance for which data is being generated.
        exp_data_path : `str`
            The file path where the experimental data will be saved.
        """
        if not pop.calc_status:
            return
        # save the calculation result in experimental data form
        x_uni = self.algo.calc_x_uni(pop)
        v_uni = self.algo.calc_v_uni(pop)
        formatted_times = write_read_exp.convert_seconds_to_time(self.algo.t_all)
        sumN_uni = np.zeros((len(x_uni)-1, len(self.algo.t_all)))
        
        for idt in self.idt_vec[1:]:
            if self.algo.smoothing:
                sumvol_uni = pop.return_distribution(t=idt, flag='sumvol_uni')[0]
                ## The volume of particles with index=0 is 0. 
                ## In theory, such particles do not exist.
                kde = self.algo.KDE_fit(x_uni[1:], sumvol_uni[1:])
                ## Recalculate the values of after smoothing
                q3 = self.algo.KDE_score(kde, x_uni[1:])
                Q3 = self.algo.calc_Q3(x_uni[1:], q3)
                sumvol_uni = self.algo.calc_sum_uni(Q3, sumvol_uni.sum())
                sumN_uni[:, idt] = sumvol_uni / v_uni[1:]
            else:
                sumN_uni[:, idt] = pop.return_num_distribution(t=idt, flag='sumN_uni')[0]
        ## Data used for initialization should not be smoothed
        for idt in self.algo.idt_init:
            ## The volume of particles with index=0 is 0. 
            ## In theory, such particles do not exist.
            sumN_uni[:, idt] = pop.return_num_distribution(t=idt, flag='sumN_uni')[0][1:]
        
        if self.algo.add_noise:
            sumN_uni = self.algo.function_noise(sumN_uni)

        df = pd.DataFrame(data=sumN_uni, index=x_uni[1:], columns=formatted_times)
        df.index.name = 'Circular Equivalent Diameter'
        # save DataFrame as Excel file
        df.to_excel(exp_data_path)
        return 
    
    # Visualize only the last time step of the specified time vector and the last used experimental data
    def visualize_distribution(self, pop, ori_params, opt_values, exp_data_paths,
                               R_NM, R_M, R01_0_scl, R03_0_scl,dist_path_1,dist_path_2, 
                               ax=None,fig=None,close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5],log_output=False):
        """
        Visualizes the distribution at the last time step.
        
        """
        ## Recalculate PSD using original parameter
        self.algo.calc_init_N = False
        self.algo.set_comp_para(R_NM=R_NM, R_M=R_M,R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl,
                                dist_path_NM=dist_path_1,dist_path_M=dist_path_2)
        ## Todo: set_comp_para with original parameter
        self.algo.calc_pop(pop, ori_params)

        x_uni_ori, q3_ori, Q3_ori, sumvol_uni_ori = pop.return_distribution(t=-1, flag='x_uni, q3, Q3,sumvol_uni')

        if self.algo.smoothing:
            kde = self.algo.KDE_fit(x_uni_ori, sumvol_uni_ori)
            q3_ori = self.algo.KDE_score(kde, x_uni_ori)
            Q3_ori = self.algo.calc_Q3(x_uni_ori, q3_ori)
            
        self.algo.calc_init_N = True
        self.algo.set_comp_para(R_NM=R_NM, R_M=R_M,R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl)
        self.algo.set_init_N(self.algo.sample_num, exp_data_paths, 'mean')
        self.algo.calc_pop(pop, opt_values)  
            
        x_uni, q3, Q3, sumvol_uni= pop.return_distribution(t=-1, flag='x_uni, q3, Q3,sumvol_uni')
        if self.algo.smoothing:
            kde = self.algo.KDE_fit(x_uni, sumvol_uni)
            q3 = self.algo.KDE_score(kde, x_uni)
            Q3 = self.algo.calc_Q3(x_uni, q3)
        
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
                               ylbl='accumulated volume distribution of agglomerates $Q3$ / $-$',
                               lbl='Q3_mod',clr='b',mrk='o')

        axQ3, fig = pt.plot_data(x_uni_ori, Q3_ori, fig=fig, ax=axQ3,
                               lbl='Q3_ori',clr='r',mrk='v')
        
        axq3.grid('minor')
        axQ3.grid('minor')
        if log_output:
            axq3.set_xscale('log')
            axQ3.set_xscale('log')
        plt.tight_layout()   
        
        return fig
    
    def save_as_png(self, fig, file_name):
        """
        Saves a figure as a PNG file.
        
        """
        file_path = os.path.join(self.base_path, file_name)
        fig.savefig(file_path, dpi=150)
        return 0
    
        
