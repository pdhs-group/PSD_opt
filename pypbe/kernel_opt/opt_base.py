# -*- coding: utf-8 -*-
"""
This script is to instantiate the :class:`opt_algo` or :class:`opt_algo_multi()` classes, pass parameters to them, 
and return optimization results. It includes additional functionalities for generating synthetic data and 
visualizing results.
"""

import numpy as np
import os
import importlib.util
import warnings
import pandas as pd
import ray
from ..pbe.dpbe_base import DPBESolver
from .opt_core import OptCore
from .opt_core_multi import OptCoreMulti
from ..utils.func.func_read_exp import write_read_exp
## For plots
import matplotlib.pyplot as plt
from ..utils.plotter import plotter as pt        

class OptBase():
    """
    A class to manage the optimization process for finding the kernel of PBE.
    
    Attributes
    ----------
    multi_flag : bool
        Flag to choose between single dimension optimization (False) and multi-dimensional optimization (True).
    
    Methods
    -------

    """
    def __init__(self, config_path=None, data_path=None):
        ## read config file and get all attribute
        self.pth = os.path.dirname( __file__ )
        config = self.check_config_path(config_path)
        self.core_params = config['algo_params']
        self.pop_params = config['pop_params']
        self.multi_flag = config['multi_flag']
        self.opt_params = config['opt_params']
        self.dim = self.core_params.get('dim', None)
        ## initialize instance of optimization algorithm and PBE
        self.init_opt_algo(data_path)
        self.init_opt_pop()
        
    def check_config_path(self, config_path):
        if config_path is None:
            config_path = os.path.join(self.pth, "..","..","config","opt_config.py")
            config_name = "opt_config"
        if not os.path.exists(config_path):
            raise Exception(f"Warning: Config file not found at: {config_path}.")
        else:
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            spec = importlib.util.spec_from_file_location(config_name, config_path)
            conf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf)
            config = conf.config
            return config
    def init_opt_algo(self, data_path=None):
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
        
        if self.dim == 1:
            print("The multi algorithm does not support 1-D pop!")
            self.multi_flag = False
        if not self.multi_flag:
            self.core = OptCore()
        else:
            self.core = OptCoreMulti()  
        for key, value in self.core_params.items():
            setattr(self.core, key, value)

        self.core.t_init = self.core.t_init.astype(float)
        self.core.t_vec = self.core.t_vec.astype(float)
        self.core.num_t_init = len(self.core.t_init)
        self.core.num_t_steps = len(self.core.t_vec)
        if self.core.delta_t_start_step < 1 or self.core.delta_t_start_step >= self.core.num_t_steps:
            raise Exception("The value of delta_t_start_step must be within the indices range of t_vec! and >0")
        ## Get the complete simulation time and get the indices corresponding 
        ## to the vec and init time vectors
        self.core.t_all = np.concatenate((self.core.t_init, self.core.t_vec))
        self.core.t_all = np.unique(self.core.t_all)
        self.core.idt_init = [np.where(self.core.t_all == t_time)[0][0] for t_time in self.core.t_init]
        self.idt_vec = [np.where(self.core.t_all == t_time)[0][0] for t_time in self.core.t_vec]
        # Set the base path for exp_data_path
        if data_path is None:
            print('Data path is not found or is None, default path will be used.')
            self.data_path = os.path.join(self.pth, "..", "data")
        else:
            self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        
    def init_opt_pop(self):
        self.core.p = DPBESolver(dim=self.dim, disc='geo', t_vec=self.core.t_vec, load_attr=False)
        ## The 1D-pop data is also used when calculating the initial N of 2/3D-pop.
        if self.dim == 2:
            self.core.create_1d_pop(self.core.t_vec, disc='geo')
        self.core.set_init_pop_para(self.pop_params)
        self.core.set_comp_para(self.data_path)
        
    def generate_data(self, pop_params=None, add_info=""):
        """
        Generates synthetic data based on simulation results, optionally adding noise.
        
        Parameters
        ----------
        sample_num : `int`, optional
            Number of synthetic data samples to generate. Default is 1.
        add_info : `str`, optional
            Additional information to append to the file name. Default is an empty string.
        """
        if pop_params is None:
            pop_params = self.pop_params
        if self.core.add_noise:
            # Modify the file name to include noise type and strength
            filename = f"Sim_{self.core.noise_type}_{self.core.noise_strength}"+add_info+".xlsx"
        else:
            # Use the default file name
            filename = "Sim"+add_info+".xlsx"

        # Combine the base path with the modified file name
        exp_data_path = os.path.join(self.data_path, filename)
        
        if not self.multi_flag:
            self.core.calc_pop(self.core.p, pop_params, self.core.t_all)
            if self.core.p.calc_status:
                for i in range(0, self.core.sample_num):
                    if self.core.sample_num != 1:
                        exp_data_path=self.core.traverse_path(i, exp_data_path)
                    # print(self.core.exp_data_path)
                    self.write_new_data(self.core.p, exp_data_path)
            else:
                return
        else:
            exp_data_paths = [
                exp_data_path,
                exp_data_path.replace(".xlsx", "_NM.xlsx"),
                exp_data_path.replace(".xlsx", "_M.xlsx")
            ]
            self.core.calc_all_pop(pop_params, self.core.t_all)
            if self.core.p.calc_status and self.core.p_NM.calc_status and self.core.p_M.calc_status:
                for i in range(0, self.core.sample_num):
                    if self.core.sample_num != 1:
                        exp_data_paths = self.core.traverse_path(i, exp_data_paths)
                        self.write_new_data(self.core.p, exp_data_paths[0])
                        self.write_new_data(self.core.p_NM, exp_data_paths[1])
                        self.write_new_data(self.core.p_M, exp_data_paths[2])
            else:
                return
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
        x_uni = pop.calc_x_uni()
        v_uni = pop.calc_v_uni()
        formatted_times = write_read_exp.convert_seconds_to_time(self.core.t_all)
        sumN_uni = np.zeros((len(x_uni)-1, len(self.core.t_all)))
        
        for idt in self.idt_vec[1:]:
            if self.core.smoothing:
                sumvol_uni = pop.return_distribution(t=idt, flag='sumvol_uni')[0]
                ## The volume of particles with index=0 is 0. 
                ## In theory, such particles do not exist.
                kde = self.core.KDE_fit(x_uni[1:], sumvol_uni[1:])
                ## Recalculate the values of after smoothing
                ## In order to facilitate subsequent processing, 
                ## a 0 needs to be filled in the first bit of q3
                q3 = self.core.KDE_score(kde, x_uni[1:])
                q3 = np.insert(q3, 0, 0.0)
                Q3 = self.core.calc_Q3(x_uni, q3)
                ## Normalize Q3 to ensure that its maximum value is 1 
                Q3 = Q3 / Q3.max()
                sumvol_uni = self.core.calc_sum_uni(Q3, sumvol_uni.sum())
                sumN_uni[:, idt] = sumvol_uni[1:] / v_uni[1:]
            else:
                sumN_uni[:, idt] = pop.return_num_distribution(t=idt, flag='sumN_uni')[0][1:]
        ## Data used for initialization should not be smoothed
        for idt in self.core.idt_init:
            ## The volume of particles with index=0 is 0. 
            ## In theory, such particles do not exist.
            sumN_uni[:, idt] = pop.return_num_distribution(t=idt, flag='sumN_uni')[0][1:]
        
        if self.core.add_noise:
            sumN_uni = self.core.function_noise(sumN_uni)

        df = pd.DataFrame(data=sumN_uni, index=x_uni[1:], columns=formatted_times)
        df.index.name = 'Circular Equivalent Diameter'
        # save DataFrame as Excel file
        df.to_excel(exp_data_path)
        return        
    def find_opt_kernels(self, method='kernels', data_names=None, known_params=None):
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
        data_names : `str`, optional
            Name of the experimental data file (without labels).
        
        Returns
        -------
        `tuple`
            A tuple containing optimized kernels in PBE and their difference to original kernels(if given).
        """
        if self.core.set_comp_para_flag is False:
            warnings.warn('Component parameters have not been set')
        if self.core.set_init_pop_para_flag is False:
            warnings.warn('Initial PBE parameters have not been set')
            
        if data_names == None:
            warnings.warn("Please specify the name of the experiment data without labels!")
        else:
            def join_paths(names):
                if isinstance(names, list):
                    return [os.path.join(self.data_path, name) for name in names]
                return os.path.join(self.data_path, names)
            
            exp_data_paths = []
            if self.multi_flag:
                if isinstance(data_names[0], list):
                    ## known_params needs to always be the same length as data_names, 
                    ## even if its contents are None.
                    if known_params is None:
                        known_params = [None] * len(data_names)
                    ## optimization for multiple exp_data
                    for data_names_ex in data_names:
                        exp_data_paths.append(join_paths(data_names_ex))
                else:
                    ## optimization for one exp_data
                    exp_data_paths = join_paths(data_names)
            else:
                if isinstance(data_names, list):
                    if known_params is None:
                        known_params = [None] * len(data_names)
                            
                exp_data_paths = join_paths(data_names)
            # if self.core.calc_init_N:
            #     self.core.set_init_N(exp_data_paths, init_flag='mean')
            # ray.init(address="auto", log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))}})
            ray.init(log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))}})    
            if method == 'kernels':
                # delta_opt_sample = np.zeros(sample_num)
                # CORR_BETA_sample = np.zeros(sample_num)
                # if self.core.p.dim == 1:
                #     alpha_prim_sample = np.zeros(sample_num)
                    
                # elif self.core.p.dim == 2:
                #     alpha_prim_sample = np.zeros((3, sample_num))
                
                # if sample_num == 1:
                #     delta_opt = self.core.optimierer_agg(self.opt_params, exp_data_paths=exp_data_paths)
                #     CORR_BETA = self.core.CORR_BETA_opt
                #     alpha_prim = self.core.alpha_prim_opt
                    
                # else:
                #     for i in range(0, sample_num):
                #         exp_data_path=self.core.traverse_path(i, exp_data_path)
                #         delta_opt_sample[i] = \
                #             self.core.optimierer_agg(self.opt_params, exp_data_path=exp_data_path)
                            
                #         CORR_BETA_sample[i] = self.core.CORR_BETA_opt
                #         if self.core.p.dim == 1:
                #             alpha_prim_sample[i] = self.core.alpha_prim_opt
                            
                #         elif self.core.p.dim == 2:
                #             alpha_prim_sample[:, i] = self.core.alpha_prim_opt

                # if not sample_num == 1:
                #     delta_opt = np.mean(delta_opt_sample)
                #     CORR_BETA = np.mean(CORR_BETA_sample)
                #     alpha_prim = np.mean(alpha_prim_sample, axis=self.core.p.dim-1)
                #     self.core.CORR_BETA_opt = CORR_BETA
                #     self.core.alpha_prim_opt = alpha_prim
                print("not coded yet")
            elif method == 'delta':
                if self.core.multi_jobs:
                    result_dict = self.core.multi_optimierer_ray(self.opt_params,exp_data_paths=exp_data_paths, 
                                                                   known_params=known_params)
                else:
                    result_dict = []
                    if isinstance(exp_data_paths[0], list):
                        for exp_data_paths_tem, known_params_tem in zip(exp_data_paths, known_params):
                            result_dict_tem = self.core.optimierer_ray(self.opt_params,exp_data_paths=exp_data_paths_tem,
                                                                        known_params=known_params_tem)
                            # result_dict_tem = self.core.optimierer_bo(self.opt_params,exp_data_paths=exp_data_paths_tem,
                            #                                             known_params=known_params_tem)
                            result_dict.append(result_dict_tem)
                    else:
                        result_dict = self.core.optimierer_ray(self.opt_params,exp_data_paths=exp_data_paths,
                                                               known_params=known_params)
                        # result_dict = self.core.optimierer_bo(self.opt_params,exp_data_paths=exp_data_paths_tem,
                        #                                            known_params=known_params)
                # delta_opt = self.core.optimierer(sample_num=sample_num, 
                #                       exp_data_path=exp_data_path)
                
                
            # if self.core.p.dim == 1:
            #     para_diff_i = np.zeros(2)
            #     para_diff_i[0] = abs(self.core.CORR_BETA_opt- self.core.CORR_BETA) / self.core.CORR_BETA
            #     para_diff_i[1] = abs(self.core.alpha_prim_opt - self.core.alpha_prim)
                
            # elif self.core.p.dim == 2:
            #     para_diff_i = np.zeros(4)
            #     para_diff_i[0] = abs(self.core.CORR_BETA_opt- self.core.CORR_BETA) / self.core.CORR_BETA
            #     para_diff_i[1:] = abs(self.core.alpha_prim_opt - self.core.alpha_prim)
            
            # corr_agg = self.core.CORR_BETA * self.core.alpha_prim
            # corr_agg_opt = self.core.CORR_BETA_opt * self.core.alpha_prim_opt
            # corr_agg_diff = abs(corr_agg_opt - corr_agg) / np.where(corr_agg == 0, 1, corr_agg)
            # para_diff=para_diff_i.mean()
            ray.shutdown()
            return result_dict
                
            # return self.core.CORR_BETA_opt, self.core.alpha_prim_opt, para_diff, delta_opt
            
    def calc_PSD_delta(self, params, exp_data_path):
        if self.core.calc_init_N:
            self.core.set_init_N(exp_data_path, init_flag='mean')
        if isinstance(exp_data_path, list):
            ## When set to multi, the exp_data_path entered here is a list 
            ## containing one 2d data name and two 1d data names.
            exp_data_path_ori = exp_data_path[0]
            x_uni_exp = []
            data_exp = []
            for exp_data_path_tem in exp_data_path:
                if self.core.exp_data:
                    x_uni_exp_tem, data_exp_tem = self.core.get_all_exp_data(exp_data_path_tem)
                else:
                    x_uni_exp_tem, data_exp_tem = self.core.get_all_synth_data(exp_data_path_tem)
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
        else:
            ## When not set to multi or optimization of 1d-data, the exp_data_path 
            ## contain the name of that data.
            exp_data_path_ori = exp_data_path
            if self.core.exp_data:
                x_uni_exp, data_exp = self.core.get_all_exp_data(exp_data_path)
            else:
                x_uni_exp, data_exp = self.core.get_all_synth_data(exp_data_path)
        delta = self.core.calc_delta(params, x_uni_exp, data_exp)
        return delta, exp_data_path_ori
    
        
