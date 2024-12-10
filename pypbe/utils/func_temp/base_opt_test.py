# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import sys
import os
import ray
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
from pypbe.kernel_opt.opt_base import OptBase
import config.opt_config as conf
import numpy as np
import pandas as pd
import time
## For plots
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt  

def normal_test():
    start_time = time.time()

    # corr_beta_opt, alpha_prim_opt, para_diff, delta_opt= \
    #     find.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    result_dict = \
        opt.find_opt_kernels(method='delta', data_names=exp_data_paths, known_params=known_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")
    
    ## Calculate PBE with exp-data and parameter from optimization
    opt.core.set_init_N(exp_data_paths, 'mean')
    opt.core.calc_pop(opt.core.p, result_dict["opt_params"], opt.core.t_vec, opt.core.init_N)
    x_uni, Q3 = return_pop_distribution()
    
    return  x_uni, Q3 , result_dict

def return_pop_distribution():
    # Berechne x_uni
    x_uni = opt.core.p.calc_x_uni()
    
    # Erstelle eine Liste der Verteilungen für die ersten 12 Zeitschritte
    Q3 = [np.array(opt.core.p.return_distribution(t=i, flag='Q3')).reshape(len(x_uni)) for i in range(len(opt.core.t_vec))]
    
    # Zusammenfügen zu einer (52, 12) Matrix
    Q3_matrix = np.column_stack(Q3)
    
    return x_uni, Q3_matrix


def calc_delta_test(var_delta=False):
    # pop_params = conf.config['pop_params']
    pop_params = {'pl_v': 0.012150989218049626,
     'pl_P1': 2.4030393550212942e-06,
     'pl_P2': 0.7962978267023504,
     'corr_agg': np.array([0.00074662])}
    opt.core.init_attr(opt.core_params)
    opt.core.init_pbe(opt.pop_params, opt.data_path) 
    if opt.core.calc_init_N:
        opt.core.set_init_N(exp_data_paths, 'mean')
    if isinstance(exp_data_paths, list):
        x_uni_exp = []
        data_exp = []
        for exp_data_path_tem in exp_data_paths:
            if opt.core.exp_data:
                x_uni_exp_tem, data_exp_tem = opt.core.get_all_exp_data(exp_data_path_tem)
            else:
                x_uni_exp_tem, data_exp_tem = opt.core.get_all_synth_data(exp_data_path_tem)
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
    else:
        if opt.core.exp_data:
            x_uni_exp, data_exp = opt.core.get_all_exp_data(exp_data_paths)
        else:
            x_uni_exp, data_exp = opt.core.get_all_synth_data(exp_data_paths)    
    # corr_agg = pop_params['CORR_BETA'] * pop_params['alpha_prim']
    # pop_params_test = {}
    # pop_params_test['corr_agg'] = corr_agg
    if var_delta:
        delta_arr = np.zeros(len(opt.core.t_vec))
        for start_step in range(1,len(opt.core.t_vec)):
            opt.core.delta_t_start_step = start_step
            delta_arr[start_step] = opt.core.calc_delta(pop_params, x_uni_exp, data_exp)
        return delta_arr
    else:
        delta = opt.core.calc_delta(pop_params, x_uni_exp, data_exp)
        x_uni, Q3 = return_pop_distribution()
        return x_uni, Q3, delta

if __name__ == '__main__':
    ## Instantiate OptBase.
    ## The OptBase class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    opt = OptBase()
    multi_flag = opt.multi_flag
    
    data_name = "Mean_data_Q3_600.xlsx"
    
    exp_data_paths = os.path.join(opt.data_path, data_name)
    if multi_flag:
        exp_data_paths = [
            exp_data_paths,
            exp_data_paths.replace(".xlsx", "_NM.xlsx"),
            exp_data_paths.replace(".xlsx", "_M.xlsx")
        ]
    
    known_params = {
        # 'CORR_BETA' : 1.0,
        # 'alpha_prim' : [1e-3,1e-3,0.1],
        # 'pl_v' : v,
        # 'pl_P1' : P1,
        # 'pl_P2' : P2,
        # 'pl_P3' : P3,
        # 'pl_P4' : P4,
        }
    
    # Run an optimization and generate graphs of the results
    # x_uni, Q3 , result_dict = normal_test()
    
    x_uni_test, Q3_test , delta = calc_delta_test(var_delta=False)
