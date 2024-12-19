# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import sys
import os
import ray
from optframework.kernel_opt.opt_base import OptBase
import config.opt_config as conf
import numpy as np
import pandas as pd
import time
## For plots
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt  

def normal_test():
    start_time = time.time()

    # corr_beta_opt, alpha_prim_opt, para_diff, delta_opt= \
    #     find.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    result_dict = \
        opt.find_opt_kernels(method='delta', data_names=exp_data_paths, known_params=known_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")
    
    ## Calculate PBE with synth-data and parameter from optimization in 2D
    opt.core.set_init_N(exp_data_paths, 'mean')
    opt.core.calc_all_pop(result_dict["opt_params"], opt.core.t_vec)
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
    pop_params = {"pl_v": 2,
            "pl_P1": 1e-2,
            "pl_P2": 1,
            "pl_P3": 1e-2,
            "pl_P4": 1,
            "G": 80,
     'corr_agg': np.array([1e-2,1e-2,1e-2])}
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
    
    data_name = "Sim_Mul_0.1_para_1_0.01_0.01_0.01_2_0.01_1_0.01_1.xlsx"
    
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
    x_uni, Q3 , result_dict = normal_test()
    
    # x_uni_test, Q3_test , delta = calc_delta_test(var_delta=False)
