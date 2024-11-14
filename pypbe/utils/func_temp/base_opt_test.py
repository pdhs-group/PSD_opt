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
    
    # fig=plt.figure()    
    # axq3=fig.add_subplot(1,2,1)
    # axQ3=fig.add_subplot(1,2,2)
    # fig_NM=plt.figure()    
    # axq3_NM=fig_NM.add_subplot(1,2,1)
    # axQ3_NM=fig_NM.add_subplot(1,2,2)
    # fig_M=plt.figure()    
    # axq3_M=fig_M.add_subplot(1,2,1)
    # axQ3_M=fig_M.add_subplot(1,2,2)
    
    # ## Calculate PBE direkt with psd-data and original parameter
    # pop_params = conf.config['pop_params']
    # param_str = data_name.split('para_')[-1]
    # param_str = param_str.rsplit('.', 1)[0] 
    # params = param_str.split('_')
    # converted_params = [float(param) if '.' in param or 'e' in param.lower() else int(param) for param in params]
    # pop_params['CORR_BETA'] = converted_params[0]
    # pop_params['alpha_prim'] = np.array(converted_params[1:4])
    # pop_params['pl_v'] = converted_params[4]
    # pop_params['pl_P1'] = converted_params[5]
    # pop_params['pl_P2'] = converted_params[6]
    # pop_params['pl_P3'] = converted_params[7]
    # pop_params['pl_P4'] = converted_params[8]
    # # opt.core.set_init_pop_para(pop_params)
    # # opt.core.calc_init_N = False
    # # opt.core.set_comp_para('r0_001', 'r0_001',R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl,
    # #                         dist_path_NM=dist_path_1,dist_path_M=dist_path_2)
    # opt.core.calc_all_pop(pop_params)
    # opt.core.p.visualize_distribution(axq3=axq3, axQ3=axQ3, fig=fig, clr='b', lbl='ori', smoothing=True)
    # opt.core.p_NM.visualize_distribution(axq3=axq3_NM, axQ3=axQ3_NM, fig=fig_NM, clr='b', lbl='ori', smoothing=True)
    # opt.core.p_M.visualize_distribution(axq3=axq3_M, axQ3=axQ3_M, fig=fig_M, clr='b', lbl='ori', smoothing=True)
    # ## Calculate PBE with exp-data and parameter from optimization
    # # opt.core.set_init_pop_para(opt_values)
    # # opt.core.calc_init_N = True
    # # opt.core.set_comp_para(R_NM=R_NM, R_M=R_M,R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl)
    # # opt.core.set_init_N(opt.core.sample_num, exp_data_paths, 'mean')
    # opt.core.calc_all_pop(result_dict["opt_params"])
    # # opt.core.calc_pop(opt.core.p, result_dict["opt_params"])
    # opt.core.p.visualize_distribution(axq3=axq3, axQ3=axQ3, fig=fig, clr='r', lbl='opt', smoothing=True)
    # opt.core.p_NM.visualize_distribution(axq3=axq3_NM, axQ3=axQ3_NM, fig=fig_NM, clr='r', lbl='opt', smoothing=True)
    # opt.core.p_M.visualize_distribution(axq3=axq3_M, axQ3=axQ3_M, fig=fig_M, clr='r', lbl='opt', smoothing=True) 
    # opt.core.p.visualize_distribution_animation(smoothing=True)
    
    return result_dict

def calc_delta_test(var_delta=False):
    # pop_params = conf.config['pop_params']
    pop_params = {'pl_v': 1.6372233629226685,
     'pl_P1': 0.0106435922581415,
     'pl_P2': 0.49002260278233983,
     'pl_P3': 0.00020077684158093307,
     'pl_P4': 1.6331881284713745,
     'corr_agg': np.array([0.00068498, 0.00086928, 0.00011673])}
    if opt.core.calc_init_N:
        opt.core.set_init_N(exp_data_paths, 'mean')
    opt.core.init_attr(opt.core_params)
    opt.core.init_pbe(opt.pop_params, opt.data_path) 
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
        return delta

if __name__ == '__main__':
    ## Instantiate OptBase.
    ## The OptBase class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    opt = OptBase()
    
    data_name = "Sim_Mul_0.1_para_1.0_0.001_0.001_0.001_1.0_0.0001_0.5_0.0001_0.5.xlsx"  
    exp_data_path = os.path.join(opt.data_path, data_name)
    exp_data_paths = [
        exp_data_path,
        exp_data_path.replace(".xlsx", "_NM.xlsx"),
        exp_data_path.replace(".xlsx", "_M.xlsx")
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
    result_dict = normal_test()
    
    # delta = calc_delta_test(var_delta=False)
