# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import sys
import os
import ray
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
import pypbe.kernel_opt.opt_find as opt
import config.opt_config as conf
import numpy as np
import pandas as pd
import time
## For plots
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt  

os.environ["NUMEXPR_MAX_THREADS"] = "8"

def normal_test():
    start_time = time.time()

    # corr_beta_opt, alpha_prim_opt, para_diff, delta_opt= \
    #     find.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    result_dict = \
        find.find_opt_kernels(method='delta', data_names=exp_data_paths)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")
    
    fig=plt.figure()    
    axq3=fig.add_subplot(1,1,1)
    fig_NM=plt.figure()    
    axq3_NM=fig_NM.add_subplot(1,1,1)
    fig_M=plt.figure()    
    axq3_M=fig_M.add_subplot(1,1,1)
    
    ## Calculate PBE direkt with psd-data and original parameter
    pop_params = conf.config['pop_params']
    param_str = data_name.split('para_')[-1]
    param_str = param_str.rsplit('.', 1)[0] 
    params = param_str.split('_')
    converted_params = [float(param) if '.' in param or 'e' in param.lower() else int(param) for param in params]
    pop_params['CORR_BETA'] = converted_params[0]
    pop_params['alpha_prim'] = np.array(converted_params[1:4])
    pop_params['pl_v'] = converted_params[4]
    pop_params['pl_P1'] = converted_params[5]
    pop_params['pl_P2'] = converted_params[6]
    pop_params['pl_P3'] = converted_params[7]
    pop_params['pl_P4'] = converted_params[8]
    # find.algo.set_init_pop_para(pop_params)
    # find.algo.calc_init_N = False
    # find.algo.set_comp_para('r0_001', 'r0_001',R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl,
    #                         dist_path_NM=dist_path_1,dist_path_M=dist_path_2)
    find.algo.calc_all_pop(pop_params)
    return_pop_distribution(find.algo.p, axq3, fig, clr='b', q3lbl='q3_ori')
    return_pop_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='b', q3lbl='q3_ori')
    return_pop_distribution(find.algo.p_M, axq3_M, fig_M, clr='b', q3lbl='q3_ori')
    ## Calculate PBE with exp-data and parameter from optimization
    # find.algo.set_init_pop_para(opt_values)
    # find.algo.calc_init_N = True
    # find.algo.set_comp_para(R_NM=R_NM, R_M=R_M,R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl)
    # find.algo.set_init_N(find.algo.sample_num, exp_data_paths, 'mean')
    find.algo.calc_all_pop(result_dict["opt_parameters"])
    find.algo.calc_pop(find.algo.p, result_dict["opt_parameters"])
    return_pop_distribution(find.algo.p, axq3, fig, clr='r', q3lbl='q3_opt')
    return_pop_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='r', q3lbl='q3_opt')
    return_pop_distribution(find.algo.p_M, axq3_M, fig_M, clr='r', q3lbl='q3_opt')   
    find.save_as_png(fig, "PSD")
    find.save_as_png(fig_NM, "PSD-NM")
    find.save_as_png(fig_M, "PSD-M")
    
    return result_dict
    
def calc_N_test():
    find.algo.calc_init_N = False
    
    fig=plt.figure()    
    axq3=fig.add_subplot(1,1,1)
    fig_NM=plt.figure()    
    axq3_NM=fig_NM.add_subplot(1,1,1)
    fig_M=plt.figure()    
    axq3_M=fig_M.add_subplot(1,1,1)
    
    ## Calculate PBE direkt with psd-data, result is raw exp-data
    pop_params = conf.config['pop_params']
    param_str = data_name.split('para_')[-1]
    param_str = param_str.rsplit('.', 1)[0] 
    params = param_str.split('_')
    converted_params = [float(param) if '.' in param or 'e' in param.lower() else int(param) for param in params]
    pop_params['CORR_BETA'] = converted_params[0]
    pop_params['alpha_prim'] = np.array(converted_params[1:4])
    pop_params['pl_v'] = converted_params[4]
    pop_params['pl_P1'] = converted_params[5]
    pop_params['pl_P2'] = converted_params[6]
    pop_params['pl_P3'] = converted_params[7]
    pop_params['pl_P4'] = converted_params[8]
    find.algo.set_init_pop_para(pop_params)
    find.algo.calc_init_N = False
    find.algo.set_comp_para(find.base_path)
    find.algo.calc_all_pop()
    # return_pop_num_distribution(find.algo.p, axq3, fig, clr='b', q3lbl='q3_psd')
    # q3_psd = return_pop_num_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='b', q3lbl='q3_psd')
    # return_pop_num_distribution(find.algo.p_M, axq3_M, fig_M, clr='b', q3lbl='q3_psd')
    q3_psd = return_pop_distribution(find.algo.p, axq3, fig, clr='b', q3lbl='q3_psd')
    return_pop_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='b', q3lbl='q3_psd')
    return_pop_distribution(find.algo.p_M, axq3_M, fig_M, clr='b', q3lbl='q3_psd')
    N_exp = find.algo.p.N
    N_exp_1D = find.algo.p_NM.N
    ## Calculate PBE with exp-data
    # find.algo.calc_init_N = True
    # find.algo.set_init_N(find.algo.sample_num, exp_data_paths, 'mean')
    find.algo.calc_all_pop()
    # return_pop_num_distribution(find.algo.p, axq3, fig, clr='r', q3lbl='q3_exp')
    # q3_exp = return_pop_num_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='r', q3lbl='q3_exp')
    # return_pop_num_distribution(find.algo.p_M, axq3_M, fig_M, clr='r', q3lbl='q3_exp')   
    q3_exp = return_pop_distribution(find.algo.p, axq3, fig, clr='r', q3lbl='q3_exp')
    return_pop_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='r', q3lbl='q3_exp')
    return_pop_distribution(find.algo.p_M, axq3_M, fig_M, clr='r', q3lbl='q3_exp')   
    N_calc = find.algo.p.N
    N_calc_1D = find.algo.p_NM.N
    
    return N_exp, N_calc, N_exp_1D, N_calc_1D, q3_psd, q3_exp

# def return_pop_num_distribution(pop, axq3=None,fig=None, clr='b', q3lbl='q3'):

#     x_uni, q3, Q3, sumvol_uni = pop.return_distribution(t=1, flag='x_uni, q3, Q3,sumvol_uni')
#     # kde = find.algo.KDE_fit(x_uni, q3)
#     # q3_sm = find.algo.KDE_score(kde, x_uni)

#     kde = find.algo.KDE_fit(x_uni, sumvol_uni)
#     q3_sm = find.algo.KDE_score(kde, x_uni)
    
#     axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
#                            xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
#                            ylbl='number distribution of agglomerates $q3$ / $-$',
#                            lbl=q3lbl,clr=clr,mrk='o')
    
#     axq3, fig = pt.plot_data(x_uni, q3_sm, fig=fig, ax=axq3,
#                             lbl=q3lbl+'_sm',clr=clr,mrk='^')
    
#     # axq3, fig = pt.plot_data(x_uni, sumN_uni, fig=fig, ax=axq3,
#     #                         xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
#     #                         ylbl='number distribution of agglomerates $q3$ / $-$',
#     #                         lbl='sumN_uni',clr='r',mrk='o') 
    
#     df = pd.DataFrame(data=q3_sm, index=x_uni)
#     axq3.grid('minor')
#     axq3.set_xscale('log')
#     # axq3.set_yscale('log')
#     return df

def return_pop_distribution(pop, axq3=None,fig=None, clr='b', q3lbl='q3'):

    x_uni = find.algo.calc_x_uni(pop)
    q3, Q3, sumvol_uni = pop.return_distribution(t=-1, flag='q3, Q3, sumvol_uni')
    # kde = find.algo.KDE_fit(x_uni, q3)
    # q3_sm = find.algo.KDE_score(kde, x_uni)

    kde = find.algo.KDE_fit(x_uni[1:], sumvol_uni[1:])
    q3_sm = find.algo.KDE_score(kde, x_uni[1:])
    q3_sm = np.insert(q3_sm, 0, 0.0)
    
    # axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
    #                        xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                        ylbl='number distribution of agglomerates $q3$ / $-$',
    #                        lbl=q3lbl,clr=clr,mrk='o')
    
    axq3, fig = pt.plot_data(x_uni, q3_sm, fig=fig, ax=axq3,
                            lbl=q3lbl+'_sm',clr=clr,mrk='^')
    
    axq3.grid('minor')
    axq3.set_xscale('log')
    # axq3, fig = pt.plot_data(x_uni, sumN_uni, fig=fig, ax=axq3,
    #                         xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                         ylbl='number distribution of agglomerates $q3$ / $-$',
    #                         lbl='sumN_uni',clr='r',mrk='o') 
    
    df = pd.DataFrame(data=q3_sm, index=x_uni)
    return df

def calc_delta_test(var_delta=False):
    pop_params = conf.config['pop_params']
    if find.algo.calc_init_N:
        find.algo.set_init_N(exp_data_paths, 'mean')
    x_uni_exp, data_exp = find.algo.get_all_exp_data(exp_data_paths)
    # corr_agg = pop_params['CORR_BETA'] * pop_params['alpha_prim']
    # pop_params_test = {}
    # pop_params_test['corr_agg'] = corr_agg
    if var_delta:
        delta_arr = np.zeros(len(find.algo.t_vec))
        for start_step in range(1,len(find.algo.t_vec)):
            find.algo.delta_t_start_step = start_step
            delta_arr[start_step] = find.algo.calc_delta_agg(pop_params, x_uni_exp, data_exp)
        return delta_arr
    else:
        delta = find.algo.calc_delta_agg(pop_params, x_uni_exp, data_exp)
        return delta

if __name__ == '__main__':
    ## Instantiate find and algo.
    ## The find class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    find = opt.OptFind()
    
    data_name = "Batchversuch_600rpm_1200rpm.xlsx"  
    exp_data_paths = os.path.join(find.data_path, data_name)
    # exp_data_paths = [
    #     exp_data_path,
    #     exp_data_path.replace(".xlsx", "_NM.xlsx"),
    #     exp_data_path.replace(".xlsx", "_M.xlsx")
    # ]
    
    # Run an optimization and generate graphs of the results
    result_dict = normal_test()
    
    # N_exp, N_calc, N_exp_1D, N_calc_1D, q3_psd, q3_exp = calc_N_test()
    
    # delta = calc_delta_test(var_delta=False)
