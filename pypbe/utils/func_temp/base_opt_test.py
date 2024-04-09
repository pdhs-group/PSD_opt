# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
import pypbe.kernel_opt.opt_find as opt
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
    delta_opt, opt_values = \
        find.find_opt_kernels(sample_num=find.algo.sample_num, method='delta', data_name=data_name)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")
    
    return delta_opt, opt_values
    
def calc_N_test():
    find.algo.calc_init_N = False
    pth = os.path.dirname( __file__ )
    dist_path_1 = os.path.join(pth, "..", "..","data", "PSD_data", conf.config['dist_scale_1'])
    find.algo.set_comp_para('r0_001', 'r0_001', dist_path_1, dist_path_1,R01_0_scl=1e-1,R03_0_scl=1e-1)
    
    fig=plt.figure()    
    axq3=fig.add_subplot(1,1,1)
    fig_NM=plt.figure()    
    axq3_NM=fig_NM.add_subplot(1,1,1)
    fig_M=plt.figure()    
    axq3_M=fig_M.add_subplot(1,1,1)
    
    ## Calculate PBE direkt with psd-data, result is raw exp-data
    find.algo.calc_all_pop()
    return_pop_num_distribution(find.algo.p, axq3, fig, clr='b', q3lbl='q3_psd')
    q3_psd = return_pop_num_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='b', q3lbl='q3_psd')
    return_pop_num_distribution(find.algo.p_M, axq3_M, fig_M, clr='b', q3lbl='q3_psd')
    # return_pop_distribution(find.algo.p, axq3, fig, clr='b', q3lbl='q3_psd')
    # q3_psd = return_pop_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='b', q3lbl='q3_psd')
    # return_pop_distribution(find.algo.p_M, axq3_M, fig_M, clr='b', q3lbl='q3_psd')
    N_exp = find.algo.p.N
    N_exp_1D = find.algo.p_NM.N
    ## Calculate PBE with exp-data
    find.algo.calc_init_N = True
    find.algo.set_comp_para(R_NM=8.68e-7, R_M=8.68e-7,R01_0_scl=1e-1,R03_0_scl=1e-1)
    find.algo.set_init_N(find.algo.sample_num, exp_data_paths, 'mean')
    find.algo.calc_all_pop()
    return_pop_num_distribution(find.algo.p, axq3, fig, clr='r', q3lbl='q3_exp')
    q3_exp = return_pop_num_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='r', q3lbl='q3_exp')
    return_pop_num_distribution(find.algo.p_M, axq3_M, fig_M, clr='r', q3lbl='q3_exp')   
    # return_pop_distribution(find.algo.p, axq3, fig, clr='r', q3lbl='q3_exp')
    # q3_exp = return_pop_distribution(find.algo.p_NM, axq3_NM, fig_NM, clr='r', q3lbl='q3_exp')
    # return_pop_distribution(find.algo.p_M, axq3_M, fig_M, clr='r', q3lbl='q3_exp')   
    N_calc = find.algo.p.N
    N_calc_1D = find.algo.p_NM.N
    
    return N_exp, N_calc, N_exp_1D, N_calc_1D, q3_psd, q3_exp

def return_pop_num_distribution(pop, axq3=None,fig=None, clr='b', q3lbl='q3'):

    x_uni, q3, Q3, sumvol_uni = pop.return_distribution(t=-1, flag='x_uni, q3, Q3,sumvol_uni')
    # kde = find.algo.KDE_fit(x_uni, q3)
    # q3_sm = find.algo.KDE_score(kde, x_uni)

    kde = find.algo.KDE_fit(x_uni, sumvol_uni)
    q3_sm = find.algo.KDE_score(kde, x_uni)
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl=q3lbl,clr=clr,mrk='o')
    
    axq3, fig = pt.plot_data(x_uni, q3_sm, fig=fig, ax=axq3,
                            lbl=q3lbl+'_sm',clr=clr,mrk='^')
    
    # axq3, fig = pt.plot_data(x_uni, sumN_uni, fig=fig, ax=axq3,
    #                         xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                         ylbl='number distribution of agglomerates $q3$ / $-$',
    #                         lbl='sumN_uni',clr='r',mrk='o') 
    
    df = pd.DataFrame(data=q3_sm, index=x_uni)
    axq3.grid('minor')
    axq3.set_xscale('log')
    # axq3.set_yscale('log')
    return df

def return_pop_distribution(pop, axq3=None,fig=None, clr='b', q3lbl='q3'):

    x_uni = find.algo.calc_x_uni(pop)
    q3, Q3, sumvol_uni = pop.return_distribution(t=-1, flag='q3, Q3, sumvol_uni')
    # kde = find.algo.KDE_fit(x_uni, q3)
    # q3_sm = find.algo.KDE_score(kde, x_uni)

    kde = find.algo.KDE_fit(x_uni, sumvol_uni)
    q3_sm = find.algo.KDE_score(kde, x_uni)
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl=q3lbl,clr=clr,mrk='o')
    
    axq3, fig = pt.plot_data(x_uni, q3_sm, fig=fig, ax=axq3,
                            lbl=q3lbl+'_sm',clr=clr,mrk='^')
    
    # axq3, fig = pt.plot_data(x_uni, sumN_uni, fig=fig, ax=axq3,
    #                         xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                         ylbl='number distribution of agglomerates $q3$ / $-$',
    #                         lbl='sumN_uni',clr='r',mrk='o') 
    
    df = pd.DataFrame(data=q3_sm, index=x_uni)
    return df

def calc_delta_test():
    find.algo.set_init_N(find.algo.sample_num, exp_data_paths, 'mean')
    
    corr_agg = pop_params['CORR_BETA'] * pop_params['alpha_prim']
    pop_params_test = {}
    pop_params_test['corr_agg'] = corr_agg
    delta = find.algo.calc_delta_agg(pop_params_test, sample_num=find.algo.sample_num, exp_data_path=exp_data_paths)
    return delta

if __name__ == '__main__':
    #%%  Input for Opt
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    
    pop_params['CORR_BETA'] = 10.0
    pop_params['alpha_prim'] = np.array([0.5, 0.5, 0.5])
    pop_params['pl_v'] = 2
    pop_params['pl_P1'] = 1e-6
    pop_params['pl_P2'] = 1e-1
    pop_params['pl_P3'] = 1e-6
    pop_params['pl_P4'] = 1e-1
    pop_params['pl_P5'] = 1e-6
    pop_params['pl_P6'] = 1.0
    
    ## Instantiate find and algo.
    ## The find class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    find = opt.opt_find()
     
    #%% Variable parameters
    ## Set the R0 particle radius and 
    ## whether to calculate the initial conditions from experimental data
    ## 0. Use only 2D Data or 1D+2D
    multi_flag = conf.config['multi_flag']
    opt_params = conf.config['opt_params']
    
    find.init_opt_algo(multi_flag, algo_params, opt_params)
    
    find.algo.set_init_pop_para(pop_params)
    
    ## 1. The diameter ratio of the primary particles can also be used as a variable
    find.algo.set_comp_para(R_NM=conf.config['R_NM'], R_M=conf.config['R_M'],R01_0_scl=1e-1,R03_0_scl=1e-1)
    ## 5. Weight of 2D data
    ## The error of 2d pop may be more important, so weight needs to be added
    find.algo.weight_2d = conf.config['weight_2d']
    
    ## 6. Method how to use the datasets, kernels or delta
    ## kernels: Find the kernel for each set of data, and then average these kernels.
    ## delta: Read all input directly and use all data to find the kernel once
    ## wait to write hier 
    data_name = f"Sim_{find.algo.noise_type}_{find.algo.noise_strength}_para_10.0_0.5_0.5_0.5_2_1e-06_0.1_1e-06_0.1_1e-06_1.0.xlsx"
        
    base_path = os.path.join(find.algo.p.pth, "data")
    
    # conf_params = {
    #     'pop_params':{
    #         'CORR_BETA' : 15,
    #         'alpha_prim' : np.array([0.2, 0.6, 0.8])
    #         }
    #     }
    # pop_params = conf_params['pop_params']
    exp_data_path = os.path.join(base_path, data_name)
    exp_data_paths = [
        exp_data_path,
        exp_data_path.replace(".xlsx", "_NM.xlsx"),
        exp_data_path.replace(".xlsx", "_M.xlsx")
    ]
    
    # find.algo.calc_init_N = False
    # pth = os.path.dirname( __file__ )
    # dist_path_1 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_1'])
    # find.algo.set_comp_para('r0_001', 'r0_001', dist_path_1, dist_path_1)
    # find.generate_data(pop_params, find.algo.sample_num, add_info='_para_15.0_0.2_0.6_0.8_1')
    
    # Run an optimization and generate graphs of the results
    delta_opt, opt_values = normal_test()
    fig_mix = find.visualize_distribution(find.algo.p, pop_params, 
                                opt_values, exp_data_path=None,log_output=True)
    fig_NM = find.visualize_distribution(find.algo.p_NM, pop_params,  
                                opt_values, exp_data_path=None,log_output=True)
    fig_M = find.visualize_distribution(find.algo.p_M, pop_params, 
                                opt_values, exp_data_path=None,log_output=True)
    find.save_as_png(fig_mix, "PSD")
    find.save_as_png(fig_NM, "PSD-NM")
    find.save_as_png(fig_M, "PSD-M")
    
        
    # N_exp, N_calc, N_exp_1D, N_calc_1D, q3_psd, q3_exp = calc_N_test()
    
    # delta = calc_delta_test()
