# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import opt_find as opt
import numpy as np
import pandas as pd
import opt_config as conf
import time
import matplotlib.pyplot as plt
import plotter.plotter as pt  

def normal_test():
    start_time = time.time()

    # corr_beta_opt, alpha_prim_opt, para_diff, delta_opt= \
    #     find.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    corr_beta_opt, alpha_prim_opt, para_diff, delta_opt, \
        corr_agg, corr_agg_opt, corr_agg_diff = \
        find.find_opt_kernels(sample_num=sample_num, method='delta', data_name=data_name)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")
        
    fig_mix = find.visualize_distribution(find.algo.p, find.algo.corr_beta, find.algo.alpha_prim, 
                                corr_beta_opt, alpha_prim_opt, exp_data_path=None)
    fig_NM = find.visualize_distribution(find.algo.p_NM, find.algo.corr_beta, find.algo.alpha_prim[0], 
                                corr_beta_opt, alpha_prim_opt[0], exp_data_path=None)
    fig_M = find.visualize_distribution(find.algo.p_M, find.algo.corr_beta, find.algo.alpha_prim[2], 
                                corr_beta_opt, alpha_prim_opt[2], exp_data_path=None)
    find.save_as_png(fig_mix, "PSD")
    find.save_as_png(fig_NM, "PSD-NM")
    find.save_as_png(fig_M, "PSD-M")
    
    return corr_beta_opt, alpha_prim_opt, para_diff, delta_opt, elapsed_time,corr_agg, corr_agg_opt, corr_agg_diff
    
def calc_N_test():
    find.algo.calc_init_N = False
    pth = os.path.dirname( __file__ )
    dist_path_1 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_1'])
    find.algo.set_comp_para('r0_001', 'r0_001', dist_path_1, dist_path_1)
    
    fig=plt.figure()    
    axq3=fig.add_subplot(1,1,1)
    fig_NM=plt.figure()    
    axq3_NM=fig_NM.add_subplot(1,1,1)
    fig_M=plt.figure()    
    axq3_M=fig_M.add_subplot(1,1,1)
    
    ## Calculate PBE direkt with psd-data, result is raw exp-data
    find.algo.calc_all_pop(find.algo.corr_beta, find.algo.alpha_prim, find.algo.t_all)
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
    find.algo.set_comp_para(R_NM=8.68e-7, R_M=8.68e-7)
    find.algo.set_init_N(sample_num, exp_data_paths, 'mean')
    find.algo.calc_all_pop(find.algo.corr_beta, find.algo.alpha_prim, find.algo.t_all)
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
    find.algo.set_init_N(sample_num, exp_data_paths, 'mean')
    corr_agg = find.algo.corr_beta * find.algo.alpha_prim
    delta = find.algo.calc_delta_agg(corr_agg, -1, sample_num, exp_data_paths)
    return delta

if __name__ == '__main__':
    #%%  Input for Opt
    dim = conf.config['dim']
    t_init = conf.config['t_init']
    t_vec = conf.config['t_vec']
    add_noise = conf.config['add_noise']
    smoothing = conf.config['smoothing']
    noise_type=conf.config['noise_type']
    noise_strength = conf.config['noise_strength']
    sample_num = conf.config['sample_num']
    
    ## Instantiate find and algo.
    ## The find class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    find = opt.opt_find()
     
    #%% Variable parameters
    ## Set the R0 particle radius and 
    ## whether to calculate the initial conditions from experimental data
    ## 0. Use only 2D Data or 1D+2D
    find.multi_flag = conf.config['multi_flag']
    find.init_opt_algo(dim, t_init, t_vec, add_noise, noise_type, noise_strength, smoothing)
    ## Iteration steps for optimierer
    find.algo.n_iter = conf.config['n_iter']
    
    ## 1. The diameter ratio of the primary particles can also be used as a variable
    find.algo.calc_init_N = conf.config['calc_init_N']
    find.algo.set_comp_para(R_NM=conf.config['R_NM'], R_M=conf.config['R_M'])
    
    ## 2. Criteria of optimization target
    ## delta_flag = q3: use q3
    ## delta_flag = Q3: use Q3
    ## delta_flag = x_10: use x_10
    ## delta_flag = x_50: use x_50
    ## delta_flag = x_90: use x_90
    find.algo.delta_flag = conf.config['delta_flag']
    
    ## 3. Optimize method: 
    ##   'BO': Bayesian Optimization with package BayesianOptimization
    find.algo.method= conf.config['method']
    
    ## 4. Type of cost function to use
    ##   'MSE': Mean Squared Error
    ##   'RMSE': Root Mean Squared Error
    ##   'MAE': Mean Absolute Error
    ##   'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
    find.algo.cost_func_type = conf.config['cost_func_type']
    
    ## 5. Weight of 2D data
    ## The error of 2d pop may be more important, so weight needs to be added
    find.algo.weight_2d = conf.config['weight_2d']
    
    ## 6. Method how to use the datasets, kernels or delta
    ## kernels: Find the kernel for each set of data, and then average these kernels.
    ## delta: Read all input directly and use all data to find the kernel once
    ## wait to write hier 
    if add_noise:
        data_name = f"Sim_{noise_type}_{noise_strength}_para_15.0_0.2_0.6_0.8_1.xlsx"
    else:
        data_name = "Sim_para_15.0_0.2_0.6_0.8_1.xlsx"
        
    base_path = os.path.join(find.algo.p.pth, "data")
    
    find.algo.corr_beta = 15
    find.algo.alpha_prim = np.array([0.2, 0.6, 0.8])
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
    # find.generate_data(sample_num, add_info='_para_15.0_0.2_0.6_0.8_1')
    
    corr_beta_opt, alpha_prim_opt, para_diff, delta_opt, elapsed_time,corr_agg, \
        corr_agg_opt, corr_agg_diff = normal_test()
        
    # N_exp, N_calc, N_exp_1D, N_calc_1D, q3_psd, q3_exp = calc_N_test()
    
    # delta = calc_delta_test()
