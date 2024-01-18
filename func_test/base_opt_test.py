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

def distribution_test(ax=None,fig=None,close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    pop = find.algo.p
    pop.calc_R()
    pop.init_N()
    find.algo.cal_pop(pop, find.algo.corr_beta, find.algo.alpha_prim)
    x_uni = find.algo.cal_x_uni(pop)
    v_uni = find.algo.cal_v_uni(pop)
    idt=len(pop.t_vec)-1
    q3 = pop.return_distribution(t=idt, flag='q3')[0]
    kde = find.algo.KDE_fit(x_uni, q3)
    sumV_uni = find.algo.KDE_score(kde, x_uni)
    _, q3, _, _, _,_ = find.algo.re_cal_distribution(x_uni, sumV_uni)
    sumvol = np.sum(v_uni * sumV_uni)
    sumN_uni = sumV_uni * sumvol / v_uni
    _, q3_num, _, _, _,_ = find.algo.re_cal_distribution(x_uni, sumN_uni)
    
    q3_num_pop = pop.return_num_distribution(t=idt, flag='q3')[0]
    kde = find.algo.KDE_fit(x_uni, q3_num_pop)
    sumN_uni = find.algo.KDE_score(kde, x_uni)
    _, q3_num_pop, _, _, _,_ = find.algo.re_cal_distribution(x_uni, sumN_uni)
    
    q3_re = find.algo.convert_dist_num_to_vol(x_uni, q3_num)
    
    plt.close('all')
    fig=plt.figure()    
    axq3=fig.add_subplot(1,1,1)   
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl='q3',clr='b',mrk='o')
    
    axq3, fig = pt.plot_data(x_uni, q3_num, fig=fig, ax=axq3,
                           lbl='q3_num',clr='g',mrk='^')
    
    axq3, fig = pt.plot_data(x_uni, q3_num_pop, fig=fig, ax=axq3,
                           lbl='q3_num_pop',clr='k',mrk='^')
    
    axq3, fig = pt.plot_data(x_uni, q3_re[:, idt], fig=fig, ax=axq3,
                           lbl='q3_re',clr='r',mrk='o')
    
    return q3, q3_num, q3_re[:, idt]
    
if __name__ == '__main__':
    #%%  Input for Opt
    dim = 2
    t_vec = np.concatenate(([0.0, 0.1, 0.3, 0.6, 0.9], np.arange(1, 602, 60, dtype=float)))
    add_noise = True
    smoothing = True
    noise_type='Mul'
    noise_strength = 0.1
    sample_num = 5
    
    ## Instantiate find and algo.
    ## The find class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    find = opt.opt_find()
    find.init_opt_algo(dim, t_vec, add_noise, noise_type, noise_strength, smoothing)
     
    
    # Iteration steps for optimierer
    find.algo.n_iter = 10
    
    #%% Variable parameters
    ## Set the R0 particle radius and 
    ## whether to calculate the initial conditions from experimental data
    ## 0. The diameter ratio of the primary particles can also be used as a variable
    find.algo.calc_init_N = True
    find.algo.set_comp_para(R_NM=2.9e-7, R_M=2.9e-7)
    
    ## 1. Use only 2D Data or 1D+2D
    multi_flag = True
    
    ## 2. Criteria of optimization target
    ## delta_flag = 1: use q3
    ## delta_flag = 2: use Q3
    ## delta_flag = 3: use x_10
    ## delta_flag = 4: use x_50
    ## delta_flag = 5: use x_90
    find.algo.delta_flag = 1
    delta_flag_target = ['','q3','Q3','x_10','x_50','x_90']
    
    ## 3. Optimize method: 
    ##   'BO': Bayesian Optimization with package BayesianOptimization
    find.method='BO'
    
    ## 4. Type of cost function to use
    ##   'MSE': Mean Squared Error
    ##   'RMSE': Root Mean Squared Error
    ##   'MAE': Mean Absolute Error
    ##   'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
    find.algo.cost_func_type = 'KL'
    
    ## 5. Weight of 2D data
    ## The error of 2d pop may be more important, so weight needs to be added
    find.algo.weight_2d = 1
    
    ## 6. Method how to use the datasets, kernels or delta
    ## kernels: Find the kernel for each set of data, and then average these kernels.
    ## delta: Read all input directly and use all data to find the kernel once
    ## wait to write hier 
    if add_noise:
        data_name = f"Sim_{noise_type}_{noise_strength}_para_10.0_1.0_1.0_0.5_1.xlsx"
    else:
        data_name = "Sim_para_10_1.0_1.0_0.5_1.xlsx"
    
    find.algo.corr_beta = 10
    find.algo.alpha_prim = np.array([1.0, 1.0, 0.5])
    
    # corr_beta_opt, alpha_prim_opt, para_diff, delta_opt, elapsed_time,corr_agg, \
    #     corr_agg_opt, corr_agg_diff = normal_test()
    
    q3, q3_num, q3_re = distribution_test()