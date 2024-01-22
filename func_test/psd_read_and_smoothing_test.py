# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:40:35 2024

@author: Administrator
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import numpy as np
from general_scripts.generate_psd import full_psd
import opt_find as opt
import opt_config as conf
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt   

def visualize_distribution_smoothing(Opt, pop, x_uni, q3, Q3, ax=None,fig=None,
                           close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    x_uni *= 1e6
    ## smoothing the results
    kde = find.algo.KDE_fit(x_uni, q3, bandwidth='scott', kernel_func='epanechnikov')
    sumN_uni = find.algo.KDE_score(kde, x_uni)
    _, q3_sm, Q3_sm, _, _,_ = find.algo.re_cal_distribution(x_uni, sumN_uni)
    
    pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    axq3=fig.add_subplot(1,2,1)   
    axQ3=fig.add_subplot(1,2,2) 
    
    # axq3, fig = pt.plot_data(x_uni, q3/np.max(q3), fig=fig, ax=axq3,
    #                        xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                        ylbl='number distribution of agglomerates $q3$ / $-$',
    #                        lbl='q3',clr='b',mrk='o')
    # axq3, fig = pt.plot_data(x_uni, q3_sm/np.max(q3_sm), fig=fig, ax=axq3,
    #                        lbl='q3_sm',clr='r',mrk='v')
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                            xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                            ylbl='number distribution of agglomerates $q3$ / $-$',
                            lbl='q3',clr='b',mrk='o')
    axq3, fig = pt.plot_data(x_uni, q3_sm, fig=fig, ax=axq3,
                            lbl='q3_sm',clr='r',mrk='v')
    
    axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='accumulated number distribution of agglomerates $Q3$ / $-$',
                           lbl='Q3',clr='b',mrk='o')
    axQ3, fig = pt.plot_data(x_uni, Q3_sm, fig=fig, ax=axQ3,
                           lbl='Q3_sm',clr='r',mrk='v')

    axq3.grid('minor')
    axQ3.grid('minor')
    plt.tight_layout() 
    
def visualize_distribution_smoothing_v(Opt, pop, v_uni, q3, Q3, ax=None,fig=None,
                           close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    v_uni *= 1e18
    ## smoothing the results
    kde = find.algo.KDE_fit(np.log10(v_uni), q3, bandwidth='scott', kernel_func='epanechnikov')
    sumN_uni = find.algo.KDE_score(kde, np.log10(v_uni))
    _, q3_sm, Q3_sm, _, _,_ = find.algo.re_cal_distribution(v_uni, sumN_uni)
    
    pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    axq3=fig.add_subplot(1,2,1)   
    axQ3=fig.add_subplot(1,2,2) 
    
    # axq3, fig = pt.plot_data(x_uni, q3/np.max(q3), fig=fig, ax=axq3,
    #                        xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                        ylbl='number distribution of agglomerates $q3$ / $-$',
    #                        lbl='q3',clr='b',mrk='o')
    # axq3, fig = pt.plot_data(x_uni, q3_sm/np.max(q3_sm), fig=fig, ax=axq3,
    #                        lbl='q3_sm',clr='r',mrk='v')
    
    axq3, fig = pt.plot_data(np.log10(v_uni), q3, fig=fig, ax=axq3,
                            xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                            ylbl='number distribution of agglomerates $q3$ / $-$',
                            lbl='q3',clr='b',mrk='o')
    axq3, fig = pt.plot_data(np.log10(v_uni), q3_sm, fig=fig, ax=axq3,
                            lbl='q3_sm',clr='r',mrk='v')
    
    axQ3, fig = pt.plot_data(np.log10(v_uni), Q3, fig=fig, ax=axQ3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='accumulated number distribution of agglomerates $Q3$ / $-$',
                           lbl='Q3',clr='b',mrk='o')
    axQ3, fig = pt.plot_data(np.log10(v_uni), Q3_sm, fig=fig, ax=axQ3,
                           lbl='Q3_sm',clr='r',mrk='v')

    axq3.grid('minor')
    axQ3.grid('minor')
    plt.tight_layout()    
    
if __name__ == '__main__':
    #%%  Input for Opt
    dim = 2
    t_vec = np.concatenate(([0.0, 0.1, 0.3, 0.6, 0.9], np.arange(1, 602, 60, dtype=float)))
    add_noise = False
    smoothing = True
    noise_type='Mul'
    noise_strength = 0.1
    sample_num = 5
    
    ## Instantiate find and algo.
    ## The find class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    find = opt.opt_find()
     
    #%% Variable parameters
    ## Set the R0 particle radius and 
    ## whether to calculate the initial conditions from experimental data
    ## 0. Use only 2D Data or 1D+2D
    find.multi_flag = True
    find.init_opt_algo(dim, t_vec, add_noise, noise_type, noise_strength, smoothing)
    ## Iteration steps for optimierer
    find.algo.n_iter = 800
    
    ## 1. The diameter ratio of the primary particles can also be used as a variable
    find.algo.calc_init_N = True
    find.algo.set_comp_para(R_NM=2.9e-7, R_M=2.9e-7)
    
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
   
    # ## Input for generating psd-data
    # x50 = 1   # /mm
    # resigma = 1
    # minscale = 1e-3
    # maxscale = 1e3
    # dist_path = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
    # # psd_dict = np.load(dist_path,allow_pickle=True).item()
    
    ## Reinitialization of pop equations using psd data  
    find.algo.calc_init_N = False
    pth = os.path.dirname( __file__ )
    dist_path_1 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_1'])
    find.algo.set_comp_para('r0_001', 'r0_001', dist_path_1, dist_path_1)
    find.algo.corr_beta = 15
    find.algo.alpha_prim = np.array([0.2, 0.6, 0.8])
    ## Calculate PBE direkt with psd-data, result is raw exp-data
    find.algo.cal_all_pop(find.algo.corr_beta, find.algo.alpha_prim)
    
    x_uni, q3, Q3, _, _, _ = find.algo.p.return_num_distribution(t=-1)
    x_uni_NM, q3_NM, Q3_NM, _, _, _ = find.algo.p_NM.return_num_distribution(t=-1)
    x_uni_M, q3_M, Q3_M, _, _, _ = find.algo.p_M.return_num_distribution(t=-1)
    v_uni = find.algo.cal_v_uni(find.algo.p)
    v_uni_NM = find.algo.cal_v_uni(find.algo.p_NM)
    v_uni_M = find.algo.cal_v_uni(find.algo.p_M)
    
    visualize_distribution_smoothing(find, find.algo.p, x_uni, q3, Q3)
    visualize_distribution_smoothing(find, find.algo.p_NM, x_uni_NM, q3_NM, Q3_NM)
    visualize_distribution_smoothing(find, find.algo.p_M, x_uni_M, q3_M, Q3_M)
    # visualize_distribution_smoothing_v(find, find.algo.p, v_uni, q3, Q3)
    # visualize_distribution_smoothing_v(find, find.algo.p_NM, v_uni_NM, q3_NM, Q3_NM)
    # visualize_distribution_smoothing_v(find, find.algo.p_M, v_uni_M, q3_M, Q3_M)
