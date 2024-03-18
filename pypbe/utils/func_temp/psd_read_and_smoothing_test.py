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

def visualize_distribution_smoothing(Opt, pop, x_uni, q3, Q3, sumvol_uni, ax=None,fig=None,
                           close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    v_uni = find.algo.calc_v_uni(pop)
    sumN_uni_sm = np.zeros(len(sumvol_uni))
    ## smoothing the results
    kde = find.algo.KDE_fit(x_uni, sumvol_uni, bandwidth='scott', kernel_func='epanechnikov')
    q3_sm = find.algo.KDE_score(kde, x_uni)
    Q3_sm = find.algo.calc_Q3(x_uni,q3_sm)
    sumvol_uni_sm = find.algo.calc_sum_uni(Q3_sm, sumvol_uni.sum())
    sumN_uni_sm[1:] = sumvol_uni_sm[1:] / v_uni
    
    noised_data = np.random.normal(1, 0.2, len(sumvol_uni))
    sumN_uni_noise = sumN_uni_sm * noised_data
    
    pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    axq3=fig.add_subplot(1,2,1)   
    axQ3=fig.add_subplot(1,2,2) 
    
    # axq3, fig = pt.plot_data(x_uni, q3/np.max(q3), fig=fig, ax=axq3,
    #                         xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                         ylbl='number distribution of agglomerates $q3$ / $-$',
    #                         lbl='q3',clr='b',mrk='o')
    # axq3, fig = pt.plot_data(x_uni, q3_sm/np.max(q3_sm), fig=fig, ax=axq3,
    #                         lbl='q3_sm',clr='r',mrk='v')
    
    # axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
    #                         xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
    #                         ylbl='number distribution of agglomerates $q3$ / $-$',
    #                         lbl='q3',clr='b',mrk='o')
    # axq3, fig = pt.plot_data(x_uni, q3_sm, fig=fig, ax=axq3,
    #                         lbl='q3_sm',clr='r',mrk='v')
    
    axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                            xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                            ylbl='accumulated number distribution of agglomerates $Q3$ / $-$',
                            lbl='Q3',clr='b',mrk='o')
    axQ3, fig = pt.plot_data(x_uni, Q3_sm, fig=fig, ax=axQ3,
                            lbl='Q3_sm',clr='r',mrk='v')
    
    axq3, fig = pt.plot_data(x_uni, sumN_uni_sm, fig=fig, ax=axq3,
                            xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                            ylbl='number density of agglomerates $q3$ / $-$',
                            lbl='data_origin',clr='r',mrk='o')
    axq3, fig = pt.plot_data(x_uni, sumN_uni_noise, fig=fig, ax=axq3,
                            lbl='data_noised',clr='k',mrk='v')

    axq3.grid('minor')
    axq3.set_xscale('log')
    axq3.set_yscale('log')
    axQ3.grid('minor')
    axQ3.set_xscale('log')
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
    axq3.set_xscale('log')
    axQ3.grid('minor')
    axQ3.set_xscale('log')
    plt.tight_layout()    
    
if __name__ == '__main__':
    #%%  Input for Opt
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    
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
    find.algo.set_comp_para(R_NM=1e-6, R_M=1e-6)
    
    delta_flag_target = ['','q3','Q3','x_10','x_50','x_90']
    
    ## 5. Weight of 2D data
    ## The error of 2d pop may be more important, so weight needs to be added
    find.algo.weight_2d = 1
    
    ## 6. Method how to use the datasets, kernels or delta
    ## kernels: Find the kernel for each set of data, and then average these kernels.
    ## delta: Read all input directly and use all data to find the kernel once
    ## wait to write hier 
   
    ## Input for generating psd-data
    # x50 = 1   # /um
    # resigma = 1.5
    # minscale = 1e-3
    # maxscale = 1e3
    # dist_path = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=True)
    # psd_dict = np.load(dist_path,allow_pickle=True).item()
    
    ## Reinitialization of pop equations using psd data  
    find.algo.calc_init_N = False
    pth = os.path.dirname( __file__ )
    # dist_path_1 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_1'])
    # dist_path_1 = dist_path
    # find.algo.set_comp_para('r0_001', 'r0_001', dist_path_1, dist_path_1,R01_0_scl=1,R03_0_scl=1)
    # find.algo.corr_beta = 150
    # find.algo.alpha_prim = np.array([1, 1, 1])
    
    ## Calculate PBE direkt with psd-data, result is raw exp-data
    find.algo.calc_all_pop()
    # find.algo.calc_pop(find.algo.p)      
    
    # ## Test the influence of Total number concentration to q3
    # find.algo.p_M.V01 *= 10
    # find.algo.cal_pop(find.algo.p_M, find.algo.corr_beta, find.algo.alpha_prim[0])
    
    x_uni, q3, Q3, sumvol_uni = find.algo.p.return_distribution(t=-1, flag='x_uni, q3, Q3, sumvol_uni')
    v_uni = find.algo.calc_v_uni(find.algo.p)
    visualize_distribution_smoothing(find, find.algo.p, x_uni, q3, Q3, sumvol_uni)
    if find.multi_flag:
        x_uni_NM, q3_NM, Q3_NM, sumvol_uni_NM = find.algo.p_NM.return_distribution(t=-1, flag='x_uni, q3, Q3, sumvol_uni')
        v_uni_NM = find.algo.calc_v_uni(find.algo.p_NM)
        visualize_distribution_smoothing(find, find.algo.p_NM, x_uni_NM, q3_NM, Q3_NM, sumvol_uni_NM)
    if find.multi_flag:    
        x_uni_M, q3_M, Q3_M, sumvol_uni_M = find.algo.p_M.return_distribution(t=-1, flag='x_uni, q3, Q3, sumvol_uni')
        v_uni_M = find.algo.calc_v_uni(find.algo.p_M)
        visualize_distribution_smoothing(find, find.algo.p_M, x_uni_M, q3_M, Q3_M, sumvol_uni_M)
    
    # visualize_distribution_smoothing_v(find, find.algo.p, v_uni, q3, Q3)
    # visualize_distribution_smoothing_v(find, find.algo.p_NM, v_uni_NM, q3_NM, Q3_NM)
    # visualize_distribution_smoothing_v(find, find.algo.p_M, v_uni_M, q3_M, Q3_M)
