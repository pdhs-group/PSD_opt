# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:40:35 2024

@author: Administrator
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import numpy as np
from generate_psd import full_psd
import opt_method as opt
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt   

def visualize_distribution_smoothing(Opt, pop, x_uni, q3, Q3, ax=None,fig=None,
                           close_all=False,clr='k',scl_a4=1,figsze=[12.8,6.4*1.5]):
    x_uni *= 1e6   
    ## smoothing the results
    kde = Opt.k.KDE_fit(x_uni, q3, bandwidth='scott', kernel_func='epanechnikov')
    sumN_uni = Opt.k.KDE_score(kde, x_uni)
    _, q3_sm, Q3_sm, _, _,_ = Opt.k.re_cal_distribution(x_uni, sumN_uni)
    
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
    
    
if __name__ == '__main__':
    ## Input for Opt
    dim = 2
    corr_beta = 10
    alpha_prim = np.array([0.8, 0.5, 0.2])
    t_vec = np.arange(1, 602, 60, dtype=float)
    delta_flag = 1
    add_noise = True
    smoothing = True
    noise_type='Multiplicative'
    noise_strength = 0.1
    sample_num = 5
    multi_flag = True
    ## Instantiate Opt
    Opt = opt.opt_method(add_noise, smoothing, dim, delta_flag, noise_type, 
                         noise_strength, t_vec, multi_flag)
    Opt.k.corr_beta = corr_beta
    Opt.k.alpha_prim = alpha_prim
    ## Input for generating psd-data
    x50 = 1   # /mm
    resigma = 1
    minscale = 1e-3
    maxscale = 1e3
    dist_path = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
    # psd_dict = np.load(dist_path,allow_pickle=True).item()
    
    ## Reinitialization of pop equations using psd data  
    dist_path_NM = dist_path
    dist_path_M = dist_path
    R01_0 = 'r0_005'
    R03_0 = 'r0_005'
    Opt.k.set_comp_para(R01_0, R03_0, dist_path_NM, dist_path_M)
    Opt.k.cal_all_pop(corr_beta, alpha_prim)
    
    x_uni, q3, Q3, _, _, _ = Opt.k.p.return_num_distribution_fixed(t=len(t_vec)-1)
    x_uni_NM, q3_NM, Q3_NM, _, _, _ = Opt.k.p_NM.return_num_distribution_fixed(t=len(t_vec)-1)
    x_uni_M, q3_M, Q3_M, _, _, _ = Opt.k.p_M.return_num_distribution_fixed(t=len(t_vec)-1)
    
    visualize_distribution_smoothing(Opt, Opt.k.p, x_uni, q3, Q3)
    visualize_distribution_smoothing(Opt, Opt.k.p_NM, x_uni_NM, q3_NM, Q3_NM)
    visualize_distribution_smoothing(Opt, Opt.k.p_M, x_uni_M, q3_M, Q3_M)
