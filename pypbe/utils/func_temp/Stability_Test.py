# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:14:44 2024

@author: px2030
"""

import sys
import os
import numpy as np
import multiprocessing
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
from pypbe.utils.general_scripts.generate_psd import full_psd
import pypbe.kernel_opt.opt_find as opt
import config.opt_config as conf
## For plots
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt 

def calc_function(var_pop_params):
    #%%  Input for Opt 
    find = opt.opt_find()
    
    find.init_opt_algo(multi_flag, algo_params, opt_params)
    
    find.algo.set_init_pop_para(pop_params)
    
    find.algo.set_comp_para(R_NM=conf.config['R_NM'], R_M=conf.config['R_M'])
    
    find.algo.weight_2d = conf.config['weight_2d']
    
    find.algo.calc_all_pop(var_pop_params, find.algo.t_vec)
    # calc_status = find.algo.p.calc_status
    # calc_NM_status = find.algo.p_NM.calc_status
    # calc_M_status = find.algo.p_M.calc_status
    t_2d = find.algo.p.t_res_tem[-1]/t_end
    t_1d = find.algo.p_NM.t_res_tem[-1]/t_end

    return t_2d,t_1d

def plot_res_1d(scl_a4=1,figsze=[12.8,6.4*1.5]):
    pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    ax1d=fig.add_subplot(1,2,1) 
    ax2d=fig.add_subplot(1,2,2) 
    ax1d, fig = pt.plot_data(var_v, res_t_1d, fig=fig, ax=ax1d,
                            xlbl='$v_\mathrm{breakage}$ / $-$',
                            lbl='stability',clr='b',mrk='o')
    
    ax2d, fig = pt.plot_data(var_v, res_t_2d, fig=fig, ax=ax2d,
                            xlbl='$v_\mathrm{breakage}$ / $-$',
                            lbl='stability',clr='r',mrk='o')
    
    ax1d.grid('minor')
    ax2d.grid('minor')
    plt.tight_layout()    
    
def plot_res_2d():
    P1, P2 = np.meshgrid(var_P1, var_P2, indexing='ij')
    
    
if __name__ == '__main__':
         
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    multi_flag = conf.config['multi_flag']
    opt_params = conf.config['opt_params']
    
    t_end = algo_params['t_vec'][-1]
    generate_new_psd = False
    
    if generate_new_psd:
        ## Input for generating psd-data
        x50 = 2   # /mm
        resigma = 0.15
        minscale = 0.5
        maxscale = 2
        dist_path_1 = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        dist_path_5 = full_psd(x50*5, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        dist_path_10 = full_psd(x50*10, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
    else:
        pth = os.path.dirname( __file__ )
        dist_path_1 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_1'])
        dist_path_5 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_5'])
        dist_path_10 = os.path.join(pth, "..","data", "PSD_data", conf.config['dist_scale_10'])

    var_corr_beta = np.array([1e-3, 1e0, 1e3])

    ## define the range of v(breakage function)
    var_v = np.arange(0.1, 2, 0.1, dtype=float)
    # var_v = np.array([0.01])
    ## define the range of P1, P2 for power law breakage rate
    var_P1 = np.array([1e-6, 1e-5,1e-4,1e-3,1e-2,1e-1, 1])
    var_P2 = np.array([1e-2,1e-1, 1, 10])
    var_P3 = np.array([1e-6, 1e-3, 1])
    var_P4 = np.array([1e-1, 1, 10])
    var_P5 = np.array([-1e-6,1e-6])
    var_P6 = np.array([1e-1])
    # var_P1 = np.array([1])
    # var_P2 = np.array([0.0])
    
    ## define the range of particle size scale and minimal size
    dist_path = [dist_path_1] # [dist_path_1, dist_path_10]
    size_scale = np.array([1, 10])
    R01_0 = 'r0_001'
    R03_0 = 'r0_001'
    res_t_1d = np.zeros((len(var_P5),len(var_P6)))
    res_t_2d = np.zeros((len(var_P5),len(var_P6)))
    # for j,corr_beta in enumerate(var_corr_beta):
    # for l,v in enumerate(var_v):
    # for m1,P1 in enumerate(var_P1):
    #     for m2,P2 in enumerate(var_P2):
    for m5,P5 in enumerate(var_P5):
        for m6,P6 in enumerate(var_P6):
                    # Set parameters for PBE
                    conf_params = {
                        'pop_params':{
                            # 'CORR_BETA' : corr_beta,
                            # 'pl_v' : v,
                            # 'pl_P1' : P1,
                            # 'pl_P2' : P2,
                            # 'pl_P3' : P1,
                            # 'pl_P4' : P2,
                            'pl_P5' : P5,
                            'pl_P6' : P6,
                            }
                        }
                    var_pop_params = conf_params['pop_params']
                    res_t_2d[m5,m6], res_t_1d[m5,m6] = calc_function(var_pop_params)
    # plot_res_1d()
    plot_res_2d()
