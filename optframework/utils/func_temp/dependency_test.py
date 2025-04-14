# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:03:49 2024

@author: px2030
"""

import sys, os
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
import pypbe.kernel_opt.opt_find as opt
import config.opt_config as conf
## For plots
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt 

def return_results(find, pop, t_frame):
    x_uni, q3, Q3, sumvol_uni = pop.return_distribution(t=t_frame, flag='x_uni, qx, Qx, sum_uni')
    # Conversion unit
    if find.algo.smoothing:
        kde = find.algo.KDE_fit(x_uni, sumvol_uni, bandwidth='scott', kernel_func='epanechnikov')    
        q3 = find.algo.KDE_score(kde, x_uni)
        Q3 = find.algo.calc_Q3(x_uni,q3)
    x_50 = np.interp(0.5, Q3, x_uni)
    
    return x_uni, q3, Q3, x_50
    
def return_all_results(paras, t_vec, t_frame):
    find.algo.calc_all_pop(paras, t_vec)
    
    x_uni, q3, Q3, x_50 = return_results(find, find.algo.p, t_frame)
    x_uni_NM, q3_NM, Q3_NM, x_50_NM = return_results(find, find.algo.p_NM, t_frame)
    x_uni_M, q3_M, Q3_M, x_50_M = return_results(find, find.algo.p_M, t_frame)
    
    return x_uni, q3, Q3, x_50, x_uni_NM, q3_NM, Q3_NM, x_50_NM, x_uni_M, q3_M, Q3_M, x_50_M
    
def depend_test(find, test_case, t_frame):
    q3 = []
    Q3 = []
    x_50 = []
    q3_NM = []
    Q3_NM = []
    x_50_NM = []
    q3_M = []
    Q3_M = []        
    x_50_M = []
    if test_case == 'pl_v':
        variable = np.arange(0.1, 2.1, 0.1, dtype=float)
    elif test_case == 'pl_P1' or test_case == 'pl_P3':
        variable_ex = np.arange(-8, -1, 1, dtype=float)
        variable = 10**variable_ex
    elif test_case == 'pl_P2' or test_case == 'pl_P4':
        variable = np.arange(0.01, 0.7, 0.05, dtype=float)
    elif test_case == 'CORR_BETA':
        variable = np.arange(1, 1002, 50, dtype=float)
    for var in variable:
        pop_params[test_case] = var
        x_uni, q3_tem, Q3_tem, x_50_tem, \
        x_uni_NM, q3_NM_tem, Q3_NM_tem, x_50_NM_tem, \
        x_uni_M, q3_M_tem, Q3_M_tem, x_50_M_tem = return_all_results(pop_params, find.algo.t_vec, t_frame)
        q3.append(q3_tem)
        Q3.append(Q3_tem)
        x_50.append(x_50_tem)
        q3_NM.append(q3_NM_tem)
        Q3_NM.append(Q3_NM_tem)
        x_50_NM.append(x_50_NM_tem)
        q3_M.append(q3_M_tem)
        Q3_M.append(Q3_M_tem)
        x_50_M.append(x_50_M_tem)

    pt.plot_init(scl_a4=1,figsze=[6.4*2.5,6.4*1.5],lnewdth=0.8,mrksze=0,use_locale=True,
                 scl=2,fontsize=5,labelfontsize=4.5,tickfontsize=4)  
    cmap = plt.cm.viridis  # or plt.cm.plasma, plt.cm.inferno, plt.cm.magma
    num_colors = len(variable)
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    
    output_dir = 'dependency_test'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, test_case)
    file_path_NM = os.path.join(output_dir, test_case+'_NM')
    file_path_M = os.path.join(output_dir, test_case+'_M')
    
    visualize_results(file_path, test_case, variable, colors, x_uni, q3, Q3, x_50)
    visualize_results(file_path_NM,test_case, variable, colors, x_uni_NM, q3_NM, Q3_NM, x_50_NM)
    visualize_results(file_path_M, test_case, variable, colors, x_uni_M, q3_M, Q3_M, x_50_M)   


def visualize_results(path, test_case, variable, colors, x_uni, q3, Q3, x_50):
    figq3, axq3 = plt.subplots()
    for i, q3_data in enumerate(q3):
        axq3, figq3 = pt.plot_data(x_uni, q3_data, fig=figq3, ax=axq3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='number density distribution of agglomerates $q3$ / $-$',
                               lbl=test_case+f'={variable[i]:.2e}',clr=colors[i])
    axq3.grid('minor')
    axq3.set_xscale('log')
    plt.tight_layout()
    figq3.savefig(f"{path}_q3.png", dpi=300)
    plt.close(figq3)
        
    figQ3, axQ3 = plt.subplots()
    for i, Q3_data in enumerate(Q3):
        axQ3, figQ3 = pt.plot_data(x_uni, Q3_data, fig=figQ3, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='number density distribution of agglomerates $q3$ / $-$',
                               lbl=test_case+f'={variable[i]:.2e}',clr=colors[i])
    axQ3.grid('minor')
    axQ3.set_xscale('log')
    plt.tight_layout()
    figQ3.savefig(f"{path}_QQ3.png", dpi=300)
    plt.close(figQ3)    
    
    figx_50, axx_50 = plt.subplots()
    axx_50, figx_50 = pt.plot_data(variable, x_50, fig=figx_50, ax=axx_50,
                           xlbl=test_case+' / $-$',
                           ylbl='Median particle size $x_50$ / $-$',
                           lbl='$x_50$',clr='red') 
    axx_50.grid('minor')
    plt.tight_layout()
    figx_50.savefig(f"{path}_x_50.png", dpi=300)
    plt.close(figx_50)        
if __name__ == '__main__':
    test_case = 'CORR_BETA'
    t_frame = -1
    #%%  Input for Opt
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    
    # pop_params['CORR_BETA'] = 1e2
    # pop_params['alpha_prim'] = np.array([0.5, 0.5, 0.5])
    # pop_params['pl_v'] = 2
    # pop_params['pl_P1'] = 1e-2
    # pop_params['pl_P2'] = 0.5
    # pop_params['pl_P3'] = 1e-2
    # pop_params['pl_P4'] = 0.5
    # pop_params['pl_P5'] = 1e-2
    # pop_params['pl_P6'] = 1e-1
    
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
    
    base_path = os.path.join(find.algo.p.pth, "data")
    if find.algo.p.process_type == 'breakage':
        USE_PSD = False
        dist_path_NM = None
        dist_path_M = None
    else:
        USE_PSD = True
        dist_path_NM = os.path.join(base_path, "PSD_data", conf.config['dist_scale_1'])
        dist_path_M = os.path.join(base_path, "PSD_data", conf.config['dist_scale_1'])
        
    R_NM = conf.config['R_NM']
    R_M=conf.config['R_M']
    R01_0_scl=conf.config['R01_0_scl']
    R03_0_scl=conf.config['R03_0_scl']
    find.algo.set_comp_para(USE_PSD, R_NM=R_NM, R_M=R_M,R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl,
                            dist_path_NM=dist_path_NM, dist_path_M=dist_path_M)
    find.algo.weight_2d = conf.config['weight_2d']
    
    depend_test(find, test_case, t_frame)

     