# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:03:49 2024

@author: px2030
"""

import sys
import os
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import opt_method as opt

import matplotlib.pyplot as plt
import plotter.plotter as pt

def return_results(Opt, pop):
    x_uni, q3, Q3, _, x_50, _ = pop.return_num_distribution_fixed(t=len(pop.t_vec)-1)
    # Conversion unit
    x_uni *= 1e6     
    x_50 *= 1e6    
    if Opt.k.smoothing:
        kde = Opt.k.KDE_fit(x_uni, q3)
        sumN_uni = Opt.k.KDE_score(kde, x_uni)
        _, q3, Q3, _, _,_ = Opt.k.re_cal_distribution(x_uni, sumN_uni)
    return x_uni, q3, Q3, x_50
    
def cal_pop(Opt, corr_beta, alpha_prim, x3x1):
    Opt.k.set_comp_para(R_NM=2.9e-7, R_M=2.9e-7*x3x1)
    Opt.k.cal_all_pop(corr_beta, alpha_prim)
    
    x_uni, q3, Q3, x_50 = return_results(Opt, Opt.k.p)
    x_uni_NM, q3_NM, Q3_NM, x_50_NM = return_results(Opt, Opt.k.p_NM)
    x_uni_M, q3_M, Q3_M, x_50_M = return_results(Opt, Opt.k.p_M)
    
    return x_uni, q3, Q3, x_50, x_uni_NM, q3_NM, Q3_NM, x_50_NM, x_uni_M, q3_M, Q3_M, x_50_M
    
def depend_test(Opt, test_case):
    corr_beta = 10
    alpha_prim = np.array([0.5, 0.5, 0.5])
    x3x1 = 1
    x_50=x_50_NM=x_50_M=np.zeros(11)
    q3 = []
    Q3 = []
    q3_NM = []
    Q3_NM = []
    q3_M = []
    Q3_M = []
    if test_case == 'corr_beta':
        corr_beta = np.zeros(11)
        for i in range(11):
            corr_beta[i] = 2 * i + 0.1
            x_uni, q3_i, Q3_i, x_50[i], \
            x_uni_NM, q3_NM_i, Q3_NM_i, x_50_NM[i], \
            x_uni_M, q3_M_i, Q3_M_i, x_50_M[i] = cal_pop(Opt, corr_beta[i], alpha_prim, x3x1)
            q3.append(q3_i)
            Q3.append(Q3_i)
            q3_NM.append(q3_NM_i)
            Q3_NM.append(Q3_NM_i)
            q3_M.append(q3_M_i)
            Q3_M.append(Q3_M_i)
            
    elif test_case == 'alpha_prim_1':
        alpha_prim_tem = np.zeros(11)
        for i in range(11):
            alpha_prim_tem[i] = 0.1 * i
            alpha_prim = np.array([alpha_prim_tem[i], 0.5, 0.5])
            x_uni, q3_i, Q3_i, x_50[i], \
            x_uni_NM, q3_NM_i, Q3_NM_i, x_50_NM[i], \
            x_uni_M, q3_M_i, Q3_M_i, x_50_M[i] = cal_pop(Opt, corr_beta, alpha_prim, x3x1)
            q3.append(q3_i)
            Q3.append(Q3_i)
            q3_NM.append(q3_NM_i)
            Q3_NM.append(Q3_NM_i)
            q3_M.append(q3_M_i)
            Q3_M.append(Q3_M_i)
            
    elif test_case == 'alpha_prim_2':
        alpha_prim_tem = np.zeros(11)
        for i in range(11):
            alpha_prim_tem[i] = 0.1 * i
            alpha_prim = np.array([0.5, alpha_prim_tem[i], 0.5])
            x_uni, q3_i, Q3_i, x_50[i], \
            x_uni_NM, q3_NM_i, Q3_NM_i, x_50_NM[i], \
            x_uni_M, q3_M_i, Q3_M_i, x_50_M[i] = cal_pop(Opt, corr_beta, alpha_prim, x3x1)
            q3.append(q3_i)
            Q3.append(Q3_i)
            q3_NM.append(q3_NM_i)
            Q3_NM.append(Q3_NM_i)
            q3_M.append(q3_M_i)
            Q3_M.append(Q3_M_i)
            
    elif test_case == 'alpha_prim_3':
        alpha_prim_tem = np.zeros(11)
        for i in range(11):
            alpha_prim_tem[i] = 0.1 * i
            alpha_prim = np.array([0.5, 0.5, alpha_prim_tem[i]])
            x_uni, q3_i, Q3_i, x_50[i], \
            x_uni_NM, q3_NM_i, Q3_NM_i, x_50_NM[i], \
            x_uni_M, q3_M_i, Q3_M_i, x_50_M[i] = cal_pop(Opt, corr_beta, alpha_prim, x3x1)
            q3.append(q3_i)
            Q3.append(Q3_i)
            q3_NM.append(q3_NM_i)
            Q3_NM.append(Q3_NM_i)
            q3_M.append(q3_M_i)
            Q3_M.append(Q3_M_i)
            
    elif test_case == 'x3x1':
        x3x1 = np.zeros(11)
        for i in range(11):
            x3x1[i] = 0.5*i + 1
            x_uni, q3_i, Q3_i, x_50[i], \
            x_uni_NM, q3_NM_i, Q3_NM_i, x_50_NM[i], \
            x_uni_M, q3_M_i, Q3_M_i, x_50_M[i] = cal_pop(Opt, corr_beta, alpha_prim, x3x1[i])
            q3.append(q3_i)
            Q3.append(Q3_i)
            q3_NM.append(q3_NM_i)
            Q3_NM.append(Q3_NM_i)
            q3_M.append(q3_M_i)
            Q3_M.append(Q3_M_i)
    
    pt.plot_init(scl_a4=1,figsze=[6.4*2.5,6.4*1.5],lnewdth=0.8,mrksze=0,use_locale=True,
                 scl=2,fontsize=5,labelfontsize=4.5,tickfontsize=4)
    fig=plt.figure()    
    fig_NM=plt.figure()   
    fig_M=plt.figure()   
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
    if test_case == 'corr_beta':
        varlabel = corr_beta
    elif test_case == 'x3x1':
        varlabel = x3x1
    else:
        varlabel = alpha_prim_tem
    
    file_name = "test_data\\" + test_case
    pth = os.path.dirname( __file__ )
    file_path = os.path.join(pth, file_name)
    file_path_NM = os.path.join(pth, file_name+'_NM')
    file_path_M = os.path.join(pth, file_name+'_M')
    
    visualize_results(file_path, fig, test_case, varlabel, colors, x_uni, q3, Q3, x3x1, x_50)
    visualize_results(file_path_NM,fig_NM, test_case, varlabel, colors, x_uni_NM, q3_NM, Q3_NM, x3x1, x_50_NM)
    visualize_results(file_path_M, fig_M, test_case, varlabel, colors, x_uni_M, q3_M, Q3_M, x3x1, x_50_M)   


def visualize_results(path, fig, test_case, varlabel, colors, x_uni, q3, Q3, x3x1, x_50):
    axq3=fig.add_subplot(2,2,1)   
    axQ3=fig.add_subplot(2,2,2)
    axq3, fig = pt.plot_data(x_uni, q3[0], fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl=test_case+f'={varlabel[0]:.1f}',clr=colors[0])
    axQ3, fig = pt.plot_data(x_uni, Q3[0], fig=fig, ax=axQ3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='accumulated number distribution of agglomerates $Q3$ / $-$',
                           lbl=test_case+f'={varlabel[0]:.1f}',clr=colors[0])
    
    for i in range(1,11):
        axq3, fig = pt.plot_data(x_uni, q3[i], fig=fig, ax=axq3,
                               lbl=test_case+f'={varlabel[i]:.1f}',clr=colors[i])
        axQ3, fig = pt.plot_data(x_uni, Q3[i], fig=fig, ax=axQ3,
                               lbl=test_case+f'={varlabel[i]:.1f}',clr=colors[i])
        
    axx_50=fig.add_subplot(2,1,2) 
    axx_50, fig = pt.plot_data(varlabel, x_50, fig=fig, ax=axx_50,
                           xlbl=test_case+' / $-$',
                           ylbl='Median particle size $x_50$ / $-$',
                           lbl='$x_50$',clr=colors[0]) 
    axq3.grid('minor')
    axQ3.grid('minor')
    # plt.tight_layout()  
    fig.savefig(path, dpi=300)
            
if __name__ == '__main__':
     dim = 2
     corr_beta = 10
     alpha_prim = np.array([0.8, 0.5, 0.2])
     t_vec = np.arange(1, 602, 60, dtype=float)

     delta_flag = 1
     add_noise = True
     smoothing = True
     noise_type='Multiplicative'
     noise_strength = 0.1
     multi_flag = True
     
     Opt = opt.opt_method(add_noise, smoothing, corr_beta, alpha_prim, dim,
                         delta_flag, noise_type, noise_strength, t_vec, multi_flag)

     Opt.k.set_comp_para(R_NM=2.9e-7, R_M=2.9e-7)
     
     test_case = 'corr_beta'
     # test_case = 'alpha_prim_1'
     # test_case = 'alpha_prim_2'
     # test_case = 'alpha_prim_3'
     # test_case = 'x3x1' # not implement
     depend_test(Opt, test_case)

     