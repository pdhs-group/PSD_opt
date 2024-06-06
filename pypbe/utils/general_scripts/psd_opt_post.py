# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
import config.opt_config as conf
import pypbe.kernel_opt.opt_find as opt
import pypbe.kernel_opt.opt_algo as algo
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
## For plots
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt  
import itertools
from matplotlib.animation import FuncAnimation

epsilon = 1e-20

def read_results(data_paths):
    results = []
    for data_path in data_paths:
        data = np.load(data_path,allow_pickle=True)
        result=data['results']
        # For comparison, CORR_BETA and alpha_prim in the original parameters are merged into corr_agg
        ori_kernels = result[:,2]
        if 'CORR_BETA' in ori_kernels[0] and 'alpha_prim' in ori_kernels[0]:
            for ori_kernel in ori_kernels:
                ori_kernel['corr_agg'] = ori_kernel['CORR_BETA'] * ori_kernel['alpha_prim']
        results.append(result)
        data.close()
    return results

def calc_diff(result):
    # delta_opt = result[:,0]
    opt_kernels_tem = result[:,1]
    ori_kernels_tem = result[:,2]
    diff_kernels = {}
    opt_kernels = {}
    ori_kernels = {}
    
    for kernel in opt_kernels_tem[0]:
        tem_opt_kernel = np.array([dic[kernel] for dic in opt_kernels_tem])
        tem_ori_kernel = np.array([dic[kernel] for dic in ori_kernels_tem])
        if tem_opt_kernel.ndim != 1:
            for i in range(tem_opt_kernel.shape[1]):
                opt_kernels[f"{kernel}_{i}"] = tem_opt_kernel[:,i]
                ori_kernels[f"{kernel}_{i}"] = tem_ori_kernel[:,i]
                diff = abs(tem_opt_kernel[:,i] - tem_ori_kernel[:,i])
                rel_diff = np.where(tem_ori_kernel[:,i] != 0, diff / (tem_ori_kernel[:,i]+epsilon), diff)
                if use_rel_diff:
                    diff = rel_diff
                diff_kernels[f"{kernel}_{i}"] = diff
        else:
            ## Change the format of the dictionary 
            ## so that it remains in the same format as diff_kernels
            opt_kernels[kernel] = tem_opt_kernel
            ori_kernels[kernel] = tem_ori_kernel
            diff = abs(tem_opt_kernel - tem_ori_kernel)
            rel_diff = np.where(tem_ori_kernel != 0, diff / (tem_ori_kernel+epsilon), diff)
            if use_rel_diff:
                diff = rel_diff
            diff_kernels[kernel] = diff
    return diff_kernels, opt_kernels, ori_kernels 

def visualize_diff_mean(results, labels):
    num_results = len(results)
    diff_mean = np.zeros(num_results)
    diff_std = np.zeros(num_results)
    diff_var = np.zeros(num_results)
    
    for i, result in enumerate(results):
        diff_kernels, _, _ = calc_diff(result)
        all_elements = np.concatenate(list(diff_kernels.values()))
        diff_mean[i] = np.mean(all_elements)
        diff_std[i] = np.std(all_elements)
        diff_var[i] = np.var(all_elements)
    
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x_pos, diff_mean, yerr=diff_std, align='center', alpha=0.7, ecolor='black', capsize=10)
    
    ax.set_ylabel('diff_mean')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()
    
def visualize_diff_kernel_value(result, eval_kernels):
    diff_kernels, opt_kernels, ori_kernels = calc_diff(result)

    pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    ax=fig.add_subplot(1,1,1)
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    markers = itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+', 'x'])
    
    for kernel in eval_kernels:
        color = next(colors)
        marker = next(markers)
        
        ori_values = np.array(ori_kernels[kernel]).reshape(-1, 1)
        error_values = np.array(diff_kernels[kernel])
        
        plt.scatter(ori_values, error_values, label=kernel, color=color, marker=marker)
        
        model = LinearRegression()
        model.fit(ori_values, error_values)
        predicted_error = model.predict(ori_values)
        
        ax, fig = pt.plot_data(ori_values,predicted_error, fig=fig, ax=ax,
                               xlbl='Original Kernel Values',
                               ylbl='Optimization Error',
                               lbl=f'{kernel} (fit)',clr=color,mrk=marker)

    plt.title('Optimization Error vs. Original kerneleter Values')
    ax.grid('minor')
    plt.tight_layout() 

def visualize_PSD(variable, pbe_type, one_frame):
    find, exp_data_paths= initial_pop(variable, pbe_type)
    fig=plt.figure()    
    axq3=fig.add_subplot(1,1,1)
    fig_NM=plt.figure()    
    axq3_NM=fig_NM.add_subplot(1,1,1)
    fig_M=plt.figure()    
    axq3_M=fig_M.add_subplot(1,1,1)
    
    if one_frame:
        return_one_frame(variable, find, exp_data_paths,fig,axq3,fig_NM,axq3_NM,fig_M,axq3_M)
    else:
        return_animation(variable, find, exp_data_paths,fig,axq3,fig_NM,axq3_NM,fig_M,axq3_M)
    
def return_animation(variable, find, exp_data_paths,fig,axq3,fig_NM,axq3_NM,fig_M,axq3_M):
    find_opt = copy.deepcopy(find)
    find.algo.calc_init_N = False
    find.algo.calc_all_pop(variable[2])
    t_vec = find.algo.t_vec
    smoothing = find.algo.smoothing
    
    find_opt.algo.calc_init_N = True
    find_opt.algo.set_init_N(find.algo.sample_num, exp_data_paths, 'mean')
    find_opt.algo.calc_all_pop(variable[1])
    ani=return_animation_distribution(t_vec,smoothing,find.algo.p,find_opt.algo.p,fig,axq3,fps)
    ani_NM=return_animation_distribution(t_vec,smoothing,find.algo.p_NM,find_opt.algo.p_NM,fig_NM,axq3_NM,fps)
    ani_M=return_animation_distribution(t_vec,smoothing,find.algo.p_M,find_opt.algo.p_M,fig_M,axq3_M,fps)
    ani.save('PSD_ani.gif', writer='imagemagick', fps=fps)
    ani_NM.save('PSD_ani_NM.gif', writer='imagemagick', fps=fps)
    ani_M.save('PSD_ani_M.gif', writer='imagemagick', fps=fps)
    
def return_one_frame(variable, find, exp_data_paths,fig,axq3,fig_NM,axq3_NM,fig_M,axq3_M ):
    ## Calculate original PSD(exp)
    find.algo.calc_init_N = False
    find.algo.calc_all_pop(variable[2])
    return_pop_distribution(find, find.algo.p, axq3, fig, clr='b', q3lbl='q3_psd')
    return_pop_distribution(find, find.algo.p_NM, axq3_NM, fig_NM, clr='b', q3lbl='q3_psd')
    return_pop_distribution(find, find.algo.p_M, axq3_M, fig_M, clr='b', q3lbl='q3_psd')
    
    ## Calculate PSD using opt_value
    find.algo.calc_init_N = True
    find.algo.set_init_N(find.algo.sample_num, exp_data_paths, 'mean')
    find.algo.calc_all_pop(variable[1])
    return_pop_distribution(find, find.algo.p, axq3, fig, clr='r', q3lbl='q3_opt')
    return_pop_distribution(find, find.algo.p_NM, axq3_NM, fig_NM, clr='r', q3lbl='q3_opt')
    return_pop_distribution(find, find.algo.p_M, axq3_M, fig_M, clr='r', q3lbl='q3_opt')   
    fig.savefig('PSD', dpi=150)
    fig_NM.savefig('PSD_NM', dpi=150)
    fig_M.savefig('PSD_M', dpi=150)
    
def initial_pop(variable, pbe_type):
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    pop_params['process_type'] = pbe_type
    find = opt.opt_find()        
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
    R01_0 = 'r0_001'
    R03_0 = 'r0_001'
    find.algo.set_comp_para(USE_PSD, R01_0, R03_0, R_NM=R_NM, R_M=R_M,R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl,
                            dist_path_NM=dist_path_NM, dist_path_M=dist_path_M)
    find.algo.weight_2d = conf.config['weight_2d']
    pop_params.update(variable[2])
    b = pop_params['CORR_BETA']
    a = pop_params['alpha_prim']
    v = pop_params['pl_v']
    p1 = pop_params['pl_P1']
    p2 = pop_params['pl_P2']
    p3 = pop_params['pl_P3']
    p4 = pop_params['pl_P4']
    data_name = f"Sim_Mul_0.1_para_{b}_{a[0]}_{a[1]}_{a[2]}_{v}_{p1}_{p2}_{p3}_{p4}.xlsx" 
    exp_data_path = os.path.join('PSD_data', pbe_type, data_name)
    exp_data_paths = [
        exp_data_path,
        exp_data_path.replace(".xlsx", "_NM.xlsx"),
        exp_data_path.replace(".xlsx", "_M.xlsx")
    ]
    return find, exp_data_paths

def return_animation_distribution(t_vec,smoothing,p,p_opt,fig,axq3,fps):
    def update(frame):
        q3lbl = f"t={t_vec[frame]}"
        q3lbl_opt = f"t_opt={t_vec[frame]}"
        while len(axq3.lines) > 0:
            axq3.lines[0].remove()
        x_uni, q3, Q3, sumvol_uni = p.return_distribution(t=frame, flag='x_uni, q3, Q3, sumvol_uni')
        x_uni_opt, q3_opt, Q3_opt, sumvol_uni_opt = p_opt.return_distribution(t=frame, flag='x_uni, q3, Q3, sumvol_uni')
        if smoothing:
            algo_ins = algo.opt_algo()
            kde = algo_ins.KDE_fit(x_uni[1:],sumvol_uni[1:],bandwidth='scott', kernel_func='epanechnikov')
            q3 = algo_ins.KDE_score(kde,x_uni[1:])
            q3 = np.insert(q3, 0, 0.0)
            kde_opt = algo_ins.KDE_fit(x_uni_opt[1:],sumvol_uni_opt[1:],bandwidth='scott', kernel_func='epanechnikov')
            q3_opt = algo_ins.KDE_score(kde_opt,x_uni_opt[1:])
            q3_opt = np.insert(q3_opt, 0, 0.0)
        
        axq3.plot(x_uni, q3, label=q3lbl, color='b', marker='o')  
        axq3.plot(x_uni_opt, q3_opt, label=q3lbl_opt, color='r', marker='^')  
        axq3.legend()
        return axq3,
    t_frame = np.arange(len(t_vec))
    axq3.set_xlabel('Agglomeration size $x_\mathrm{A}$ / $-$')
    axq3.set_ylabel('number distribution of agglomerates $q3$ / $-$')
    axq3.grid('minor')
    axq3.set_xscale('log')
    plt.tight_layout()  
    
    ani = FuncAnimation(fig, update, frames=t_frame, blit=False)
    return ani
    
    
def return_pop_distribution(find, pop, axq3=None,fig=None, clr='b', q3lbl='q3'):
    x_uni, q3, Q3, sumvol_uni = pop.return_distribution(t=t_return, flag='x_uni, q3, Q3, sumvol_uni')

    kde = find.algo.KDE_fit(x_uni[1:], sumvol_uni[1:])
    q3_sm = find.algo.KDE_score(kde, x_uni[1:])
    q3_sm = np.insert(q3_sm, 0, 0.0)
    axq3, fig = pt.plot_data(x_uni, q3_sm, fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl=q3lbl+'_sm',clr=clr,mrk='o')
    if not find.algo.smoothing:    
        axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                                lbl=q3lbl,clr=clr,mrk='^')
    
    axq3.grid('minor')
    axq3.set_xscale('log')
    
if __name__ == '__main__': 
    use_rel_diff = False
    results_pth = 'Parameter_study'
    pbe_type = 'mix'
    # pbe_type = 'breakage'
    # pbe_type = 'mix'
    # file_names = [
    #     'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'MSE\')]_BO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'MAE\')]_BO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'RMSE\')]_BO_wight_1_iter_400.npz',
    #     'multi_[(\'QQ3\', \'KL\')]_BO_wight_1_iter_400.npz',
    #     # 'multi_[(\'QQ3\', \'MSE\')]_BO_wight_1_iter_400.npz',
    #     'multi_[(\'x_50\', \'MSE\')]_BO_wight_1_iter_400.npz',
    #     # 'multi_[(\'q3\', \'KL\'), (\'Q3\', \'KL\'), (\'x_50\', \'MSE\')]_BO_wight_1_iter_400.npz',
    #     ]
    # labels = [
    #     'q3_KL',
    #     'q3_MSE',
    #     'q3_MAE',
    #     'q3_RMSE',
    #     'Q3_KL',
    #     # 'Q3_MSE',
    #     'x_50_MSE',
    #     # 'q3_KL_Q3_KL_x_50_MSE',
    #     ]
    # file_names = [
    #     'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_400.npz',
    #     # 'multi_[(\'q3\', \'KL\')]_BO_wight_5_iter_400.npz',
    #     'multi_[(\'q3\', \'KL\')]_BO_wight_10_iter_400.npz',
    #     ]
    # labels = [
    #     'wight_1',
    #     # 'wight_5',
    #     'wight_10',
    #     ]
    
    file_names = [
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_50.npz',
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_100.npz',
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_200.npz',
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_400.npz',
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_800.npz',
        ]
    labels = [
        'iter_50',
        'iter_100',
        'iter_200',
        'iter_400',
        'iter_800',
        ]
    
    data_paths = [os.path.join(results_pth, pbe_type, file_name) for file_name in file_names]
    # 'results' saves the results of all reading files. 
    # The first column in each result is the value of the optimized criteria. 
    # The second column is the value of the optimization kernels. 
    # The third column is the kernel value (target value) of the original pbe.
    results = read_results(data_paths)
    
    visualize_diff_mean(results, labels)
    ## kernel: corr_agg_0, corr_agg_1, corr_agg_2, pl_v, pl_P1, pl_P2, pl_P3, pl_P4
    result_to_analyse = results[-1]
    visualize_diff_kernel_value(result_to_analyse, eval_kernels=['corr_agg_0','corr_agg_1','corr_agg_2'])
    visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_v'])
    visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P1','pl_P3'])
    visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P2','pl_P4'])
    
    variable_to_analyse = result_to_analyse[-1]
    one_frame = False
    t_return = -1
    fps = 5
    variable_to_analyse = visualize_PSD(variable_to_analyse, pbe_type, one_frame)
    
