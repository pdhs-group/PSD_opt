# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
import config.opt_config as conf
from pypbe.kernel_opt.opt_base import OptBase
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
## For plots
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt  
import itertools
import multiprocessing
from matplotlib.animation import FuncAnimation

epsilon = 1e-20

def read_results(data_paths):
    post_results = []
    elapsed_time = []
    for data_path in data_paths:
        data = np.load(data_path,allow_pickle=True)
        results=data['results']
        tem_time = data['time']
        results_tem = np.empty((len(results), 3), dtype=object)
        for i in range(results.shape[0]):
            # results_tem[i, 0] = results[i, 0]['opt_score']
            # results_tem[i, 1] = results[i, 0]['opt_parameters']
            # results_tem[i, 2] = results[i, 1]
            results_tem[i, 0] = results[i]['opt_score']
            results_tem[i, 1] = results[i]['opt_params']
            filename = results[i]['file_path'] 
            if isinstance(filename, list):
                data_name = filename[0]
            else:
                data_name = filename
            results_tem[i, 2] = get_kernels_form_data_name(data_name)
            
        # For comparison, CORR_BETA and alpha_prim in the original parameters are merged into corr_agg
        # ori_kernels = results_tem[:,2]
        # if 'CORR_BETA' in ori_kernels[0] and 'alpha_prim' in ori_kernels[0]:
        #     for ori_kernel in ori_kernels:
        #         ori_kernel['corr_agg'] = ori_kernel['CORR_BETA'] * ori_kernel['alpha_prim']
        post_results.append(results_tem)
        elapsed_time.append(tem_time)
        data.close()
    return post_results, elapsed_time

def get_kernels_form_data_name(data_name):
    kernels = {}
    param_str = data_name.split('para_')[-1]
    param_str = param_str.rsplit('.', 1)[0] 
    params = param_str.split('_')
    converted_params = [float(param) if '.' in param or 'e' in param.lower() else int(param) for param in params]
    CORR_BETA = converted_params[0]
    alpha_prim = np.array(converted_params[1:4])
    kernels['corr_agg'] = CORR_BETA * alpha_prim
    kernels['pl_v'] = converted_params[4]
    kernels['pl_P1'] = converted_params[5]
    kernels['pl_P2'] = converted_params[6]
    kernels['pl_P3'] = converted_params[7]
    kernels['pl_P4'] = converted_params[8]
    return kernels
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
                if diff_type == 'rel':
                    rel_diff = np.where(tem_ori_kernel[:,i] != 0, diff / (tem_ori_kernel[:,i]+epsilon), diff)
                    diff = rel_diff
                elif diff_type == 'scaled':
                    scaled_diff = diff / max(tem_ori_kernel[:,i])
                    diff = scaled_diff
                diff_kernels[f"{kernel}_{i}"] = diff
        else:
            ## Change the format of the dictionary 
            ## so that it remains in the same format as diff_kernels
            opt_kernels[kernel] = tem_opt_kernel
            ori_kernels[kernel] = tem_ori_kernel
            diff = abs(tem_opt_kernel - tem_ori_kernel)
            if diff_type=='rel':
                rel_diff = np.where(tem_ori_kernel != 0, diff / (tem_ori_kernel+epsilon), diff)
                diff = rel_diff
            elif diff_type == 'scaled':
                scaled_diff = diff / max(tem_ori_kernel)
                diff = scaled_diff
            diff_kernels[kernel] = diff
    return diff_kernels, opt_kernels, ori_kernels 

# def visualize_diff_mean(results, labels):
#     num_results = len(results)
#     diff_mean = np.zeros(num_results)
#     diff_std = np.zeros(num_results)
#     diff_var = np.zeros(num_results)
    
#     for i, result in enumerate(results):
#         if vis_criteria == 'kernels':
#             diff_kernels, _, _ = calc_diff(result)
#             all_elements = np.concatenate(list(diff_kernels.values()))
#             ylabel = 'Kernels Error'
#         elif vis_criteria == 'mse':
#             # all_elements = result[:,3]
#             all_elements = result[:,0]
#             ylabel = 'MSE of PSD'
#         diff_mean[i] = np.mean(all_elements)
#         diff_std[i] = np.std(all_elements)
#         diff_var[i] = np.var(all_elements)
    
#     x_pos = np.arange(len(labels))
#     fig, ax = plt.subplots(figsize=(16, 8))
#     ax.bar(x_pos, diff_mean, yerr=diff_std, align='center', alpha=0.7, ecolor='black', capsize=10)
    
#     ax.set_ylabel(ylabel, fontsize=20)
#     ax.tick_params(axis='y', labelsize=15)
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(labels, fontsize=20)
    
#     plt.tight_layout()
#     plt.show()
    
def visualize_diff_mean(results, labels):
    num_results = len(results)
    diff_mean_kernels = np.zeros(num_results)
    diff_std_kernels = np.zeros(num_results)
    diff_var_kernels = np.zeros(num_results)
    
    diff_mean_mse = np.zeros(num_results)
    diff_std_mse = np.zeros(num_results)
    diff_var_mse = np.zeros(num_results)

    for i, result in enumerate(results):
        diff_kernels, _, _ = calc_diff(result)
        all_elements_kernels = np.concatenate(list(diff_kernels.values()))
        diff_mean_kernels[i] = np.mean(all_elements_kernels)
        diff_std_kernels[i] = np.std(all_elements_kernels)
        diff_var_kernels[i] = np.var(all_elements_kernels)
        all_elements_mse = result[:, 0]
        diff_mean_mse[i] = np.mean(all_elements_mse)
        diff_std_mse[i] = np.std(all_elements_mse)
        diff_var_mse[i] = np.var(all_elements_mse)
    
    x_pos = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # ax1.set_xlabel('Labels')
    ax1.set_ylabel('Kernels Error', color='tab:blue', fontsize=20)
    ax1.bar(x_pos - 0.2, diff_mean_kernels, yerr=diff_std_kernels, width=0.4, align='center', alpha=0.7, ecolor='black', capsize=10, color='tab:blue', label='Kernels Error')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=20)
    ax1.axhline(0, color='black', linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.set_ylabel('MSE of PSD', color='tab:red', fontsize=20)
    ax2.bar(x_pos + 0.2, diff_mean_mse, yerr=diff_std_mse, width=0.4, align='center', alpha=0.7, ecolor='black', capsize=10, color='tab:red', label='MSE of PSD')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=15)
    ax2.axhline(0, color='black', linewidth=0.8)

    
    all_lims = [0]
    all_lims.extend(diff_mean_kernels - diff_std_kernels-0.1)
    all_lims.extend(diff_mean_kernels + diff_std_kernels+0.1)
    all_lims.extend(diff_mean_mse - diff_std_mse-0.1)
    all_lims.extend(diff_mean_mse + diff_std_mse+0.1)
    y1lim = [min(all_lims), max(all_lims)]
    scale_y2 = max(diff_std_mse / diff_std_kernels)*1.5
    y2lim = [min(all_lims)*scale_y2, max(all_lims)*scale_y2] 
    ax1.set_ylim(y1lim)
    ax2.set_ylim(y2lim)
    
    fig.tight_layout()
    plt.show()
    
def visualize_diff_kernel_mse(result):
    diff_kernels, _, _ = calc_diff(result)
    mean_diff_kernels_tem = []
    for key, diff_kernel in diff_kernels.items():
        mean_diff_kernels_tem.append(diff_kernel)
    mean_diff_kernels = np.array(mean_diff_kernels_tem).mean(axis=0)
    mse = result[:,0]
    num_exp = np.arange(len(mean_diff_kernels))
    
    sorted_indices = np.argsort(mean_diff_kernels)
    sorted_mean_diff_kernels = mean_diff_kernels[sorted_indices]
    sorted_mse = mse[sorted_indices]
    
    fig, ax1 = plt.subplots(figsize=(16, 8))   
    # ax1.set_xlabel('Experiment Number')
    ax1.set_ylabel('Mean Diff Kernels', color='tab:blue', fontsize=20)
    ax1.plot(num_exp, sorted_mean_diff_kernels, color='tab:blue', label='Mean Diff Kernels', linewidth= 1.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=15)
    ax1.grid('minor')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('MSE', color='tab:red', fontsize=20)
    ax2.plot(num_exp, sorted_mse, color='tab:red', label='MSE', linewidth= 1.5)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=15)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1)  # 设置主轴的粗细
        ax1.spines[axis].set_color('black')  # 设置主轴的颜色
    
    fig.tight_layout()
    plt.show()

def calc_save_PSD_delta(results, data_paths):
    for i, result in enumerate(results):
        opt_kernels = result[:,1]
        delta = np.zeros(len(result))
        # for j, variable in enumerate(result):
        j = 351
        variable = result[j]
        opt, exp_data_paths = initial_pop(variable, pbe_type)
        opt.core.set_init_N(opt.core.sample_num, exp_data_paths, 'mean')
        delta[j] = opt.core.calc_delta_agg(opt_kernels[j], sample_num=opt.core.sample_num, exp_data_path=exp_data_paths)
        new_result = np.column_stack((result, delta))
        results[i] = new_result    
        np.savez(data_paths[i], results=new_result)
        
def calc_ori_mse():
    opt = OptBase()
    
    var_corr_beta = np.array([1e-3,1e-2,1e-1])
    values = np.array([0.5, 1.0])
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]
    unique_alpha_prim = []
    for comp in var_alpha_prim:
        comp_reversed = comp[::-1]  
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_alpha_prim):
            unique_alpha_prim.append(comp)
    var_alpha_prim = np.array(unique_alpha_prim)
    var_v = np.array([0.7,1.0,2.0])
    var_P1 = np.array([1e-3,1e-2,1e-1])
    var_P2 = np.array([0.5,1.0,2.0])
    var_P3 = np.array([1e-3,1e-2,1e-1])
    var_P4 = np.array([0.5,1.0,2.0])
    
    func_list = []
    for j,corr_beta in enumerate(var_corr_beta):
        for k,alpha_prim in enumerate(var_alpha_prim):
            for l,v in enumerate(var_v):
                for m1,P1 in enumerate(var_P1):
                    for m2,P2 in enumerate(var_P2):
                        for m3,P3 in enumerate(var_P3):
                            for m4,P4 in enumerate(var_P4):
                                ori_params = {
                                    'CORR_BETA' : corr_beta,
                                    'alpha_prim' : alpha_prim,
                                    'pl_v' : v,
                                    'pl_P1' : P1,
                                    'pl_P2' : P2,
                                    'pl_P3' : P3,
                                    'pl_P4' : P4,
                                    }
                                data_name = f"Sim_Mul_0.1_para_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_{v}_{P1}_{P2}_{P3}_{P4}.xlsx"
                                exp_data_path = os.path.join('PSD_data', pbe_type, 'data', data_name)
                                exp_data_paths = [
                                    exp_data_path,
                                    exp_data_path.replace(".xlsx", "_NM.xlsx"),
                                    exp_data_path.replace(".xlsx", "_M.xlsx")
                                ]
                                # results = opt.calc_PSD_delta(ori_params, exp_data_paths)
                                func_list.append((ori_params,exp_data_paths))
    pool = multiprocessing.Pool(processes=24)
    results = pool.starmap(opt.calc_PSD_delta, func_list) 
    np.savez('ori_mse.npz', 
          results=results, 
          )     
    return results
    
def do_remove_small_results(results):
    indices_to_remove = set()
    if pbe_type == 'agglomeration':
        for i in range(len(results[0])):
            corr_agg = results[0][i, 2]['corr_agg']
            if corr_agg[0] * corr_agg[1] * corr_agg[2] < 1:
                indices_to_remove.add(i)
    elif pbe_type == 'breakage':
        for i in range(len(results[0])):
            pl_P1 = results[0][i, 2]['pl_P1']
            pl_P3 = results[0][i, 2]['pl_P3']
            if pl_P1 * pl_P3 < 1e-9:
                indices_to_remove.add(i)
          
    for idx in sorted(indices_to_remove, reverse=True):
        for j in range(len(results)):
            results[j] = np.delete(results[j], idx, axis=0)
                
    return results



#%% VISUALIZE KERNEL DIFFERENCE
#%%%VISUALZE IN RADAR
def visualize_diff_mean_radar(results, data_labels):
    diff_mean = []
    if vis_criteria == 'kernels':
        for i, result in enumerate(results):
            diff_mean_tem =[]
            kernels_labels = []
            diff_kernels, _, _ = calc_diff(result)
            for key, array in diff_kernels.items():
                avg  = np.mean(array)
                diff_mean_tem.append(avg)
                kernels_labels.append(key) 
            diff_mean.append(np.array(diff_mean_tem))
            title = 'Kernels Error'
        radar_chart(diff_mean, data_labels, kernels_labels, title)
    else:
        return
    
def radar_chart(data, data_labels, kernels_labels, title):
    # Number of variables
    num_vars = len(kernels_labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, d in enumerate(data):
        values = d.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=data_labels[i])

    # Draw one axe per variable + add labels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(kernels_labels, fontsize=15)

    plt.title(title, size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=18)
    plt.show()

#%%%VISUALZE IN BLOCK
def visualize_diff_kernel_value(result, eval_kernels, log_axis=False):
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
        opt_values = np.array(opt_kernels[kernel])
        
        plt.scatter(ori_values, opt_values, label=kernel, color=color, marker=marker)
        
        model = LinearRegression()
        model.fit(ori_values, opt_values)
        predicted_opt = model.predict(ori_values)
        
        ax, fig = pt.plot_data(ori_values, predicted_opt, fig=fig, ax=ax,
                               xlbl='Original Kernel Values',
                               ylbl='Optimized Kernel Values',
                               lbl=f'{kernel} (fit)',clr=color,mrk=marker)
        ax, fig = pt.plot_data(ori_values, ori_values, fig=fig, ax=ax,
                                lbl=f'{kernel} (correct)',clr='k',mrk='x')    
    
    if log_axis:
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.title('Optimized Kernel Values vs. Original kerneleter Values')
    ax.grid('minor')
    plt.tight_layout() 
    return diff_kernels
    
#%% RETURN PSD IN FRAME/ANIMATION
def initial_pop(variable, pbe_type):
    opt = OptBase()
    
    pop_params = opt.core.check_corr_agg(variable[2])
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
    return opt, exp_data_paths

def visualize_PSD(variable, pbe_type, one_frame):
    opt, exp_data_paths= initial_pop(variable, pbe_type)
    
    if one_frame:
        return_one_frame(variable, opt, exp_data_paths)
    else:
        return_animation(variable, opt, exp_data_paths)
    
def return_animation(variable, opt, exp_data_paths):
    opt_opt = copy.deepcopy(opt)
    calc_init_N_tem = opt.core.calc_init_N
    opt.core.calc_init_N = False
    opt.core.calc_all_pop(variable[2])
    
    opt_opt.core.calc_init_N = calc_init_N_tem
    if opt_opt.core.calc_init_N:
        opt_opt.core.set_init_N(exp_data_paths, 'mean')
    opt_opt.core.calc_all_pop(variable[1])
    ani=opt.core.p.visualize_distribution_animation(smoothing=opt.core.smoothing,fps=fps,others=[opt_opt.core.p],other_labels=['opt'])
    ani_NM=opt.core.p_NM.visualize_distribution_animation(smoothing=opt.core.smoothing,fps=fps,others=[opt_opt.core.p_NM],other_labels=['opt'])
    ani_M=opt.core.p_M.visualize_distribution_animation(smoothing=opt.core.smoothing,fps=fps,others=[opt_opt.core.p_M],other_labels=['opt'])
    
    ani.save('PSD_ani.gif', writer='imagemagick', fps=fps)
    ani_NM.save('PSD_ani_NM.gif', writer='imagemagick', fps=fps)
    ani_M.save('PSD_ani_M.gif', writer='imagemagick', fps=fps)
    
def return_one_frame(variable, opt, exp_data_paths):
    fig=plt.figure()    
    axq3=fig.add_subplot(1,2,1)
    axQ3=fig.add_subplot(1,2,2)
    fig_NM=plt.figure()    
    axq3_NM=fig_NM.add_subplot(1,2,1)
    axQ3_NM=fig_NM.add_subplot(1,2,2)
    fig_M=plt.figure()    
    axq3_M=fig_M.add_subplot(1,2,1)
    axQ3_M=fig_M.add_subplot(1,2,2)
    
    calc_init_N_tem = opt.core.calc_init_N
    ## Calculate original PSD(exp)
    opt.core.calc_init_N = False
    opt.core.calc_all_pop(variable[2])
    opt.core.p.visualize_distribution(smoothing=opt.core.smoothing,axq3=axq3,axQ3=axQ3,fig=fig,clr='b',lbl='PSD_ori')
    opt.core.p_NM.visualize_distribution(smoothing=opt.core.smoothing,axq3=axq3_NM,axQ3=axQ3_NM,fig=fig_NM,clr='b',lbl='PSD_ori')
    opt.core.p_M.visualize_distribution(smoothing=opt.core.smoothing,axq3=axq3_M,axQ3=axQ3_M,fig=fig_M,clr='b',lbl='PSD_ori')
    
    ## Calculate PSD using opt_value
    opt.core.calc_init_N = calc_init_N_tem
    if opt.core.calc_init_N:
        opt.core.set_init_N(exp_data_paths, 'mean')
    opt.core.calc_all_pop(variable[1])
    opt.core.p.visualize_distribution(smoothing=True,axq3=axq3,axQ3=axQ3,fig=fig,clr='r',lbl='PSD_opt')
    opt.core.p_NM.visualize_distribution(smoothing=True,axq3=axq3_NM,axQ3=axQ3_NM,fig=fig_NM,clr='r',lbl='PSD_opt')
    opt.core.p_M.visualize_distribution(smoothing=True,axq3=axq3_M,axQ3=axQ3_M,fig=fig_M,clr='r',lbl='PSD_opt')
    fig.savefig('PSD', dpi=150)
    fig_NM.savefig('PSD_NM', dpi=150)
    fig_M.savefig('PSD_M', dpi=150)
#%% MAIN FUNCTION
if __name__ == '__main__': 
    # diff_type = 'rel'
    # diff_type = 'abs'
    diff_type = 'scaled'
    
    remove_small_results = False
    results_pth = 'Parameter_study'
    calc_criteria = False
    vis_criteria = 'kernels'
    # vis_criteria = 'mse'

    # pbe_type = 'agglomeration'
    # pbe_type = 'breakage'
    pbe_type = 'mix'
    
    results_mse = calc_ori_mse()
    # file_names = [
    #     'multi_[(\'q3\', \'KL\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'MAE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'RMSE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'QQ3\', \'KL\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'QQ3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'QQ3\', \'MAE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'QQ3\', \'RMSE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'x_50\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
    #     # 'multi_[(\'q3\', \'MSE\'), (\'Q3\', \'MSE\'), (\'x_50\', \'MSE\')]_BO_wight_1_iter_400.npz',
    #     ]
    # labels = [
    #     'q3_KL',
    #     'q3_MSE',
    #     'q3_MAE',
    #     'q3_RMSE',
    #     'Q3_KL',
    #     'Q3_MSE',
    #     'Q3_MAE',
    #     'Q3_RMSE',
    #     'x_50_MSE',
    #     # 'q3_MSE_Q3_MSE_x_50_MSE',
    #     ]
    # file_names = [
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_2_iter_400.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_3_iter_400.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_5_iter_400.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_10_iter_400.npz',
    #     ]
    # labels = [
    #     'wight_1',
    #     'wight_2',
    #     'wight_3',
    #     'wight_5',
    #     'wight_10',
    #     ]
    
    # file_names = [
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_50.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_100.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_200.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
    #     'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
    #     ]
    # labels = [
    #     'iter_50',
    #     'iter_100',
    #     'iter_200',
    #     'iter_400',
    #     'iter_800',
    #     ]
    
    # file_names = [
    #     '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_50.npz',
    #     '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_100.npz',
    #     '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_200.npz',
    #     '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
    #     '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
    #     ]
    # labels = [
    #     'iter_50',
    #     'iter_100',
    #     'iter_200',
    #     'iter_400',
    #     'iter_800',
    #     ]
        
    file_names = [
        'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_800.npz',
        'multi_[(\'q3\', \'MSE\')]_NSGA_wight_1_iter_800.npz',
        'multi_[(\'q3\', \'MSE\')]_QMC_wight_1_iter_800.npz',
        'multi_[(\'q3\', \'MSE\')]_TPS_wight_1_iter_800.npz',
        'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_800.npz',
        'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
        ]
    labels = [
        'GP',
        'NSGA',
        'QMC',
        'TPS',
        'Cmaes',
        'HEBO',
        ]
    
    data_paths = [os.path.join(results_pth, pbe_type, file_name) for file_name in file_names]
    # 'results' saves the results of all reading files. 
    # The first column in each result is the value of the optimized criteria. 
    # The second column is the value of the optimization kernels. 
    # The third column is the kernel value (target value) of the original pbe.
    results, elapsed_time = read_results(data_paths)
    
    if calc_criteria:
        calc_save_PSD_delta(results, data_paths)
    if remove_small_results:
        results = do_remove_small_results(results)
        
    # visualize_diff_mean(results, labels)
    
    # kernel: corr_agg_0, corr_agg_1, corr_agg_2, pl_v, pl_P1, pl_P2, pl_P3, pl_P4
    result_to_analyse = results[-1]
    # if pbe_type == 'agglomeration' or pbe_type == 'mix':
    #     corr_agg_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['corr_agg_0','corr_agg_1','corr_agg_2'])
    # if pbe_type == 'breakage' or pbe_type == 'mix':
    #     pl_v_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_v'])
    #     pl_P13_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P1','pl_P3'], log_axis=False)
    #     pl_P24_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P2','pl_P4'])
    
    # visualize_diff_mean_radar(results, labels)
    # visualize_diff_kernel_mse(result_to_analyse)
    
    variable_to_analyse = result_to_analyse[2]
    one_frame = False
    # calc_init = False
    t_return = -1
    fps = 5
    # visualize_PSD(variable_to_analyse, pbe_type, one_frame)

