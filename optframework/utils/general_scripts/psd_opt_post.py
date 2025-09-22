# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import sys, os
import re
import opt_config as conf
from optframework.kernel_opt.opt_base import OptBase
import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
## For plots
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt  
import itertools
import multiprocessing
from matplotlib.animation import FuncAnimation
from scipy.stats import pearsonr, spearmanr
from typing import Iterable, List, Tuple, Dict

epsilon = 1e-20

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
                max_search, min_search = get_search_range(f"{kernel}_{i}")
                diff = abs(tem_opt_kernel[:,i] - tem_ori_kernel[:,i])
                if diff_type == 'rel':
                    rel_diff = np.where(tem_ori_kernel[:,i] != 0, diff / (tem_ori_kernel[:,i]+epsilon), diff)
                    diff = rel_diff
                elif diff_type == 'scaled':
                    # scaled_diff = diff / (max_search - min_search)
                    scaled_diff = diff / (max(tem_ori_kernel[:,i]) - min(tem_ori_kernel[:,i]))
                    diff = scaled_diff
                diff_kernels[f"{kernel}_{i}"] = diff
        else:
            ## Change the format of the dictionary 
            ## so that it remains in the same format as diff_kernels
            opt_kernels[kernel] = tem_opt_kernel
            ori_kernels[kernel] = tem_ori_kernel
            max_search, min_search = get_search_range(kernel)
            diff = abs(tem_opt_kernel - tem_ori_kernel)
            if diff_type=='rel':
                rel_diff = np.where(tem_ori_kernel != 0, diff / (tem_ori_kernel+epsilon), diff)
                diff = rel_diff
            elif diff_type == 'scaled':
                # scaled_diff = diff / (max_search - min_search)
                scaled_diff = diff / (max(tem_ori_kernel) - min(tem_ori_kernel))
                diff = scaled_diff
            diff_kernels[kernel] = diff
    return diff_kernels, opt_kernels, ori_kernels 

def visualize_sampler_iter():
    def plot_metric_vs_iterations(iterations, metric_means, metric_stds, samplers, ylabel, title):
        # 绘制每个采样器的曲线和方差范围
        for i, sampler in enumerate(samplers):
            mean_values = metric_means[i]
            std_values = metric_stds[i]
            
            # 绘制平均值曲线
            plt.plot(iterations, mean_values, label=sampler)
            
            # 绘制方差范围的半透明区域
            plt.fill_between(
                iterations,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.2
            )
    
        plt.xlabel("Iterations")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(title="Samplers")
        plt.grid(True)
        plt.show()
    samplers = ['HEBO', 'GP', 'NSGA', 'QMC', 'TPE', 'Cmaes']
    iterations = [50, 100, 200, 400, 800] 
    file_names = [
    f"multi_[(\'q3\', \'MSE\')]_{sampler}_wight_1_iter_{iter_count}.npz"
    for sampler in samplers
    for iter_count in iterations
    ]
    data_paths = [os.path.join(results_pth, pbe_type, file_name) for file_name in file_names]
    results, elapsed_time = read_results(data_paths)
    
    num_results = len(results)
    diff_mean_kernels = np.zeros(num_results)
    diff_std_kernels = np.zeros(num_results)
    pearson_corrs = np.zeros(num_results)
    
    diff_mean_mse = np.zeros(num_results)
    diff_std_mse = np.zeros(num_results)

    for i, result in enumerate(results):
        diff_kernels, _, _ = calc_diff(result)
        all_elements_kernels = np.concatenate(list(diff_kernels.values()))
        diff_mean_kernels[i] = np.mean(all_elements_kernels)
        diff_std_kernels[i] = np.std(all_elements_kernels)
        all_elements_mse = result[:, 0]
        diff_mean_mse[i] = np.mean(all_elements_mse)
        diff_std_mse[i] = np.std(all_elements_mse)
        pearson_corrs[i] = correlation_analysis(result)
        
    num_samplers = len(samplers)
    num_iterations = len(iterations)
    diff_mean_mse = diff_mean_mse.reshape(num_samplers, num_iterations)
    diff_std_mse = diff_std_mse.reshape(num_samplers, num_iterations)
    diff_mean_kernels = diff_mean_kernels.reshape(num_samplers, num_iterations)
    diff_std_kernels = diff_std_kernels.reshape(num_samplers, num_iterations)
    # 绘制平均 MSE
    plot_metric_vs_iterations(
        iterations=iterations,
        metric_means=diff_mean_mse,
        metric_stds=diff_std_mse,
        samplers=samplers,
        ylabel="Mean MSE",
        title="Mean MSE vs Iterations"
    )
    
    # # 绘制平均 Kernels
    # plot_metric_vs_iterations(
    #     iterations=iterations,
    #     metric_means=diff_mean_kernels,
    #     metric_stds=diff_std_kernels,
    #     samplers=samplers,
    #     ylabel="Mean Kernels",
    #     title="Mean Kernels vs Iterations"
    # )
    # 构建DataFrame，包含所有数据
    data = {
        "Sampler": np.repeat(samplers, num_iterations),
        "Iterations": iterations * num_samplers,
        "Mean_Kernels": diff_mean_kernels.flatten(),
        "Std_Kernels": diff_std_kernels.flatten(),
        "Mean_MSE": diff_mean_mse.flatten(),
        "Std_MSE": diff_std_mse.flatten(),
    }
    
    df = pd.DataFrame(data)
    
    # 保存为CSV文件
    df.to_csv("results_for_origin.csv", index=False)
    
    return pearson_corrs
    
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
        diff_std_kernels[i] = np.std(all_elements_kernels) / np.sqrt(len(all_elements_kernels))
        diff_var_kernels[i] = np.var(all_elements_kernels)
        all_elements_mse = result[:, 0]
        diff_mean_mse[i] = np.mean(all_elements_mse)
        diff_std_mse[i] = np.std(all_elements_mse)  / np.sqrt(len(all_elements_mse))
        diff_var_mse[i] = np.var(all_elements_mse)
    
    x_pos = np.arange(len(labels))
    fig=plt.figure() 
    ax1=fig.add_subplot(1,1,1)

    ax1.set_ylabel('$\overline{k_{\delta}}$')
    ax1.bar(x_pos - 0.2, diff_mean_kernels, yerr=diff_std_kernels, width=0.4, align='center', alpha=0.7, ecolor='black', capsize=10, color='tab:blue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.axhline(0, color='black', linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.set_ylabel('$\overline{MSE_{q3}}$')
    ax2.bar(x_pos + 0.2, diff_mean_mse, yerr=diff_std_mse, width=0.4, align='center', alpha=0.7, ecolor='black', capsize=10, color='tab:red')
    ax2.axhline(0, color='black', linewidth=0.8)
    ## Indicator line for theoretical minimum mse
    ax2.axhline(1, color='red', linestyle='--', linewidth=1.5)
    ax2.text(len(labels) - 0.5, 1.1, '$\overline{MSE_{q3}} = 1$', color='red', fontsize=15, va='bottom', ha='right')
    
    all_lims = [0]
    # y1lim_tem = diff_mean_kernels - diff_std_kernels
    # y1lim_tem.extend(diff_mean_kernels + diff_std_kernels)
    # y2lim_tem = diff_mean_mse - diff_std_mse
    # y2lim_tem.extend(diff_mean_mse + diff_std_mse)
    all_lims.extend(diff_mean_kernels - diff_std_kernels)
    all_lims.extend(diff_mean_kernels + diff_std_kernels)
    all_lims.extend(diff_mean_mse - diff_std_mse)
    all_lims.extend(diff_mean_mse + diff_std_mse)
    ## Slightly shift the upper or lower limit
    min_lim=min(all_lims)
    max_lim=max(all_lims)+0.1

    y2lim = [min_lim, max_lim]
    if diff_type == 'abs':
        scale_y2 = min(diff_std_mse / diff_std_kernels)
        y1lim = [min_lim/scale_y2, max_lim/scale_y2] 
    elif diff_type == 'scaled':
        y1lim = [0.0, 1.0] 
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
        ax1.spines[axis].set_linewidth(1)  
        ax1.spines[axis].set_color('black')  
    
    fig.tight_layout()
    plt.show()
  
def visualize_correlation(results, labels):
    pearson_corrs = np.zeros(len(results))
    for i, result in enumerate(results):
        pearson_corrs[i] = correlation_analysis(result)
    fig=plt.figure()   
    ax=fig.add_subplot(1,1,1)

    ax, fig = pt.plot_data(labels, pearson_corrs, fig=fig, ax=ax,
                           xlbl='',
                           ylbl='Pearson value',lbl='Pearson correlation coefficient',
                           clr='k',mrk='o')
    ax.grid('minor')
    plt.tight_layout() 
        
    return pearson_corrs
def correlation_analysis(result, plot=False):
    diff_kernels, _, _ = calc_diff(result)
    mean_diff_kernels_tem = []
    for key, diff_kernel in diff_kernels.items():
        mean_diff_kernels_tem.append(diff_kernel)
    mean_diff_kernels = np.array(mean_diff_kernels_tem).mean(axis=0)
    mse = np.array(result[:,0], dtype=float)
    # Calculate the Pearson correlation coefficient
    pearson_corr, _ = pearsonr(mse, mean_diff_kernels)
    spearman_corr, _ = spearmanr(mse, mean_diff_kernels)
    # m = np.row_stack((mse,mean_diff_kernels))
    # kovar = np.cov(m)
    m=0
    b=0
    if plot:
        fig=plt.figure()   
        ax=fig.add_subplot(1,1,1)
        # plot data points
        plt.scatter(mse, mean_diff_kernels, color='blue', label='Data Points')
        # Fit a straight line to show the trend
        m, b = np.polyfit(mse, mean_diff_kernels, 1)
        ax, fig = pt.plot_data(mse, m*mse + b, fig=fig, ax=ax,
                               xlbl='$MSE_{q3}$',
                               ylbl='k$_{\delta}$',lbl='Fit Line (y = {m:.2f}x + {b:.2f})',
                               tit='$k_{\delta}$ vs $MSE_{q3}$'+f'(Pearson r = {pearson_corr:.2f})',
                               clr='r',mrk='')
        ax.grid('minor')
        plt.tight_layout() 
    return pearson_corr,m,b,spearman_corr
    
def correlation_analysis_delta(results):
    pearson_corr_list = []
    spearman_corr_list = []
    
    mean_diff_kernels_old = None
    mse_old = None
    
    for result_to_analyse in results:
        diff_kernels, _, _ = calc_diff(result_to_analyse)
        mean_diff_kernels = np.mean(list(diff_kernels.values()), axis=0)
        mse = np.array(result_to_analyse[:, 0], dtype=float)
    
        if mean_diff_kernels_old is not None and mse_old is not None:
            mean_diff_kernels_delta = mean_diff_kernels - mean_diff_kernels_old
            mse_delta = mse - mse_old
            pearson_corr, _ = pearsonr(mse_delta, mean_diff_kernels_delta)
            spearman_corr, _ = spearmanr(mse_delta, mean_diff_kernels_delta)
            pearson_corr_list.append(pearson_corr)
            spearman_corr_list.append(spearman_corr)
    
        mean_diff_kernels_old = mean_diff_kernels
        mse_old = mse
    return pearson_corr_list,spearman_corr_list

def correlation_analysis_sliding(results, window_size=10):
    """
    results: list，每个元素是一个 np.ndarray，表示一个时间步的所有样本数据
             result[:,0] 是 mse； calc_diff(result) 返回 (diff_kernels, _, _)
    window_size: 每个时间窗口包含的时间点数量（默认 10）

    返回:
        pearson_corr_list, spearman_corr_list
        每个元素是一个窗口的平均相关值
    """
    num_steps = len(results)
    pearson_corr_list = []
    spearman_corr_list = []

    # 以滑动窗口的方式划分数据
    for start in range(0, num_steps, window_size):
        end = start + window_size
        if end > num_steps:
            break  # 不足一个窗口就不算
        window_results = results[start:end]  # 这个窗口的所有时间点
        
        # 计算所有样本在该窗口内的 mean_diff_kernels 和 mse 序列
        # 转置结构： sample -> (time, features)
        num_samples = window_results[0].shape[0]

        # 初始化：每个样本的时间序列
        mse_series = np.zeros((num_samples, window_size))
        diff_series = np.zeros((num_samples, window_size))

        for t, res in enumerate(window_results):
            diff_kernels, _, _ = calc_diff(res)
            mean_diff_kernels = np.mean(list(diff_kernels.values()), axis=0)
            mse = np.array(res[:, 0], dtype=float)

            mse_series[:, t] = mse
            diff_series[:, t] = mean_diff_kernels

        # 对每个样本计算 Pearson 和 Spearman 相关
        sample_pearsons = []
        sample_spearmans = []
        for s in range(num_samples):
            # 至少要有两个点才能算相关
            if np.all(mse_series[s] == mse_series[s][0]) or np.all(diff_series[s] == diff_series[s][0]):
                p = 0.0
                sp = 0.0
            else:
                p, _ = pearsonr(mse_series[s], diff_series[s])
                sp, _ = spearmanr(mse_series[s], diff_series[s])
            sample_pearsons.append(p)
            sample_spearmans.append(sp)
            if end == 10 and s == 100:
                fig=plt.figure()   
                ax=fig.add_subplot(1,1,1)
                # plot data points
                plt.scatter(mse_series[s], diff_series[s], color='blue', label='Data Points')
                # Fit a straight line to show the trend
                # m, b = np.polyfit(mse, mean_diff_kernels, 1)
                # ax, fig = pt.plot_data(mse, m*mse + b, fig=fig, ax=ax,
                #                        xlbl='$MSE_{q3}$',
                #                        ylbl='k$_{\delta}$',lbl='Fit Line (y = {m:.2f}x + {b:.2f})',
                #                        tit='$k_{\delta}$ vs $MSE_{q3}$'+f'(Pearson r = {pearson_corr:.2f})',
                #                        clr='r',mrk='')
                ax.grid('minor')
                plt.tight_layout() 

        # 求这个窗口的平均相关
        if sample_pearsons:  # 避免空列表报错
            pearson_corr_list.append(np.nanmean(sample_pearsons))
            spearman_corr_list.append(np.nanmean(sample_spearmans))
        else:
            pearson_corr_list.append(np.nan)
            spearman_corr_list.append(np.nan)

    return pearson_corr_list, spearman_corr_list

def calc_save_PSD_delta(results, data_paths):
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    # config_path = os.path.join(my_pth, '../../../tests/config/opt_config.py')
    data_path = r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_data\New_CAMES\data"
    config_path = r"C:\Users\px2030\Code\PSD_opt\tests\config\opt_config.py"
    opt = OptBase(config_path=config_path, data_path=data_path)
    for i, result in enumerate(results):
        func_list = []
        # delta = np.zeros(len(result))
        # path = np.empty(len(result),dtype=str)
        for j, _ in enumerate(result):
            variable = result[j]
            if isinstance(variable[3], list):
                file_names = [os.path.basename(file_path) for file_path in variable[3]]
                exp_data_paths = [os.path.join(data_path, file_name) for file_name in file_names]
                exp_data_path = exp_data_paths[0]
            else:
                data_name = "Sim_" + variable[3] + ".xlsx"
                exp_data_path = os.path.join(data_path, data_name)
                exp_data_paths = [
                    exp_data_path,
                    exp_data_path.replace(".xlsx", "_NM.xlsx"),
                    exp_data_path.replace(".xlsx", "_M.xlsx")
                ]
            if opt.multi_flag:
                _path = exp_data_paths
            else:
                _path = exp_data_path
            func_list.append((variable[1], _path))
            # delta, path = opt.calc_PSD_delta(variable[1], _path)
            # return delta,opt
        pool = multiprocessing.Pool(processes=8)
        try:
            delta = pool.starmap(opt.calc_PSD_delta, func_list)
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
        finally:
            pool.close()
            pool.join() 
        new_result = np.column_stack((result, delta))
        np.savez(data_paths[i], results=new_result)
    return new_result
        
def calc_save_PSD_delta_test(results, data_paths):
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    # config_path = os.path.join(my_pth, '../../../tests/config/opt_config.py')
    data_path = r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_data\New_CAMES\data"
    config_path = r"C:\Users\px2030\Code\PSD_opt\tests\config\opt_config.py"
    opt = OptBase(config_path=config_path, data_path=data_path)
    result = results[0]
    variable = result[0]
    if isinstance(variable[3], list):
        file_names = [os.path.basename(file_path) for file_path in variable[3]]
        exp_data_paths = [os.path.join(data_path, file_name) for file_name in file_names]
        exp_data_path = exp_data_paths[0]
    else:
        data_name = "Sim_" + variable[3] + ".xlsx"
        exp_data_path = os.path.join(data_path, data_name)
        exp_data_paths = [
            exp_data_path,
            exp_data_path.replace(".xlsx", "_NM.xlsx"),
            exp_data_path.replace(".xlsx", "_M.xlsx")
        ]
    if opt.multi_flag:
        _path = exp_data_paths
    else:
        _path = exp_data_path
    delta, path = opt.calc_PSD_delta(variable[1], _path)
    print(f"delta = {delta}")
    print(f"path = {path}")
    return delta, opt

def calc_ori_mse():
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    # config_path = os.path.join(my_pth, '../../../tests/config/opt_config.py')
    data_path = r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_data\New_CAMES\data"
    config_path = r"C:\Users\px2030\Code\PSD_opt\tests\config\opt_config.py"
    opt = OptBase(config_path=config_path, data_path=data_path)
    
    var_corr_beta = np.array([1.0])
    values = np.array([1e-3,1e-1])
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]
    unique_alpha_prim = []
    for comp in var_alpha_prim:
        comp_reversed = comp[::-1]  
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_alpha_prim):
            unique_alpha_prim.append(comp)
    var_alpha_prim = np.array(unique_alpha_prim)
    var_alpha_prim = np.array([[1e-3,1e-1,1e-3]])
    var_v = np.array([1.0])
    var_P1 = np.array([1e-2])
    var_P2 = np.array([2.0])
    var_P3 = np.array([1e-2])
    var_P4 = np.array([0.5])
    
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
                                if opt.core.add_noise:
                                    prefix = f"Sim_{opt.core.noise_type}_{opt.core.noise_strength}_para"
                                else:
                                    prefix = "Sim_para"
                                data_name = f"{prefix}_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_{v}_{P1}_{P2}_{P3}_{P4}.xlsx"
                                exp_data_path = os.path.join(data_path, data_name)
                                exp_data_paths = [
                                    exp_data_path,
                                    exp_data_path.replace(".xlsx", "_NM.xlsx"),
                                    exp_data_path.replace(".xlsx", "_M.xlsx")
                                ]
                                if opt.multi_flag:
                                    _path = exp_data_paths
                                else:
                                    _path = exp_data_path
                                # print(data_name)
                                results = opt.calc_PSD_delta(ori_params, _path)
    #                             func_list.append((ori_params,_path))
    # pool = multiprocessing.Pool(processes=6)
    # results = pool.starmap(opt.calc_PSD_delta, func_list) 
    # np.savez('ori_mse.npz', 
    #       results=results, 
    #       )     
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

def write_origin_data(results, labels, group_flag):
    # 初始化summary数据
    summary_data = {
        "sheet name": labels,
    }
    
    # 获取所有key作为列名
    sample_result = results[0]
    diff_kernels, _, _ = calc_diff(sample_result)
    keys = list(diff_kernels.keys())
    
    # 初始化diff_kernels keys的平均值列
    for key in keys:
        summary_data[key] = []
    
    # 添加MSE，MSE_error，Kernels，Kernels_error列
    summary_data["MSE"] = []
    summary_data["MSE_error"] = []
    summary_data["Kernels"] = []
    summary_data["Kernels_error"] = []
    
    # 创建一个 Excel writer
    with pd.ExcelWriter(f"post_{group_flag}.xlsx") as writer:
        # 遍历每个result，处理每个sheet
        for i, result in enumerate(results):
            # 从result计算出需要保存的数据
            diff_kernels, _, _ = calc_diff(result)
            all_elements_mse = result[:, 0]
            
            # 计算diff_kernels每个key的平均值并添加到summary
            for key in keys:
                mean_value = np.mean(diff_kernels[key])
                summary_data[key].append(mean_value)
            
            # 计算MSE平均值和标准误差
            mse_mean = np.mean(all_elements_mse)
            mse_error = np.std(all_elements_mse) / np.sqrt(len(all_elements_mse))
            summary_data["MSE"].append(mse_mean)
            summary_data["MSE_error"].append(mse_error)
            
            # 计算diff_kernels所有key的总平均值和标准误差
            all_kernels = np.concatenate(list(diff_kernels.values()))
            kernels_mean = np.mean(all_kernels)
            kernels_error = np.std(all_kernels) / np.sqrt(len(all_kernels))
            summary_data["Kernels"].append(kernels_mean)
            summary_data["Kernels_error"].append(kernels_error)
            
            # 将diff_kernels和MSE数据转换为DataFrame写入sheet
            df = pd.DataFrame(diff_kernels)
            df["MSE"] = all_elements_mse
            sheet_name = labels[i]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 将summary数据写入summary sheet
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        
def parse_group_and_label(fname: str):
    """
    从文件名中解析分组前缀和 label。
    规则：<GROUP>_<LABEL>.npz，例如 MSE_50.npz -> ('MSE', '50')
    允许带路径，自动取 basename。
    """
    base = os.path.basename(fname)
    if not base.lower().endswith(".npz"):
        raise ValueError(f"不是 npz 文件: {fname}")
    stem = base[:-4]  # 去掉 .npz
    if "_" not in stem:
        raise ValueError(f"文件名不含下划线，无法解析前缀与标签: {fname}")
    group, label = stem.split("_", 1)
    # 只保留 label 中的数字（例如 '50', '1600'）；如果含别的字符，也尽量提取末尾数字块
    m = re.search(r"(\d+)$", label)
    if m:
        label_clean = m.group(1)
    else:
        label_clean = label  # 兜底，不强制数字
    return group, label_clean
            
def process_all(file_names: Iterable[str], base_dir: str | None = None):
    """
    自动分组并批量后处理：
      1) 按 <GROUP>_<LABEL>.npz 分组；
      2) 同组内按 LABEL（数字优先，否则字典序）排序；
      3) 用 read_results(data_paths) 一次性读取同组文件；
      4) 调用 write_origin_data(results, labels, group_flag) 生成 post_<group>.xlsx
    """
    # 收集：group -> [(label, fullpath)]
    groups: Dict[str, List[Tuple[str, str]]] = {}

    for fn in file_names:
        if not fn or not fn.endswith(".npz"):
            continue
        fullpath = os.path.join(base_dir, fn) if base_dir and not os.path.isabs(fn) else fn
        if not os.path.exists(fullpath):
            print(f"[SKIP] 文件不存在: {fullpath}")
            continue
        try:
            group, label = parse_group_and_label(fullpath)
        except Exception as e:
            print(f"[WARN] 解析失败，跳过 {fullpath}: {e}")
            continue
        groups.setdefault(group, []).append((label, fullpath))

    if not groups:
        print("[WARN] 没有可处理的分组。")
        return

    # 逐组处理
    for group_flag, items in sorted(groups.items()):
        # 排序键：可转 int 的放前面并按数值排；否则按字符串
        def _label_key(t: Tuple[str, str]):
            lbl = t[0]
            try:
                return (0, int(lbl))
            except ValueError:
                return (1, lbl)

        items_sorted = sorted(items, key=_label_key)

        # 检查 label 重复
        seen = set()
        dups = [lbl for lbl, _ in items_sorted if (lbl in seen) or seen.add(lbl)]
        if dups:
            print(f"[WARN] 分组 {group_flag} 含重复 label：{sorted(set(dups))}")

        labels = [lbl for lbl, _ in items_sorted]
        data_paths = [p for _, p in items_sorted]

        print(f"[INFO] 处理分组 {group_flag}: {len(data_paths)} 个文件, labels={labels}")

        # 用你的读取器一次性读取该组所有文件
        results, _ = read_results(data_paths)  # -> List[results]，顺序需与 data_paths 对齐

        # 生成该组 Excel
        write_origin_data(results, labels, group_flag)

    print("[DONE] 全部分组处理完成。")
        
def which_group(group_flag):
    if group_flag == "_recalc":
        file_names = [
            # 'MSE_50.npz',
            # 'MSE_100.npz',
            # 'MSE_200.npz',
            # 'MSE_400.npz',
            # 'MSE_800.npz',
            # 'MSE_1600.npz',
            # 'MSE_2400.npz',
            # 'MSE_3200.npz',
            
            # 'KL_50.npz',
            # 'KL_100.npz',
            # 'KL_200.npz',
            # 'KL_400.npz',
            # 'KL_800.npz',
            # 'KL_1600.npz',
            
            # 'MAE_50.npz',
            # 'MAE_100.npz',
            # 'MAE_200.npz',
            # 'MAE_400.npz',
            # 'MAE_800.npz',
            # 'MAE_1600.npz',
            
            # 'RMSE_50.npz',
            # 'RMSE_100.npz',
            # 'RMSE_200.npz',
            # 'RMSE_400.npz',
            # 'RMSE_800.npz',
            # 'RMSE_1600.npz',
            
            # 'QMSE_50.npz',
            # 'QMSE_100.npz',
            # 'QMSE_200.npz',
            # 'QMSE_400.npz',
            # 'QMSE_800.npz',
            # 'QMSE_1600.npz',
            
            # 'QMAE_50.npz',
            # 'QMAE_100.npz',
            # 'QMAE_200.npz',
            # 'QMAE_400.npz',
            # 'QMAE_800.npz',
            # 'QMAE_1600.npz',
            
            # 'QRMSE_50.npz',
            # 'QRMSE_100.npz',
            # 'QRMSE_200.npz',
            # 'QRMSE_400.npz',
            # 'QRMSE_800.npz',
            # 'QRMSE_1600.npz',
            
            # 'xMSE_50.npz',
            # 'xMSE_100.npz',
            # 'xMSE_200.npz',
            # 'xMSE_400.npz',
            # 'xMSE_800.npz',
            # 'xMSE_1600.npz',
            
            # 'xMAE_50.npz',
            # 'xMAE_100.npz',
            # 'xMAE_200.npz',
            # 'xMAE_400.npz',
            # 'xMAE_800.npz',
            # 'xMAE_1600.npz',
            
            # 'xRMSE_50.npz',
            # 'xRMSE_100.npz',
            # 'xRMSE_200.npz',
            # 'xRMSE_400.npz',
            # 'xRMSE_800.npz',
            # 'xRMSE_1600.npz',
            
            # 'wbMSE_50.npz',
            # 'wbMSE_100.npz',
            # 'wbMSE_200.npz',
            # 'wbMSE_400.npz',
            # 'wbMSE_800.npz',
            # 'wbMSE_1600.npz',
            
            # 'wbMAE_50.npz',
            # 'wbMAE_100.npz',
            # 'wbMAE_200.npz',
            # 'wbMAE_400.npz',
            # 'wbMAE_800.npz',
            # 'wbMAE_1600.npz',
            
            # 'wbRMSE_50.npz',
            # 'wbRMSE_100.npz',
            # 'wbRMSE_200.npz',
            # 'wbRMSE_400.npz',
            # 'wbRMSE_800.npz',
            # 'wbRMSE_1600.npz',
            
            # 'kc1_50.npz',
            # 'kc1_100.npz',
            # 'kc1_200.npz',
            # 'kc1_400.npz',
            # 'kc1_800.npz',
            # 'kc1_1600.npz',
            # 'kc1_2400.npz',
            # 'kc1_3200.npz',
            
            # 'kP1_50.npz',
            # 'kP1_100.npz',
            # 'kP1_200.npz',
            # 'kP1_400.npz',
            # 'kP1_800.npz',
            # 'kP1_1600.npz',
            # 'kP1_2400.npz',
            # 'kP1_3200.npz',
            
            # 'kv_50.npz',
            # 'kv_100.npz',
            # 'kv_200.npz',
            # 'kv_400.npz',
            # 'kv_800.npz',
            # 'kv_1600.npz',
            # 'kv_2400.npz',
            # 'kv_3200.npz',
            
            # 'nomul_50.npz',
            # 'nomul_100.npz',
            # 'nomul_200.npz',
            # 'nomul_400.npz',
            # 'nomul_800.npz',
            # 'nomul_1600.npz',
            # 'nomul_2400.npz',
            # 'nomul_3200.npz',
            
            # 'nonoise_50.npz',
            # 'nonoise_100.npz',
            # 'nonoise_200.npz',
            # 'nonoise_400.npz',
            # 'nonoise_800.npz',
            # 'nonoise_1600.npz',
            # 'nonoise_2400.npz',
            # 'nonoise_3200.npz',
            
            # 'random_50.npz',
            # 'random_100.npz',
            # 'random_200.npz',
            # 'random_400.npz',
            # 'random_800.npz',
            # 'random_1600.npz',
            # 'random_2400.npz',
            # 'random_3200.npz',
            
            # 'se4_50.npz',
            # 'se4_100.npz',
            # 'se4_200.npz',
            # 'se4_400.npz',
            # 'se4_800.npz',
            # 'se4_1600.npz',
            # 'se4_2400.npz',
            # 'se4_3200.npz',
            
            # 'se16_50.npz',
            # 'se16_100.npz',
            # 'se16_200.npz',
            # 'se16_400.npz',
            # 'se16_800.npz',
            # 'se16_1600.npz',
            # 'se16_2400.npz',
            # 'se16_3200.npz',
            
            # 'se64_50.npz',
            # 'se64_100.npz',
            # 'se64_200.npz',
            # 'se64_400.npz',
            # 'se64_800.npz',
            # 'se64_1600.npz',
            # 'se64_2400.npz',
            # 'se64_3200.npz',
            
            # 'se256_50.npz',
            # 'se256_100.npz',
            # 'se256_200.npz',
            # 'se256_400.npz',
            # 'se256_800.npz',
            # 'se256_1600.npz',
            # 'se256_2400.npz',
            # 'se256_3200.npz',
            
            # 'se1024_50.npz',
            # 'se1024_100.npz',
            # 'se1024_200.npz',
            # 'se1024_400.npz',
            # 'se1024_800.npz',
            # 'se1024_1600.npz',
            # 'se1024_2400.npz',
            # 'se1024_3200.npz',
            
            # 'se4096_50.npz',
            # 'se4096_100.npz',
            # 'se4096_200.npz',
            # 'se4096_400.npz',
            # 'se4096_800.npz',
            # 'se4096_1600.npz',
            # 'se4096_2400.npz',
            # 'se4096_3200.npz',
            
            # 'kP2_50.npz',
            # 'kP2_200.npz',
            # 'kP2_200.npz',
            # 'kP2_400.npz',
            # 'kP2_800.npz',
            # 'kP2_1600.npz',
            # 'kP2_2400.npz',
            # 'kP2_3200.npz',
            
            'kc0_50.npz',
            'kc0_200.npz',
            'kc0_200.npz',
            'kc0_400.npz',
            'kc0_800.npz',
            'kc0_1600.npz',
            'kc0_2400.npz',
            'kc0_3200.npz',
            ]
        labels = [
            '50',
            '100',
            '200',
            '400',
            '800',
            '1600',
            '2400',
            '3200',
            ]
    elif group_flag == "seed800":
        file_names = [
            'MSE_800.npz',
            'se4_800.npz',
            'se16_800.npz',
            'se64_800.npz',
            'se256_800.npz',
            'se1024_800.npz',
            'se4096_800.npz',
            ]
        labels = [
            '1',
            '4',
            '16',
            '64',
            '256',
            '1024',
            '4096',
            ]
    elif group_flag == "seed3200":
        file_names = [
            'MSE_3200.npz',
            'se4_3200.npz',
            'se16_3200.npz',
            'se64_3200.npz',
            'se256_3200.npz',
            'se1024_3200.npz',
            'se4096_3200.npz',
            ]
        labels = [
            '1',
            '4',
            '16',
            '64',
            '256',
            '1024',
            '4096',
            ]
    elif group_flag == "target800":
        file_names = [
            'KL_800.npz',
            'MSE_800.npz',
            'MAE_800.npz',
            'RMSE_800.npz',
            'QMSE_800.npz',
            'QMAE_800.npz',
            'QRMSE_800.npz',
            'xMSE_800.npz',
            'xMAE_800.npz',
            'xRMSE_800.npz',
            'wbMSE_800.npz',
            'wbMAE_800.npz',
            'wbRMSE_800.npz',
            ]
        labels = [
            'KL',
            'MSE',
            'MAE',
            'RMSE',
            'QMSE',
            'QMAE',
            'QRMSE',
            'xMSE',
            'xMAE',
            'xRMSE',
            'wbMSE',
            'wbMAE',
            'wbRMSE',
            ]
    elif group_flag == "MSE_nomul":
        file_names = [
            'MSE_nm_50.npz',
            'MSE_nm_100.npz',
            'MSE_nm_200.npz',
            'MSE_nm_400.npz',
            'MSE_nm_800.npz',
            'MSE_nm_1600.npz',
            'MSE_nm_2400.npz',
            'MSE_nm_3200.npz',
            ]
        labels = [
            '50',
            '100',
            '200',
            '400',
            '800',
            '1600',
            '2400',
            '3200',
            ]
    elif group_flag == "pearson":
        file_names = [
            # 'kva_50.npz',
            # 'kva_100.npz',
            # 'kva_200.npz',
            # 'kva_400.npz',
            # 'kva_800.npz',
            # 'kva_1600.npz',
            # 'kva_2400.npz',
            # 'kva_3200.npz',
            # 'kva_4800.npz',
            # 'kva_6400.npz',
            
            # 'MSEa_50.npz',
            # 'MSEa_100.npz',
            # 'MSEa_200.npz',
            # 'MSEa_400.npz',
            # 'MSEa_800.npz',
            # 'MSEa_1600.npz',
            # 'MSEa_2400.npz',
            # 'MSEa_3200.npz',
            # 'MSEa_4800.npz',
            # 'MSEa_6400.npz',
            # 'kva_5.npz','kva_10.npz','kva_15.npz','kva_20.npz','kva_25.npz','kva_30.npz','kva_35.npz','kva_40.npz','kva_45.npz','kva_50.npz',
            # 'kva_55.npz','kva_60.npz','kva_65.npz','kva_70.npz','kva_75.npz','kva_80.npz','kva_85.npz','kva_90.npz','kva_95.npz','kva_100.npz',
            # 'kva_110.npz','kva_120.npz','kva_130.npz','kva_140.npz','kva_150.npz','kva_160.npz','kva_170.npz','kva_180.npz','kva_190.npz','kva_200.npz',
            # 'kva_220.npz','kva_240.npz','kva_260.npz','kva_280.npz','kva_300.npz','kva_320.npz','kva_340.npz','kva_360.npz','kva_380.npz','kva_400.npz',
            # 'kva_440.npz','kva_480.npz','kva_520.npz','kva_560.npz','kva_600.npz','kva_640.npz','kva_680.npz','kva_720.npz','kva_760.npz','kva_800.npz',
            # 'kva_880.npz','kva_960.npz','kva_1040.npz','kva_1120.npz','kva_1200.npz','kva_1280.npz','kva_1360.npz','kva_1440.npz','kva_1520.npz','kva_1600.npz',
            # 'kva_1680.npz','kva_1760.npz','kva_1840.npz','kva_1920.npz','kva_2000.npz','kva_2080.npz','kva_2160.npz','kva_2240.npz','kva_2320.npz','kva_2400.npz',
            # 'kva_2480.npz','kva_2560.npz','kva_2640.npz','kva_2720.npz','kva_2800.npz','kva_2880.npz','kva_2960.npz','kva_3040.npz','kva_3120.npz','kva_3200.npz',
            # 'kva_3360.npz','kva_3520.npz','kva_3680.npz','kva_3840.npz','kva_4000.npz','kva_4160.npz','kva_4320.npz','kva_4480.npz','kva_4640.npz','kva_4800.npz',
            # 'kva_4960.npz','kva_5120.npz','kva_5280.npz','kva_5440.npz','kva_5600.npz','kva_5760.npz','kva_5920.npz','kva_6080.npz','kva_6240.npz','kva_6400.npz'
            
            'MSEa_5.npz','MSEa_10.npz','MSEa_15.npz','MSEa_20.npz','MSEa_25.npz','MSEa_30.npz','MSEa_35.npz','MSEa_40.npz','MSEa_45.npz','MSEa_50.npz',
            'MSEa_55.npz','MSEa_60.npz','MSEa_65.npz','MSEa_70.npz','MSEa_75.npz','MSEa_80.npz','MSEa_85.npz','MSEa_90.npz','MSEa_95.npz','MSEa_100.npz',
            'MSEa_110.npz','MSEa_120.npz','MSEa_130.npz','MSEa_140.npz','MSEa_150.npz','MSEa_160.npz','MSEa_170.npz','MSEa_180.npz','MSEa_190.npz','MSEa_200.npz',
            'MSEa_220.npz','MSEa_240.npz','MSEa_260.npz','MSEa_280.npz','MSEa_300.npz','MSEa_320.npz','MSEa_340.npz','MSEa_360.npz','MSEa_380.npz','MSEa_400.npz',
            'MSEa_440.npz','MSEa_480.npz','MSEa_520.npz','MSEa_560.npz','MSEa_600.npz','MSEa_640.npz','MSEa_680.npz','MSEa_720.npz','MSEa_760.npz','MSEa_800.npz',
            'MSEa_880.npz','MSEa_960.npz','MSEa_1040.npz','MSEa_1120.npz','MSEa_1200.npz','MSEa_1280.npz','MSEa_1360.npz','MSEa_1440.npz','MSEa_1520.npz','MSEa_1600.npz',
            'MSEa_1680.npz','MSEa_1760.npz','MSEa_1840.npz','MSEa_1920.npz','MSEa_2000.npz','MSEa_2080.npz','MSEa_2160.npz','MSEa_2240.npz','MSEa_2320.npz','MSEa_2400.npz',
            'MSEa_2480.npz','MSEa_2560.npz','MSEa_2640.npz','MSEa_2720.npz','MSEa_2800.npz','MSEa_2880.npz','MSEa_2960.npz','MSEa_3040.npz','MSEa_3120.npz','MSEa_3200.npz',
            'MSEa_3360.npz','MSEa_3520.npz','MSEa_3680.npz','MSEa_3840.npz','MSEa_4000.npz','MSEa_4160.npz','MSEa_4320.npz','MSEa_4480.npz','MSEa_4640.npz','MSEa_4800.npz',
            'MSEa_4960.npz','MSEa_5120.npz','MSEa_5280.npz','MSEa_5440.npz','MSEa_5600.npz','MSEa_5760.npz','MSEa_5920.npz','MSEa_6080.npz','MSEa_6240.npz','MSEa_6400.npz'
            
            ]
        labels = [
            # '10',
            # '50',
            # '100',
            # '200',
            # '400',
            # '800',
            # '1600',
            # '2400',
            # '3200',
            # '4800',
            # '6400',
            '5','10','15','20','25','30','35','40','45','50',
            '55','60','65','70','75','80','85','90','95','100',
            '110','120','130','140','150','160','170','180','190','200',
            '220','240','260','280','300','320','340','360','380','400',
            '440','480','520','560','600','640','680','720','760','800',
            '880','960','1040','1120','1200','1280','1360','1440','1520','1600',
            '1680','1760','1840','1920','2000','2080','2160','2240','2320','2400',
            '2480','2560','2640','2720','2800','2880','2960','3040','3120','3200',
            '3360','3520','3680','3840','4000','4160','4320','4480','4640','4800',
            '4960','5120','5280','5440','5600','5760','5920','6080','6240','6400'
            ]
    elif group_flag == "no_noise":
        file_names = [
            # 'nna_10.npz',
            # 'nna_50.npz',
            # 'nna_100.npz',
            # 'nna_200.npz',
            # 'nna_400.npz',
            # 'nna_800.npz',
            # 'nna_1600.npz',
            # 'nna_2400.npz',
            # 'nna_3200.npz',
            # 'nna_4800.npz',
            # 'nna_6400.npz',
            
            'nna_5.npz','nna_10.npz','nna_15.npz','nna_20.npz','nna_25.npz','nna_30.npz','nna_35.npz','nna_40.npz','nna_45.npz','nna_50.npz',
            'nna_55.npz','nna_60.npz','nna_65.npz','nna_70.npz','nna_75.npz','nna_80.npz','nna_85.npz','nna_90.npz','nna_95.npz','nna_100.npz',
            'nna_110.npz','nna_120.npz','nna_130.npz','nna_140.npz','nna_150.npz','nna_160.npz','nna_170.npz','nna_180.npz','nna_190.npz','nna_200.npz',
            'nna_220.npz','nna_240.npz','nna_260.npz','nna_280.npz','nna_300.npz','nna_320.npz','nna_340.npz','nna_360.npz','nna_380.npz','nna_400.npz',
            'nna_440.npz','nna_480.npz','nna_520.npz','nna_560.npz','nna_600.npz','nna_640.npz','nna_680.npz','nna_720.npz','nna_760.npz','nna_800.npz',
            'nna_880.npz','nna_960.npz','nna_1040.npz','nna_1120.npz','nna_1200.npz','nna_1280.npz','nna_1360.npz','nna_1440.npz','nna_1520.npz','nna_1600.npz',
            'nna_1680.npz','nna_1760.npz','nna_1840.npz','nna_1920.npz','nna_2000.npz','nna_2080.npz','nna_2160.npz','nna_2240.npz','nna_2320.npz','nna_2400.npz',
            'nna_2480.npz','nna_2560.npz','nna_2640.npz','nna_2720.npz','nna_2800.npz','nna_2880.npz','nna_2960.npz','nna_3040.npz','nna_3120.npz','nna_3200.npz',
            'nna_3360.npz','nna_3520.npz','nna_3680.npz','nna_3840.npz','nna_4000.npz','nna_4160.npz','nna_4320.npz','nna_4480.npz','nna_4640.npz','nna_4800.npz',
            'nna_4960.npz','nna_5120.npz','nna_5280.npz','nna_5440.npz','nna_5600.npz','nna_5760.npz','nna_5920.npz','nna_6080.npz','nna_6240.npz','nna_6400.npz'
            ]
        labels = [
            # '10',
            # '50',
            # '100',
            # '200',
            # '400',
            # '800',
            # '1600',
            # '2400',
            # '3200',
            # '4800',
            # '6400',
            '5','10','15','20','25','30','35','40','45','50',
            '55','60','65','70','75','80','85','90','95','100',
            '110','120','130','140','150','160','170','180','190','200',
            '220','240','260','280','300','320','340','360','380','400',
            '440','480','520','560','600','640','680','720','760','800',
            '880','960','1040','1120','1200','1280','1360','1440','1520','1600',
            '1680','1760','1840','1920','2000','2080','2160','2240','2320','2400',
            '2480','2560','2640','2720','2800','2880','2960','3040','3120','3200',
            '3360','3520','3680','3840','4000','4160','4320','4480','4640','4800',
            '4960','5120','5280','5440','5600','5760','5920','6080','6240','6400'
            ]
    elif group_flag == "samplers":
        file_names = [
            'GP_50.npz',
            'GP_100.npz',
            'GP_200.npz',
            'GP_400.npz',
            'GP_800.npz',
            'GP_1600.npz',
            
            # 'TPE_50.npz',
            # 'TPE_100.npz',
            # 'TPE_200.npz',
            # 'TPE_400.npz',
            # 'TPE_800.npz',
            # 'TPE_1600.npz',
            
            # 'QMC_50.npz',
            # 'QMC_100.npz',
            # 'QMC_200.npz',
            # 'QMC_400.npz',
            # 'QMC_800.npz',
            # 'QMC_1600.npz',
            
            # 'NSGA_50.npz',
            # 'NSGA_100.npz',
            # 'NSGA_200.npz',
            # 'NSGA_400.npz',
            # 'NSGA_800.npz',
            # 'NSGA_1600.npz',
            
            # 'TPE_50.npz',
            # 'TPE_100.npz',
            # 'TPE_200.npz',
            # 'TPE_400.npz',
            # 'TPE_800.npz',
            # 'TPE_1600.npz',
            ]
        labels = [
            '50',
            '100',
            '200',
            '400',
            '800',
            '1600',
            ]
    return file_names, labels
        

#%% PRE-POCESSING
def read_results(data_paths):
    if group_flag == "MSE_nomul":
        ori_mse_path = os.path.join(results_pth, pbe_type, 'ori_mse_no_multi.npz')
    else:
        ori_mse_path = os.path.join(results_pth, pbe_type, 'ori_mse.npz')
    ori_mse = np.load(ori_mse_path,allow_pickle=True)['results']
    ori_mse_tem = np.empty(ori_mse.shape, dtype=object)
    ori_mse_tem[:,0] = ori_mse[:,0]
    # if group_flag == "no_multi":
    #     ori_mse_tem[78,0] = 0
    # else:
    #     # Unicluster have some problem on this results(not convergent)
    #     # Used results from my PC instead
    #     ori_mse_tem[78,0] = 1.267987945506276e-05
        
    for i, data_name in enumerate(ori_mse[:,1]):
        ori_mse_tem[i, 1] = get_kernels_form_data_name(data_name)
        
    post_results = []
    elapsed_time = []
    for data_path in data_paths:
        data = np.load(data_path,allow_pickle=True)
        results=data['results']
        if 'time' in data:
            tem_time = data['time']
        else:
            tem_time = 0
        results_tem = np.empty((len(results), 4), dtype=object)
        if results.ndim == 1:
            for i in range(results.shape[0]):
                # results_tem[i, 0] = results[i, 0]['opt_score']
                # results_tem[i, 1] = results[i, 0]['opt_parameters']
                # results_tem[i, 2] = results[i, 1]
                results_tem[i, 0] = results[i]['opt_score']
                results_tem[i, 1] = results[i]['opt_params']
                del results_tem[i, 1]['actor_wait']
                del results_tem[i, 1]['wait_time']
                del results_tem[i, 1]['max_reuse']
                del results_tem[i, 1]['__exp_paths']
                del results_tem[i, 1]['__known_params']
                filename = results[i]['file_path'] 
                results_tem[i, 3] = filename
                if isinstance(filename, list):
                    data_name = filename[0]
                else:
                    data_name = filename
                results_tem[i, 2] = get_kernels_form_data_name(data_name)
        else:
            for i in range(results.shape[0]):
                results_tem[i, 0] = results[i,-2]
                results_tem[i, 1] = results[i,1]
                # del results_tem[i, 1]['actor_wait']
                # del results_tem[i, 1]['wait_time']
                # del results_tem[i, 1]['max_reuse']
                # del results_tem[i, 1]['__exp_paths']
                # del results_tem[i, 1]['__known_params']
                # results_tem[i, 2] = results[i,2]
                results_tem[i, 3] = results[i,-1]
                if isinstance(results_tem[i, 3], list):
                    data_name = results_tem[i, 3][0]
                else:
                    data_name = results_tem[i, 3]
                results_tem[i, 2] = get_kernels_form_data_name(data_name)
        ## convert absolute mse into relative mse
        if group_flag != "no_noise":
            results_tem = calc_rel_mse(results_tem, ori_mse_tem)
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
    if data_name.lower().endswith(".xlsx"):
        param_str = param_str.rsplit('.', 1)[0] 
    params = param_str.split('_')
    converted_params = [float(param) if '.' in param or 'e' in param.lower() else int(param) for param in params]
    CORR_BETA = converted_params[0]
    alpha_prim = np.array(converted_params[1:4])
    # kernels['corr_agg'] = CORR_BETA * alpha_prim
    kernels['corr_agg_0'] = CORR_BETA * alpha_prim[0]
    kernels['corr_agg_1'] = CORR_BETA * alpha_prim[1]
    kernels['corr_agg_2'] = CORR_BETA * alpha_prim[2]
    kernels['pl_v'] = converted_params[4]
    kernels['pl_P1'] = converted_params[5]
    kernels['pl_P2'] = converted_params[6]
    kernels['pl_P3'] = converted_params[7]
    kernels['pl_P4'] = converted_params[8]
    return kernels

def calc_rel_mse(results_tem, ori_mse_tem):
    for i in range(results_tem.shape[0]):
        current_dict = results_tem[i, 2]  
        for j in range(ori_mse_tem.shape[0]):
            if compare_dicts(ori_mse_tem[j, 1], current_dict): 
                results_tem[i, 0] = float(results_tem[i, 0]) / float(ori_mse_tem[j, 0])
                break  
    return results_tem

def compare_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2): 
                return False
        else:
            if val1 != val2: 
                return False
    return True

def get_search_range(kernel):
    # 获取kernel对应的子字典
    param_info = conf.config["opt_params"][kernel]
    
    # 检查子字典是否存在
    if not param_info:
        raise ValueError(f"Key '{kernel}' not found in 'opt_params'.")
    
    # 获取bounds和log_scale
    bounds = param_info['bounds']
    log_scale = param_info['log_scale']
    
    # 如果log_scale为True，转换为10的次幂
    if log_scale:
        min_val, max_val = 10 ** bounds[0], 10 ** bounds[1]
    else:
        min_val, max_val = bounds
    
    # 返回最大值和最小值
    return max(max_val, min_val), min(max_val, min_val)
#%% VISUALIZE KERNEL DIFFERENCE
#%%%VISUALZE IN RADAR
def visualize_diff_mean_radar(results, data_labels):
    diff_mean = []

    for i, result in enumerate(results):
        diff_mean_tem =[]
        kernels_labels = []
        diff_kernels, _, _ = calc_diff(result)
        for key, array in diff_kernels.items():
            avg  = np.mean(array)
            diff_mean_tem.append(avg)
            kernels_labels.append(key) 
        diff_mean.append(np.array(diff_mean_tem))
        title = '$\overline{k_{j,\delta}}$'
    radar_chart(diff_mean, data_labels, kernels_labels, title)

    
def radar_chart(data, data_labels, kernels_labels, title):
    # Number of variables
    num_vars = len(kernels_labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, d in enumerate(data):
        values = d.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=data_labels[i])

    # Draw one axe per variable + add labels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(kernels_labels)

    plt.title(title, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()

#%%%VISUALZE IN BLOCK
def visualize_diff_kernel_value_old(result, eval_kernels, log_axis=False):
    diff_kernels, opt_kernels, ori_kernels = calc_diff(result)

    pt.plot_init(scl_a4=2,figsze=[6,4*2,4,8*2],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.5)
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

def visualize_diff_kernel_value(result, eval_kernels, log_axis=False):
    diff_kernels, opt_kernels, ori_kernels = calc_diff(result)

    fig=plt.figure()    
    ax=fig.add_subplot(1,1,1)
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    width_factor = 0.0
    for kernel in eval_kernels:
        color = next(colors)
        
        ori_values = np.array(ori_kernels[kernel]).reshape(-1, 1)
        opt_values = np.array(opt_kernels[kernel])
        
        mean_opt = []
        std_opt = []
        ori_value_list = []
        width_factor += (ori_values.max() - ori_values.min()) / 40
        
        # Iterate over each unique original kernel value
        for ori_value in np.unique(ori_values):
            opt_values_for_ori = opt_values[ori_values.flatten() == ori_value]
            
            # Calculate statistics
            q25, q75 = np.percentile(opt_values_for_ori, [25, 75])
            mean_val = np.mean(opt_values_for_ori)
            std_val = np.std(opt_values_for_ori)
            
            # Draws a rectangle ranging from 25% to 75%
            ax.fill_between([ori_value - width_factor, ori_value + width_factor], q25, q75, color=color, alpha=0.3)
            
            # Record the mean and standard deviation
            mean_opt.append(mean_val)
            std_opt.append(std_val)
            ori_value_list.append(ori_value)
        
        
        # Plot the average and right value
        ax.plot(ori_value_list, mean_opt, label=f'{kernel} (mean)', color=color, marker='o')
        ax.plot(ori_value_list, ori_value_list, label=f'{kernel} (right)', color='k', marker='v')
        
        # Mark the standard deviation range at the mean
        ax.errorbar(ori_value_list, mean_opt, yerr=std_opt, fmt='none', ecolor=color, capsize=5)
        
    ax.set_xlabel('Original Kernel Values')
    ax.set_ylabel('Optimized Kernel Values')
    if log_axis:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.grid('minor')
    plt.title('Optimized Kernel Values vs. Original Kernel Values')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return diff_kernels
    
#%% RETURN PSD IN FRAME/ANIMATION
def visualize_PSD(variable, pbe_type, one_frame):
    data_path = r"C:\Users\px2030\Code\PSD_opt\pypbe\data"
    opt = OptBase(data_path=data_path)
    file_names = [os.path.basename(file_path) for file_path in variable[3]]
    exp_data_paths = [os.path.join(data_path, file_name) for file_name in file_names]
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
    ## 对于不是使用MSE或者不同权重计算的数据，需要让calc_criteria为True运行以下，重新计算MSE
    ## npz数据会被重新生成，格式会有所更改，然后就可以直接使用了，对应地读取和修改已经在
    ## 读入文件的方函数中写好了
    # diff_type = 'rel'
    # diff_type = 'abs'
    diff_type = 'scaled'
    
    my_pth = os.path.dirname( __file__ )
    results_pth = os.path.join(my_pth, 'Parameter_study')
    remove_small_results = False
    calc_criteria = False
    visualize_sampler_iter_flag = False
    export_in_origin = True

    # pbe_type = 'agglomeration'
    # pbe_type = 'breakage'
    pbe_type = 'mix'
    # pbe_type = 'test'
    
    group_flag = "_recalc"
    # group_flag = "seed800"
    # group_flag = "seed3200"
    # group_flag = "target800"
    # group_flag = "MSE_nomul"
    # group_flag = "pearson"
    # group_flag = "no_noise"
    # group_flag = "samplers"
    
    # results_mse = calc_ori_mse()
    
    file_names, labels = which_group(group_flag=group_flag)
    
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
    
    data_paths = [os.path.join(results_pth, pbe_type, file_name) for file_name in file_names]
    # 'results' saves the results of all reading files. 
    # The first column in each result is the value of the optimized criteria. 
    # The second column is the value of the optimization kernels. 
    # The third column is the kernel value (target value) of the original pbe.
    results, elapsed_time = read_results(data_paths)
    
    if calc_criteria:
        results = calc_save_PSD_delta(results, data_paths)
        # delta,opt = calc_save_PSD_delta_test(results, data_paths)
    if remove_small_results:
        results = do_remove_small_results(results)
    if export_in_origin:
        # process_all(file_names, os.path.join(results_pth, pbe_type))
        write_origin_data(results, labels, group_flag)
    
    pt.plot_init(scl_a4=1,figsze=[6.4*2,4.8*2],lnewdth=0.8,mrksze=5,use_locale=True,scl=2)
    
    if visualize_sampler_iter_flag:
        pearson_corrs = visualize_sampler_iter()
        
    visualize_diff_mean(results, labels)
    
    # # kernel: corr_agg_0, corr_agg_1, corr_agg_2, pl_v, pl_P1, pl_P2, pl_P3, pl_P4
    # result_to_analyse = results[-1]
    # if pbe_type == 'agglomeration' or pbe_type == 'mix':
    #     corr_agg_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['corr_agg_0','corr_agg_1','corr_agg_2'])
    # if pbe_type == 'breakage' or pbe_type == 'mix':
    #     pl_v_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_v'])
    #     pl_P13_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P1','pl_P3'], log_axis=False)
    #     pl_P24_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P2','pl_P4'])
    visualize_diff_mean_radar(results, labels)
    
    # pearson_corrs = visualize_correlation(results, labels)
    # pearson_corrs_list = []
    # spearman_corr_list = []
    # m_list = []
    # b_list = []
    # for result_to_analyse in results:
    #     pearson_corrs,m,b,spearman_corr = correlation_analysis(result_to_analyse,plot=True)
    #     pearson_corrs_list.append(pearson_corrs)
    #     spearman_corr_list.append(spearman_corr)
    #     m_list.append(m)
    #     b_list.append(b)
    
    # pearson_corrs_list, spearman_corr_list = correlation_analysis_sliding(results)
    
    
    # pearson_corrs_ar=np.array(pearson_corrs_list)
    # spearman_corr_ar=np.array(spearman_corr_list)
    # visualize_diff_kernel_mse(result_to_analyse)
    
    # variable_to_analyse = result_to_analyse[1]
    # one_frame = False
    # # calc_init = False
    # t_return = -1
    # fps = 5
    # visualize_PSD(variable_to_analyse, pbe_type, one_frame)

