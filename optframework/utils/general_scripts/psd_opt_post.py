# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import sys, os
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
from scipy.stats import pearsonr

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
                scaled_diff
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
    scale_y2 = min(diff_std_mse / diff_std_kernels)
    y1lim = [min_lim/scale_y2, max_lim/scale_y2] 
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
    return pearson_corr,m,b

def calc_save_PSD_delta(results, data_paths):
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    data_path = r"C:\Users\px2030\Code\PSD_opt\pypbe\data"
    opt = OptBase(data_path=data_path)
    for i, result in enumerate(results):
        func_list = []
        # delta = np.zeros(len(result))
        # path = np.empty(len(result),dtype=str)
        for j, _ in enumerate(result):
            if i==1 and j == 0:
                variable = result[j]
                file_names = [os.path.basename(file_path) for file_path in variable[3]]
                exp_data_paths = [os.path.join(data_path, file_name) for file_name in file_names]
                func_list.append((variable[1], exp_data_paths))
                # delta, path = opt.calc_PSD_delta(variable[1], exp_data_paths)
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
        
def calc_ori_mse():
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    data_path = r"C:\Users\px2030\Code\PSD_opt\pypbe\data"
    opt = OptBase(data_path=data_path)
    
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
                                exp_data_path = os.path.join(data_path, data_name)
                                exp_data_paths = [
                                    exp_data_path,
                                    exp_data_path.replace(".xlsx", "_NM.xlsx"),
                                    exp_data_path.replace(".xlsx", "_M.xlsx")
                                ]
                                # print(data_name)
                                # results = opt.calc_PSD_delta(ori_params, exp_data_paths)
                                func_list.append((ori_params,exp_data_paths))
    pool = multiprocessing.Pool()
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
        
def which_group(group_flag):
    if group_flag == "iter":    
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_50.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_100.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_200.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_1000.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_1200.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_1600.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_2400.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_3200.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_6400.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1000',
            'iter_1200',
            'iter_1600',
            'iter_2400',
            'iter_3200',
            'iter_6400',
            ]
    if group_flag == "iter_Cmaes":    
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_50.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_100.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_200.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_1600.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_2400.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_3200.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_4000.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_4800.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_5000.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_6400.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1600',
            'iter_2400',
            'iter_3200',
            'iter_4000',
            'iter_4800',
            'iter_5000',
            # 'iter_6400',
            ]
    elif group_flag == "sampler400":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_NSGA_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_QMC_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_TPE_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400.npz',
            ]
        labels = [
            'HEBO',
            'GP',
            'NSGA',
            'QMC',
            'TPS',
            'Cmaes',
            ]
    elif group_flag == "sampler800":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_NSGA_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_QMC_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_TPE_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_800.npz',
            ]
        labels = [
            'HEBO',
            'GP',
            'NSGA',
            'QMC',
            'TPS',
            'Cmaes',
            ]   
    elif group_flag == "sampler400rand":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400random.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400random.npz',
            'multi_[(\'q3\', \'MSE\')]_NSGA_wight_1_iter_400random.npz',
            'multi_[(\'q3\', \'MSE\')]_QMC_wight_1_iter_400random.npz',
            'multi_[(\'q3\', \'MSE\')]_TPE_wight_1_iter_400random.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400random.npz',
            ]
        labels = [
            'HEBO',
            'GP',
            'NSGA',
            'QMC',
            'TPS',
            'Cmaes',
            ]
    elif group_flag == "sampler800rand":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800randomt.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_800randomt.npz',
            'multi_[(\'q3\', \'MSE\')]_NSGA_wight_1_iter_800randomt.npz',
            'multi_[(\'q3\', \'MSE\')]_QMC_wight_1_iter_800randomt.npz',
            'multi_[(\'q3\', \'MSE\')]_TPE_wight_1_iter_800randomt.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_800randomt.npz',
            ]
        labels = [
            'HEBO',
            'GP',
            'NSGA',
            'QMC',
            'TPS',
            'Cmaes',
            ] 
    elif group_flag == "target400":
        file_names = [
            'multi_[(\'q3\', \'KL\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MAE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'RMSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'QQ3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'QQ3\', \'MAE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'QQ3\', \'RMSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'x_50\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'x_50\', \'MAE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'x_50\', \'RMSE\')]_HEBO_wight_1_iter_400.npz',
            # 'multi_[(\'q3\', \'MSE\'), (\'Q3\', \'MSE\'), (\'x_50\', \'MSE\')]_BO_wight_1_iter_400.npz',
            ]
        labels = [
            'q3_KL',
            'q3_MSE',
            'q3_MAE',
            'q3_RMSE',
            'QQ3_MSE',
            'QQ3_MAE',
            'QQ3_RMSE',
            'x_50_MSE',
            'x_50_MAE',
            'x_50_RMSE',
            # 'q3_MSE_Q3_MSE_x_50_MSE',
            ]
    elif group_flag == "target800":
        file_names = [
            'multi_[(\'q3\', \'KL\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'MAE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'q3\', \'RMSE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'QQ3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'QQ3\', \'MAE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'QQ3\', \'RMSE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'x_50\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'x_50\', \'MAE\')]_HEBO_wight_1_iter_800.npz',
            'multi_[(\'x_50\', \'RMSE\')]_HEBO_wight_1_iter_800.npz',
            # 'multi_[(\'q3\', \'MSE\'), (\'Q3\', \'MSE\'), (\'x_50\', \'MSE\')]_BO_wight_1_iter_400.npz',
            ]
        labels = [
            'q3_KL',
            'q3_MSE',
            'q3_MAE',
            'q3_RMSE',
            'QQ3_MSE',
            'QQ3_MAE',
            'QQ3_RMSE',
            'x_50_MSE',
            'x_50_MAE',
            'x_50_RMSE',
            # 'q3_MSE_Q3_MSE_x_50_MSE',
            ]
    elif group_flag == "no_multi":
        file_names = [
            '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_50.npz',
            '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_100.npz',
            '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_200.npz',
            '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800.npz',
            '[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_1600.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1600',
            ]
        
    elif group_flag == "no_noise":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_50no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_100no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_200no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_1600no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_3200no_noise.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1600',
            'iter_3200',
            ]

    elif group_flag == "no_noise_Cmaes":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_50no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_100no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_200no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_800no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_1600no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_2400no_noise.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_3000no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_3200no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_4000no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_4500no_noise.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_5000no_noise.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_5600no_noise.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_6400no_noise.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1600',
            'iter_2400',
            # 'iter_3000',
            'iter_3200',
            'iter_4000',
            'iter_4500',
            'iter_5000',
            # 'iter_5600',
            # 'iter_6400',
            ] 
        
    elif group_flag == "wight":    
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_2_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_2_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_3_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_4_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_5_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_6_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_7_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_8_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_9_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_10_iter_400.npz',
            ]
        labels = [
            'wight_1',
            'wight_2',
            'wight_2',
            'wight_3',
            'wight_4',
            'wight_5',
            'wight_6',
            'wight_7',
            'wight_8',
            'wight_9',
            'wight_10',
            ]
    elif group_flag == "P1P3":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_50P1P3.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_100P1P3.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_200P1P3.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400P1P3.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800P1P3.npz',
            # 'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_1600P1P3.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            # 'iter_1600',
            ]
    elif group_flag == "v":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_50v.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_100v.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_200v.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400v.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_800v.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_1600v.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1600',
            ]
     
    elif group_flag == "v_Cmaes":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_50v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_100v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_200v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_800v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_1600v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_2400v.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_3200v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_4000v.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_4500v.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_5000v.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_5600v.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_6400v.npz',

            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1600',
            'iter_2400',
            # 'iter_3200',
            'iter_4000',
            # 'iter_4500',
            'iter_5000',
            # 'iter_5600',
            # 'iter_6400',
            ]
        
    elif group_flag == "HEBOseed":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400seed2.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400seed4.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400seed8.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400seed16.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400seed32.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400seed64.npz',
            'multi_[(\'q3\', \'MSE\')]_HEBO_wight_1_iter_400seed128.npz',
            ]
        labels = [
            '1',
            '2',
            '4',
            '8',
            '16',
            '32',
            '64',
            '128',
            ]
    elif group_flag == "GPseed":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400seed2.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400seed4.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400seed8.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400seed16.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400seed32.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400seed64.npz',
            'multi_[(\'q3\', \'MSE\')]_GP_wight_1_iter_400seed128.npz',
            ]
        labels = [
            '1',
            '2',
            '4',
            '8',
            '16',
            '32',
            '64',
            '128',
            ]
    elif group_flag == "Cmaesseed":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed2.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed4.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed8.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed16.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed32.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed64.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed128.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed256.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed512.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed1024.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed2048.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed4096.npz',
            ]
        labels = [
            '1',
            '2',
            # '4',
            '8',
            '16',
            '32',
            # '64',
            '128',
            '256',
            '512',
            # '1024',
            '2048',
            '4096',
            ]
    elif group_flag == "Cmaesseed4096":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_50seed4096.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_100seed4096.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_200seed4096.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_400seed4096.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_800seed4096.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_1600seed4096.npz',
            ]
        labels = [
            'iter_50',
            'iter_100',
            'iter_200',
            'iter_400',
            'iter_800',
            'iter_1600',
            ]
    elif group_flag == "_test":
        file_names = [
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_50.npz',
            'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_50_20241125_125014.npz',
            # 'multi_[(\'q3\', \'MSE\')]_Cmaes_wight_1_iter_50_20241124_142740.npz',
            # 'multi_[(\'q3\', \'MSE\')]_TPE_wight_1_iter_50.npz',
            # 'multi_[(\'q3\', \'MSE\')]_TPE_wight_1_iter_50_20241125_112019.npz',
            ]
        labels = [
            'Cmaes',
            'Cmaes',
            # 'Cmaes',
            # 'TPE',
            # 'TPE',
            ]
    return file_names, labels
        

#%% PRE-POCESSING
def read_results(data_paths):
    if group_flag == "no_multi":
        ori_mse_path = os.path.join(results_pth, pbe_type, 'no_multi_ori_mse.npz')
    else:
        ori_mse_path = os.path.join(results_pth, pbe_type, 'ori_mse.npz')
    ori_mse = np.load(ori_mse_path,allow_pickle=True)['results']
    ori_mse_tem = np.empty(ori_mse.shape, dtype=object)
    ori_mse_tem[:,0] = ori_mse[:,0]
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
                results_tem[i, 2] = results[i,2]
                results_tem[i, 3] = results[i,3]
        ## convert absolute mse into relative mse   
        if not group_flag == "no_noise":
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
    
    # group_flag = "iter"
    # group_flag = "iter_Cmaes"
    # group_flag = "sampler400"
    # group_flag = "sampler800"
    # group_flag = "sampler400rand"
    # group_flag = "sampler800rand"
    # group_flag = "target400"
    # group_flag = "target800"
    # group_flag = "no_multi"
    # group_flag = "no_noise"
    # group_flag = "no_noise_Cmaes"
    # group_flag = "wight"
    # group_flag = "P1P3"
    # group_flag = "v"
    group_flag = "v_Cmaes"
    # group_flag = "HEBOseed"
    # group_flag = "GPseed"
    # group_flag = "Cmaesseed"
    # group_flag = "Cmaesseed4096"
    # group_flag = "_test"
    
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
        delta,opt = calc_save_PSD_delta(results, data_paths)
    if remove_small_results:
        results = do_remove_small_results(results)
    if export_in_origin:
        write_origin_data(results, labels, group_flag)
    
    pt.plot_init(scl_a4=1,figsze=[6.4*2,4.8*2],lnewdth=0.8,mrksze=5,use_locale=True,scl=2)
    
    if visualize_sampler_iter_flag:
        pearson_corrs = visualize_sampler_iter()
        
    visualize_diff_mean(results, labels)
    # # kernel: corr_agg_0, corr_agg_1, corr_agg_2, pl_v, pl_P1, pl_P2, pl_P3, pl_P4
    result_to_analyse = results[-3]
    # if pbe_type == 'agglomeration' or pbe_type == 'mix':
    #     corr_agg_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['corr_agg_0','corr_agg_1','corr_agg_2'])
    # if pbe_type == 'breakage' or pbe_type == 'mix':
    #     pl_v_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_v'])
    #     pl_P13_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P1','pl_P3'], log_axis=False)
    #     pl_P24_diff = visualize_diff_kernel_value(result_to_analyse, eval_kernels=['pl_P2','pl_P4'])
    
    visualize_diff_mean_radar(results, labels)
    # pearson_corrs = visualize_correlation(results, labels)
    pearson_corrs,m,b = correlation_analysis(result_to_analyse,plot=True)
    # visualize_diff_kernel_mse(result_to_analyse)
    
    # variable_to_analyse = result_to_analyse[1]
    # one_frame = False
    # # calc_init = False
    # t_return = -1
    # fps = 5
    # visualize_PSD(variable_to_analyse, pbe_type, one_frame)

