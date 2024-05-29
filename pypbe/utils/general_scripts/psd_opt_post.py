# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
import numpy as np
from sklearn.linear_model import LinearRegression
## For plots
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt  
import itertools

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
    # 画图
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
            
if __name__ == '__main__': 
    use_rel_diff = True
    results_pth = 'Parameter_study'
    pbe_type = 'mix'
    # pbe_type = 'breakage'
    # pbe_type = 'mix'
    file_names = [
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_50.npz',
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_100.npz',
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_200.npz',
        'multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_400.npz',
        ]
    labels = [
        'iter_50',
        'iter_100',
        'iter_200',
        'iter_400',
        ]
    
    data_paths = [os.path.join(results_pth, pbe_type, file_name) for file_name in file_names]
    # 'results' saves the results of all reading files. 
    # The first column in each result is the value of the optimized criteria. 
    # The second column is the value of the optimization kernels. 
    # The third column is the kernel value (target value) of the original pbe.
    results = read_results(data_paths)
    
    visualize_diff_mean(results, labels)
    ## kernel: corr_agg_0, corr_agg_1, corr_agg_2, pl_v, pl_P1, pl_P2, pl_P3, pl_P4
    visualize_diff_kernel_value(results[3], eval_kernels=['corr_agg_0','corr_agg_1','corr_agg_2'])
    visualize_diff_kernel_value(results[3], eval_kernels=['pl_v'])
    visualize_diff_kernel_value(results[3], eval_kernels=['pl_P1','pl_P3'])
    visualize_diff_kernel_value(results[3], eval_kernels=['pl_P2','pl_P4'])
    
    # visualize_results(data_paths1, labels1)
    # visualize_results(data_paths2, labels2)
    # visualize_results(data_paths3, labels3)
