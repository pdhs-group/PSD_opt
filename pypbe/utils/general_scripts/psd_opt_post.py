# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_results(data_paths, labels):
    diff_means = []
    diff_stds = []
    diff_vars = []
    
    for data_path in data_paths:
        data = np.load(data_path,allow_pickle=True)
        results=data['results']
        
        delta_opt = results[:,0]
        opt_values = results[:,1]
        ori_params = results[:,2]
        
        if 'corr_agg' in opt_values[0]:
            opt_corr_agg = [dic['corr_agg'] for dic in opt_values]
            opt_corr_agg = np.array(opt_corr_agg)
            ori_alpha = [dic['alpha_prim'] for dic in ori_params]
            ori_alpha = np.array(ori_alpha)
            ori_beta = [dic['CORR_BETA'] for dic in ori_params]
            ori_beta = np.array(ori_beta)
            ori_corr_agg = ori_alpha * ori_beta.reshape(-1, 1)
            diff = abs(opt_corr_agg - ori_corr_agg)
            rel_diff = np.where(ori_corr_agg != 0, diff / ori_corr_agg, diff)
        
        data.close()
        
        if use_rel_diff:
            diff = rel_diff
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        diff_var = np.var(diff)

        diff_means.append(diff_mean)
        diff_stds.append(diff_std)
        diff_vars.append(diff_var)
    
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x_pos, diff_means, yerr=diff_stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    
    ax.set_ylabel('diff_mean')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__': 
    use_rel_diff = True
    data_paths1 = [
                  'Parameter_study/agg/multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_200.npz',
                  'Parameter_study/agg/multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_400.npz',
                  ]
    
    labels1 = [
              'iter_200',
              'iter_400',
                  ]
    
    data_paths2 = [
                  'Parameter_study/breakage/multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_200.npz',
                  'Parameter_study/breakage/multi_[(\'q3\', \'KL\')]_BO_wight_1_iter_400.npz',
                  ]
    
    labels2 = ['iter_200',
               'iter_400',
                  ]
    
    # data_paths3 = ['Parameter_study/multi_q3_BO_KL_wight_1_iter_400.npz',
    #               'Parameter_study/multi_Q3_BO_KL_wight_1_iter_400 (2).npz',
    #               'Parameter_study/q3_BO_KL_wight_1_iter_400.npz',
    #               'Parameter_study/Q3_BO_KL_wight_1_iter_400 (2).npz',
    #               ]
    
    # labels3 = ['q3_multi',
    #           'Q3_multi',
    #           'q3_simple',
    #           'Q3_simple',
    #               ]
    
    visualize_results(data_paths1, labels1)
    # visualize_results(data_paths2, labels2)
    # visualize_results(data_paths3, labels3)
