# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_results(data_paths, labels):
    para_diff_means = []
    para_diff_stds = []
    para_diff_vars = []
    
    for data_path in data_paths:
        data = np.load(data_path)
        corr_beta_opt = data['corr_beta_opt']
        alpha_prim_opt = data['alpha_prim_opt']
        para_diff = data['para_diff']
        delta_opt = data['delta_opt']
        corr_agg = data['corr_agg']
        corr_agg_opt = data['corr_agg_opt']
        rel_agg_diff = data['corr_agg_diff'][0]
        data.close()
        
        agg_diff = abs(corr_agg_opt - corr_agg)
        # agg_diff = rel_agg_diff
        para_diff_mean = np.mean(agg_diff)
        para_diff_std = np.std(agg_diff)
        para_diff_var = np.var(agg_diff)

        para_diff_means.append(para_diff_mean)
        para_diff_stds.append(para_diff_std)
        para_diff_vars.append(para_diff_var)
    
    x_pos = np.arange(len(labels))
    values = [para_diff_mean]
    
    fig, ax = plt.subplots()
    ax.bar(x_pos, values, yerr=para_diff_std, align='center', alpha=0.7, ecolor='black', capsize=10)
    
    ax.set_ylabel('para_diff_mean')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()
    
    
data_paths1 = ['Parameter_study/multi_q3_BO_KL_wight_1_iter_20.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_50.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_100.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_200.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_400.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_800.npz',
              ]

labels1 = ['iter_20',
          'iter_50',
          'iter_100',
          'iter_200',
          'iter_400',
          'iter_800',
              ]

visualize_results(data_paths1, labels1)
