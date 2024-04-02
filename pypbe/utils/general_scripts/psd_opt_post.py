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
        
        # agg_diff = abs(corr_agg_opt - corr_agg)
        agg_diff = rel_agg_diff
        para_diff_mean = np.mean(agg_diff)
        para_diff_std = np.std(agg_diff)
        para_diff_var = np.var(agg_diff)

        para_diff_means.append(para_diff_mean)
        para_diff_stds.append(para_diff_std)
        para_diff_vars.append(para_diff_var)
    
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x_pos, para_diff_means, yerr=para_diff_stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    
    ax.set_ylabel('para_diff_mean')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()
    
    
data_paths1 = ['Parameter_study/multi_q3_BO_KL_wight_1_iter_1.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_5.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_10.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_20.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_40.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_50.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_100.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_200.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_400.npz',
              'Parameter_study/multi_q3_BO_KL_wight_1_iter_800.npz',
              ]

labels1 = ['iter_1',
          'iter_5',
          'iter_10',
          'iter_20',
          'iter_40',
          'iter_50',
          'iter_100',
          'iter_200',
          'iter_400',
          'iter_800',
              ]

data_paths2 = ['Parameter_study/multi_q3_BO_KL_wight_1_iter_400.npz',
              'Parameter_study/multi_Q3_BO_KL_wight_1_iter_400 (2).npz',
              'Parameter_study/multi_x_10_BO_MSE_wight_1_iter_400.npz',
              'Parameter_study/multi_x_50_BO_MSE_wight_1_iter_400.npz',
              'Parameter_study/multi_x_90_BO_MSE_wight_1_iter_400.npz',
              'Parameter_study/multi_q3_BO_MAE_wight_1_iter_400.npz',
              'Parameter_study/multi_q3_BO_MSE_wight_1_iter_400.npz',
              'Parameter_study/multi_q3_BO_RMSE_wight_1_iter_400.npz',
              ]

labels2 = ['q3_KL',
          'Q3_KL',
          'x_10_MSE',
          'x_50_MSE',
          'x_90_MSE',
          'q3_MAE',
          'q3_MSE',
          'q3_RMSE',
              ]

data_paths3 = ['Parameter_study/multi_q3_BO_KL_wight_1_iter_400.npz',
              'Parameter_study/multi_Q3_BO_KL_wight_1_iter_400 (2).npz',
              'Parameter_study/q3_BO_KL_wight_1_iter_400.npz',
              'Parameter_study/Q3_BO_KL_wight_1_iter_400 (2).npz',
              ]

labels3 = ['q3_multi',
          'Q3_multi',
          'q3_simple',
          'Q3_simple',
              ]

visualize_results(data_paths1, labels1)
visualize_results(data_paths2, labels2)
visualize_results(data_paths3, labels3)