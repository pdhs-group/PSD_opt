# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load('Parameter_study/multi_q3_BO_KL_wight_1_iter_400.npz')

corr_beta_opt = data['corr_beta_opt']
alpha_prim_opt = data['alpha_prim_opt']
para_diff = data['para_diff']
delta_opt = data['delta_opt']
corr_agg = data['corr_agg'],
corr_agg_opt = data['corr_agg_opt'],
rel_agg_diff = data['corr_agg_diff'][0],
rel_agg_diff2 = abs(corr_agg_opt[0] - corr_agg[0]) / corr_agg[0]
data.close()

abs_agg_diff = abs(corr_agg_opt[0] - corr_agg[0])
para_diff_mean = np.mean(abs_agg_diff)
para_diff_std = np.std(abs_agg_diff)
para_diff_var = np.var(abs_agg_diff)

labels = ['test1']
x_pos = np.arange(len(labels))
values = [para_diff_mean]

fig, ax = plt.subplots()
ax.bar(x_pos, values, yerr=para_diff_std, align='center', alpha=0.7, ecolor='black', capsize=10)

ax.set_ylabel('para_diff_mean')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)

plt.tight_layout()
plt.show()