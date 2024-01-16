# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:41:37 2024

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load('multi_q3_BO_KL_wight_1.npz')

corr_beta_opt = data['corr_beta_opt']
alpha_prim_opt = data['alpha_prim_opt']
para_diff = data['para_diff']
delta_opt = data['delta_opt']
elapsed_time = data['elapsed_time']

data.close()

para_diff_mean = np.mean(para_diff)
para_diff_std = np.std(para_diff)
para_diff_var = np.var(para_diff)

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