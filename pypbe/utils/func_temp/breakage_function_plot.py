# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:37:03 2024

@author: px2030
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import numpy as np
import matplotlib.pyplot as plt
import math
import plotter.plotter as pt 
from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, KIT_black_green_white

def beta_func(x, y):
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)

# 定义 b 函数
def b_1d(x, y, v):
    ## Power law
    # theta = (v + 1) * (x / y) ** (v - 1)
    ## Product function of power law
    # q = 10
    # euler_beta = beta_func(q,q*(v-1))
    # z = x/y
    # theta = v * z**(q-1) * (1-z)**(q*(v-1)-1) / euler_beta
    ## Parabolic
    theta = (v + 2) * (v + 1) * (x / y) ** (v - 1) * (1 - x / y)
    return theta / y

def b_2d(x3,x1,y1,y3,v):
    z = x1*x3 / (y1*y3)
    theta = (v+2)*(v+1)*z**(v-1)*(1-z)*v/2 
    return theta / (y1*y3)

# 设定 y 的值
y1 = 1
y3 = 1

# x 的取值范围
x1_var = np.linspace(0, y1, 40)
x3_var = np.linspace(0, y3, 40)

# v 的不同值
v_values = np.linspace(0.1, 2.0, 20)

# 创建图像
plt.figure(figsize=(10, 6))

# 对每个 v 值绘制图像
for v in v_values:
    plt.plot(x1_var, b_1d(x1_var, y1, v), label=f'v = {v:.1f}')
# b = np.zeros((20,20))
# v=2
# for i,x1 in enumerate(x1_var):
#     for j,x3 in enumerate(x3_var):
#         b[i,j] = b_2d(x3,x1,y1,y3,v)

# X, Y = np.meshgrid(x1_var, x3_var)
# x_flat = X.flatten()
# y_flat = Y.flatten()
# b_flat = b.flatten()

# 设置图例
plt.legend()

# 设置图像标题和坐标轴标签
plt.title('Function Graphs for Different v Values')
plt.xlabel('relative particle size')
plt.ylabel('value of breakage function')

# 显示图像
plt.show()

# ax1, fig1, cb1, H1, xe1, ye1 = pt.plot_2d_hist(x=x_flat,y=y_flat,bins=(20,20),w=b_flat,
#                                                scale=('lin','lin'), clr=KIT_black_green_white.reversed(), 
#                                                xlbl='Partial Volume 1 $V_1$ / $\mathrm{m^3}$', norm=False,
#                                                ylbl='Partial Volume 2 $V_2$ / $\mathrm{m^3}$', grd=True,
#                                                scale_hist='lin', hist_thr=1e-4)