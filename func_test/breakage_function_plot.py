# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:37:03 2024

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def beta_func(x, y):
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)

# 定义 b 函数
def b(x, y, v):
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

# 设定 y 的值
y = 10

# x 的取值范围
x = np.linspace(0, 10, 100)

# v 的不同值
v_values = np.linspace(1, 2, 5)

# 创建图像
plt.figure(figsize=(10, 6))

# 对每个 v 值绘制图像
for v in v_values:
    plt.plot(x, b(x, y, v), label=f'v = {v}')

# 设置图例
plt.legend()

# 设置图像标题和坐标轴标签
plt.title('Function Graphs for Different v Values')
plt.xlabel('x')
plt.ylabel('b(x, 10, v)')

# 显示图像
plt.show()