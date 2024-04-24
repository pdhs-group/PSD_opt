# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:35:00 2024

@author: px2030
"""

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

def load_and_process_data(directory):
    scaler = MinMaxScaler()
    all_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            F = np.load(os.path.join(directory, filename))
            total_area = F[:, 0]
            X1 = F[:, 1]
            X2 = F[:, 2]
            x = total_area * X1
            y = total_area * X2
            ## normalization processing
            xy = np.vstack((x, y)).T
            xy_scaled = scaler.fit_transform(xy)
            all_data.append(xy_scaled)

    aggregated_data = np.vstack(all_data)
    return aggregated_data

def train_model(data):
    # 假设你已经有一个模型构建的函数
    # 这里只是一个训练模型的示例框架
    return

if __name__ == '__main__':
    # 使用函数
    directory = os.path.join('simulation_data','NS_15_S_2_V11_STR_0.6_0.8_0.2_FRAG_4')
    data = load_and_process_data(directory)
    train_model(data)