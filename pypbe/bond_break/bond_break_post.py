# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:14:13 2024

@author: px2030
"""
import numpy as np
import os
# import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# from scipy.integrate import quad
from .bond_break_generate_data import calc_2d_V

def breakage_func(NO_FRAG,kde,x):
    return kde(x) * NO_FRAG

def kde_psd(NS, S, V01, V03,NO_FRAG,data_path):
    # PSD = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
    # X1 = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, NS**2))
    x = np.linspace(0, 1, 1000)
    
    _,_,_,_,V,_ = calc_2d_V(NS, S, V01, V03)
    for i in range(NS):
        # for j in range(NS):
            j = i
            if i <= 1 or j <= 1:
                continue
            file_name = f"i{i}_j{j}.npy"
            file_path = os.path.join(data_path,file_name)
            data = np.load(file_path,allow_pickle=True)
            # Using the relitve particle size 
            PSD=data[:,0] / V[i,j]
            # X1=data[:,1]
            
            kde = gaussian_kde(PSD)  
            y = breakage_func(NO_FRAG,kde,x)
            plt.plot(x, y, label=f'i{i}_j{j}', color=colors[i*NS + j - 2])
            
    plt.legend(title='Data Series', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Kernel Density Estimates of Fragment Sizes')
    plt.xlabel('Relative Volume')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # # Verify integral constraints
    # integral_total = quad(breakage_func, 0, 1)[0]
    # integral_volume = quad(lambda x: x * breakage_func(x), 0, 1)[0]
    # print("Integral of the breakage_func over [0,1]:", integral_total)
    # print("Integral of x * breakage_func over [0,1]:", integral_volume)
    
def direkt_psd(NS, S, STR, NO_FRAG, N_GRIDS, N_FRACS, V01, V03, data_path):
    int_B_F = np.zeros((NS, NS, NS, NS))
    intx_B_F = np.zeros((NS, NS, NS, NS))
    inty_B_F = np.zeros((NS, NS, NS, NS))
    V1,V3,V_e1,V_e3,V,_ = calc_2d_V(NS, S, V01, V03) 
    V_e1_tem = np.copy(V_e1)
    V_e1_tem[0] = 0.0
    V_e3_tem = np.copy(V_e3)
    V_e3_tem[0] = 0.0
    NO_TESTS = N_GRIDS*N_FRACS
    # PSD = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
    # X1 = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
    for i in range(NS):
        if i <= 1:
            continue
        file_name = f"{STR[0]}_{STR[1]}_{STR[2]}_{NO_FRAG}_i{i}.npy"
        file_path = os.path.join(data_path,file_name)
        data = np.load(file_path,allow_pickle=True)
        PSD=data[:,0]
        X1=data[:,1]
        x1 = PSD * X1
        counts, x1_vol_sum, x3_vol_sum= calc_int_BF(NO_TESTS,x1,V_e1_tem,V3=V3[1])
        counts, x1_vol_sum = adjust_BF(counts, x1_vol_sum, V1[i])
        int_B_F[:,0,i,0] = counts
        int_B_F[:,1,i,1] = counts
        ## Use relative value
        intx_B_F[:,0,i,0] = x1_vol_sum #/ V1[i]
        intx_B_F[:,1,i,1] = x1_vol_sum # / V1[i]
        ## Because the particle with coordinate 1 always carries a primary particle. 
        ## So it's equivalent to appending a constant integral.
        ## But in fact, it is assumed that this primary particle will be broken evenly(divide by NO_FRAG). 
        ## There is no problem with mathematics. Have questions about physics?
        inty_B_F[:,1,i,1] = x3_vol_sum / NO_FRAG
        if V01 == V03:
            int_B_F[0,:,0,i] = int_B_F[:,0,i,0]
            int_B_F[1,:,1,i] = int_B_F[:,1,i,1]
            inty_B_F[0,:,0,i] = intx_B_F[:,0,i,0]
            inty_B_F[1,:,1,i] = intx_B_F[:,1,i,1]
            intx_B_F[1,:,1,i] = inty_B_F[:,1,i,1]

        for j in range(NS):
            if j <= 1:
                continue
            file_name = f"{STR[0]}_{STR[1]}_{STR[2]}_{NO_FRAG}_i{i}_j{j}.npy"
            file_path = os.path.join(data_path,file_name)
            data = np.load(file_path,allow_pickle=True)
            PSD=data[:,0]
            X1=data[:,1]
            x1 = PSD * X1
            x3 = PSD * (1 - X1)
            counts, x1_vol_sum, x3_vol_sum = calc_int_BF(NO_TESTS,x1,V_e1_tem,x3,V_e3_tem)
            counts, x1_vol_sum, x3_vol_sum = adjust_BF(counts, x1_vol_sum, V1[i],x3_vol_sum,V3[j])
            int_B_F[:,:,i,j] = counts
            ## Use relative value
            intx_B_F[:,:,i,j] = x1_vol_sum # / V[i,j]
            inty_B_F[:,:,i,j] = x3_vol_sum # / V[i,j]
    ## TODO: if V01 != V03        
    save_path = os.path.join(f'{STR[0]}_{STR[1]}_{STR[2]}_{NO_FRAG}_int_B_F')
    
    np.savez(save_path,
             STR=STR,
             NO_FRAG=NO_FRAG,
             int_B_F=int_B_F,
             intx_B_F = intx_B_F,
             inty_B_F = inty_B_F)
    return int_B_F,intx_B_F,inty_B_F
        
def calc_int_BF(NO_TESTS,x1,e1,x3=None,e3=None,V3=None):
    if x3 is None and e3 is None:
        counts, _ = np.histogram(x1, e1)

        # 初始化数组用于存储每个区间的x和y的总和
        x1_vol_sum = np.zeros(counts.shape)
        x3_vol_sum = np.zeros(counts.shape)
        
        # 使用np.digitize找出每个数据点的区间索引
        idxs = np.digitize(x1, e1) - 1
        
        # 根据索引累加到对应的区间总和中
        for idx, x_vol in zip(idxs,x1):
            if 0 <= idx < counts.shape[0]:
                x1_vol_sum[idx] += x_vol
                x3_vol_sum[idx] += V3
    else:
        counts, _, _ = np.histogram2d(x1, x3, bins=[e1, e3])
    
        # 初始化数组用于存储每个区间的x和y的总和
        x1_vol_sum = np.zeros(counts.shape)
        x3_vol_sum = np.zeros(counts.shape)
        
        # 使用np.digitize找出每个数据点的区间索引
        idxs1 = np.digitize(x1, e1) - 1
        idxs3 = np.digitize(x3, e3) - 1
        
        # 根据索引累加到对应的区间总和中
        for (idx1, idx3), (x1_vol, x3_vol) in zip(zip(idxs1, idxs3), zip(x1, x3)):
            if 0 <= idx1 < counts.shape[0] and 0 <= idx3 < counts.shape[1]:
                x1_vol_sum[idx1, idx3] += x1_vol
                x3_vol_sum[idx1, idx3] += x3_vol
                
    return counts/NO_TESTS, x1_vol_sum/NO_TESTS, x3_vol_sum/NO_TESTS
    
def adjust_BF(counts, x1_vol_sum, V1, x3_vol_sum=None, V3=None):
    # 计算当前统计的总体积
    current_total_x1 = np.sum(x1_vol_sum)
    if x3_vol_sum is not None:
        current_total_x3 = np.sum(x3_vol_sum)
    else:
        V3 = 0.0
        current_total_x3 = 0.0
    
    scale_factor = (V1 + V3) / (current_total_x1 + current_total_x3)
    
    # 调整B_F的值
    adjested_counts = counts * scale_factor
    adjusted_x1_vol_sum = x1_vol_sum * scale_factor
    if x3_vol_sum is not None:
        adjusted_x3_vol_sum = x3_vol_sum * scale_factor
        return adjested_counts, adjusted_x1_vol_sum, adjusted_x3_vol_sum
    else:
        return adjested_counts, adjusted_x1_vol_sum
