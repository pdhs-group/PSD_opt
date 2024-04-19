# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:14:13 2024

@author: px2030
"""
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from bond_break_generate_data import calc_2d_V

def breakage_func(NO_FRAG,kde,x):
    return kde(x) * NO_FRAG

def kde_psd(NS, S, V01, V03):
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
    
def direkt_psd(NS, S, V01, V03):
    int_B_F = np.zeros((NS, NS, NS, NS))
    intx_B_F = np.zeros((NS, NS, NS, NS))
    inty_B_F = np.zeros((NS, NS, NS, NS))
    V1,V3,V_e1,V_e3,V,_ = calc_2d_V(NS, S, V01, V03) 
    V_e1_tem = np.copy(V_e1)
    V_e1_tem[0] = 0.0
    V_e3_tem = np.copy(V_e3)
    V_e3_tem[0] = 0.0
    # PSD = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
    # X1 = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
    for i in range(NS):
        if i <= 1:
            continue
        file_name = f"i{i}.npy"
        file_path = os.path.join(data_path,file_name)
        data = np.load(file_path,allow_pickle=True)
        PSD=data[:,0]
        X1=data[:,1]
        x1 = PSD * X1
        counts, x1_vol_sum, x2_vol_sum= calc_int_BF(x1,V_e1_tem,V3=V3[1])
        int_B_F[:,0,i,0] = counts / NO_TESTS
        int_B_F[:,1,i,1] = counts / NO_TESTS
        ## Use relative value
        intx_B_F[:,0,i,0] = x1_vol_sum / NO_TESTS #/ V1[i]
        intx_B_F[:,1,i,1] = x1_vol_sum / NO_TESTS# / V1[i]
        ## Because the particle with coordinate 1 always carries a primary particle. 
        ## So it's equivalent to appending a constant integral.
        ## But in fact, it is assumed that this primary particle will be broken evenly(divide by NO_FRAG). 
        ## There is no problem with mathematics. Have questions about physics?
        inty_B_F[:,1,i,1] = x2_vol_sum / NO_TESTS / NO_FRAG
        if V01 == V03:
            int_B_F[0,:,0,i] = int_B_F[:,0,i,0]
            int_B_F[1,:,1,i] = int_B_F[:,1,i,1]
            inty_B_F[0,:,0,i] = intx_B_F[:,0,i,0]
            inty_B_F[1,:,1,i] = intx_B_F[:,1,i,1]
            intx_B_F[1,:,1,i] = inty_B_F[:,1,i,1]

        for j in range(NS):
            if j <= 1:
                continue
            file_name = f"i{i}_j{j}.npy"
            file_path = os.path.join(data_path,file_name)
            data = np.load(file_path,allow_pickle=True)
            PSD=data[:,0]
            X1=data[:,1]
            x1 = PSD * X1
            x3 = PSD * (1 - X1)
            counts, x1_vol_sum, x3_vol_sum = calc_int_BF(x1,V_e1_tem,x3,V_e3_tem)
            int_B_F[:,:,i,j] = counts / NO_TESTS
            ## Use relative value
            intx_B_F[:,:,i,j] = x1_vol_sum / NO_TESTS# / V[i,j]
            inty_B_F[:,:,i,j] = x3_vol_sum / NO_TESTS#/ V[i,j]
    ## TODO: if V01 != V03        
    save_path = os.path.join('int_B_F')
    
    np.savez(save_path,
             int_B_F=int_B_F,
             intx_B_F = intx_B_F,
             inty_B_F = inty_B_F)
    return int_B_F,intx_B_F,inty_B_F
 
# def direkt_psd_new(NS, S, V01, V03):
#     int_B_F = np.zeros((NS, NS, NS, NS))
#     intx_B_F = np.zeros((NS, NS, NS, NS))
#     inty_B_F = np.zeros((NS, NS, NS, NS))
#     R01,R03 = 8.68e-7,8.68e-7
#     V01,V03=4*math.pi*R01**3/3, 4*math.pi*R03**3/3
#     V1,V3,V_e1,V_e3,V,_ = calc_2d_V(NS, S, V01, V03) 
#     V_e1_tem = np.copy(V_e1)
#     V_e1_tem[0] = 0.0
#     V_e3_tem = np.copy(V_e3)
#     V_e3_tem[0] = 0.0
#     # PSD = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
#     # X1 = np.zeros((NO_FRAG*NO_TESTS, NS-2,NS-2))
#     for i in range(NS):
#         if i <= 1:
#             continue
#         file_name = f"i{i}.npy"
#         file_path = os.path.join(data_path,file_name)
#         data = np.load(file_path,allow_pickle=True)
#         PSD=data[:,0]
#         X1=data[:,1]
#         x1 = PSD * X1 * V01
#         counts, x1_vol_sum= calc_int_BF(x1,V_e1_tem)
#         int_B_F[:,0,i,0] = counts / NO_TESTS
#         int_B_F[:,1,i,1] = counts / NO_TESTS
#         ## Use relative value
#         intx_B_F[:,0,i,0] = x1_vol_sum / NO_TESTS# / V1[i]
#         intx_B_F[:,1,i,1] = x1_vol_sum / NO_TESTS #/ V1[i]
#         if V01 == V03:
#             int_B_F[0,:,0,i] = int_B_F[:,0,i,0]
#             int_B_F[1,:,1,i] = int_B_F[:,1,i,1]
#             inty_B_F[0,:,0,i] = intx_B_F[:,0,i,0]
#             inty_B_F[1,:,1,i] = intx_B_F[:,1,i,1]

#         for j in range(NS):
#             if j <= 1:
#                 continue
#             file_name = f"i{i}_j{j}.npy"
#             file_path = os.path.join(data_path,file_name)
#             data = np.load(file_path,allow_pickle=True)
#             PSD=data[:,0]
#             X1=data[:,1]
#             x1 = PSD * X1 * V01
#             x3 = PSD * (1 - X1) * V03
#             counts, x1_vol_sum, x3_vol_sum = calc_int_BF(x1,V_e1_tem,x3,V_e3_tem)
#             int_B_F[:,:,i,j] = counts / NO_TESTS
#             ## Use relative value
#             intx_B_F[:,:,i,j] = x1_vol_sum / NO_TESTS# / V[i,j]
#             inty_B_F[:,:,i,j] = x3_vol_sum / NO_TESTS# / V[i,j]
#     ## TODO: if V01 != V03        
#     # save_path = os.path.join('int_B_F')
    
#     # np.savez(save_path,
#     #          int_B_F=int_B_F,
#     #          intx_B_F = intx_B_F,
#     #          inty_B_F = inty_B_F)
#     return int_B_F,intx_B_F,inty_B_F
   
def calc_int_BF(x1,e1,x3=None,e3=None,V3=None):
    if x3 is None and e3 is None:
        counts, _ = np.histogram(x1, e1)

        # 初始化数组用于存储每个区间的x和y的总和
        x_vol_sum = np.zeros(counts.shape)
        y_vol_sum = np.zeros(counts.shape)
        
        # 使用np.digitize找出每个数据点的区间索引
        idxs = np.digitize(x1, e1) - 1
        
        # 根据索引累加到对应的区间总和中
        for idx, x_vol in zip(idxs,x1):
            if 0 <= idx < counts.shape[0]:
                x_vol_sum[idx] += x_vol
                y_vol_sum[idx] += V3
        return counts, x_vol_sum, y_vol_sum
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
        return counts, x1_vol_sum, x3_vol_sum

if __name__ == '__main__':
    # Parameters of MC-Bond-Break
    NS = 15
    S = 2
    V01 = 1
    V03 = 1
    data_path = os.path.join('simulation_data','NS_15_S_2_V11_STR_0.6_0.8_0.2_FRAG_4')
    NO_FRAG = 4
    N_GRIDS, N_FRACS = 500, 100
    NO_TESTS = N_GRIDS*N_FRACS
    
    # kde_psd(NS, S, V01, V03)
    int_B_F,intx_B_F,inty_B_F = direkt_psd(NS, S, V01, V03)
    # int_B_F_new,intx_B_F_new,inty_B_F_new = direkt_psd_new(NS, S, V01, V03)