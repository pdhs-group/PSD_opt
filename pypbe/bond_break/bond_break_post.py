# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:14:13 2024

@author: px2030
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad

def breakage_func(NO_FRAG,kde,x):
    return kde(x) * NO_FRAG

def kde_psd():
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, NS**2))
    x = np.linspace(0, 1, 1000)
    
    for i in range(NS):
        # for j in range(NS):
            j = i
            if i <= 1 or j <= 1:
                continue
            file_name = f"i{i}_j{j}.npy"
            file_path = os.path.join(data_path,file_name)
            data = np.load(file_path,allow_pickle=True)
            PSD[:,i-2,j-2]=data[:,0]
            X1[:,i-2,j-2]=data[:,1]
            
            kde = gaussian_kde(PSD[:,i-2,j-2])  
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
    
def direkt_psd():
    for i in range(NS):
        # for j in range(NS):
            j = i
            if i <= 1 or j <= 1:
                continue
            file_name = f"i{i}_j{j}.npy"
            file_path = os.path.join(data_path,file_name)
            data = np.load(file_path,allow_pickle=True)
            PSD[:,i-2,j-2]=data[:,0]
            X1[:,i-2,j-2]=data[:,1]
            
if __name__ == '__main__':
    # Parameters of MC-Bond-Break
    NS = 10
    data_path = 'simulation_data'
    PSD = np.zeros((4*500*100, NS-2,NS-2))
    X1 = np.zeros((4*500*100, NS-2,NS-2))
    NO_FRAG = 4
    
    