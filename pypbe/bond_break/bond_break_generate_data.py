# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:37:52 2024

@author: px2030
"""

import numpy as np
import itertools
import os
from .bond_break_jit import MC_breakage
import multiprocessing

def generate_dataset():
    # Define variable parameters
    A0 = 0.025
    A_values = [2*A0, 4*A0, 8*A0, 16*A0, 64*A0, 256*A0, 2048*A0, 32768*A0] # 这里A0是一个具体的值
    NO_FRAG_values = [2, 4, 8]
    X1_values = [0, 0.5, 1]
    STR_elements = [1e-3, 0.5, 1]
    
    # Define other unchanged parameters
    N_GRIDS, N_FRACS = 200, 100
    INIT_BREAK_RANDOM = False
    
    # Make sure there is a directory to save the data
    output_dir = 'simulation_data'
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    # go through all the combination of parameters
    for A, X1, NO_FRAG, STR1, STR2, STR3 in itertools.product(A_values, X1_values, NO_FRAG_values, STR_elements, STR_elements, STR_elements):
        if A/A0 < NO_FRAG:
            continue
        
        args = (A, X1, output_dir, STR1, STR2,STR3,NO_FRAG, N_GRIDS, N_FRACS, A0, INIT_BREAK_RANDOM)
        tasks.append(args)
    # Use a pool of workers to execute simulations in parallel
    pool = multiprocessing.Pool(processes=12)
    pool.map(generate_one_data, tasks)    
        
def generate_one_data(args):
    A, X1, output_dir, STR1, STR2,STR3,NO_FRAG, N_GRIDS, N_FRACS, A0, INIT_BREAK_RANDOM = args
    X2 = 1 - X1
    STR = np.array([STR1, STR2, STR3])
    # Optimize MC-Bond-Break Simulation
    F = MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                    A0=A0, init_break_random=INIT_BREAK_RANDOM)
    # convert absolute volume to relative volume
    # F[:,0] /= A 
    # construkt the file name
    file_name = f"A{A}_X1{X1}_NO_FRAG{NO_FRAG}_STR{STR1}_{STR2}_{STR3}.npy"
    file_path = os.path.join(output_dir, file_name)
    # save array
    np.save(file_path, F)
def generate_complete_2d_data(NS,S,STR,NO_FRAG,N_GRIDS, N_FRACS, V01,V03):
    V1,V3,_,_,V,X1_vol = calc_2d_V(NS, S, V01, V03)
    # Define other unchanged parameters
    INIT_BREAK_RANDOM = False
    A0 = min(V1[1],V3[1])/ NO_FRAG
    
    # Make sure there is a directory to save the data
    output_dir = 'simulation_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    tasks = []
    for idx, A in np.ndenumerate(V):
        if idx[0] <= 1 or idx[1] <= 1:
            continue
        args = (idx, A, X1_vol[idx], V1, V3, output_dir, STR, NO_FRAG, N_GRIDS, N_FRACS, A0, INIT_BREAK_RANDOM)
        tasks.append(args)
    
    # Use a pool of workers to execute simulations in parallel
    pool = multiprocessing.Pool(processes=12)
    pool.map(generate_one_2d_data, tasks)

def calc_2d_V(NS,S,V01,V03):
    V_e1 = np.zeros(NS+1)
    V_e3 = np.zeros(NS+1)
    V1 = np.zeros(NS)
    V3 = np.zeros(NS)
    V_e1[0] = -V01
    V_e3[0] = -V03
    for i in range(NS):
        V_e1[i+1] = S**(i)*V01
        V_e3[i+1] = S**(i)*V03
        V1[i] = (V_e1[i] + V_e1[i+1]) / 2
        V3[i] = (V_e3[i] + V_e3[i+1]) / 2
    
    # Initialize V, R and ratio matrices
    V = np.zeros((NS,NS))
    X1_vol = np.copy(V)
    # Write V1 and V3 in respective "column" of V
    V[:,0] = V1 
    V[0,:] = V3 
    # Calculate remaining entries of V and other matrices
    # range(1,X) excludes X itself -> NS+2
    for i in range(NS):
        for j in range(NS):
            V[i,j] = V1[i]+V3[j]
            if i==0 and j==0:
                X1_vol[i,j] = 0
            else:
                X1_vol[i,j] = V1[i]/V[i,j]
    return V1,V3,V_e1,V_e3,V,X1_vol
    
def generate_one_2d_data(args):
    idx, A, X1, V1, V3, output_dir, STR, NO_FRAG, N_GRIDS, N_FRACS, A0, INIT_BREAK_RANDOM = args
    X2 = 1 - X1
    
    # 假设MC_breakage是一个已定义的函数，可以进行蒙特卡罗破碎模拟
    F = MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                    A0=A0, init_break_random=INIT_BREAK_RANDOM)
    
    # 构建文件名并保存结果
    file_name = f"{STR[0]}_{STR[1]}_{STR[2]}_{NO_FRAG}_i{idx[0]}_j{idx[1]}.npy"
    file_path = os.path.join(output_dir, file_name)
    np.save(file_path, F) 

def generate_complete_1d_data(NS,S,STR,NO_FRAG,N_GRIDS, N_FRACS):
    V,_ = calc_1d_V(NS, S)
    # Define other unchanged parameters
    INIT_BREAK_RANDOM = False
    A0 = V[1]/ NO_FRAG
    
    # Make sure there is a directory to save the data
    output_dir = 'simulation_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    tasks = []
    for idx, A in enumerate(V):
        if idx <= 1:
            continue
        args = (idx, A, V, output_dir, STR, NO_FRAG, N_GRIDS, N_FRACS, A0, INIT_BREAK_RANDOM)
        tasks.append(args)
    
    # Use a pool of workers to execute simulations in parallel
    pool = multiprocessing.Pool(processes=12)
    pool.map(generate_one_1d_data, tasks)

def calc_1d_V(NS,S):
    V_e = np.zeros(NS+1)
    V = np.zeros(NS)
    V_e[0] = -1
    for i in range(NS):
        V_e[i+1] = S**(i)
        V[i] = (V_e[i] + V_e[i+1]) / 2
    
    return V,V_e
    
def generate_one_1d_data(args):
    idx, A, V, output_dir, STR, NO_FRAG, N_GRIDS, N_FRACS, A0, INIT_BREAK_RANDOM = args
    X1 = 1
    X2 = 1 - X1
    
    F = MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                    A0=A0, init_break_random=INIT_BREAK_RANDOM)

    file_name = f"{STR[0]}_{STR[1]}_{STR[2]}_{NO_FRAG}_i{idx}.npy"
    file_path = os.path.join(output_dir, file_name)
    np.save(file_path, F) 
       
if __name__ == '__main__':
    # generate_dataset()
    NS = 15
    S = 2
    STR = np.array([0.5,1,0.5])
    NO_FRAG = 4
    N_GRIDS, N_FRACS = 200, 100
    generate_complete_1d_data(NS,S,STR,NO_FRAG, N_GRIDS, N_FRACS)
    generate_complete_2d_data(NS,S,STR,NO_FRAG,N_GRIDS, N_FRACS, V01=1,V03=1)
    

    