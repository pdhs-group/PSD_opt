# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:37:52 2024

@author: px2030
"""

import numpy as np
import math
import itertools
import os
from bond_break_jit import MC_breakage

def generate_dataset():
    # Define variable parameters
    A0 = 0.025
    A_values = [4*A0, 16*A0, 64*A0, 256*A0, 1024*A0] # 这里A0是一个具体的值
    X1_values = [0, 0.5, 1]
    STR_elements = [1e-3, 0.5, 1]
    
    # Define other unchanged parameters
    NO_FRAG = 4
    N_GRIDS, N_FRACS = 500, 100
    INIT_BREAK_RANDOM = False
    
    # Make sure there is a directory to save the data
    output_dir = 'simulation_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # go through all the combination of parameters
    for A, X1, STR1, STR2, STR3 in itertools.product(A_values, X1_values, STR_elements, STR_elements, STR_elements):
        X2 = 1 - X1
        STR = np.array([STR1, STR2, STR3])
        
        # Optimize MC-Bond-Break Simulation
        F = MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                        A0=A0, init_break_random=INIT_BREAK_RANDOM)
        # convert absolute volume to relative volume
        F[:,0] /= A 
        # construkt the file name
        file_name = f"A{A}_X1{X1}_STR{STR1}_{STR2}_{STR3}.npy"
        file_path = os.path.join(output_dir, file_name)
        
        # save array
        np.save(file_path, F)
        
def generate_one_complete_2d_data(NS,S,V01,V03):
    
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
                # X1_vol[i,j] = V1[i]/V[i,j]
                X1_vol[i,j] = 0.6
    
    
    # Define other unchanged parameters
    STR = np.array([0.6, 0.8, 0.2])
    NO_FRAG = 4
    N_GRIDS, N_FRACS = 500, 100
    INIT_BREAK_RANDOM = False
    A0 = min(V1[1],V3[1])/ NO_FRAG
    
    # Make sure there is a directory to save the data
    output_dir = 'simulation_data'
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, A in np.ndenumerate(V):
        if idx[0] <= 1 or idx[1] <= 1:
            continue
        X1 = X1_vol[idx]
        X2 = 1 - X1

        # Optimize MC-Bond-Break Simulation
        F = MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                        A0=A0, init_break_random=INIT_BREAK_RANDOM)
        # convert absolute volume to relative volume
        F[:,0] /= A 
        # construkt the file name
        file_name = f"i{idx[0]}_j{idx[1]}.npy"
        file_path = os.path.join(output_dir, file_name)
        
        # save array
        np.save(file_path, F)
        
if __name__ == '__main__':
    
    # generate_dataset()
    # generate_one_complete_2d_data(NS=10,S=2,V01=1,V03=1)
    
    # Define variable parameters
    A0 = 0.025
    A = A0*1e4 # 这里A0是一个具体的值
    X1 = 0.6
    
    # Define other unchanged parameters
    NO_FRAG = 4
    N_GRIDS, N_FRACS = 500, 100
    INIT_BREAK_RANDOM = False
    X2 = 1 - X1
    STR = np.array([1, 1, 1])
    
    # Optimize MC-Bond-Break Simulation
    F = MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                    A0=A0, init_break_random=INIT_BREAK_RANDOM)
    