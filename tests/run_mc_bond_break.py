# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:25:22 2024

@author: px2030
"""
import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import numpy as np
import pypbe.bond_break.bond_break_generate_data as mc_gen
import pypbe.bond_break.bond_break_post as mc_post

if __name__ == '__main__':
    NS = 15
    S = 2
    STR = np.array([0.5,1,0.5])
    NO_FRAG = 4
    V01 = 1
    V03 = 1
    N_GRIDS, N_FRACS = 200, 100
    data_path = 'simulation_data'
    
    values = np.array([0.1, 0.5, 1])
    STR1, STR2, STR3 = np.meshgrid(values, values, values, indexing='ij')
    var_STR = np.column_stack((STR1.flatten(), STR2.flatten(), STR3.flatten()))
    sorted_var_STR = np.sort(var_STR, axis=1)
    unique_var_STR = np.unique(sorted_var_STR, axis=0)
    filtered_var_STR = unique_var_STR[~(unique_var_STR[:, 0] == unique_var_STR[:, 1]) | ~(unique_var_STR[:, 1] == unique_var_STR[:, 2])]
    
    for STR in filtered_var_STR:
    ## Generate complete data
    # mc_gen.generate_complete_1d_data(NS, S, STR, NO_FRAG, N_GRIDS, N_FRACS)
        mc_gen.generate_complete_2d_data(NS, S, STR, NO_FRAG, N_GRIDS, N_FRACS, V01, V03)
        
        ## Calculate breakage function using calculated data
        int_B_F,intx_B_F,inty_B_F = mc_post.direkt_psd(NS, S, STR, NO_FRAG, N_GRIDS, N_FRACS, V01, V03, data_path)
            