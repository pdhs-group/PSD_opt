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
    data_path = 'test_data'
    
    ## generate dataset with variable parameters
    mc_gen.generate_dataset()
    
    # ## Generate complete data for a grid
    # # mc_gen.generate_complete_1d_data(NS, S, STR, NO_FRAG, N_GRIDS, N_FRACS, data_path)
    #     mc_gen.generate_complete_2d_data(NS, S, STR, NO_FRAG, N_GRIDS, N_FRACS, V01, V03, data_path)
        
    #     ## Calculate breakage function using calculated data
    #     int_B_F,intx_B_F,inty_B_F = mc_post.direkt_psd(NS, S, STR, NO_FRAG, N_GRIDS, N_FRACS, V01, V03, data_path)
            