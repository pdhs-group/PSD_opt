# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:26:36 2023

@author: px2030
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from pop import population
from PSD_Exp import write_read_exp


if __name__ == '__main__':
    
    p=population(2, disc='geo')
    p.USE_PSD=True
    p.full_init()
    p.solve_PBE()

    
    '''
    # test für conservation of mass(volume)
    sum_vol_t = np.zeros(len(p.t_vec))
    for idt in range(len(p.t_vec)):
        x_uni, q3, Q3, x_10, x_50, x_90, sum_vol_t[idt] = p.return_num_distribution(t=idt)
    
    # p.visualize_sumvol_t(sumvol=sum_vol_t)
    p.visualize_distribution(x_uni, q3, Q3)
    '''
    

    # read and interpolate exp_data
    exp_data_path = os.path.join(p.pth,"data\\")+'CED_focus_Sim.xlsx'
    compare = write_read_exp(exp_data_path, read=True)
    q3_exp = compare.get_exp_data(max(p.t_vec))
    Q3_exp = q3_exp.cumsum()

    
    '''
    # write simulation result to data in exp form
    x_uni, _, _, _, _, _, _ = p.return_distribution(t=len(p.t_vec)-1)
    sim_data_path = os.path.join(p.pth,"data\\")+'CED_focus_Sim.xlsx'
    df = pd.DataFrame(index=x_uni*1e6)
    df.index.name = 'Circular Equivalent Diameter'
    formatted_times = write_read_exp.convert_seconds_to_time(p.t_vec)

    for idt in range(1, len(p.t_vec)):
        _, q3, _, _, _, _, _ = p.return_distribution(t=idt)

        if len(q3) < len(x_uni):
            # Pad all arrays to the same length
            q3 = np.pad(q3, (0, len(x_uni) - len(q3)), 'constant')
        
        df[formatted_times[idt]] = q3

    # save DataFrame as Excel file
    df.to_excel(sim_data_path)
    '''