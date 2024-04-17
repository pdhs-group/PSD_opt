# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:57:57 2024

@author: px2030
"""
import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import numpy as np
from pypbe.dpbe import population as pop

#%% MAIN   
if __name__ == "__main__":
    p = pop(dim=2)
    
    ## Set the PBE parameters
    t_vec = np.arange(0, 6, 1, dtype=float)
    p.NS = 15
    p.S = 4
    p.BREAKRVAL= 4
    p.BREAKFVAL= 5
    p.aggl_crit= 10
    p.process_type= "mix"
    p.pl_v= 0.1
    p.pl_P1= 1e-4
    p.pl_P2= 0.5
    p.pl_P3= 1e-4
    p.pl_P4= 0.5
    p.pl_P5= 1e-4
    p.pl_P6= 1
    p.COLEVAL= 2
    p.EFFEVAL= 1
    p.SIZEEVAL= 2
    p.alpha_primp = np.array([1, 1, 1])
    # p.alpha_primp = 0.5
    p.CORR_BETA= 1 
    p.R01 = 8.68e-8
    p.R03 = 8.68e-8
    
    ## If you need to read PSD data as initial conditions, set the PSD data path
    p.USE_PSD = True
    p.DIST1 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
    p.DIST3 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
    
    ## Additional modifications for testing
    # Total volume concentration of component, original value = 0.0001
    # Used to increase/decrease the overall order of magnitude of a calculated value(N)
    p.V01 *= 1e-3
    p.V03 *= 1e-3
    
    # var_v = np.array([0.1,1,2])
    # # var_v = np.array([0.01])
    # ## define the range of P1, P2 for power law breakage rate
    # var_P1 = np.array([1e-4,1e-2])
    # var_P2 = np.array([0.1,0.5])
    # var_P3 = np.array([1e-4,1e-2])
    # var_P4 = np.array([0.1,0.5])
    # var_P5 = np.array([1e-4,1e-2])
    # var_P6 = np.array([0.1,1])
    # for v in var_v:
    #     for P1 in var_P1:
    #         for P2 in var_P2:
    #             for P3 in var_P3:
    #                 for P4 in var_P4:
    #                     for P5 in var_P5:
    #                         for P6 in var_P6:
    #                             p.pl_v= v
    #                             p.pl_P1= P1
    #                             p.pl_P2= P2
    #                             p.pl_P3= P3
    #                             p.pl_P4= P4
    #                             p.pl_P5= P5
    #                             p.pl_P6= P6
    
    ## Initialize the PBE
    p.full_init(calc_alpha=False)
    ## solve the PBE
    p.solve_PBE(t_vec=t_vec)
    ## View number concentration of partikel
    N = p.N
    
    V_p = p.V
    N0 = N[:,:,0]
    NE = N[:,:,-1]
    print('### Total Volume before and after..')
    print(np.sum(N0*V_p), np.sum(NE*V_p))
    
    