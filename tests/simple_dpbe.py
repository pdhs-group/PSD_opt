# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:57:57 2024

@author: px2030
"""
import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import numpy as np
from pypbe.dpbe import population as pop
## for plotter
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt
#%% MAIN   
if __name__ == "__main__":
    dim=2
    p = pop(dim=dim)
    
    ## Set the PBE parameters
    t_vec = np.arange(0, 2001, 100, dtype=float)
    p.NS = 15
    p.S = 4
    p.BREAKRVAL= 4
    p.BREAKFVAL= 5
    p.aggl_crit= 100
    p.process_type= "agglomeration"
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
    p.CORR_BETA= 100 
    size_scale = 1e0
    p.R01 = 8.68e-7*size_scale
    p.R03 = 8.68e-7*size_scale
    
    ## If you need to read PSD data as initial conditions, set the PSD data path
    p.USE_PSD = False
    p.DIST1 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
    p.DIST3 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
    
    ## Additional modifications for testing
    # Total volume concentration of component, original value = 0.0001
    # Used to increase/decrease the overall order of magnitude of a calculated value(N)
    # p.V01 *= 1e-3
    # p.V03 *= 1e-3
    
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
    N_res_tem = p.N_res_tem
    t_res_tem = p.t_res_tem
    rate_res_tem = p.rate_res_tem
    error_res_tem = p.error_res_tem
    V_p = p.V
    
    if dim == 2:
        N0 = N[:,:,0]
        NE = N[:,:,-1]
    elif dim == 1:
       N0 = N[:,0]
       NE = N[:,-1]
    print('### Total Volume before and after..')
    print(np.sum(N0*V_p), np.sum(NE*V_p))
    
    ## visualize the convergence rate and error_norm
    pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    rate=fig.add_subplot(1,2,1)   
    error_norm=fig.add_subplot(1,2,2) 
    
    rate, fig = pt.plot_data(t_res_tem[1:], rate_res_tem, fig=fig, ax=rate,
                            xlbl='time  / $s$',
                            ylbl='convergence_rate',
                            clr='b',mrk='o')
    
    error_norm, fig = pt.plot_data(t_res_tem[1:], error_res_tem, fig=fig, ax=error_norm,
                            xlbl='time  / $s$',
                            ylbl='error_norm',
                            clr='b',mrk='o')
    rate.grid('minor')
    error_norm.grid('minor')
    plt.tight_layout()   