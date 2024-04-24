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

def visualize_distribution(axq3=None,fig=None, clr='b', q3lbl='q3'):
    x_uni, q3, Q3, sumvol_uni = p.return_distribution(t=-1, flag='x_uni, q3, Q3,sumvol_uni')
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl=q3lbl,clr=clr,mrk='o')
    
    axq3.grid('minor')
    axq3.set_xscale('log')

def visualize_convergence():
    
    pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    rate=fig.add_subplot(1,2,1)   
    error_norm=fig.add_subplot(1,2,2) 
    
    rate, fig = pt.plot_data(t_res_tem[1:], rate_res_tem, fig=fig, ax=rate,
                            xlbl='time  / $s$',
                            ylbl='convergence_rate',
                            lbl='',
                            clr='b',mrk='o')
    
    error_norm, fig = pt.plot_data(t_res_tem[1:], error_res_tem, fig=fig, ax=error_norm,
                            xlbl='time  / $s$',
                            ylbl='error_norm',
                            lbl='',
                            clr='b',mrk='o')
    rate.grid('minor')
    error_norm.grid('minor')
    plt.tight_layout()  
    
#%% MAIN   
if __name__ == "__main__":
    dim=2
    p = pop(dim=dim)
    
    ## Set the PBE parameters
    t_vec = np.arange(0, 61, 10, dtype=float)
    # Note that it must correspond to the settings of MC-Bond-Break.
    p.NS = 15
    p.S = 2
    
    p.BREAKRVAL= 4
    p.BREAKFVAL= 5
    p.aggl_crit= 100
    p.process_type= "mix"
    p.pl_v= 2
    p.pl_P1= 1e-2
    p.pl_P2= 0.5
    p.pl_P3= 1e-2
    p.pl_P4= 0.5
    # p.pl_P5= 1e-2
    # p.pl_P6= 1
    p.COLEVAL= 2
    p.EFFEVAL= 1
    p.SIZEEVAL= 1
    if dim == 2:
        p.alpha_primp = np.array([0.5,0.5,1.0])
    elif dim == 1:
        p.alpha_primp = 0.5
    p.CORR_BETA= 100
    ## The original value is the particle size at 1% of the PSD distribution. 
    ## The position of this value in the coordinate system can be adjusted by multiplying by size_scale.
    size_scale = 1e-1
    p.R01 = 8.68e-7*size_scale
    p.R03 = 8.68e-7*size_scale
    
    ## If you need to read PSD data as initial conditions, set the PSD data path
    p.USE_PSD = True
    p.DIST1 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
    p.DIST3 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
    
    ## Use the breakage function calculated by the MC-Bond-Break method
    # p.USE_MC_BOND = True
    
    ## Additional modifications for testing
    ## Total volume concentration of component, original value = 0.0001
    ## Used to increase/decrease the overall order of magnitude of a calculated value(N)
    ## Reducing the magnitude of N can improve the stability of calculation
    p.N_scale = 1e-14
    
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
    
    ## Visualize particle distribution at the first and the last time point 
    ## Visualize the convergence rate and error_norm
    visualize_distribution()
    visualize_convergence()
 