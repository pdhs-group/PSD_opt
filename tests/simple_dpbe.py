# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:57:57 2024

@author: px2030
"""
import sys, os
import time
import numpy as np
from optframework.pbe import DPBESolver
## for plotter
import matplotlib.pyplot as plt
import optframework.kernel_opt.opt_core as core
import optframework.utils.plotter.plotter as pt
from matplotlib.animation import FuncAnimation

def visualize_distribution(t_frame=-1, axq3=None,fig=None, clr='b', q3lbl='q3'):
    x_uni, q3, Q3, sumvol_uni = p.return_distribution(t=t_frame, flag='x_uni, q3, Q3,sumvol_uni')
    if smoothing:
        core_ins = core.OptCore()
        kde = core_ins.KDE_fit(x_uni[1:],sumvol_uni[1:],bandwidth='scott', kernel_func='epanechnikov')
        q3 = core_ins.KDE_score(kde,x_uni[1:])
        q3 = np.insert(q3, 0, 0.0)
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                           xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl=q3lbl,clr=clr,mrk='o')
    
    axq3.grid('minor')
    axq3.set_xscale('log')
    plt.tight_layout()  
    plt.show()

def animation_distribution(t_vec, fps=10):
    def update(frame):
        q3lbl = f"t={t_vec[frame]}"
        while len(axq3.lines) > 0:
            axq3.lines[0].remove()
        x_uni, q3, Q3, sumvol_uni = p.return_distribution(t=frame, flag='x_uni, q3, Q3, sumvol_uni')
        if smoothing:
            core_ins = core.OptCore()
            kde = core_ins.KDE_fit(x_uni[1:],sumvol_uni[1:],bandwidth='scott', kernel_func='epanechnikov')
            q3 = core_ins.KDE_score(kde,x_uni[1:])
            q3 = np.insert(q3, 0, 0.0)
        
        axq3.plot(x_uni, q3, label=q3lbl, color=clr, marker='o')  
        axq3.legend()
        return axq3,

    fig, axq3 = plt.subplots()
    clr = 'b'  
    t_frame = np.arange(1, len(t_vec))
    axq3.set_xlabel(r'Agglomeration size $x_\mathrm{A}$ / $-$')
    axq3.set_ylabel('number distribution of agglomerates $q3$ / $-$')
    axq3.grid('minor')
    axq3.set_xscale('log')
    plt.tight_layout()
    
    ani = FuncAnimation(fig, update, frames=t_frame, blit=False)
    ani.save('distribution_animation.gif', writer='imagemagick', fps=fps)
    
def visualize_convergence():
    fig=plt.figure()    
    rate=fig.add_subplot(1,2,1)   
    error_norm=fig.add_subplot(1,2,2) 
    
    rate, fig = pt.plot_data(t_res_tem[1:], rate_res_tem, fig=fig, ax=rate,
                            xlbl='time  / $s$',
                            ylbl='convergence_rate',
                            lbl='convergence_rate',
                            clr='b',mrk='o')
    
    error_norm, fig = pt.plot_data(t_res_tem[1:], error_res_tem, fig=fig, ax=error_norm,
                            xlbl='time  / $s$',
                            ylbl='error_norm',
                            lbl='error_norm',
                            clr='b',mrk='o')
    rate.grid('minor')
    error_norm.grid('minor')
    plt.tight_layout()  
    plt.show()
  
def visualize_N():
    fig=plt.figure()    
    N_t=fig.add_subplot(1,1,1)   
    if dim == 1:
        N[0,:] = 0
        N_sum = N.sum(axis=0)
    elif dim == 2:
        N[0,0,:] = 0
        N_sum = N.sum(axis=0).sum(axis=0)
    N_t, fig = pt.plot_data(t_vec, N_sum, fig=fig, ax=N_t,
                            xlbl='time  / $s$',
                            ylbl='total particle nummer',
                            lbl='total particle nummer',
                            clr='b',mrk='o')
    
    N_t.grid('minor')
    plt.tight_layout()  

#%% MAIN   
if __name__ == "__main__":
    dim=2
    p = DPBESolver(dim=dim)
    smoothing = True
    t_start = time.time()
    p.full_init(calc_alpha=False)
    t = time.time() - t_start
    print(f"initilization takes {t} second")
    t_vec = p.t_vec
    ## solve the PBE
    p.solve_PBE()
    ## View number concentration of partikel
    N = p.N
    if p.solver == "radau":
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
    pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    visualize_distribution(t_frame=-1)
    visualize_N()
    animation_distribution(t_vec,fps=5)
    if p.solver == "radau":
        visualize_convergence()
