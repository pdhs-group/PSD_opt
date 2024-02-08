# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:47:06 2024

@author: px2030
"""

# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import math
import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import plotter.plotter as pt
from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

from numba import jit
from numba.extending import overload, register_jitable

pt.close()
pt.plot_init(mrksze=8,lnewdth=1)
    
#%% PARAM
t = np.arange(0, 1001, 100, dtype=float)
NS = 30
S = 2
R01, R02 = 1, 1
V01, V02 = 2e-9, 1e-6
dim = 1
BREAKFVAL = 2
BREAKRVAL = 2

#%% FUNCTIONS
def dNdt_1D(t,N,V_p,V_e,B_R,B_F,BREAKFVAL):
    dNdt = np.zeros(N.shape)
    B_c = np.zeros(N.shape)
    M_c = np.zeros(N.shape)
    v = np.zeros(N.shape)
    D = np.zeros(N.shape)
    B = np.zeros(N.shape)
    
    # Loop through all edges
    # -1 to make sure nothing overshoots (?) CHECK THIS
    for e in range(0, len(V_p)):
        if e != 0:
            b = B_F[e,e]
            b_int = b_integrate(V_p[e], V_e[e], b)
            xb_int = xb_integrate(V_p[e], V_e[e], b)
            B_c[e] += B_R[e]*b_int*N[e]
            M_c[e] += B_R[e]*xb_int*N[e]
        
        ## if breakage function is independent of parent particle, 
        ## then the its integral only needs to be calculated once
        # Loop through all pivots (twice)
        for i in range(e+1, len(V_p)):
            b = B_F[e,i]
            b_int = b_integrate(V_e[e+1], V_e[e], b)
            xb_int = xb_integrate(V_e[e+1], V_e[e], b)
            B_c[e] += B_R[i]*b_int*N[i]
            M_c[e] += B_R[i]*xb_int*N[i]
                    
        D[e] = -B_R[e]*N[e]
    D[0] = 0
    v[B_c != 0] = M_c[B_c != 0]/B_c[B_c != 0]
    
    # print(B_c)
    
    # Assign BIRTH on each pivot
    for i in range(len(V_p)):            
        # Add contribution from LEFT cell (if existent)
        B[i] += B_c[i]*lam(v[i], V_p, i, 'm')*heaviside_jit(V_p[i]-v[i],0.5)
        B[i] += B_c[i]*lam(v[i], V_p, i, 'p')*heaviside_jit(v[i]-V_p[i],0.5)
        if i != 0:
            # Same Cell, left half
            # Left Cell, right half
            B[i] += B_c[i-1]*lam(v[i-1], V_p, i, 'm')*heaviside_jit(v[i-1]-V_p[i-1],0.5)
            
        # Add contribution from RIGHT cell (if existent)
        if i != len(V_p)-1:
            # Same Cell, right half
            # Right Cell, left half
            B[i] += B_c[i+1]*lam(v[i+1], V_p, i, 'p')*heaviside_jit(V_p[i+1]-v[i+1],0.5)
            
    dNdt = B + D
    
    return dNdt 

def b_integrate(x_up,x_low,b):
    return (x_up - x_low)*b
    
def xb_integrate(x_up,x_low,b):
    return (x_up**2 - x_low**2)*0.5*b
    
@jit(nopython=True)
def lam(v, V_p, i, case):
    if case == 'm':
        return (v-V_p[i-1])/(V_p[i]-V_p[i-1])
    elif case == 'p':        
        return (v-V_p[i+1])/(V_p[i]-V_p[i+1])
    else:
        print('WRONG CASE FOR LAM')
        
@jit(nopython=True)
def heaviside_jit(x1, x2):
    if x1 < 0:
        return 0.0
    elif x1 > 0:
        return 1.0
    else:
        return x2
    
#%% NEW 1D
if dim == 1:
    # V_e: Volume of EDGES
    V_e = np.zeros(NS+1)
    # V_p: Volume of PIVOTS
    V_p = np.zeros(NS)
    # SOLUTION N is saved on pivots
    N = np.zeros((NS,len(t)))
    #N[0,0] = 0.1
    N[-1,0] = 1
    #N[2,0] = 0.2
    
    for i in range(1,NS+1):
        V_e[i] = S**(i-1)*V01
        # ith pivot is mean between ith and (i+1)th edge
        V_p[i-1] = (V_e[i] + V_e[i-1])/2
    X1_vol = np.ones(NS)
    
    # N[:,0] = np.exp(-V_p)
    
    B_R = np.zeros(NS)
    B_F = np.zeros((NS,NS))
    if BREAKRVAL == 1:
        B_R[1:] = 1
    elif BREAKRVAL == 2:
        for idx, tmp in np.ndenumerate(B_R):
            B_R[idx] = V_p[idx]
            
    ## Validation: breakage function dependent only on parent particle
    for idx, tmp in np.ndenumerate(B_F):
        a = idx[0]; i = idx[1]
        if BREAKFVAL == 1:  
            B_F[idx] = 4 / (V_p[i])
        elif BREAKFVAL == 2:
            B_F[idx] = 2 / (V_p[i])
        
    # SOLVE    
    import scipy.integrate as integrate
    RES = integrate.solve_ivp(dNdt_1D,
                              [0, max(t)], 
                              N[:,0], t_eval=t,
                              args=(V_p,V_e,B_R,B_F,BREAKFVAL),
                              method='RK23',first_step=0.1,rtol=1e-1)
    
    # Reshape and save result to N and t_vec
    N = RES.y
    t = RES.t
    
    N0 = N[:,0]
    NE = N[:,-1]
    print('### Total Volume before and after..')
    print(np.sum(N0*V_p), np.sum(NE*V_p))
    
    print('### Initial dNdt..')
    dNdt0=dNdt_1D(0,N[:,0],V_p,V_e,B_R,B_F,BREAKFVAL)
    print(dNdt0)   
    
    fig=plt.figure(figsize=[10,2])    
    ax=fig.add_subplot(1,1,1) 
    ax.scatter(V_e, np.ones(V_e.shape), marker='|', color='k', label='edges')    
    ax.scatter(V_p, np.ones(V_p.shape), marker='d', color=c_KIT_green, label='pivots')
    ax.set_xscale('log')
    ax.set_ylim([0.99,1.02])
    ax.legend()
    plt.tight_layout()
    
    fig2=plt.figure(figsize=[4,3])    
    ax2=fig2.add_subplot(1,1,1) 
    
    mu1 = np.zeros(t.shape)
    mu0 = np.zeros(t.shape)
    for ti in range(len(t)):
        mu1[ti] = np.sum(V_p*N[:,ti])/np.sum(V_p*N[:,0])
        mu0[ti] = np.sum(N[:,ti])
    
    ax2.plot(t, mu0, color=c_KIT_green, label='$\mu_0$ (numerical)') 
    
    ## see Kumar Dissertation A.1
    # N_as = np.zeros((NS,len(t)))
    # delta = np.zeros(NS)
    # theta = np.zeros(NS)
    # delta[-1] = 1
    # theta[:-1] = 1
    # for i in range(len(V_p)):
    #     for j in range(len(t)):
    #         # N_as[i,j] = np.exp(-t[j]*V_p[i])*(delta[i]+(2*t[j]+t[j]**2*(V_p[-1]-V_p[i]))*theta[i])
    #         N_as[i,j] = np.exp(-V_p[i]*(1+t[j]))*(1+t[j])**2
    # mu0_as = 2/(2+np.sum(N[:,0])*t)
    
    mu0_as = np.exp(t)
    
    # ax2.plot(t, mu0_as, color='k', linestyle='-.', label='$\mu_0$ (analytical)')
    # ax2.plot(t, mu1_as, color='b', linestyle='-.', label='$\mu_1$ (analytical)')
    ax2.plot(t, mu1, color=c_KIT_red, label='$\mu_1$')     
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.legend()
    plt.tight_layout()