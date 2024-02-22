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
t = np.arange(0, 11, 1, dtype=float)
NS = 30
S = 2.5
V01, V02 = 2e-9, 2e-9
dim = 1
## BREAKRVAL == 1: 1, constant breakage rate
## BREAKRVAL == 2: x*y or x + y, breakage rate is related to particle size
BREAKRVAL = 2
## BREAKFVAL == 1: 4/x'y', meet the first cross moment
## BREAKFVAL == 1: 2/x'y', meet the first moment/ mass conversation
BREAKFVAL = 2

#%% FUNCTIONS
# @jit(nopython=True)
def dNdt_1D(t,N,V_p,V_e,B_R,B_F,BREAKFVAL):
    dNdt = np.zeros(N.shape)
    B_c = np.zeros(NS+1)
    M_c = np.zeros(N.shape)
    v = np.zeros(NS+1)
    D = np.zeros(N.shape)
    B = np.zeros(N.shape)
    V_p_ex = np.zeros(NS+1)
    
    ## only to check volume conservation
    # D_M = np.zeros(N.shape)
                
    # Loop through all edges
    for e in range(len(V_p)):
        b = B_F[e,e]
        b_int = b_integrate(V_p[e], V_e[e], b=b)
        xb_int = xb_integrate(V_p[e], V_e[e], b=b)
        B_c[e] += B_R[e]*b_int*N[e]
        M_c[e] += B_R[e]*xb_int*N[e]
        
        ## if breakage function is independent of parent particle, 
        ## then the its integral only needs to be calculated once
        # Loop through all pivots (twice)
        for i in range(e+1, len(V_p)):
            b = B_F[e,i]
            b_int = b_integrate(V_e[e+1], V_e[e], b=b)
            xb_int = xb_integrate(V_e[e+1], V_e[e], b=b)
            B_c[e] += B_R[i]*b_int*N[i]
            M_c[e] += B_R[i]*xb_int*N[i]
                    
        D[e] = -B_R[e]*N[e]
        if B_c[e] != 0:
            v[e] = M_c[e] / B_c[e]
        
        ## only to check volume conservation
        # D_M[e] = -B_R[e]*N[e]*V_p[e]
    # volume_erro = M_c.sum() + D_M.sum()
    # print(f'the erro of mass conservation = {volume_erro}')
    
    V_p_ex[:-1] = V_p
    # Assign BIRTH on each pivot
    for i in range(len(V_p)):            
        # Add contribution from LEFT cell (if existent)
        B[i] += B_c[i]*lam(v[i], V_p, i, 'm')*heaviside_jit(V_p[i]-v[i],0.5)
        # Left Cell, right half
        B[i] += B_c[i-1]*lam(v[i-1], V_p, i, 'm')*heaviside_jit(v[i-1]-V_p[i-1],0.5)
        # Add contribution from RIGHT cell (if existent)
        # Same Cell, right half
        B[i] += B_c[i]*lam(v[i], V_p_ex, i, 'p')*heaviside_jit(v[i]-V_p[i],0.5)
        # Right Cell, left half
        B[i] += B_c[i+1]*lam(v[i+1], V_p_ex, i, 'p')*heaviside_jit(V_p_ex[i+1]-v[i+1],0.5)
            
    dNdt = B + D
    
    return dNdt 
@jit(nopython=True)
def b_integrate(x_up,x_low,y_up=None,y_low=None,b=None):
    if y_up is None or y_low is None:
        return (x_up - x_low)*b
    else:
        return (x_up- x_low)*(y_up - y_low)*b
@jit(nopython=True)    
def xb_integrate(x_up,x_low,y_up=None,y_low=None,b=None):
    if y_up is None or y_low is None:
        return (x_up**2 - x_low**2)*0.5*b
    else:
        return (x_up**2- x_low**2)*(y_up - y_low)*0.5*b
@jit(nopython=True)    
def yb_integrate(x_up,x_low,y_up=None,y_low=None,b=None):
    return (y_up**2 - y_low**2)*(x_up-x_low)*0.5*b    
    
@jit(nopython=True)
def lam(v, V_p, i, case):
    if case == 'm':
        return (v-V_p[i-1])/(V_p[i]-V_p[i-1])
    elif case == 'p':        
        return (v-V_p[i+1])/(V_p[i]-V_p[i+1])
    else:
        print('WRONG CASE FOR LAM')
        
@jit(nopython=True)
def lam_2d(x,y,Vx,Vy,i,j,m1,m2):

    if m1 == "+":
        if m2 == "+":
            lam = (x-Vx[i+1])*(y-Vy[j+1])/((Vx[i]-Vx[i+1])*(Vy[j]-Vy[j+1]))
        else:
            lam = (x-Vx[i+1])*(y-Vy[j-1])/((Vx[i]-Vx[i+1])*(Vy[j]-Vy[j-1]))
    else:
        if m2 == "+":
            lam = (x-Vx[i-1])*(y-Vy[j+1])/((Vx[i]-Vx[i-1])*(Vy[j]-Vy[j+1]))
        else:
            lam = (x-Vx[i-1])*(y-Vy[j-1])/((Vx[i]-Vx[i-1])*(Vy[j]-Vy[j-1]))
                
                
    if math.isnan(lam):
        lam = 0 
        print('lam is NaN!')        
    
    return lam  
        
@jit(nopython=True)
def heaviside_jit(x1, x2):
    if x1 < 0:
        return 0.0
    elif x1 > 0:
        return 1.0
    else:
        return x2
    
@jit(nopython=True)    
def dNdt_2D(t,NN,V_p1,V_p2,V_e1,V_e2,B_R,B_F,BREAKFVAL):
    N = np.copy(NN) 
    N = np.reshape(N,(NS,NS))
    dNdt = np.zeros(N.shape)
    B_c = np.zeros((NS+1,NS+1))
    B_c_x = np.zeros(NS+1)
    B_c_y = np.zeros(NS+1)
    M_c_x = np.zeros(NS)
    M_c_y = np.zeros(NS)
    vx = np.zeros(NS+1)
    vy = np.zeros(NS+1)
    M1_c = np.zeros(np.shape(N))
    M2_c = np.zeros(np.shape(N))
    v1 = np.zeros((NS+1,NS+1))
    v2 = np.zeros((NS+1,NS+1))
    D = np.zeros(N.shape)
    B = np.zeros(N.shape)
    V_p1_ex = np.zeros(NS+1)
    V_p2_ex = np.zeros(NS+1)
    V_p1_ex[:-1] = V_p1
    V_p2_ex[:-1] = V_p2
    
    ## only to check volume conservation
    # D_M = np.zeros(N.shape)
    ## Because the cells on the lower boundary (e1=0 or e2=0)are not allowed to break outward, 
    ## 1d calculations need to be performed on the two lower boundaries.
    for e1 in range(0, len(V_p1)):
        for e2 in range(0, len(V_p2)):
            ## The contribution of self-fragmentation
            b = B_F[e1,e2,e1,e2]
            S = B_R[e1,e2]
            if e1 == 0:
                b_int = b_integrate(V_p2[e2], V_e2[e2], b=b)
                yb_int = xb_integrate(V_p2[e2], V_e2[e2], b=b)
                B_c_y[e2] += S*b_int*N[e1,e2]
                M_c_y[e2] += S*yb_int*N[e1,e2]
                # D_M[e1,e2] = -S*N[e1,e2]*(V_p2[e2])
            elif e2 == 0:
                b_int = b_integrate(V_p1[e1], V_e1[e1], b=b)
                xb_int = xb_integrate(V_p1[e1], V_e1[e1], b=b)
                B_c_x[e1] += S*b_int*N[e1,e2]
                M_c_x[e1] += S*xb_int*N[e1,e2]
                # D_M[e1,e2] = -S*N[e1,e2]*(V_p1[e1])
            else:
                b_int = b_integrate(V_p1[e1], V_e1[e1], V_p2[e2], V_e2[e2], b)
                xb_int = xb_integrate(V_p1[e1], V_e1[e1], V_p2[e2], V_e2[e2], b)
                yb_int = yb_integrate(V_p1[e1], V_e1[e1], V_p2[e2], V_e2[e2], b)
                B_c[e1,e2] += S*b_int*N[e1,e2]
                M1_c[e1,e2] += S*xb_int*N[e1,e2]
                M2_c[e1,e2] += S*yb_int*N[e1,e2]
                # D_M[e1,e2] = -S*N[e1,e2]*(V_p1[e1]+V_p2[e2])
            # calculate death rate    
            D[e1,e2] = -S*N[e1,e2]
            
            ## The contributions of fragments on the same y-axis
            for i in range(e1+1,len(V_p1)):
                b = B_F[e1,e2,i,e2]
                S = B_R[i,e2]
                if e2 == 0:
                    b_int = b_integrate(V_e1[e1+1], V_e1[e1], b=b)
                    xb_int = xb_integrate(V_e1[e1+1], V_e1[e1], b=b)
                    B_c_x[e1] += S*b_int*N[i,e2]
                    M_c_x[e1] += S*xb_int*N[i,e2]

                else:
                    b_int = b_integrate(V_e1[e1+1], V_e1[e1], V_p2[e2], V_e2[e2], b)
                    xb_int = xb_integrate(V_e1[e1+1], V_e1[e1], V_p2[e2], V_e2[e2], b)
                    yb_int = yb_integrate(V_e1[e1+1], V_e1[e1], V_p2[e2], V_e2[e2], b)
                    B_c[e1,e2] += S*b_int*N[i,e2] 
                    M1_c[e1,e2] += S*xb_int*N[i,e2]
                    M2_c[e1,e2] += S*yb_int*N[i,e2]
            ## The contributions of fragments on the same x-axis
            for j in range(e2+1,len(V_p2)):
                b = B_F[e1,e2,e1,j]
                S = B_R[e1,j]
                if e1 == 0:
                    b_int = b_integrate(V_e2[e2+1], V_e2[e2], b=b)
                    yb_int = xb_integrate(V_e2[e2+1], V_e2[e2], b=b)
                    B_c_y[e2] += S*b_int*N[e1,j]
                    M_c_y[e2] += S*yb_int*N[e1,j] 

                else:
                    b_int = b_integrate(V_p1[e1], V_e1[e1], V_e2[e2+1], V_e2[e2], b)
                    xb_int = xb_integrate(V_p1[e1], V_e1[e1], V_e2[e2+1], V_e2[e2], b)
                    yb_int = yb_integrate(V_p1[e1], V_e1[e1], V_e2[e2+1], V_e2[e2], b)
                    B_c[e1,e2] += S*b_int*N[e1,j]
                    M1_c[e1,e2] += S*xb_int*N[e1,j]
                    M2_c[e1,e2] += S*yb_int*N[e1,j] 
            ## The contribution from the fragments of large particles on the upper right side         
            for i in range(e1+1, len(V_p1)):
                for j in range(e2+1,len(V_p2)):  
                    b = B_F[e1,e2,i,j]
                    S = B_R[i,j]
                    b_int = b_integrate(V_e1[e1+1], V_e1[e1], V_e2[e2+1], V_e2[e2], b)
                    xb_int = xb_integrate(V_e1[e1+1], V_e1[e1], V_e2[e2+1], V_e2[e2], b)
                    yb_int = yb_integrate(V_e1[e1+1], V_e1[e1], V_e2[e2+1], V_e2[e2], b)
                    B_c[e1,e2] += S*b_int*N[i,j]
                    M1_c[e1,e2] += S*xb_int*N[i,j]
                    M2_c[e1,e2] += S*yb_int*N[i,j]  
            if B_c[e1,e2]!=0:
                v1[e1,e2] = M1_c[e1,e2]/B_c[e1,e2]
                v2[e1,e2] = M2_c[e1,e2]/B_c[e1,e2]
            if B_c_x[e1] !=0:
                vx[e1] = M_c_x[e1] / B_c_x[e1]
            if B_c_y[e2] !=0:
                vy[e2] = M_c_y[e2] / B_c_y[e2]
    
    # volume_erro_xy = M1_c.sum() + M2_c.sum() + M_c_x.sum() + M_c_y.sum() + D_M.sum()
    # volume_erro = M1_c.sum() + M2_c.sum() + D_M.sum()
    # print(volume_erro_xy)
    # print(volume_erro)
    
    # Assign BIRTH on each pivot
    for i in range(len(V_p1)):
        for j in range(len(V_p2)): 
            for p in range(2):
                for q in range(2):
                    B[i,j] += B_c[i-p,j-q] \
                        *lam_2d(v1[i-p,j-q],v2[i-p,j-q],V_p1,V_p2,i,j,"-","-") \
                        *heaviside_jit((-1)**p*(V_p1[i-p]-v1[i-p,j-q]),0.5) \
                        *heaviside_jit((-1)**q*(V_p2[j-q]-v2[i-p,j-q]),0.5) 
                          
                    B[i,j] += B_c[i-p,j+q] \
                        *lam_2d(v1[i-p,j+q],v2[i-p,j+q],V_p1,V_p2,i,j,"-","+") \
                        *heaviside_jit((-1)**p*(V_p1[i-p]-v1[i-p,j+q]),0.5) \
                        *heaviside_jit((-1)**(q+1)*(V_p2_ex[j+q]-v2[i-p,j+q]),0.5) 

                    B[i,j] += B_c[i+p,j-q] \
                        *lam_2d(v1[i+p,j-q],v2[i+p,j-q],V_p1,V_p2,i,j,"+","-") \
                        *heaviside_jit((-1)**(p+1)*(V_p1_ex[i+p]-v1[i+p,j-q]),0.5) \
                        *heaviside_jit((-1)**q*(V_p2[j-q]-v2[i+p,j-q]),0.5)

                    B[i,j] += B_c[i+p,j+q] \
                        *lam_2d(v1[i+p,j+q],v2[i+p,j+q],V_p1,V_p2,i,j,"+","+") \
                        *heaviside_jit((-1)**(p+1)*(V_p1_ex[i+p]-v1[i+p,j+q]),0.5) \
                        *heaviside_jit((-1)**(q+1)*(V_p2_ex[j+q]-v2[i+p,j+q]),0.5)

    e1 = 0
    for j in range(0,len(V_p2)): 
        B_tem = 0
        B_tem += B_c_y[j]*lam(vy[j], V_p2, j, 'm')*heaviside_jit(V_p2[j]-vy[j],0.5)
        B_tem += B_c_y[j-1]*lam(vy[j-1], V_p2, j, 'm')*heaviside_jit(vy[j-1]-V_p2[j-1],0.5)
        B_tem += B_c_y[j]*lam(vy[j], V_p2, j, 'p')*heaviside_jit(vy[j]-V_p2[j],0.5)
        B_tem += B_c_y[j+1]*lam(vy[j+1], V_p2, j, 'p')*heaviside_jit(V_p2_ex[j+1]-vy[j+1],0.5)
        if j == 0:
            ## It seems that the calculation of 1d will be repeated on the 
            ## smallest particle(for two axis). 
            ## The case that the x and y coordinates are the same can be 
            ## handled by dividing by 2. 
            ## But other cases are not clear!
            B_tem /= 2
        B[e1,j] += B_tem
    e2 = 0
    for i in range(0,len(V_p1)):  
        B_tem = 0          
        B_tem += B_c_x[i]*lam(vx[i], V_p1, i, 'm')*heaviside_jit(V_p1[i]-vx[i],0.5)
        B_tem += B_c_x[i-1]*lam(vx[i-1], V_p1, i, 'm')*heaviside_jit(vx[i-1]-V_p1[i-1],0.5)
        B_tem += B_c_x[i]*lam(vx[i], V_p1, i, 'p')*heaviside_jit(vx[i]-V_p1[i],0.5)
        B_tem += B_c_x[i+1]*lam(vx[i+1], V_p1, i, 'p')*heaviside_jit(V_p1_ex[i+1]-vx[i+1],0.5)  
        if i == 0:
            B_tem /= 2
        B[i,e2] += B_tem

    dNdt = B + D
    
    return dNdt.reshape(-1)  

if __name__ == "__main__":    
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
        
        for i in range(NS):
            V_e[i+1] = S**i*V01
            V_p[i] = (V_e[i]+V_e[i+1]) / 2
            # N[i,0] = -np.exp(-V_e[i+1]) - (-np.exp(-V_e[i]))
            
        X1_vol = np.ones(NS)
        
        B_R = np.zeros(NS)
        B_F = np.zeros((NS,NS))
        
        if BREAKRVAL == 1:
            B_R[1:] = 1
        elif BREAKRVAL == 2:
            for idx, tmp in np.ndenumerate(B_R):
                if idx[0] == 0:
                    continue
                else:
                    B_R[idx] = V_p[idx]
                
        ## Validation: breakage function dependent only on parent particle
        for idx, tmp in np.ndenumerate(B_F):
            a = idx[0]; i = idx[1]
            # if i != 0:
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
        
        if BREAKFVAL == 2 and BREAKRVAL == 2:
            # see Kumar Dissertation A.1
            N_as = np.zeros((NS,len(t)))
            V_sum = np.zeros((NS,len(t)))
            delta = np.zeros(NS)
            theta = np.zeros(NS)
            delta[-1] = 1
            theta[:-1] = 1
            for i in range(0, len(V_p)):
                for j in range(len(t)):
                    ## integrate the analytical solution for n(t,x) with exp-distribution initial condition
                    # N_as[i,j] = np.exp(-V_e[i+1]*(1+t[j]))*(-t[j]-1) -\
                    #     np.exp(-V_e[i]*(1+t[j]))*(-t[j]-1)
                    # V_sum[i,j] = N_as[i,j] * V_p[i]
                    ## integrate the analytical solution for n(t,x) with mono-disperse initial condition
                    if i != len(V_p)-1:
                        N_as[i,j] = (-(t[j]*V_p[-1]+1)+t[j]*V_e[i+1])*np.exp(-V_e[i+1]*t[j])-\
                            (-(t[j]*V_p[-1]+1)+t[j]*V_e[i])*np.exp(-V_e[i]*t[j])
                    else:
                        N_as[i,j] = (-(t[j]*V_p[-1]+1)+t[j]*V_p[i])*np.exp(-V_p[i]*t[j])-\
                            (-(t[j]*V_p[-1]+1)+t[j]*V_e[i])*np.exp(-V_e[i]*t[j]) + \
                            (np.exp(-t[j]*V_p[i]))
                    V_sum[i,j] = N_as[i,j] * V_p[i]
        mu1_as = V_sum.sum(axis=0)  
        mu0_as = N_as.sum(axis=0)
        ax2.plot(t, mu0_as, color='k', linestyle='-.', label='$\mu_0$ (analytical)')
        # ax2.plot(t, mu1_as, color='b', linestyle='-.', label='$\mu_1$ (analytical)')
        ax2.plot(t, mu1, color=c_KIT_red, label='$\mu_1$')     
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.legend()
        plt.tight_layout()
        
        nE = NE / (V_e[1:] - V_e[:-1])
        NE_as = N_as[:,-1]
        nE_as = NE_as / (V_e[1:] - V_e[:-1])
        fig3=plt.figure(figsize=[4,3])    
        ax3=fig3.add_subplot(1,1,1) 
        ax3.plot(V_p, nE, color=c_KIT_green, label='$\Particle numerber$ (numerical)')
        ax3.plot(V_p, nE_as, color='k', linestyle='-.', label='$\Particle numerber$ (analytical)')
        ax3.set_xscale('log')
        plt.tight_layout()
        
    #%% NEW 2D    
    if dim == 2:
        # V_e: Volume of EDGES
        V_e1 = np.zeros(NS+1) #np.zeros(NS+1)
        V_e2 = np.zeros(NS+1) #np.zeros(NS+1)  
        # V_e1[0], V_e2[0] =-V01, -V02
        
        # V_p: Volume of PIVOTS
        V_p1 = np.zeros(NS)#np.zeros(NS)
        V_p2 = np.zeros(NS)#np.zeros(NS)
        V_p = np.ones((NS,NS))#np.zeros((NS,NS)) 
        
        # Volume fractions
        X1 = np.zeros((NS,NS))#np.zeros((NS,NS)) 
        X2 = np.zeros((NS,NS))#np.zeros((NS,NS)) 
        
        # SOLUTION N is saved on pivots
        N = np.zeros((NS,NS,len(t)))#np.zeros((NS,NS,len(t)))
        N[-1,-1,0] = 1
        # N[0,-1,0] = 1
        # N[-1,0,0] = 1
        
        for i in range(NS):
            V_e1[i+1] = S**i*V01
            V_e2[i+1] = S**i*V02
            V_p1[i] = (V_e1[i] + V_e1[i+1]) / 2
            V_p2[i] = (V_e2[i] + V_e2[i+1]) / 2
            
        # V_e1[0], V_e2[0] = 0.0, 0.0
        # Calculate remaining entries of V_e and V_p and other matrices
        for i in range(NS): #range(NS)
            for j in range(NS): #range(NS)
                V_p[i,j] = V_p1[i]+V_p2[j]
                if i==0 and j==0: #i==0 and j==0
                    X1[i,j] = 0
                    X2[i,j] = 0
                else:
                    X1[i,j] = V_p1[i]/V_p[i,j]
                    X2[i,j] = V_p2[j]/V_p[i,j]
        
        B_R = np.zeros((NS,NS))
        B_F = np.zeros((NS,NS,NS,NS))
        
                
        for idx, tmp in np.ndenumerate(B_F):
            a = idx[0]; b = idx[1] ; i = idx[2]; j = idx[3] 
            if BREAKFVAL == 1: 
                # if i == 0 and j == 0:
                #     continue
                # elif i == 0:
                #     B_F[idx] = 4 / (V_p2[j])
                # elif j == 0:
                #     B_F[idx] = 4 / (V_p1[i])
                # else:
                B_F[idx] = 4 / (V_p1[i]*V_p2[j])
            elif BREAKFVAL == 2:
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    B_F[idx] = 2 / (V_p2[j])
                elif j == 0:
                    B_F[idx] = 2 / (V_p1[i])
                else:
                    B_F[idx] = 2 / (V_p1[i]*V_p2[j])
                    
        if BREAKRVAL == 1:
            B_R[:,:] = 1
            B_R[0,0] = 0
        elif BREAKRVAL == 2:
            for idx, tmp in np.ndenumerate(B_R):
                a = idx[0]; b = idx[1]
                if a == 0 and b == 0:
                    continue
                elif a == 0:
                    B_R[idx] = V_p2[b]
                elif b == 0:
                    B_R[idx] = V_p1[a]
                else:
                    if BREAKFVAL == 1:
                        B_R[idx] = V_p1[a]*V_p2[b]
                    else:
                        B_R[idx] = V_p1[a] + V_p2[b]
                
        
        # SOLVE    
        import scipy.integrate as integrate
        RES = integrate.solve_ivp(dNdt_2D,
                                  [0, max(t)], 
                                  N[:,:,0].reshape(-1), t_eval=t,
                                  args=(V_p1,V_p2,V_e1,V_e2,B_R,B_F,BREAKFVAL),
                                  method='RK23',first_step=0.1,rtol=1e-3)
        
        # Reshape and save result to N and t_vec
        N = RES.y.reshape((NS,NS,len(t))) #RES.y.reshape((NS,NS,len(t)))
        t = RES.t
        
        N0 = N[:,:,0]
        NE = N[:,:,-1]
        print('### Total Volume before and after..')
        print(np.sum(N0*V_p), np.sum(NE*V_p))
        
        print('### Initial dNdt..')
        dNdt0=dNdt_2D(0,N[:,:,0].reshape(-1),V_p1,V_p2,V_e1,V_e2,B_R,B_F,BREAKFVAL).reshape(NS,NS)#.reshape(NS,NS)
        print(dNdt0)
        
        VE2, VE1 = np.meshgrid(V_e2, V_e1)
        VP2, VP1 = np.meshgrid(V_p2, V_p1)
    
        fig=plt.figure(figsize=[5,5])    
        ax=fig.add_subplot(1,1,1) 
        ax.scatter(VE2.flatten(), VE1.flatten(), marker='x', color='k', label='edges')    
        ax.scatter(VP2.flatten(), VP1.flatten(), marker='d', color=c_KIT_green, label='pivots')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()
             
        fig2=plt.figure(figsize=[4,3])    
        ax2=fig2.add_subplot(1,1,1) 
        
        mu1 = np.zeros(t.shape)
        mu0 = np.zeros(t.shape)
        mu11 = np.zeros(t.shape)
        for ti in range(len(t)): 
            mu1[ti] = np.sum(V_p*N[:,:,ti])/np.sum(V_p*N[:,:,0])  
            mu0[ti] = np.sum(N[:,:,ti])/np.sum(N[:,:,0]) 
            mu11[ti] = np.sum(V_p1*V_p2*N[:,:,ti])/np.sum(V_p1*V_p2*N[:,:,0])  
            
        ax2.plot(t, mu0, color=c_KIT_green, label='$\mu_0$ (numerical)') 
        mu_as = np.zeros((2,2,len(t)))
        if BREAKRVAL == 1 and BREAKFVAL == 1:
            for k in range(2):
                for l in range(2):
                    mu_as[k,l,:] = np.exp((4/((k+1)*(l+1))-1)*t)
        elif BREAKRVAL == 1 and BREAKFVAL == 2:
            for k in range(2):
                for l in range(2):
                    mu_as[k,l,:] = np.exp((2/((k+1)*(l+1))-1)*t)
        elif BREAKRVAL == 2 and BREAKFVAL == 1:
            mu_as[0,0,:] = 1.0 + 3*V_p[-1,-1]*t
            mu_as[1,1,:] = V_p1[-1]*V_p2[-1]
        else: 
            mu_as[0,0,:] = 1.0 + V_p[-1,-1]*t
            
        ax2.plot(t, mu_as[0,0,:], color='k', linestyle='-.', label='$\mu_0$ (analytical)')
        ax2.plot(t, mu1, color=c_KIT_red, label='$\mu_1$')     
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.legend()
        plt.tight_layout()