# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:35:05 2024

@author: xy0264
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
S = 2
R01, R02 = 1, 1
V01, V02 = 1e-9, 1e-9
V_crit = 1e-6
corr_beta = 1
dim = 2
## BREAKRVAL == 1: 1, constant breakage rate
## BREAKRVAL == 2: x*y or x + y, breakage rate is related to particle size
BREAKRVAL = 2
## BREAKFVAL == 1: 4/x'y', meet the first cross moment
## BREAKFVAL == 1: 2/x'y', meet the first moment/ mass conversation
BREAKFVAL = 2
# art_flag = "agglomeration"
# art_flag = "breakage"
art_flag = "mix"

#%% FUNCTIONS
@jit(nopython=True)
def calc_1d_agglomeration(N,V_p,V_e,F_M,B_c,M_c,D,agg_crit):
    for e in range(1,agg_crit):
        # Loop through all pivots (twice)
        for i in range(agg_crit):
                for j in range(agg_crit):
                    F=F_M[i,j]
                    # Check if the agglomeration of i and j produce an 
                    # agglomerate inside the current cell 
                    # Use upper edge as "reference point"
                    if V_p[i]+V_p[j] >= V_p[agg_crit]:
                        continue
                    if V_e[e] <= V_p[i]+V_p[j] < V_e[e+1]:
                        # Total birthed agglomerates
                        B_c[e] += F*N[i]*N[j]/2
                        M_c[e] += F*N[i]*N[j]*(V_p[i]+V_p[j])/2
                        D[j] -= F*N[i]*N[j]
    return B_c, M_c, D
@jit(nopython=True)
def calc_1d_breakage(N,V_p,V_e,B_R,bf_int,xbf_int,B_c,M_c,D):
    #  Loop through all pivots
    for e in range(1,len(V_p)):
        b = bf_int[e-1,e-1]
        xb = xbf_int[e-1,e-1]
        S = B_R[e-1]
        B_c[e] += S*b*N[e]
        M_c[e] += S*xb*N[e]
        D[e] = -S*N[e]
        for i in range(e+1, len(V_p)):
            b = bf_int[e-1,i-1]
            xb = xbf_int[e-1,i-1]
            S = B_R[i-1]
            B_c[e] += S*b*N[i]
            M_c[e] += S*xb*N[i]
    return B_c, M_c, D
  
@jit(nopython=True)      
def dNdt_1D(t,N,NS,V_p,V_e,F_M,B_R,bf_int,xbf_int,art_flag,agg_crit):
    dNdt = np.zeros(N.shape)
    M_c = np.zeros(V_e.shape)
    D = np.zeros(N.shape)
    B = np.zeros(N.shape)
    B_c = np.zeros(NS+1)
    v = np.zeros(NS+1)
    V_p_ex = np.zeros(NS+1)
    V_p_ex[:-1] = V_p
    
    if art_flag == "agglomeration":
        B_c, M_c, D = calc_1d_agglomeration(N,V_p,V_e,F_M,B_c,M_c,D,agg_crit)
    elif art_flag == "breakage":
        B_c, M_c, D = calc_1d_breakage(N,V_p,V_e,B_R,bf_int,xbf_int,B_c,M_c,D)
    elif art_flag == "mix":
        B_c, M_c, D = calc_1d_agglomeration(N, V_p, V_e, F_M, B_c, M_c, D,agg_crit)
        B_c, M_c, D = calc_1d_breakage(N, V_p, V_e, B_R, bf_int,xbf_int, B_c, M_c, D)
    else:
        raise Exception("Current art_flag is not supported")
                    
    v[B_c != 0] = M_c[B_c != 0]/B_c[B_c != 0]
    
    # Assign BIRTH on each pivot
    for i in range(len(V_p)):
        # Add contribution from LEFT cell (if existent)
        B[i] += B_c[i]*lam(v[i], V_p, i, 'm')*heaviside_jit(V_p[i]-v[i],0.5)
        # Left Cell, right half
        B[i] += B_c[i-1]*lam(v[i-1], V_p, i, 'm')*heaviside_jit(v[i-1]-V_p[i-1],0.5)
        # Same Cell, right half
        B[i] += B_c[i]*lam(v[i], V_p_ex, i, 'p')*heaviside_jit(v[i]-V_p[i],0.5)
        # Right Cell, left half
        B[i] += B_c[i+1]*lam(v[i+1], V_p_ex, i, 'p')*heaviside_jit(V_p_ex[i+1]-v[i+1],0.5)
            
    dNdt = B + D
    
    return dNdt    

@jit(nopython=True)
def heaviside_jit(x1, x2):
    if x1 < 0:
        return 0.0
    elif x1 > 0:
        return 1.0
    else:
        return x2
@jit(nopython=True)
def calc_2d_agglomeration(N,V_p,V_e1,V_e2,F_M,B_c,M1_c,M2_c,D,agg_crit):
    x_crit = agg_crit[0]
    y_crit = agg_crit[1]
    # Loop through all edges
    # Go till len()-1 to make sure nothing is collected in the BORDER
    # This automatically solves border issues. If B_c is 0 in the border, nothing has to be distributed
    for e1 in range(x_crit):
        for e2 in range(y_crit):
            # Loop through all pivots (twice)
            for i in range(x_crit):
                for j in range(y_crit):
                    # a <= i and b <= j (equal is allowed!) 
                    for a in range(x_crit): #i+1
                        for b in range(y_crit): #j+1
                            if V_p[i,j] + V_p[a,b] >= V_p[x_crit,y_crit]:
                                continue
                            # Check if the agglomeration of ij and ab produce an 
                            # agglomerate inside the current cell 
                            # Use upper edge as "reference point"
                            if (V_e1[e1] <= V_p[i,0]+V_p[a,0] < V_e1[e1+1]) and \
                                (V_e2[e2] <= V_p[0,j]+V_p[0,b] < V_e2[e2+1]):
                                F = F_M[i,j,a,b]
                                B_c[e1,e2] += F*N[i,j]*N[a,b]/2
                                M1_c[e1,e2] += F*N[i,j]*N[a,b]*(V_p[i,0]+V_p[a,0])/2
                                M2_c[e1,e2] += F*N[i,j]*N[a,b]*(V_p[0,j]+V_p[0,b])/2
                                # Track death 
                                D[i,j] -= F*N[i,j]*N[a,b]

    return B_c,M1_c,M2_c,D
@jit(nopython=True)
def calc_2d_breakage(N,V_p,V_e1,V_e2,B_R,bf_int,xbf_int,ybf_int,B_c,M1_c,M2_c,D):
    ## only to check volume conservation
    # D_M = np.zeros(N.shape)
    for e1 in range(1, len(V_p1)):
        for e2 in range(1, len(V_p2)):
            ## The contribution of self-fragmentation
            b = bf_int[e1-1,e2-1,e1-1,e2-1]
            xb = xbf_int[e1-1,e2-1,e1-1,e2-1]
            yb = ybf_int[e1-1,e2-1,e1-1,e2-1]
            S = B_R[e1-1,e2-1]
            if e1 != 1 and e2 != 1:
                B_c[e1,e2] += S*b*N[e1,e2]
                M1_c[e1,e2] += S*xb*N[e1,e2]
                M2_c[e1,e2] += S*yb*N[e1,e2]
                # D_M[e1,e2] = -S*N[e1,e2]*(V_p1[e1]+V_p2[e2])
            # calculate death rate    
                D[e1,e2] = -S*N[e1,e2]
            
            ## The contributions of fragments on the same horizontal axis
            for i in range(e1+1,len(V_p1)):
                b = bf_int[e1-1,e2-1,i-1,e2-1]
                xb = xbf_int[e1-1,e2-1,i-1,e2-1]
                yb = ybf_int[e1-1,e2-1,i-1,e2-1]
                S = B_R[i-1,e2-1]
                if e2 != 1:
                    B_c[e1,e2] += S*b*N[i,e2] 
                    M1_c[e1,e2] += S*xb*N[i,e2]
                    M2_c[e1,e2] += S*yb*N[i,e2]
            ## The contributions of fragments on the same vertical axis
            for j in range(e2+1,len(V_p2)):
                b = bf_int[e1-1,e2-1,e1-1,j-1]
                xb = xbf_int[e1-1,e2-1,e1-1,j-1]
                yb = ybf_int[e1-1,e2-1,e1-1,j-1]
                S = B_R[e1-1,j-1]
                if e1 != 1:
                    B_c[e1,e2] += S*b*N[e1,j]
                    M1_c[e1,e2] += S*xb*N[e1,j]
                    M2_c[e1,e2] += S*yb*N[e1,j] 
            ## The contribution from the fragments of large particles on the upper right side         
            for i in range(e1+1, len(V_p1)):
                for j in range(e2+1,len(V_p2)):  
                    b = bf_int[e1-1,e2-1,i-1,j-1]
                    xb = xbf_int[e1-1,e2-1,i-1,j-1]
                    yb = ybf_int[e1-1,e2-1,i-1,j-1]
                    S = B_R[i-1,j-1]
                    B_c[e1,e2] += S*b*N[i,j]
                    M1_c[e1,e2] += S*xb*N[i,j]
                    M2_c[e1,e2] += S*yb*N[i,j]    
    # volume_erro = M1_c.sum() + M2_c.sum() + D_M.sum()    
    # print(volume_erro)            
    return B_c,M1_c,M2_c,D


@jit(nopython=True)
def dNdt_2D(t,NN,NS,V_p,V_e1,V_e2,F_M,B_R,bf_int,xbf_int,ybf_int,art_flag,agg_crit):
  
    N = np.copy(NN) 
    N = np.reshape(N,(NS,NS))
    dNdt = np.zeros(np.shape(N))
    D = np.zeros(np.shape(N))
    B = np.zeros(np.shape(N))
    ## Add an extra zero edges on the outer boundary to prevent mass leakage
    B_c = np.zeros((NS+1,NS+1)) 
    v1 = np.zeros((NS+1,NS+1))
    v2 = np.zeros((NS+1,NS+1))
    M1_c = np.zeros((NS+1,NS+1))
    M2_c = np.zeros((NS+1,NS+1))
    V_p_ex = np.zeros((NS+1,NS+1))
    V_p_ex[:-1,:-1] = V_p
    ## variable for 1d-breakage on the left(y) and low(x) boundary
    dNdt_bound_x = np.zeros(NS)
    dNdt_bound_y = np.zeros(NS)
    
    if art_flag == "agglomeration":
        B_c, M1_c,M2_c, D = calc_2d_agglomeration(N,V_p,V_e1,V_e2,F_M,B_c,M1_c,M2_c,D,agg_crit)
    elif art_flag == "breakage":
        ## Because the cells on the lower boundary (e1=0 or e2=0)are not allowed to break outward, 
        ## 1d calculations need to be performed on the two lower boundaries.
        dNdt_bound_x = dNdt_1D(t,N[:,1],NS,V_p[:,0],V_e1,F_M[:,0,:,0],B_R[:,0],bf_int[:,0,:,0],xbf_int[:,0,:,0],"breakage",agg_crit[0])
        dNdt_bound_y = dNdt_1D(t,N[1,:],NS,V_p[0,:],V_e2,F_M[:,0,:,0],B_R[0,:],bf_int[0,:,0,:],ybf_int[0,:,0,:],"breakage",agg_crit[1])
        ## the same to B /= 2, because there is no death on primary particle
        dNdt_bound_x[1] /= 2
        dNdt_bound_y[1] /= 2
        B_c, M1_c,M2_c, D = calc_2d_breakage(N,V_p,V_e1,V_e2,B_R,bf_int,xbf_int,ybf_int,B_c,M1_c,M2_c,D)
    elif art_flag == "mix":
        B_c, M1_c,M2_c, D = calc_2d_agglomeration(N,V_p,V_e1,V_e2,F_M,B_c,M1_c,M2_c,D,agg_crit)
        ## Because the cells on the lower boundary (e1=0 or e2=0)are not allowed to break outward, 
        ## 1d calculations need to be performed on the two lower boundaries.
        dNdt_bound_x = dNdt_1D(t,N[:,1],NS,V_p[:,0],V_e1,F_M[:,0,:,0],B_R[:,0],bf_int[:,0,:,0],xbf_int[:,0,:,0],"breakage",agg_crit[0])
        dNdt_bound_y = dNdt_1D(t,N[1,:],NS,V_p[0,:],V_e2,F_M[:,0,:,0],B_R[0,:],bf_int[0,:,0,:],ybf_int[0,:,0,:],"breakage",agg_crit[1])
        ## the same to B /= 2, because there is no death on primary particle
        dNdt_bound_x[1] /= 2
        dNdt_bound_y[1] /= 2
        B_c, M1_c,M2_c, D = calc_2d_breakage(N,V_p,V_e1,V_e2,B_R,bf_int,xbf_int,ybf_int,B_c,M1_c,M2_c,D)
    else:
        raise Exception("Current art_flag is not supported")
    
    for i in range(NS+1):
        for j in range(NS+1):
            if B_c[i,j] != 0:
                v1[i,j] = M1_c[i,j]/B_c[i,j]
                v2[i,j] = M2_c[i,j]/B_c[i,j]
    # # Assign BIRTH on each pivot
    for i in range(len(V_p[:,0])):
        for j in range(len(V_p[0,:])): 
            for p in range(2):
                for q in range(2):
                    # Actual modification calculation
                    # if (i!=0) and (j!=0):
                        B[i,j] += B_c[i-p,j-q] \
                            *lam_2d(v1[i-p,j-q],v2[i-p,j-q],V_p[:,0],V_p[0,:],i,j,"-","-") \
                            *heaviside_jit((-1)**p*(V_p[i-p,0]-v1[i-p,j-q]),0.5) \
                            *heaviside_jit((-1)**q*(V_p[0,j-q]-v2[i-p,j-q]),0.5) 
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B1', B[i,j])
                    # if (i!=0) and (j!=len(V_p[0,:])-1):                           
                        B[i,j] += B_c[i-p,j+q] \
                            *lam_2d(v1[i-p,j+q],v2[i-p,j+q],V_p[:,0],V_p[0,:],i,j,"-","+") \
                            *heaviside_jit((-1)**p*(V_p[i-p,0]-v1[i-p,j+q]),0.5) \
                            *heaviside_jit((-1)**(q+1)*(V_p_ex[0,j+q]-v2[i-p,j+q]),0.5) 
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B2', B[i,j])
                    # if (i!=len(V_p[:,0])-1) and (j!=0):
                        B[i,j] += B_c[i+p,j-q] \
                            *lam_2d(v1[i+p,j-q],v2[i+p,j-q],V_p[:,0],V_p[0,:],i,j,"+","-") \
                            *heaviside_jit((-1)**(p+1)*(V_p_ex[i+p,0]-v1[i+p,j-q]),0.5) \
                            *heaviside_jit((-1)**q*(V_p[0,j-q]-v2[i+p,j-q]),0.5)
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B3', B[i,j])
                    # if (i!=len(V_p[:,0])-1) and (j!=len(V_p[0,:])-1): 
                        B[i,j] += B_c[i+p,j+q] \
                            *lam_2d(v1[i+p,j+q],v2[i+p,j+q],V_p[:,0],V_p[0,:],i,j,"+","+") \
                            *heaviside_jit((-1)**(p+1)*(V_p_ex[i+p,0]-v1[i+p,j+q]),0.5) \
                            *heaviside_jit((-1)**(q+1)*(V_p_ex[0,j+q]-v2[i+p,j+q]),0.5)
                            
    dNdt = B + D
    dNdt[:,1] += dNdt_bound_x
    dNdt[1,:] += dNdt_bound_y
    
    return dNdt.reshape(-1)   

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
def delta(i,j,a,b):
    if i==a and j==b:
        return 0.5
    else:
        return 1
    
@jit(nopython=True)
def b_integrate(x_up,x_low,y_up=None,y_low=None,bf=None):
    if y_up is None or y_low is None:
        return (x_up - x_low)*bf
    else:
        return (x_up- x_low)*(y_up - y_low)*bf
@jit(nopython=True)    
def xb_integrate(x_up,x_low,y_up=None,y_low=None,bf=None):
    if y_up is None or y_low is None:
        return (x_up**2 - x_low**2)*0.5*bf
    else:
        return (x_up**2- x_low**2)*(y_up - y_low)*0.5*bf
@jit(nopython=True)    
def yb_integrate(x_up,x_low,y_up=None,y_low=None,bf=None):
    return (y_up**2 - y_low**2)*(x_up-x_low)*0.5*bf  

if __name__ == "__main__":    
    #%% NEW 1D
    if dim == 1:
        # V_e: Volume of EDGES
        V_e = np.zeros(NS+1) 
        V_e[0] = -V01
        # V_p: Volume of PIVOTS
        V_p = np.zeros(NS)
        # SOLUTION N is saved on pivots
        N = np.zeros((NS,len(t)))
        #N[0,0] = 0.1
        if art_flag == "agglomeration":
            N[1,0] = 0.3
        elif art_flag == "breakage":
            N[-1,0] = 1
        else:
            # N[1,0] = 0.3
            N[-1,0] = 1
        #N[2,0] = 0.2
        
        for i in range(1,NS+1):
            V_e[i] = S**(i-1)*V01
            # ith pivot is mean between ith and (i+1)th edge
            V_p[i-1] = (V_e[i] + V_e[i-1])/2
            
        index_crit = np.where(V_p < V_crit)[0]
        agg_crit = index_crit[-1] if (index_crit.size > 0 and index_crit.size < len(V_p)) else (len(V_p) -1)
        agg_crit = agg_crit
        ## Let the integration range associated with the breakage function start from zero 
        ## to ensure mass conservation  
        V_e_tem = np.zeros(NS) 
        V_e_tem[:] = V_e[1:]
        V_e_tem[0] = 0.0
        
        F_M = np.zeros((NS-1,NS-1))
        ## constant kernal
        # F_M[:,:] = 1
        ## sum kernal
        for idx, tmp in np.ndenumerate(F_M):
            if idx[0]==0 or idx[1]==0:
                continue
            a = idx[0]; b = idx[1]
            F_M[idx] = (V_p[a] + V_p[b])/V_p[1]
        
        B_R = np.zeros(NS-1)
        B_F = np.zeros((NS-1,NS-1))
        bf_int = np.zeros((NS-1,NS-1))
        xbf_int = np.zeros((NS-1,NS-1))
        
        if BREAKRVAL == 1:
            B_R[1:] = 1
        elif BREAKRVAL == 2:
            for idx, tmp in np.ndenumerate(B_R):
                a = idx[0]
                if idx[0] != 0:
                    B_R[idx] = V_p[a+1]
                
        ## Validation: breakage function dependent only on parent particle
        for idx, tmp in np.ndenumerate(B_F):
            a = idx[0]; i = idx[1]
            if i != 0 and a <= i:
                if BREAKFVAL == 1:  
                    B_F[idx] = 4 / (V_p[i+1])
                elif BREAKFVAL == 2:
                    # B_F[idx] = 2 / (V_p[i+1])
                    bf = 2 / (V_p[i+1])
                    if a == i:
                        bf_int[idx] = b_integrate(V_p[a+1],V_e_tem[a],bf=bf)
                        xbf_int[idx] = xb_integrate(V_p[a+1],V_e_tem[a],bf=bf)
                    else:
                        bf_int[idx] = b_integrate(V_e_tem[a+1],V_e_tem[a],bf=bf)
                        xbf_int[idx] = xb_integrate(V_e_tem[a+1],V_e_tem[a],bf=bf)
        # SOLVE    
        import scipy.integrate as integrate
        RES = integrate.solve_ivp(dNdt_1D,
                                  [0, max(t)], 
                                  N[:,0], t_eval=t,
                                  args=(NS,V_p,V_e,F_M,B_R,bf_int,xbf_int,art_flag,agg_crit),
                                  method='RK45',first_step=0.1,rtol=1e-3)
        
        # Reshape and save result to N and t_vec
        N = RES.y
        t = RES.t
        
        N0 = N[:,0]
        NE = N[:,-1]
        print('### Total Volume before and after..')
        print(np.sum(N0*V_p), np.sum(NE*V_p))
        
        print('### Initial dNdt..')
        dNdt0=dNdt_1D(0,N[:,0],NS,V_p,V_e,F_M,B_R,bf_int,xbf_int,art_flag,agg_crit)
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
            mu0[ti] = np.sum(N[:,ti])/np.sum(N[:,0]) 
        
        ax2.plot(t, mu0, color=c_KIT_green, label='$\mu_0$ (numerical)') 
        
        # mu0_as = 2/(2+np.sum(N[:,0])*t)
        if art_flag == "agglomeration":
            mu0_as = np.exp(-np.sum(N[:,0])*t)  
        elif art_flag == "breakage":
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
            mu0_as = N_as.sum(axis=0)
        else:
            mu0_as = 1*t
        
        ax2.plot(t, mu0_as, color='k', linestyle='-.', label='$\mu_0$ (analytical)')
        ax2.plot(t, mu1, color=c_KIT_red, label='$\mu_1$')     
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.legend()
        plt.tight_layout()
        
        nE = NE #/ (V_e[1:] - V_e[:-1])
        fig3=plt.figure(figsize=[4,3])    
        ax3=fig3.add_subplot(1,1,1) 
        ax3.plot(V_p, nE, color=c_KIT_green, label='$\Particle numerber$ (numerical)')
        # ax3.set_xscale('log')
        plt.tight_layout()
    
    #%% NEW 2D    
    if dim == 2:
        # V_e: Volume of EDGES
        V_e1 = np.zeros(NS+1) #np.zeros(NS+1)
        V_e2 = np.zeros(NS+1) #np.zeros(NS+1)    
        # Make first cell symmetric --> pivots fall on 0
        V_e1[0], V_e2[0] = -V01, -V02
        
        # V_p: Volume of PIVOTS
        V_p1 = np.zeros(NS)#np.zeros(NS)
        V_p2 = np.zeros(NS)#np.zeros(NS)
        V_p = np.ones((NS,NS))#np.zeros((NS,NS)) 
        
        # Volume fractions
        X1 = np.zeros((NS,NS))#np.zeros((NS,NS)) 
        X2 = np.zeros((NS,NS))#np.zeros((NS,NS)) 
        
        # SOLUTION N is saved on pivots
        N = np.zeros((NS,NS,len(t)))#np.zeros((NS,NS,len(t)))
        
        for i in range(0,NS):#range(1,NS+1)
            # ith pivot is mean between ith and (i+1)th edge
            V_e1[i+1] = S**(i)*V01
            V_e2[i+1] = S**(i)*V02
            V_p1[i] = (V_e1[i] + V_e1[i+1]) / 2#S**(i-1)*V01
            V_p2[i] = (V_e2[i] + V_e2[i+1]) / 2#S**(i-1)*V02
        if art_flag == "agglomeration":
            N[0,1,0] = 0.3
            N[1,0,0] = 0.3
        elif art_flag == "breakage":
            N[-1,-1,0] = 1
        else:
            N[-1,-1,0] = 1   
            N[0,1,0] = 0.3
            N[1,0,0] = 0.3
        
        ## Let the integration range associated with the breakage function start from zero 
        ## to ensure mass conservation   
        V_e1_tem = np.zeros(NS) 
        V_e2_tem = np.zeros(NS) 
        V_e1_tem[:] = V_e1[1:]
        V_e2_tem[:] = V_e2[1:]
        V_e1_tem[0] = 0.0
        V_e2_tem[0] = 0.0

        V_p[:,0] = V_p1 
        V_p[0,:] = V_p2 
        index1_crit = np.where(V_p1 < V_crit)[0]
        index2_crit = np.where(V_p2 < V_crit)[0]
        agg_crit = np.zeros(2, dtype=int)
        agg_crit[0] = index1_crit[-1] if (index1_crit.size > 0 and index1_crit.size < len(V_p1)) else (len(V_p1) -1)
        agg_crit[1] = index2_crit[-1] if (index2_crit.size > 0 and index2_crit.size < len(V_p2)) else (len(V_p2) -1)
        
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
        ## To save computing resources, do not set parameters for corresponding particles 
        ## which do not participate in calculations.
        ## That means, F_M does not include the right and upper boundaries
        ## B_R and B_F do not include the left and low boundaries
        F_M = np.zeros((NS-1,NS-1,NS-1,NS-1))
        B_R = np.zeros((NS-1,NS-1))
        B_F = np.zeros((NS-1,NS-1,NS-1,NS-1))
        bf_int = np.zeros((NS-1,NS-1,NS-1,NS-1))
        xbf_int = np.zeros((NS-1,NS-1,NS-1,NS-1))
        ybf_int = np.zeros((NS-1,NS-1,NS-1,NS-1))
        
        F_M_tem=1
        for idx, tmp in np.ndenumerate(B_F):
            a = idx[0]; b = idx[1] ; i = idx[2]; j = idx[3] 
            if a + b != 0 and i + j != 0:
                F_M[idx] = F_M_tem
            # F_M[idx] = (V_p[a,b] + V_p[i,j]) * corr_beta
            if i + j != 0 and a<=i or b <= j:
                if BREAKFVAL == 1: 
                    # if i == 0 and j == 0:
                    #     continue
                    # elif i == 0:
                    #     B_F[idx] = 4 / (V_p2[j])
                    # elif j == 0:
                    #     B_F[idx] = 4 / (V_p1[i])
                    # else:
                    B_F[idx] = 4 / (V_p1[i+1]*V_p2[j+1])
                elif BREAKFVAL == 2:
                    if i == 0:
                        ## for left boundary/y
                        # B_F[idx] = 2 / (V_p2[j+1])
                        bf = 2 / (V_p2[j+1])
                        if j == 0:
                            continue
                        elif b == j:
                            bf_int[idx] = b_integrate(V_p2[b+1],V_e2_tem[b],bf=bf)
                            ybf_int[idx] = xb_integrate(V_p2[b+1],V_e2_tem[b],bf=bf)
                        else:
                            bf_int[idx] = b_integrate(V_e2_tem[b+1],V_e2_tem[b],bf=bf)
                            ybf_int[idx] = xb_integrate(V_e2_tem[b+1],V_e2_tem[b],bf=bf)
                    elif j == 0:
                        ## for low boundary/x
                        # B_F[idx] = 2 / (V_p1[i+1])
                        bf = 2 / (V_p1[i+1])
                        if a == i:
                            bf_int[idx] = b_integrate(V_p1[a+1],V_e1_tem[a],bf=bf)
                            xbf_int[idx] = xb_integrate(V_p1[a+1],V_e1_tem[a],bf=bf)
                        else:
                            bf_int[idx] = b_integrate(V_e1_tem[a+1],V_e1_tem[a],bf=bf)
                            xbf_int[idx] = xb_integrate(V_e1_tem[a+1],V_e1_tem[a],bf=bf)
                    else:
                        # B_F[idx] = 2 / (V_p1[i+1]*V_p2[j+1])
                        bf = 2 / (V_p1[i+1]*V_p2[j+1])
                        ## The contributions of fragments on the same vertical axis
                        if a == i and b == j:
                            bf_int[idx] = b_integrate(V_p1[a+1], V_e1_tem[a], V_p2[b+1], V_e2_tem[b], bf)
                            xbf_int[idx] = xb_integrate(V_p1[a+1], V_e1_tem[a], V_p2[b+1], V_e2_tem[b], bf)
                            ybf_int[idx] = yb_integrate(V_p1[a+1], V_e1_tem[a], V_p2[b+1], V_e2_tem[b], bf)
                        elif a == i:
                            bf_int[idx] = b_integrate(V_p1[a+1], V_e1_tem[a], V_e2_tem[b+1], V_e2_tem[b], bf)
                            xbf_int[idx] = xb_integrate(V_p1[a+1], V_e1_tem[a], V_e2_tem[b+1], V_e2_tem[b], bf)
                            ybf_int[idx] = yb_integrate(V_p1[a+1], V_e1_tem[a], V_e2_tem[b+1], V_e2_tem[b], bf)
                        ## The contributions of fragments on the same horizontal axis
                        elif b == j:   
                            bf_int[idx] = b_integrate(V_e1_tem[a+1], V_e1_tem[a], V_p2[b+1], V_e2_tem[b], bf)
                            xbf_int[idx] = xb_integrate(V_e1_tem[a+1], V_e1_tem[a], V_p2[b+1], V_e2_tem[b], bf)
                            ybf_int[idx] = yb_integrate(V_e1_tem[a+1], V_e1_tem[a], V_p2[b+1], V_e2_tem[b], bf)
                        ## The contribution from the fragments of large particles on the upper right side 
                        else:
                            bf_int[idx] = b_integrate(V_e1_tem[a+1], V_e1_tem[a], V_e2_tem[b+1], V_e2_tem[b], bf)
                            xbf_int[idx] = xb_integrate(V_e1_tem[a+1], V_e1_tem[a], V_e2_tem[b+1], V_e2_tem[b], bf)
                            ybf_int[idx] = yb_integrate(V_e1_tem[a+1], V_e1_tem[a], V_e2_tem[b+1], V_e2_tem[b], bf)
                        
        if BREAKRVAL == 1:
            B_R[:,:] = 1
            B_R[0,0] = 0
        elif BREAKRVAL == 2:
            for idx, tmp in np.ndenumerate(B_R):
                a = idx[0]; b = idx[1]
                if a == 0 and b == 0:
                    continue
                elif a == 0:
                    B_R[idx] = V_p2[b+1]
                elif b == 0:
                    B_R[idx] = V_p1[a+1]
                else:
                    if BREAKFVAL == 1:
                        B_R[idx] = V_p1[a+1]*V_p2[b+1]
                    else:
                        B_R[idx] = V_p1[a+1] + V_p2[b+1]
            
        # SOLVE    
        import scipy.integrate as integrate
        RES = integrate.solve_ivp(dNdt_2D,
                                  [0, max(t)], 
                                  N[:,:,0].reshape(-1), t_eval=t,
                                  args=(NS,V_p,V_e1,V_e2,F_M,B_R,bf_int,xbf_int,ybf_int,art_flag,agg_crit),
                                  method='RK23',first_step=0.1,rtol=1e-3)
        
        # Reshape and save result to N and t_vec
        N = RES.y.reshape((NS,NS,len(t))) #RES.y.reshape((NS,NS,len(t)))
        t = RES.t
        
        N0 = N[:,:,0]
        NE = N[:,:,-1]
        print('### Total Volume before and after..')
        print(np.sum(N0*V_p), np.sum(NE*V_p))
        
        print('### Initial dNdt..')
        dNdt0=dNdt_2D(0,N[:,:,0].reshape(-1),NS,V_p,V_e1,V_e2,F_M,B_R,bf_int,xbf_int,ybf_int,art_flag,agg_crit).reshape(NS,NS)#.reshape(NS,NS)
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
        for ti in range(len(t)): 
            mu1[ti] = np.sum(V_p*N[:,:,ti])/np.sum(V_p*N[:,:,0])  
            mu0[ti] = np.sum(N[:,:,ti])/np.sum(N[:,:,0]) 
            
        ax2.plot(t, mu0, color=c_KIT_green, label='$\mu_0$ (numerical)') 
        if art_flag == "agglomeration":
            mu0_as = 2/(2+F_M_tem*np.sum(N[:,:,0])*t)  
        elif art_flag == "breakage":
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
            mu0_as = mu_as[0,0,:]
        else:
            mu0_as = 1*t
        
        ax2.plot(t, mu0_as, color='k', linestyle='-.', label='$\mu_0$ (analytical)')
        ax2.plot(t, mu1, color=c_KIT_red, label='$\mu_1$')     
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.legend()
        plt.tight_layout()


