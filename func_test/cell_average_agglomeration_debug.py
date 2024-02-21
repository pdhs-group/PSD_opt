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
t = np.arange(0, 601, 60, dtype=float)
NS = 12
S = 2
R01, R02 = 1, 1
V01, V02 = 1, 1
dim = 2

#%% FUNCTIONS
def dNdt_1D(t,N,V_p,V_e,F_M):
    dNdt = np.zeros(N.shape)
    B_c = np.zeros(V_e.shape)
    M_c = np.zeros(V_e.shape)
    v = np.zeros(V_e.shape)
    D = np.zeros(N.shape)
    B = np.zeros(N.shape)
    
    # Loop through all edges
    # -1 to make sure nothing overshoots (?) CHECK THIS
    for e in range(1, len(V_e)-1):
        # Loop through all pivots (twice)
        for i in range(len(V_p)):
            for j in range(len(V_p)):
                # Unequal sized processes are counted double --> Set zeta to 2
                # if i==j:
                #     zeta = 1
                # else:
                #     zeta = 2
                
                # Setting zeta=2 for all combinations seems correct
                # See commented code above, I though we need to set it to 1 sometimes..
                zeta = 2
                
                F=F_M[i,j]
                # Check if the agglomeration of i and j produce an 
                # agglomerate inside the current cell 
                # Use upper edge as "reference point"
                if V_e[e-1] <= V_p[i]+V_p[j] < V_e[e]:
                    # Total birthed agglomerates
                    B_c[e] += F*N[i]*N[j]/zeta
                    M_c[e] += F*N[i]*N[j]*(V_p[i]+V_p[j])/(zeta)
                    
                    # Track death 
                    D[i] -= F*N[i]*N[j]/zeta
                    D[j] -= F*N[i]*N[j]/zeta
                    
                v[B_c != 0] = M_c[B_c != 0]/B_c[B_c != 0]
    
    # print(B_c)
    
    # Assign BIRTH on each pivot
    for i in range(len(V_p)):            
        # Add contribution from LEFT cell (if existent)
        if i != 0:
            # Same Cell, left half
            B[i] += B_c[i+1]*lam(v[i+1], V_p, i, 'm')*np.heaviside(V_p[i]-v[i+1],0.5)
            # Left Cell, right half
            B[i] += B_c[i]*lam(v[i], V_p, i, 'm')*np.heaviside(v[i]-V_p[i-1],0.5)
            
        # Add contribution from RIGHT cell (if existent)
        if i != len(V_p)-1:
            # Same Cell, right half
            B[i] += B_c[i+1]*lam(v[i+1], V_p, i, 'p')*np.heaviside(v[i+1]-V_p[i],0.5)
            # Right Cell, left half
            B[i] += B_c[i+2]*lam(v[i+2], V_p, i, 'p')*np.heaviside(V_p[i+1]-v[i+2],0.5)
            
    dNdt = B + D
    
    return dNdt    

# Define np.heaviside for JIT compilation
# @overload(np.heaviside)
# def np_heaviside(x1, x2):
#     @register_jitable
#     def heaviside_impl(x1, x2):
#         if x1 < 0:
#             return 0.0
#         elif x1 > 0:
#             return 1.0
#         else:
#             return x2

#     return heaviside_impl 

@jit(nopython=True)
def heaviside_jit(x1, x2):
    if x1 < 0:
        return 0.0
    elif x1 > 0:
        return 1.0
    else:
        return x2

@jit(nopython=True)
def dNdt_2D(t,NN,V_p,V1_e,V3_e,NS,F_M):
  
    N = np.copy(NN) 
    N = np.reshape(N,(NS,NS))
    dNdt = np.zeros(np.shape(N))
    D = np.zeros(np.shape(N))
    # shape_mit_boundary = tuple(dim + 1 for dim in N.shape)
    B_c = np.zeros(np.shape(N)) #np.shape(V_e) (old)
    M1_c = np.zeros(np.shape(N))
    M2_c = np.zeros(np.shape(N))
    v1 = np.zeros(np.shape(N))
    v2 = np.zeros(np.shape(N))
    D = np.zeros(np.shape(N))
    B = np.zeros(np.shape(N))
    
    # Loop through all edges
    # Go from 2 till len()-1 to make sure nothing is collected in the BORDER
    # This automatically solves border issues. If B_c is 0 in the border, nothing has to be distributed
    for e1 in range(len(V_p[:,0])-1):
        for e2 in range(len(V_p[0,:])-1):
            # Loop through all pivots (twice)
            for i in range(len(V_p[:,0])):
                for j in range(len(V_p[0,:])):
                    # a <= i and b <= j (equal is allowed!) 
                    for a in range(len(V_p[:,0])): #i+1
                        for b in range(len(V_p[:,0])): #j+1
                            # Check if the agglomeration of ij and ab produce an 
                            # agglomerate inside the current cell 
                            # Use upper edge as "reference point"
                            if (V1_e[e1] <= V_p[i,0]+V_p[a,0] < V1_e[e1+1]) and \
                                (V3_e[e2] <= V_p[0,j]+V_p[0,b] < V3_e[e2+1]):
                                # if abs(V_p[a,b]+V_p[i,j]-V_p[e1,e2]) < 1e-5*V_p[1,0]:
                                #     #print(i,a,c,'|',j,b,d)
                                #     zeta = 1
                                # else:
                                #     zeta = 2
                                zeta = 2    
                                F = F_M[i,j,a,b]
                                # Total birthed agglomerates
                                # e1 and e2 start at 1 (upper edge) --> corresponds to cell "below"
                                # Use B_c[e1-1, e2-1] (e.g. e1=1, e2=1 corresponds to the first CELL [0,0])
                                B_c[e1,e2] += F*N[i,j]*N[a,b]/zeta
                                M1_c[e1,e2] += F*N[i,j]*N[a,b]*(V_p[i,0]+V_p[a,0])/zeta
                                M2_c[e1,e2] += F*N[i,j]*N[a,b]*(V_p[0,j]+V_p[0,b])/zeta
     
                                # Track death 
                                D[i,j] -= F*N[i,j]*N[a,b]
                                # D[a,b] -= F*N[i,j]*N[a,b]/zeta
                                #print(D)
            # Average volume 
            if B_c[e1,e2]!=0:
                v1[e1,e2] = M1_c[e1,e2]/B_c[e1,e2]
                v2[e1,e2] = M2_c[e1,e2]/B_c[e1,e2]
                                
    # 30.01.24 B_c, D and v1 seem correct (judging from first dNdt)
    # print(B_c)
    # print(D)
    # print(v1)
    
    # # Assign BIRTH on each pivot
    for i in range(len(V_p[:,0])-1):
        for j in range(len(V_p[0,:])-1): 
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
                            *heaviside_jit((-1)**(q+1)*(V_p[0,j+q]-v2[i-p,j+q]),0.5) 
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B2', B[i,j])
                    # if (i!=len(V_p[:,0])-1) and (j!=0):
                        B[i,j] += B_c[i+p,j-q] \
                            *lam_2d(v1[i+p,j-q],v2[i+p,j-q],V_p[:,0],V_p[0,:],i,j,"+","-") \
                            *heaviside_jit((-1)**(p+1)*(V_p[i+p,0]-v1[i+p,j-q]),0.5) \
                            *heaviside_jit((-1)**q*(V_p[0,j-q]-v2[i+p,j-q]),0.5)
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B3', B[i,j])
                    # if (i!=len(V_p[:,0])-1) and (j!=len(V_p[0,:])-1): 
                        B[i,j] += B_c[i+p,j+q] \
                            *lam_2d(v1[i+p,j+q],v2[i+p,j+q],V_p[:,0],V_p[0,:],i,j,"+","+") \
                            *heaviside_jit((-1)**(p+1)*(V_p[i+p,0]-v1[i+p,j+q]),0.5) \
                            *heaviside_jit((-1)**(q+1)*(V_p[0,j+q]-v2[i+p,j+q]),0.5)
    i = len(V_p[0,:]) - 1
    for j in range(len(V_p[0,:])-1):
        for p in range(2):
            for q in range(2):
                B[i,j] += B_c[i-p,j-q] \
                    *lam_2d(v1[i-p,j-q],v2[i-p,j-q],V_p[:,0],V_p[0,:],i,j,"-","-") \
                    *heaviside_jit((-1)**p*(V_p[i-p,0]-v1[i-p,j-q]),0.5) \
                    *heaviside_jit((-1)**q*(V_p[0,j-q]-v2[i-p,j-q]),0.5)  
                B[i,j] += B_c[i-p,j+q] \
                    *lam_2d(v1[i-p,j+q],v2[i-p,j+q],V_p[:,0],V_p[0,:],i,j,"-","+") \
                    *heaviside_jit((-1)**p*(V_p[i-p,0]-v1[i-p,j+q]),0.5) \
                    *heaviside_jit((-1)**(q+1)*(V_p[0,j+q]-v2[i-p,j+q]),0.5) 
    j = len(V_p[0,:]) - 1
    for i in range(len(V_p[0,:])-1):
       for p in range(2):
           for q in range(2):
              B[i,j] += B_c[i-p,j-q] \
                  *lam_2d(v1[i-p,j-q],v2[i-p,j-q],V_p[:,0],V_p[0,:],i,j,"-","-") \
                  *heaviside_jit((-1)**p*(V_p[i-p,0]-v1[i-p,j-q]),0.5) \
                  *heaviside_jit((-1)**q*(V_p[0,j-q]-v2[i-p,j-q]),0.5)  
              B[i,j] += B_c[i+p,j-q] \
                  *lam_2d(v1[i+p,j-q],v2[i+p,j-q],V_p[:,0],V_p[0,:],i,j,"+","-") \
                  *heaviside_jit((-1)**(p+1)*(V_p[i+p,0]-v1[i+p,j-q]),0.5) \
                  *heaviside_jit((-1)**q*(V_p[0,j-q]-v2[i+p,j-q]),0.5) 
    i = len(V_p[0,:]) - 1
    j = len(V_p[0,:]) - 1
    for p in range(2):
        for q in range(2):     
           B[i,j] += B_c[i-p,j-q] \
               *lam_2d(v1[i-p,j-q],v2[i-p,j-q],V_p[:,0],V_p[0,:],i,j,"-","-") \
               *heaviside_jit((-1)**p*(V_p[i-p,0]-v1[i-p,j-q]),0.5) \
               *heaviside_jit((-1)**q*(V_p[0,j-q]-v2[i-p,j-q]),0.5)  
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==3 and j==0:
                        #     print('B4', B[i,j])
                        # if i==3 and j==1 and p==1 and q==0:
                        #     print(f'i={i}, j={j}, p={p}, q={q} (transfer from B_c[{i+p},{j+q}]')
                        #     print('B_c', B_c[i+p,j+q])                            
                        #     print('lam', lam_2d(v1[i+p,j+q],v2[i+p,j+q],V_p[:,1],V_p[1,:],i,j,"+","+"))
                        #     print('heavi 1',heaviside_jit((-1)**(p+1)*(V_p[i+p,1]-v1[i+p,j+q]),0.5))
                        #     print('heavi 2',heaviside_jit((-1)**(q+1)*(V_p[1,j+q]-v2[i+p,j+q]),0.5))
    
    # Combine birth and death
    dNdt = B + D
    
    return dNdt.reshape(-1)   

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
        N[1,0] = 0.3
        #N[2,0] = 0.2
        
        for i in range(1,NS+1):
            V_e[i] = S**(i-1)*V01
            # ith pivot is mean between ith and (i+1)th edge
            V_p[i-1] = (V_e[i] + V_e[i-1])/2
            
        F_M = np.zeros((NS,NS))
        ## constant kernal
        # F_M[:,:] = 1
        ## sum kernal
        for idx, tmp in np.ndenumerate(F_M):
            if idx[0]==0 or idx[1]==0:
                continue
            a = idx[0]
            b = idx[1]
            F_M[idx] = (V_p[a] + V_p[b])/V_p[1]
            
        # SOLVE    
        import scipy.integrate as integrate
        RES = integrate.solve_ivp(dNdt_1D,
                                  [0, max(t)], 
                                  N[:,0], t_eval=t,
                                  args=(V_p,V_e,F_M),
                                  method='LSODA',first_step=0.1,rtol=1e-1)
        
        # Reshape and save result to N and t_vec
        N = RES.y
        t = RES.t
        
        N0 = N[:,0]
        NE = N[:,-1]
        print('### Total Volume before and after..')
        print(np.sum(N0*V_p), np.sum(NE*V_p))
        
        print('### Initial dNdt..')
        dNdt0=dNdt_1D(0,N[:,0],V_p,V_e,F_M)
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
        mu0_as = np.exp(-np.sum(N[:,0])*t)  
        
        ax2.plot(t, mu0_as, color='k', linestyle='-.', label='$\mu_0$ (analytical)')
        ax2.plot(t, mu1, color=c_KIT_red, label='$\mu_1$')     
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.legend()
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
        
        F_M_tem=1
        F = np.zeros((NS,NS,NS,NS))
        for idx, tmp in np.ndenumerate(F):
            if idx[0]+idx[1]==0 or idx[2]+idx[3]==0:
                continue
            F[idx] = F_M_tem
        
        # for i in range(0,NS-1):#range(1,NS+1)
            # ith pivot is mean between ith and (i+1)th edge
            # V_p1[i+1] = S**(i)*V01
            # V_p2[i+1] = S**(i)*V02
            # V_e1[i+1] = (V_p1[i] + V_p1[i+1]) / 2#S**(i-1)*V01
            # V_e2[i+1] = (V_p2[i] + V_p2[i+1]) / 2#S**(i-1)*V02
        for i in range(0,NS):#range(1,NS+1)
            # ith pivot is mean between ith and (i+1)th edge
            V_e1[i+1] = S**(i)*V01
            V_e2[i+1] = S**(i)*V02
            V_p1[i] = (V_e1[i] + V_e1[i+1]) / 2#S**(i-1)*V01
            V_p2[i] = (V_e2[i] + V_e2[i+1]) / 2#S**(i-1)*V02
                
        # V_e1[-1] = V_p1[-1] * (1 + S) / 2
        # V_e2[-1] = V_p2[-1] * (1 + S) / 2
        # Write V1 and V3 in respective "column" of V
        #V_e[:,0] = V_e1 
        #V_e[0,:] = V_e2
        V_p[:,0] = V_p1 #V_p[:,0] = V_p1  
        V_p[0,:] = V_p2 #V_p[0,:] = V_p2
        # N[0,1,0] = 0.3
        # N[1,0,0] = 0.3
        N[0,1:,0] = np.exp(-V_p1[1:])
        N[1:,0,0] = np.exp(-V_p2[1:])
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
        
        # SOLVE    
        import scipy.integrate as integrate
        RES = integrate.solve_ivp(dNdt_2D,
                                  [0, max(t)], 
                                  N[:,:,0].reshape(-1), t_eval=t,
                                  args=(V_p, V_e1, V_e2, NS, F),
                                  method='RK23',first_step=0.1,rtol=1e-3)
        
        # Reshape and save result to N and t_vec
        N = RES.y.reshape((NS,NS,len(t))) #RES.y.reshape((NS,NS,len(t)))
        t = RES.t
        
        N0 = N[:,:,0]
        NE = N[:,:,-1]
        print('### Total Volume before and after..')
        print(np.sum(N0*V_p), np.sum(NE*V_p))
        
        print('### Initial dNdt..')
        dNdt0=dNdt_2D(0,N[:,:,0].reshape(-1),V_p,V_e1,V_e2,NS, F).reshape(NS,NS)#.reshape(NS,NS)
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
        mu0_as = 2/(2+F_M_tem*np.sum(N[:,:,0])*t)  
    
        ax2.plot(t, mu0_as, color='k', linestyle='-.', label='$\mu_0$ (analytical)')
        ax2.plot(t, mu1, color=c_KIT_red, label='$\mu_1$')     
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.legend()
        plt.tight_layout()


