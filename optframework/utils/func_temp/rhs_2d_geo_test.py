# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:35:05 2024

@author: xy0264
"""

# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import math

from numba import jit
from numba.extending import overload, register_jitable

    
#%% PARAM
t = np.arange(0, 301, 60, dtype=float)
NS = 12
S = 2
R01, R02 = 1, 1
V01, V02 = 1, 1
dim = 2
rel_tol = V01*1e-6

#%% FUNCTIONS
def dNdt_2D_b(t,NN,V_p,V_e1,V_e2,NS,F_M):
    
    N = np.copy(NN) 
    dNdt = np.zeros(np.shape(N))
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
    for e1 in range(2, len(V_e1)-1):
        for e2 in range(2, len(V_e2)-1):
            # Loop through all pivots (twice)
            for i in range(1, len(V_p[:,0])):
                for j in range(1, len(V_p[0,:])):
                    # a <= i and b <= j (equal is allowed!) 
                    for a in range(1, len(V_p[:,0])): #i+1
                        for b in range(1, len(V_p[:,0])): #j+1
                            # Unequal sized processes are counted double --> Set zeta to 2
                            # if i==a and j==b:
                            #     zeta = 1
                            # else:
                            #     zeta = 2
                            
                            # Setting zeta=2 for all combinations seems correct
                            # See commented code above, I though we need to set it to 1 sometimes..
                            zeta = 2
                            F = F_M[i-1,j-1,a-1,b-1]
                            # Check if the agglomeration of ij and ab produce an 
                            # agglomerate inside the current cell 
                            # Use upper edge as "reference point"
                            if (V_e1[e1-1] <= V_p[i,1]+V_p[a,1] < V_e1[e1]) and \
                                (V_e2[e2-1] <= V_p[1,j]+V_p[1,b] < V_e2[e2]):
                                
            
                                # Total birthed agglomerates
                                # e1 and e2 start at 1 (upper edge) --> corresponds to cell "below"
                                # Use B_c[e1-1, e2-1] (e.g. e1=1, e2=1 corresponds to the first CELL [0,0])
                                B_c[e1-1,e2-1] += F*N[i,j]*N[a,b]/zeta
                                M1_c[e1-1,e2-1] += F*N[i,j]*N[a,b]*(V_p[i,1]+V_p[a,1])/zeta
                                M2_c[e1-1,e2-1] += F*N[i,j]*N[a,b]*(V_p[1,j]+V_p[1,b])/zeta
                                                                
                                # Average volume 
                                if B_c[e1-1,e2-1]!=0:
                                    v1[e1-1,e2-1] = M1_c[e1-1,e2-1]/B_c[e1-1,e2-1]
                                    v2[e1-1,e2-1] = M2_c[e1-1,e2-1]/B_c[e1-1,e2-1]
                                    
                                # Track death 
                                D[i,j] -= F*N[i,j]*N[a,b]
                                # D[a,b] -= F*N[i,j]*N[a,b]/zeta
                                #print(D)
                                
    # 30.01.24 B_c, D and v1 seem correct (judging from first dNdt)
    # print(B_c)
    # print(D)
    # print(v1)
    
    # # Assign BIRTH on each pivot
    for i in range(len(V_p[:,0])):
        for j in range(len(V_p[0,:])): 
            for p in range(2):
                for q in range(2):
                    # Actual modification calculation
                    if (i!=0) and (j!=0):
                        B[i,j] += B_c[i-p,j-q] \
                            *lam_2d(v1[i-p,j-q],v2[i-p,j-q],V_p[:,1],V_p[1,:],i,j,"-","-") \
                            *heaviside_jit((-1)**p*(V_p[i-p,1]-v1[i-p,j-q]),0.5) \
                            *heaviside_jit((-1)**q*(V_p[1,j-q]-v2[i-p,j-q]),0.5) 
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B1', B[i,j])
                    if (i!=0) and (j!=len(V_p[0,:])-1):                           
                        B[i,j] += B_c[i-p,j+q] \
                            *lam_2d(v1[i-p,j+q],v2[i-p,j+q],V_p[:,1],V_p[1,:],i,j,"-","+") \
                            *heaviside_jit((-1)**p*(V_p[i-p,1]-v1[i-p,j+q]),0.5) \
                            *heaviside_jit((-1)**(q+1)*(V_p[1,j+q]-v2[i-p,j+q]),0.5) 
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B2', B[i,j])
                    if (i!=len(V_p[:,0])-1) and (j!=0):
                        B[i,j] += B_c[i+p,j-q] \
                            *lam_2d(v1[i+p,j-q],v2[i+p,j-q],V_p[:,1],V_p[1,:],i,j,"+","-") \
                            *heaviside_jit((-1)**(p+1)*(V_p[i+p,1]-v1[i+p,j-q]),0.5) \
                            *heaviside_jit((-1)**q*(V_p[1,j-q]-v2[i+p,j-q]),0.5)
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i==2 and j==0:
                        #     print('B3', B[i,j])
                    if (i!=len(V_p[:,0])-1) and (j!=len(V_p[0,:])-1): 
                        B[i,j] += B_c[i+p,j+q] \
                            *lam_2d(v1[i+p,j+q],v2[i+p,j+q],V_p[:,1],V_p[1,:],i,j,"+","+") \
                            *heaviside_jit((-1)**(p+1)*(V_p[i+p,1]-v1[i+p,j+q]),0.5) \
                            *heaviside_jit((-1)**(q+1)*(V_p[1,j+q]-v2[i+p,j+q]),0.5)
                            
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
    
    return dNdt, B_c  

def dNdt_2D_nob(t,NN,V_p,V_e1,V_e2,NS,F_M):
    
    N = np.copy(NN) 
    dNdt = np.zeros(np.shape(N))
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
        for e2 in range(len(V_p[:,0])-1):
            # Loop through all pivots (twice)
            for i in range(len(V_p[:,0])):
                for j in range(len(V_p[0,:])):
                    # a <= i and b <= j (equal is allowed!) 
                    for a in range(len(V_p[:,0])): #i+1
                        for b in range(len(V_p[:,0])): #j+1
                            # Unequal sized processes are counted double --> Set zeta to 2
                            # if i==a and j==b:
                            #     zeta = 1
                            # else:
                            #     zeta = 2
                            
                            # Setting zeta=2 for all combinations seems correct
                            # See commented code above, I though we need to set it to 1 sometimes..
                            zeta = 2
                            F = F_M[i,j,a,b]
                            # Check if the agglomeration of ij and ab produce an 
                            # agglomerate inside the current cell 
                            # Use upper edge as "reference point"
                            if (V_e1[e1] <= V_p[i,0]+V_p[a,0] < V_e1[e1+1]) and \
                                (V_e2[e2] <= V_p[0,j]+V_p[0,b] < V_e2[e2+1]):
                                
            
                                # Total birthed agglomerates
                                # e1 and e2 start at 1 (upper edge) --> corresponds to cell "below"
                                # Use B_c[e1-1, e2-1] (e.g. e1=1, e2=1 corresponds to the first CELL [0,0])
                                B_c[e1,e2] += F*N[i,j]*N[a,b]/zeta
                                M1_c[e1,e2] += F*N[i,j]*N[a,b]*(V_p[i,0]+V_p[a,0])/zeta
                                M2_c[e1,e2] += F*N[i,j]*N[a,b]*(V_p[0,j]+V_p[0,b])/zeta
                                                                
                                # Average volume 
                                if B_c[e1,e2]!=0:
                                    v1[e1,e2] = M1_c[e1,e2]/B_c[e1,e2]
                                    v2[e1,e2] = M2_c[e1,e2]/B_c[e1,e2]
                                    
                                # Track death 
                                D[i,j] -= F*N[i,j]*N[a,b]
                                # D[a,b] -= F*N[i,j]*N[a,b]/zeta
                                #print(D)
                                
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
    
    return dNdt, B_c   


@jit(nopython=True)
def heaviside_jit(x1, x2):
    if x1 < -rel_tol:
        return 0.0
    elif x1 > rel_tol:
        return 1.0
    else:
        return x2
    

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

if __name__ == "__main__":    
    F_M_tem=1e-5
    F = np.zeros((NS,NS,NS,NS))
    for idx, tmp in np.ndenumerate(F):
        if idx[0]+idx[1]==0 or idx[2]+idx[3]==0:
            continue
        F[idx] = F_M_tem
        
    normal_dist = np.random.normal(0.5, 0.3, NS-1)
    normal_dist[ normal_dist<0 ] = 0
    #%% BOUNDARY  
    # SOLUTION N is saved on pivots
    N = np.zeros((NS+1,NS+1,len(t)))#np.zeros((NS,NS,len(t)))
    N[1,2:,0] = normal_dist
    N[2:,1,0] = normal_dist
        
    # V_e: Volume of EDGES
    V_e1 = np.zeros(NS+2) #np.zeros(NS+1)
    V_e2 = np.zeros(NS+2) #np.zeros(NS+1)    
    # Make first cell symmetric --> pivots fall on 0
    #V_e1[0], V_e2[0] = -V01, -V02
    V_e1[0], V_e2[0] = -2*V01, -2*V02
    V_e1[1], V_e2[1] = -V01, -V02
    
    # V_p: Volume of PIVOTS
    V_p1 = np.zeros(NS+1)#np.zeros(NS)
    V_p2 = np.zeros(NS+1)#np.zeros(NS)
    V_p = -1.5*np.ones((NS+1,NS+1))#np.zeros((NS,NS)) 
    V_p1[0], V_p2[0] = -1.5*V01, -1.5*V02
    V_p[1,1] = 0
    
    # Volume fractions
    X1 = np.zeros((NS+1,NS+1))#np.zeros((NS,NS)) 
    X2 = np.zeros((NS+1,NS+1))#np.zeros((NS,NS)) 
    
    
    for i in range(2,NS+2):#range(1,NS+1)
        V_e1[i] = S**(i-2)*V01#S**(i-1)*V01
        V_e2[i] = S**(i-2)*V01#S**(i-1)*V02
        # ith pivot is mean between ith and (i+1)th edge
        V_p1[i-1] = (V_e1[i] + V_e1[i-1])/2
        V_p2[i-1] = (V_e2[i] + V_e2[i-1])/2
    
    # Write V1 and V3 in respective "column" of V
    #V_e[:,0] = V_e1 
    #V_e[0,:] = V_e2
    V_p[:,1] = V_p1 #V_p[:,0] = V_p1  
    V_p[1,:] = V_p2 #V_p[0,:] = V_p2
    
    # Calculate remaining entries of V_e and V_p and other matrices
    for i in range(1,NS+1): #range(NS)
        for j in range(1,NS+1): #range(NS)
            V_p[i,j] = V_p1[i]+V_p2[j]
            if i==0 or j == 0 or (i==1 and j==1): #i==0 and j==0
                X1[i,j] = 0
                X2[i,j] = 0
            else:
                X1[i,j] = V_p1[i]/V_p[i,j]
                X2[i,j] = V_p2[j]/V_p[i,j]
    
    dNdt_b, B_C_b=dNdt_2D_b(0,N[:,:,0],V_p,V_e1,V_e2,NS,F)#.reshape(NS,NS)
        
    #%% NOBOUNDARY 
    # SOLUTION N is saved on pivots
    N = np.zeros((NS,NS,len(t)))#np.zeros((NS,NS,len(t)))
    N[0,1:,0] = normal_dist
    N[1:,0,0] = normal_dist
    # V_e: Volume of EDGES
    V_e1 = np.zeros(NS+1)
    V_e2 = np.zeros(NS+1)    
    # Make first cell symmetric --> pivots fall on 0
    V_e1[0], V_e2[0] = -V01, -V02
    
    # V_p: Volume of PIVOTS
    V_p1 = np.zeros(NS)
    V_p2 = np.zeros(NS)
    V_p = np.zeros((NS,NS)) 
    
    # Volume fractions
    X1 = np.zeros((NS,NS)) 
    X2 = np.zeros((NS,NS)) 
    
    for i in range(1,NS+1):
        V_e1[i] = S**(i-1)*V01
        V_e2[i] = S**(i-1)*V02
        # ith pivot is mean between ith and (i+1)th edge
        V_p1[i-1] = (V_e1[i] + V_e1[i-1])/2
        V_p2[i-1] = (V_e2[i] + V_e2[i-1])/2
    
    # Write V1 and V3 in respective "column" of V
    #V_e[:,0] = V_e1 
    #V_e[0,:] = V_e2
    
    # Calculate remaining entries of V_e and V_p and other matrices
    for i in range(NS):
        for j in range(NS):
            V_p[i,j] = V_p1[i]+V_p2[j]
            if i==0 and j==0:
                X1[i,j] = 0
                X2[i,j] = 0
            else:
                X1[i,j] = V_p1[i]/V_p[i,j]
                X2[i,j] = V_p2[j]/V_p[i,j]
    
    dNdt_nob, B_C_nob=dNdt_2D_nob(0,N[:,:,0],V_p,V_e1,V_e2,NS,F)
