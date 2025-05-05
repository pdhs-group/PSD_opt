# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:24:30 2024

@author: px2030
"""
import numpy as np
import math
from numba import jit, njit, float64, int64

@njit
def get_dNdt_1d_geo(t,N,NS,V_p,V_e,F_M,B_R,bf_int,xbf_int,type_flag,agg_crit):
    dNdt = np.zeros(N.shape)
    M_c = np.zeros(V_e.shape)
    D = np.zeros(N.shape)
    B = np.zeros(N.shape)
    B_c = np.zeros(NS+1)
    v = np.zeros(NS+1)
    V_p_ex = np.zeros(NS+1)-1
    V_p_ex[:-1] = V_p
    # N_scale_ex = np.ones(NS+1)
    # N_scale_ex[:-1] = N_scale
    
    if type_flag == "agglomeration":
    ## Because each item contains N^2 when calculating agglomeration, 
    ## it is equivalent to multiplying each item by an additional N_scale 
    ## and needs to be manually removed(divide by N_scale) to correct it.    
        B_c, M_c, D = calc_1d_agglomeration(N,V_p,V_e,F_M,B_c,M_c,D,agg_crit)
        # B_c /= N_scale_ex
        # M_c /= N_scale_ex
        # D /= N_scale
    elif type_flag == "breakage":
        B_c, M_c, D = calc_1d_breakage(N,V_p,V_e,B_R,bf_int,xbf_int,B_c,M_c,D)
    elif type_flag == "mix":
        B_c, M_c, D = calc_1d_agglomeration(N, V_p, V_e, F_M, B_c, M_c, D,agg_crit)
        # B_c /= N_scale_ex
        # M_c /= N_scale_ex
        # D /= N_scale
        B_c, M_c, D = calc_1d_breakage(N, V_p, V_e, B_R, bf_int,xbf_int, B_c, M_c, D)
    else:
        raise Exception("Current type_flag is not supported")
                    
    v[B_c != 0] = M_c[B_c != 0]/B_c[B_c != 0]
    
    # Assign BIRTH on each pivot
    for i in range(len(V_p)):
        # Add contribution from LEFT cell (if existent)
        B[i] += B_c[i]*lam(v[i], V_p_ex, i, "+")*heaviside(V_p[i]-v[i],0.5)
        # Left Cell, right half
        B[i] += B_c[i-1]*lam(v[i-1], V_p_ex, i, "+")*heaviside(v[i-1]-V_p[i-1],0.5)
        # Same Cell, right half
        B[i] += B_c[i]*lam(v[i], V_p_ex, i, "-")*heaviside(v[i]-V_p[i],0.5)
        # Right Cell, left half
        B[i] += B_c[i+1]*lam(v[i+1], V_p_ex, i, "-")*heaviside(V_p_ex[i+1]-v[i+1],0.5)
    ## Particles with a volume of zero will not have any impact on physical processes 
    ## and mass conservation, but may affect the convergence of differential equations, 
    ## so they are set to zero manually.
    dNdt[0] = 0.0        
    dNdt = B + D
    
    return dNdt 

@njit
def get_dNdt_2d_geo(t,NN,NS,V_p,V_e1,V_e2,F_M,B_R,bf_int,xbf_int,ybf_int,type_flag,agg_crit):       
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
    V_p_ex = np.zeros((NS+1,NS+1))-1
    V_p_ex[:-1,:-1] = V_p
    # N_scale_ex = np.ones((NS+1,NS+1))
    # N_scale_ex[:-1,:-1] = N_scale
    
    if type_flag == "agglomeration":
        B_c, M1_c,M2_c, D = calc_2d_agglomeration(N,V_p,V_e1,V_e2,F_M,B_c,M1_c,M2_c,D,agg_crit)
        # B_c /= N_scale_ex
        # M1_c /= N_scale_ex
        # M2_c /= N_scale_ex
        # D /= N_scale
    elif type_flag == "breakage":
        B_c, M1_c,M2_c, D = calc_2d_breakage(N,V_p,V_e1,V_e2,B_R,bf_int,xbf_int,ybf_int,B_c,M1_c,M2_c,D)
    elif type_flag == "mix":
        B_c, M1_c,M2_c, D = calc_2d_agglomeration(N,V_p,V_e1,V_e2,F_M,B_c,M1_c,M2_c,D,agg_crit)
        # B_c /= N_scale_ex
        # M1_c /= N_scale_ex
        # M2_c /= N_scale_ex
        # D /= N_scale
        B_c, M1_c,M2_c, D = calc_2d_breakage(N,V_p,V_e1,V_e2,B_R,bf_int,xbf_int,ybf_int,B_c,M1_c,M2_c,D)
    else:
        raise Exception("Current type_flag is not supported")
    
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
                    B[i,j] += B_c[i-p,j-q] \
                        *lam_2d(v1[i-p,j-q],v2[i-p,j-q],V_p_ex[:,0],V_p_ex[0,:],i,j,"-","-") \
                        *heaviside((-1)**p*(V_p[i-p,0]-v1[i-p,j-q]),0.5) \
                        *heaviside((-1)**q*(V_p[0,j-q]-v2[i-p,j-q]),0.5) 
                    # B[i,j] += tem 
                    ## PRINTS FOR DEBUGGING / TESTING
                    # if i-p==0 and j-q==0:
                    #     print(f'mass flux in [{i},{j}] is', tem)
                    B[i,j] += B_c[i-p,j+q] \
                        *lam_2d(v1[i-p,j+q],v2[i-p,j+q],V_p_ex[:,0],V_p_ex[0,:],i,j,"-","+") \
                        *heaviside((-1)**p*(V_p[i-p,0]-v1[i-p,j+q]),0.5) \
                        *heaviside((-1)**(q+1)*(V_p_ex[0,j+q]-v2[i-p,j+q]),0.5) 
                    # B[i,j] += tem 
                    ## PRINTS FOR DEBUGGING / TESTING
                    # if i-p==0 and j+q==0:
                    #     print(f'mass flux in [{i},{j}] is', tem)
                    B[i,j] += B_c[i+p,j-q] \
                        *lam_2d(v1[i+p,j-q],v2[i+p,j-q],V_p_ex[:,0],V_p_ex[0,:],i,j,"+","-") \
                        *heaviside((-1)**(p+1)*(V_p_ex[i+p,0]-v1[i+p,j-q]),0.5) \
                        *heaviside((-1)**q*(V_p[0,j-q]-v2[i+p,j-q]),0.5)
                    # B[i,j] += tem
                    ## PRINTS FOR DEBUGGING / TESTING
                    # if i+p==0 and j-q==0:
                    #     print(f'mass flux in [{i},{j}] is', tem)
                    B[i,j] += B_c[i+p,j+q] \
                        *lam_2d(v1[i+p,j+q],v2[i+p,j+q],V_p_ex[:,0],V_p_ex[0,:],i,j,"+","+") \
                        *heaviside((-1)**(p+1)*(V_p_ex[i+p,0]-v1[i+p,j+q]),0.5) \
                        *heaviside((-1)**(q+1)*(V_p_ex[0,j+q]-v2[i+p,j+q]),0.5)
                    # B[i,j] += tem
                    # if i+p==0 and j+q==0:
                    #     print(f'mass flux in [{i},{j}] is', tem)
                            
    dNdt = B + D
    ## Particles with a volume of zero will not have any impact on physical processes 
    ## and mass conservation, but may affect the convergence of differential equations, 
    ## so they are set to zero manually.
    dNdt[0,0] = 0.0
    # volume_error = (dNdt*V_p).sum()
    # print('volume error after assignment is ', volume_error)
    
    return dNdt.reshape(-1)   


@njit
def get_dNdt_3d_geo(t,NN,V,V1,V2,V3,F_M,NS,THR):       
        
    N = NN.copy() 
    N = N.reshape((NS+3,NS+3,NS+3))
    
    # Initialize DN with zeros
    DN = np.zeros(np.shape(N))
    #NS = len(N[:,1])-3
        
    # Initialize other temporary variables
    B_3D = np.copy(DN); D_3D = np.copy(DN); Bmod_3D = np.copy(DN) 
    Mx_3D = np.copy(DN); My_3D = np.copy(DN); Mz_3D=np.copy(DN)
    xx_3D = np.copy(DN); yy_3D = np.copy(DN); zz_3D = np.copy(DN)
    
    # Go through all possible combinations to calculate B, M and D matrix
    # First loop: All indices of DN to sum and reference actual concentration change
    for i in range(1,NS+2):
        for j in range(1,NS+2):
            for k in range(1,NS+2):
            
                # Second loop: All agglomeration partners 1 [a,c,e] and 2 [b,d,f] 
                # with index (<=i,<=j)
                for a in range(1,i+1):
                    for c in range(1,j+1):
                        for e in range(1,k+1):
                            for b in range(1,i+1): 
                                for d in range(1,j+1):
                                    for f in range(1,k+1):
                                                    
                                        # Only calculate if result of (a,c)+(b,d) yields agglomerate in 
                                        # range(i-1:i+1,j-1:j+1) with respect to total volume
                                        if V1[i-1]<V1[a]+V1[b]<V1[i+1] and V2[j-1]<V2[c]+V2[d]<V2[j+1] and V3[k-1]<V3[e]+V3[f]<V3[k+1]:
                                                                   
                                            # When birth exactly collides with [i,j,k] set zeta to 1, otherwise
                                            # set zeta to 2 (zeta defines whether B needs further distribution)
                                            #if V[a,c,e]+V[b,d,f]==V[i,j,k]:
                                            if abs(V[a,c,e]+V[b,d,f]-V[i,j,k]) < 1e-5*(V[2,1,1]):
                                                zeta = 1
                                            else:
                                                zeta = 2
                                        
                                            # Get corresponding F from F_M. Attention: F_M is defined without -1 borders
                                            # and ist a (NS+1)^6 matrix. a-f represent "real" indices of the N or V
                                            # matrix --> It is necessary to subtract 1 in order to get the right entry
                                            F = F_M[a-1,c-1,e-1,b-1,d-1,f-1]
                                            
                                            # Calculate raw birth term as well as volumetric fluxes M.
                                            # Division by 2 resulting from loop definition (don't count processes double)
                                            B_3D[i,j,k] = B_3D[i,j,k]+F*N[a,c,e]*N[b,d,f]/(2*zeta)
                                            Mx_3D[i,j,k] = Mx_3D[i,j,k]+F*N[a,c,e]*N[b,d,f]*(V1[a]+V1[b])/(2*zeta)
                                            My_3D[i,j,k] = My_3D[i,j,k]+F*N[a,c,e]*N[b,d,f]*(V2[c]+V2[d])/(2*zeta)
                                            Mz_3D[i,j,k] = Mz_3D[i,j,k]+F*N[a,c,e]*N[b,d,f]*(V3[e]+V3[f])/(2*zeta)
                                            
                                            # Average volume of all newborn agglomerates in the [i,j,k]th cell
                                            if not B_3D[i,j,k]==0:
                                                xx_3D[i,j,k] = Mx_3D[i,j,k]/B_3D[i,j,k]
                                                yy_3D[i,j,k] = My_3D[i,j,k]/B_3D[i,j,k]
                                                zz_3D[i,j,k] = Mz_3D[i,j,k]/B_3D[i,j,k]
                                                
                                            # Calculate death term of [a,c,e] and [b,d,f]. 
                                            # D_3D is defined positively (addition) and subtracted later
                                            D_3D[a,c,e]=D_3D[a,c,e]+F*N[a,c,e]*N[b,d,f]/(2*zeta)
                                            D_3D[b,d,f]=D_3D[b,d,f]+F*N[a,c,e]*N[b,d,f]/(2*zeta)  
                    
    # Modification of the birth term. Currently mass conservation is not given. 
    # The generated agglomerate mass needs to be distributed to neighboring nodes.
    # Loop: All indices of B_2D need to be modified + get all combinations of p and q 
    # element of  [0,1]: 
    for i in range(1,NS+2):
        for j in range(1,NS+2):
            for k in range(1,NS+2):
                for p in range(2):
                    for q in range(2):
                        for r in range(2):
                        
                            # Actual modification calculation
                            Bmod_3D[i,j,k]=Bmod_3D[i,j,k] \
                                + B_3D[i-p,j-q,k-r] \
                                    *get_lam_3d(xx_3D[i-p,j-q,k-r],yy_3D[i-p,j-q,k-r],zz_3D[i-p,j-q,k-r],V1,V2,V3,i,j,k,"-","-","-") \
                                    *np.heaviside((-1)**p*(V1[i-p]-xx_3D[i-p,j-q,k-r]),0.5) \
                                    *np.heaviside((-1)**q*(V2[j-q]-yy_3D[i-p,j-q,k-r]),0.5) \
                                    *np.heaviside((-1)**r*(V3[k-r]-zz_3D[i-p,j-q,k-r]),0.5) \
                                + B_3D[i-p,j-q,k+r] \
                                    *get_lam_3d(xx_3D[i-p,j-q,k+r],yy_3D[i-p,j-q,k+r],zz_3D[i-p,j-q,k+r],V1,V2,V3,i,j,k,"-","-","+") \
                                    *np.heaviside((-1)**p*(V1[i-p]-xx_3D[i-p,j-q,k+r]),0.5) \
                                    *np.heaviside((-1)**q*(V2[j-q]-yy_3D[i-p,j-q,k+r]),0.5) \
                                    *np.heaviside((-1)**(r+1)*(V3[k+r]-zz_3D[i-p,j-q,k+r]),0.5) \
                                + B_3D[i-p,j+q,k-r] \
                                    *get_lam_3d(xx_3D[i-p,j+q,k-r],yy_3D[i-p,j+q,k-r],zz_3D[i-p,j+q,k-r],V1,V2,V3,i,j,k,"-","+","-") \
                                    *np.heaviside((-1)**p*(V1[i-p]-xx_3D[i-p,j+q,k-r]),0.5) \
                                    *np.heaviside((-1)**(q+1)*(V2[j+q]-yy_3D[i-p,j+q,k-r]),0.5) \
                                    *np.heaviside((-1)**r*(V3[k-r]-zz_3D[i-p,j+q,k-r]),0.5) \
                                + B_3D[i-p,j+q,k+r] \
                                    *get_lam_3d(xx_3D[i-p,j+q,k+r],yy_3D[i-p,j+q,k+r],zz_3D[i-p,j+q,k+r],V1,V2,V3,i,j,k,"-","+","+") \
                                    *np.heaviside((-1)**p*(V1[i-p]-xx_3D[i-p,j+q,k+r]),0.5) \
                                    *np.heaviside((-1)**(q+1)*(V2[j+q]-yy_3D[i-p,j+q,k+r]),0.5) \
                                    *np.heaviside((-1)**(r+1)*(V3[k+r]-zz_3D[i-p,j+q,k+r]),0.5) \
                                + B_3D[i+p,j+q,k+r] \
                                    *get_lam_3d(xx_3D[i+p,j+q,k+r],yy_3D[i+p,j+q,k+r],zz_3D[i+p,j+q,k+r],V1,V2,V3,i,j,k,"+","+","+") \
                                    *np.heaviside((-1)**(p+1)*(V1[i+p]-xx_3D[i+p,j+q,k+r]),0.5) \
                                    *np.heaviside((-1)**(q+1)*(V2[j+q]-yy_3D[i+p,j+q,k+r]),0.5) \
                                    *np.heaviside((-1)**(r+1)*(V3[k+r]-zz_3D[i+p,j+q,k+r]),0.5) \
                                + B_3D[i+p,j+q,k-r] \
                                    *get_lam_3d(xx_3D[i+p,j+q,k-r],yy_3D[i+p,j+q,k-r],zz_3D[i+p,j+q,k-r],V1,V2,V3,i,j,k,"+","+","-") \
                                    *np.heaviside((-1)**(p+1)*(V1[i+p]-xx_3D[i+p,j+q,k-r]),0.5) \
                                    *np.heaviside((-1)**(q+1)*(V2[j+q]-yy_3D[i+p,j+q,k-r]),0.5) \
                                    *np.heaviside((-1)**r*(V3[k-r]-zz_3D[i+p,j+q,k-r]),0.5) \
                                + B_3D[i+p,j-q,k+r] \
                                    *get_lam_3d(xx_3D[i+p,j-q,k+r],yy_3D[i+p,j-q,k+r],zz_3D[i+p,j-q,k+r],V1,V2,V3,i,j,k,"+","-","+") \
                                    *np.heaviside((-1)**(p+1)*(V1[i+p]-xx_3D[i+p,j-q,k+r]),0.5) \
                                    *np.heaviside((-1)**q*(V2[j-q]-yy_3D[i+p,j-q,k+r]),0.5) \
                                    *np.heaviside((-1)**(r+1)*(V3[k+r]-zz_3D[i+p,j-q,k+r]),0.5) \
                                + B_3D[i+p,j-q,k-r] \
                                    *get_lam_3d(xx_3D[i+p,j-q,k-r],yy_3D[i+p,j-q,k-r],zz_3D[i+p,j-q,k-r],V1,V2,V3,i,j,k,"+","-","-") \
                                    *np.heaviside((-1)**(p+1)*(V1[i+p]-xx_3D[i+p,j-q,k-r]),0.5) \
                                    *np.heaviside((-1)**q*(V2[j-q]-yy_3D[i+p,j-q,k-r]),0.5) \
                                    *np.heaviside((-1)**r*(V3[k-r]-zz_3D[i+p,j-q,k-r]),0.5)         
            
    # Calculate final result and return 
    DN = Bmod_3D-D_3D
    
    # Due to numerical issues it is necessary to define a threshold for DN
    for i in range(NS+3): 
        for j in range(NS+3):
            for k in range(NS+3):
                if abs(DN[i,j,k])<THR: DN[i,j,k] = 0
    
    return DN.reshape(-1) 

@njit
def get_dNdt_1d_uni(t,N,V,B_R,B_F,F_M,NS,agg_crit,process_type):       

    # Initialize DN with zeros
    DN = np.zeros(np.shape(N))
        
    # Initialize other temporary variables (birth and death term)
    B = np.copy(DN); D = np.copy(DN); BR = np.copy(DN)
    
    # Go through all possible combinations to calculate B and D matrix
    # First loop: All indices of DN 
    if process_type == 'agglomeration' or process_type == 'mix':
        for e in range(1,agg_crit):
            # Second loop: All agglomeration partners that are smaller or equally sized
            for i in range(1,agg_crit):
                # Corresponding partner that i is produced (b = i-a + 1, because border) 
                for j in range(1,agg_crit):              
                    if V[i]+V[j] >= V[agg_crit]:
                        continue                    
                    # Only calculate if result of (a)+(b) yields agglomerate in 
                    # class i with respect to total volume
                    # if V[a]+V[b] == V[i]:
                    if abs(V[i]+V[j]-V[e]) < 1e-5*V[1]:
                        # Get corresponding F from F_M. Attention: F_M is defined without -1 borders
                        # and is a (NS+1)^2 matrix. a and b represent "real" indices of the N or V
                        # matrix --> It is necessary to subtract 1 in order to get the right entry
                        F = F_M[i,j]
                        # Calculate raw birth term as well as volumetric fluxes M.
                        # Division by 2 resulting from loop definition (don't count processes double)
                        # B[e] += F*N[i]*N[j]/2/N_scale[e] 
                        B[e] += F*N[i]*N[j]/2
                        # Calculate death term of [a] and [b]. 
                        # D is defined positively (addition) and subtracted later
                        # D[j] -= F*N[i]*N[j]/N_scale[e]   
                        D[j] -= F*N[i]*N[j]
    if process_type == 'breakage' or process_type == 'mix':
        for e in range(1,NS):
            S = B_R[e]
            D[e] -= S*N[e]
            for i in range(e, NS):
                # Calculate breakage if current index is larger than 2 (primary particles cannot break)
                # ATTENTION: This is only valid for binary breakage and s=2! (Test purposes)
                    S = B_R[i]
                    B[e] += S * B_F[e,i] * N[i]
    # Calculate final result and return 
    DN = B+D+BR
    # Due to numerical issues it is necessary to define a threshold for DN
    # for i in range(NS+3): 
    #     if abs(DN[i])<THR: DN[i] = 0
    return DN

@njit
def get_dNdt_2d_uni(t,NN,V,V1,V3,F_M,NS,THR):       
  
    N = NN.copy() 
    N = N.reshape((NS,NS))
    
    # Initialize DN with zeros
    DN = np.zeros(np.shape(N))
        
    # Initialize other temporary variables
    B = np.copy(DN); D = np.copy(DN)
    
    # Go through all possible combinations to calculate B and D matrix
    # First loop: All indices of DN to sum and reference actual concentration change
    for i in range(0,NS):
        for j in range(0,NS):
            
            # Second loop: All agglomeration partners 1 [a,c] and 2 [b,d] 
            # with index (<=i,<=j)
            for a in range(0,i+1):
                for c in range(0,j+1):
                    for b in range(0,i+1): 
                        for d in range(0,j+1): 
                                            
                            # Only calculate if result of (a,c)+(b,d) yields agglomerate in 
                            # range(i-1:i+1,j-1:j+1) with respect to total volume
                            #if V1[a]+V1[b] == V1[i] and V3[c]+V3[d] == V3[j]:
                            if abs(V1[a]+V1[b]-V1[i]) < 1e-5*V1[2] and abs(V3[c]+V3[d]-V3[j]) < 1e-5*V3[2]:
                                
                                # Get corresponding F from F_M. Attention: F_M is defined without -1 borders
                                # and is a (NS+1)^4 matrix. a-d represent "real" indices of the N or V
                                # matrix --> It is necessary to subtract 1 in order to get the right entry
                                F = F_M[a,c,b,d]
                                
                                # Calculate raw birth term
                                # Division by 2 resulting from loop definition (don't count processes double)
                                B[i,j] = B[i,j]+F*N[a,c]*N[b,d]/2
                                    
                                # Calculate death term of [a,c] and [b,d]. 
                                # D is defined positively (addition) and subtracted later
                                D[a,c] = D[a,c]+F*N[a,c]*N[b,d]/2
                                D[b,d] = D[b,d]+F*N[a,c]*N[b,d]/2    
            
    # Calculate final result and return 
    DN = B-D
    
    # Due to numerical issues it is necessary to define a threshold for DN
    for i in range(NS): 
        for j in range(NS):
            if abs(DN[i,j])<THR: DN[i,j] = 0
    
    return DN.reshape(-1) 

@njit
def get_dNdt_3d_uni(t,NN,V,V1,V2,V3,F_M,NS,THR):       
        
    N = NN.copy() 
    N = N.reshape((NS+3,NS+3,NS+3))
    
    # Initialize DN with zeros
    DN = np.zeros(np.shape(N))
        
    # Initialize other temporary variables
    B = np.copy(DN); D = np.copy(DN)
    
    # Go through all possible combinations to calculate B, M and D matrix
    # First loop: All indices of DN to sum and reference actual concentration change
    for i in range(1,NS+2):
        for j in range(1,NS+2):
            for k in range(1,NS+2):
            
                # Second loop: All agglomeration partners 1 [a,c,e] and 2 [b,d,f] 
                # with index (<=i,<=j)
                for a in range(1,i+1):
                    for c in range(1,j+1):
                        for e in range(1,k+1):
                            for b in range(1,i+1): 
                                for d in range(1,j+1):
                                    for f in range(1,k+1):
                                                    
                                        # Only calculate if result of (a,c,e)+(b,d,f) yields agglomerate (i,j,k)
                                        #if V1[a]+V1[b]==V1[i] and V2[c]+V2[d]==V2[j] and V3[e]+V3[f]==V3[k]:
                                        if abs(V1[a]+V1[b]-V1[i]) < 1e-5*V1[2] and abs(V2[c]+V2[d]-V2[j]) < 1e-5*V2[2] and abs(V3[e]+V3[f]-V3[k]) < 1e-5*V3[2]:                                             
                                            # Get corresponding F from F_M. Attention: F_M is defined without -1 borders
                                            # and ist a (NS+1)^6 matrix. a-f represent "real" indices of the N or V
                                            # matrix --> It is necessary to subtract 1 in order to get the right entry
                                            F = F_M[a-1,c-1,e-1,b-1,d-1,f-1]
                                            
                                            # Calculate raw birth term as well as volumetric fluxes M.
                                            # Division by 2 resulting from loop definition (don't count processes double)
                                            B[i,j,k] = B[i,j,k]+F*N[a,c,e]*N[b,d,f]/2
                                            
                                            # Calculate death term of [a,c,e] and [b,d,f]. 
                                            # D_3D is defined positively (addition) and subtracted later
                                            D[a,c,e]=D[a,c,e]+F*N[a,c,e]*N[b,d,f]/2
                                            D[b,d,f]=D[b,d,f]+F*N[a,c,e]*N[b,d,f]/2 
                    
    # Calculate final result and return 
    DN = B-D
    
    # Due to numerical issues it is necessary to define a threshold for DN
    for i in range(NS+3): 
        for j in range(NS+3):
            for k in range(NS+3):
                if abs(DN[i,j,k])<THR: DN[i,j,k] = 0
    
    return DN.reshape(-1) 

@njit
def lam(v, V_p, i, m):
    if m == "+":
        return (v-V_p[i-1])/(V_p[i]-V_p[i-1])
    else:        
        return (v-V_p[i+1])/(V_p[i]-V_p[i+1])
    # else:
    #     print('WRONG CASE FOR LAM')
        
@njit
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
        # print('lam is NaN!')        
    
    return lam 

@njit
def heaviside(x1, x2):
    if x1 < 0:
        return 0.0
    elif x1 > 0:
        return 1.0
    else:
        return x2

# @njit
# def b_integrate(x_up,x_low,y_up=None,y_low=None,bf=None):
#     if y_up is None or y_low is None:
#         return (x_up - x_low)*bf
#     else:
#         return (x_up- x_low)*(y_up - y_low)*bf
# @njit    
# def xb_integrate(x_up,x_low,y_up=None,y_low=None,bf=None):
#     if y_up is None or y_low is None:
#         return (x_up**2 - x_low**2)*0.5*bf
#     else:
#         return (x_up**2- x_low**2)*(y_up - y_low)*0.5*bf
# @njit    
# def yb_integrate(x_up,x_low,y_up=None,y_low=None,bf=None):
#     return (y_up**2 - y_low**2)*(x_up-x_low)*0.5*bf 

@njit
def get_lam_3d(x,y,z,Vx,Vy,Vz,i,j,k,m1,m2,m3):

    if m1 == "+":
        if m2 == "+":
            if m3 == "+":
                # (+,+,+)
                lam=(x-Vx[i+1])*(y-Vy[j+1])*(z-Vz[k+1])/((Vx[i]-Vx[i+1])*(Vy[j]-Vy[j+1])*(Vz[k]-Vz[k+1]))
            else:
                # (+,+,-)
                lam=(x-Vx[i+1])*(y-Vy[j+1])*(z-Vz[k-1])/((Vx[i]-Vx[i+1])*(Vy[j]-Vy[j+1])*(Vz[k]-Vz[k-1]))
        else:
            if m3 == "+":
                # (+,-,+)
                lam=(x-Vx[i+1])*(y-Vy[j-1])*(z-Vz[k+1])/((Vx[i]-Vx[i+1])*(Vy[j]-Vy[j-1])*(Vz[k]-Vz[k+1]))
            else:
                # (+,-,-)
                lam=(x-Vx[i+1])*(y-Vy[j-1])*(z-Vz[k-1])/((Vx[i]-Vx[i+1])*(Vy[j]-Vy[j-1])*(Vz[k]-Vz[k-1]))
    else:
        if m2 == "+":
            if m3 == "+":
                # (-,+,+)
                lam=(x-Vx[i-1])*(y-Vy[j+1])*(z-Vz[k+1])/((Vx[i]-Vx[i-1])*(Vy[j]-Vy[j+1])*(Vz[k]-Vz[k+1]))
            else:
                # (-,+,-)
                lam=(x-Vx[i-1])*(y-Vy[j+1])*(z-Vz[k-1])/((Vx[i]-Vx[i-1])*(Vy[j]-Vy[j+1])*(Vz[k]-Vz[k-1]))
        else:
            if m3 == "+":
                # (-,-,+)
                lam=(x-Vx[i-1])*(y-Vy[j-1])*(z-Vz[k+1])/((Vx[i]-Vx[i-1])*(Vy[j]-Vy[j-1])*(Vz[k]-Vz[k+1]))
            else:
                # (-,-,-)
                lam=(x-Vx[i-1])*(y-Vy[j-1])*(z-Vz[k-1])/((Vx[i]-Vx[i-1])*(Vy[j]-Vy[j-1])*(Vz[k]-Vz[k-1]))
                
                
    if math.isnan(lam):
        lam=0         
    
    #print(m1,m2,m3,lam)
    return lam




@njit
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
@njit
def calc_1d_breakage(N,V_p,V_e,B_R,bf_int,xbf_int,B_c,M_c,D):
    # D_M = np.zeros(N.shape)
    #  Loop through all pivots
    for e in range(len(V_p)):
        S = B_R[e]
        D[e] -= S*N[e]
        # D_M[e] = D[e]*V_p[e]
        for i in range(e, len(V_p)):
            b = bf_int[e,i]
            xb = xbf_int[e,i]
            S = B_R[i]
            B_c[e] += S*b*N[i]
            M_c[e] += S*xb*N[i]
    # volume_erro = M_c.sum() + D_M.sum()    
    # print(volume_erro)
    return B_c, M_c, D
 
@njit
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
@njit
def calc_2d_breakage(N,V_p,V_e1,V_e2,B_R,bf_int,xbf_int,ybf_int,B_c,M1_c,M2_c,D):
    ## only to check volume conservation
    # D_M = np.zeros(N.shape)
    for e1 in range(len(V_p[:,0])):
        for e2 in range(len(V_p[0,:])):
            # calculate death rate    * ``pop.int_B_F``: (2D)The integral of the breakage function from class ab to class ij. Result is stored in ``int_B_F[a,b,i,j]`` 
            S = B_R[e1,e2]
            D[e1,e2] -= S*N[e1,e2]
            # D_M[e1,e2] = D[e1,e2]*V_p[e1,e2]
            for i in range(e1, len(V_p[:,0])):
                for j in range(e2,len(V_p[0,:])):
                    b = bf_int[e1,e2,i,j]
                    xb = xbf_int[e1,e2,i,j]
                    yb = ybf_int[e1,e2,i,j]
                    S = B_R[i,j]
                    B_c[e1,e2] += S*b*N[i,j]
                    M1_c[e1,e2] += S*xb*N[i,j]
                    M2_c[e1,e2] += S*yb*N[i,j]    
    # volume_erro = M1_c.sum() + M2_c.sum() + D_M.sum()    
    # print('volume_error before assignment is ', volume_erro)            
    return B_c,M1_c,M2_c,D    