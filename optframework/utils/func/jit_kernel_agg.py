# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:13:47 2024

@author: px2030
"""
import numpy as np
import math
from numba import jit, njit, float64, int64

@jit(nopython=True)
def calc_F_M_1D(NS, COLEVAL, CORR_BETA, G, R, alpha_prim, EFFEVAL, SIZEEVAL, X_SEL, Y_SEL):
    # To avoid mass leakage at the boundary in CAT, boundary cells are not directly involved in the calculation. 
    # So there is no need to define the corresponding F_M at boundary. F_M is (NS-1)^2 instead (NS)^2
    F_M = np.zeros((NS-1, NS-1))
    
    # calc_beta = prepare_calc_beta(COLEVAL, CORR_BETA, G)
    # Go through all agglomeration partners 1 [a] and 2 [i]
    # The current index tuple idx stores them as (a, i)
    for idx, tmp in np.ndenumerate(F_M):
        # Indices [a]=[0] and [i]=[0] not allowed!
        if idx[0] == 0 or idx[1] == 0:
            continue
        
        # Calculate the corresponding agglomeration efficiency
        # Add one to indices to account for borders
        a, i = idx[0], idx[1]
        # r1, r2 = R[a], R[i]
        # beta = calc_beta(r1, r2)
        beta = calc_beta(COLEVAL, CORR_BETA, G, R, a, i)                
        # Calculate collision efficiency depending on EFFEVAL. 
        if EFFEVAL == 1:
            alpha = alpha_prim
        elif EFFEVAL == 2:
            alpha = alpha_prim
        
        # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
        if R[a] <= R[i]:
            lam = R[a] / R[i]
        else:
            lam = R[i] / R[a]
            
        if SIZEEVAL == 1:
            # No size dependency of alpha
            corr_size = 1
        elif SIZEEVAL == 2:
            # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
            corr_size = np.exp(-X_SEL * (1 - lam) ** 2) / ((R[a] * R[i] / (R[1] ** 2)) ** Y_SEL)
        
        # Store result
        F_M[idx] = beta * alpha * corr_size

    return F_M
            
@jit(nopython=True)
def calc_F_M_2D(NS,COLEVAL,CORR_BETA,G,R,X1,X3,EFFEVAL,alpha_prim,SIZEEVAL,X_SEL,Y_SEL):
    # To avoid mass leakage at the boundary in CAT, boundary cells are not directly involved in the calculation. 
    # So there is no need to define the corresponding F_M at boundary. F_M is (NS-1)^4 instead (NS)^4
    F_M = np.zeros((NS-1,NS-1,NS-1,NS-1))
    
    # calc_beta = prepare_calc_beta(COLEVAL)
    # Go through all agglomeration partners 1 [a,b] and 2 [i,j]
    # The current index tuple idx stores them as (a,b,i,j)
    for idx, tmp in np.ndenumerate(F_M):
        # # Indices [a,b]=[0,0] and [i,j]=[0,0] not allowed!
        if idx[0]+idx[1]==0 or idx[2]+idx[3]==0:
            continue
        
        # Calculate the corresponding agglomeration efficiency
        # Add one to indices to account for borders
        a, b, i, j = idx[0], idx[1], idx[2], idx[3]
        # r1, r2 = R[a, b], R[i, j]
        # beta = calc_beta(r1, r2, CORR_BETA, G)
        beta = calc_beta(COLEVAL, CORR_BETA, G, R, (a, b), (i, j))
        
        # Calculate probabilities, that particle 1 [a,b] is colliding as
        # nonmagnetic 1 (NM1) or magnetic (M). Repeat for
        # particle 2 [i,j]. Use area weighted composition.
        # Calculate probability vector for all combinations. 
        # Indices: 
        # 1) a:N1 <-> i:N1  -> X1[a,b]*X1[i,j]
        # 2) a:N1 <-> i:M   -> X1[a,b]*X3[i,j]
        # 3) a:M  <-> i:N1  -> X3[a,b]*X1[i,j]
        # 4) a:M  <-> i:M   -> X3[a,b]*X3[i,j]
        p=np.array([X1[a,b]*X1[i,j],\
            X1[a,b]*X3[i,j],\
            X3[a,b]*X1[i,j],\
            X3[a,b]*X3[i,j]])
        
        # Calculate collision effiecieny depending on EFFEVAL. 
        # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun_2d
        # Case(2): Reduced model. Calculation only based on primary particles
        # Case(3): Alphas are pre-fed from ANN or other source.
        if EFFEVAL == 1:
            # Not coded here
            alpha = np.sum(p*alpha_prim)
        if EFFEVAL == 2 or EFFEVAL == 3:
            alpha = np.sum(p*alpha_prim)
        
        # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
        # Calculate lam
        if R[a,b]<=R[i,j]:
            lam = R[a,b]/R[i,j]
        else:
            lam = R[i,j]/R[a,b]
            
        if SIZEEVAL == 1:
            # No size dependency of alpha
            corr_size = 1
        if SIZEEVAL == 2:
            # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
            # with model parameters x and y. corr_size is lowered with lowered
            # value of lambda (numerator) and with increasing particles size (denominator)
            corr_size = np.exp(-X_SEL*(1-lam)**2)/((R[a,b]*R[i,j]/(np.min(np.array([R[2,1],R[1,2]]))**2))**Y_SEL)
        
        # Store result
        # alpha[idx] = alpha
        # beta[idx] = beta
        F_M[idx] = beta*alpha*corr_size

    return F_M
                
@jit(nopython=True)
def calc_F_M_3D(NS,COLEVAL,CORR_BETA,G,R,X1,X2,X3,EFFEVAL,alpha_prim,SIZEEVAL,X_SEL,Y_SEL):
                       
    # Initialize F_M Matrix. NOTE: F_M is defined without the border around the calculation grid
    # as e.g. N or V are (saving memory and calculations). 
    # Thus, F_M is (NS+1)^6 instead of (NS+3)^6. As reference, V is (NS+3)^3.
    F_M = np.zeros((NS+1,NS+1,NS+1,NS+1,NS+1,NS+1))
    
    # Go through all agglomeration partners 1 [a,b,c] and 2 [i,j,k]
    # The current index tuple idx stores them as (a,b,c,i,j,k)
    for idx, tmp in np.ndenumerate(F_M):
        # # Indices [a,b,c]=[0,0,0] and [i,j,k]=[0,0,0] not allowed!
        if idx[0]+idx[1]+idx[2]==0 or idx[3]+idx[4]+idx[5]==0:
            continue
        
        # Calculate the corresponding agglomeration efficiency
        # Add one to indices to account for borders
        a = idx[0]+1; b = idx[1]+1; c = idx[2]+1;
        i = idx[3]+1; j = idx[4]+1; k = idx[5]+1;
        
        # Calculate collision frequency beta depending on COLEVAL
        if COLEVAL == 1:
            # Chin 1998 (shear induced flocculation in stirred tanks)
            # Optional reduction factor.
            # corr_beta=1;
            beta = CORR_BETA*G*2.3*(R[a,b,c]+R[i,j,k])**3 # [m^3/s]
        if COLEVAL == 2:
            # Tsouris 1995 Brownian diffusion as controlling mechanism
            # Optional reduction factor
            # corr_beta=1;
            beta = CORR_BETA*2*1.38*(10**-23)*293*(R[a,b,c]+R[i,j,k])**2/(3*(10**-3)*(R[a,b,c]*R[i,j,k])) # [m^3/s] | KT= 1.38*(10**-23)*293 | MU_W=10**-3
        if COLEVAL == 3:
            # Use a constant collision frequency given by CORR_BETA
            beta = CORR_BETA
        if COLEVAL == 4:
            # Sum-Kernal (for validation) scaled by CORR_BETA
            beta = CORR_BETA*4*math.pi*(R[a,b,c]**3+R[i,j,k]**3)/3
        
        # Calculate probabilities, that particle 1 [a,b,c] is colliding as
        # nonmagnetic 1 (NM1), nonmagnetic 2 (NM2) or magnetic (M). Repeat for
        # particle 2 [i,j,k]. Use area weighted composition.
        # Calculate probability vector for all combinations. 
        # Indices: 
        # 1) a:N1 <-> i:N1  -> X1[a,b,c]*X1[i,j,k]
        # 2) a:N1 <-> i:N2  -> X1[a,b,c]*X2[i,j,k]
        # 3) a:N1 <-> i:M   -> X1[a,b,c]*X3[i,j,k]
        # 4) a:N2 <-> i:N1  -> X2[a,b,c]*X1[i,j,k] 
        # 5) a:N2 <-> i:N2  -> X2[a,b,c]*X2[i,j,k]
        # 6) a:N2 <-> i:M   -> X2[a,b,c]*X3[i,j,k]
        # 7) a:M  <-> i:N1  -> X3[a,b,c]*X1[i,j,k]
        # 8) a:M  <-> i:N2  -> X3[a,b,c]*X2[i,j,k]
        # 9) a:M  <-> i:M   -> X3[a,b,c]*X3[i,j,k]
        p=np.array([X1[a,b,c]*X1[i,j,k],\
                    X1[a,b,c]*X2[i,j,k],\
                    X1[a,b,c]*X3[i,j,k],\
                    X2[a,b,c]*X1[i,j,k],\
                    X2[a,b,c]*X2[i,j,k],\
                    X2[a,b,c]*X3[i,j,k],\
                    X3[a,b,c]*X1[i,j,k],\
                    X3[a,b,c]*X2[i,j,k],\
                    X3[a,b,c]*X3[i,j,k]])
        
        # Calculate collision effiecieny depending on EFFEVAL. 
        # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun
        # Case(2): Reduced model. Calculation only based on primary particles
        # Case(3): Alphas are pre-fed from ANN or other source.
        if EFFEVAL == 1:
            # Not coded here
            alpha = np.sum(p*alpha_prim)
        if EFFEVAL == 2 or EFFEVAL == 3:
            alpha = np.sum(p*alpha_prim)
        
        # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
        # Calculate lam
        if R[a,b,c]<=R[i,j,k]:
            lam = R[a,b,c]/R[i,j,k]
        else:
            lam = R[i,j,k]/R[a,b,c]
            
        if SIZEEVAL == 1:
            # No size dependency of alpha
            corr_size = 1
        if SIZEEVAL == 2:
            # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
            # with model parameters x and y. corr_size is lowered with lowered
            # value of lambda (numerator) and with increasing particles size (denominator)
            corr_size = np.exp(-X_SEL*(1-lam)**2)/((R[a,b,c]*R[i,j,k]/(np.min(np.array([R[2,1,1],R[1,2,1],R[1,1,2]]))**2))**Y_SEL)
        
        # Store result
        F_M[idx] = beta*alpha*corr_size
    
    return F_M

# @jit(nopython=True)
# def prepare_calc_beta(COLEVAL):
#     """
#     Prepare a fast version of calc_beta based on the COLEVAL value.
#     Returns a function specific to the given COLEVAL to reduce runtime checks.
#     """
#     if COLEVAL == 1:
#         def calc_beta(r1, r2, CORR_BETA, G):
#             return CORR_BETA * G * 2.3 * (r1 + r2) ** 3
#     elif COLEVAL == 2:
#         def calc_beta(r1, r2, CORR_BETA, G):
#             return CORR_BETA * 2 * 1.38e-23 * 293 * (r1 + r2) ** 2 / (3e-3 * (r1 * r2))
#     elif COLEVAL == 3:
#         def calc_beta(r1, r2, CORR_BETA, G):
#             return CORR_BETA
#     elif COLEVAL == 4:
#         def calc_beta(r1, r2, CORR_BETA, G):
#             return CORR_BETA * 4 * math.pi * (r1 ** 3 + r2 ** 3) / 3
#     else:
#         def calc_beta(r1, r2, CORR_BETA, G):
#             return 0.0
#     return calc_beta

@jit(nopython=True)
def calc_beta(COLEVAL, CORR_BETA, G, R, idx1, idx2):
    """
    Calculate beta based on collision model with flexible support for 1D or 2D R arrays.
    Parameters:
        COLEVAL : int - Collision model evaluation type.
        CORR_BETA : float - Correction factor for beta.
        G : float - Shear rate or related parameter.
        R : array - Particle radius/size array. Can be 1D or 2D.
        idx1, idx2 : int - Indices of the two colliding particles.
    Returns:
        beta : float - Collision frequency.
    """
    # Handle 1D case for Monte Carlo: R[idx] directly corresponds to size
    if R.ndim == 1:
        r1, r2 = R[idx1], R[idx2]
    # Handle 2D case for traditional methods: R[a, b] represents grid-based sizes
    elif R.ndim == 2:
        r1, r2 = R[idx1[0], idx1[1]], R[idx2[0], idx2[1]]
    else:
        raise ValueError("R array must be 1D or 2D.")
    if r1 <= 0 or r2 <= 0:
        return 0.0
    # Collision frequency calculation based on COLEVAL
    if COLEVAL == 1:
        beta = CORR_BETA * G * 2.3 * (r1 + r2) ** 3
    elif COLEVAL == 2:
        beta = CORR_BETA * 2 * 1.38e-23 * 293 * (r1 + r2) ** 2 / (3e-3 * (r1 * r2))
    elif COLEVAL == 3:
        beta = CORR_BETA
    elif COLEVAL == 4:
        beta = CORR_BETA * 4 * math.pi * (r1 ** 3 + r2 ** 3) / 3
    else:
        beta = 0.0  # Default case if COLEVAL is invalid.
    return beta


