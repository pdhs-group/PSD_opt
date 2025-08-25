# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:31:38 2024

@author: px2030
"""
import numpy as np
import math
from numba import njit
from scipy.integrate import quad, dblquad

@njit
def beta_func(x, y):
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)

@njit
def breakage_func_1d(x,y,v,q,BREAKFVAL):
    # number of daughters
    # p = 0.0
    # Conservation of Hypervolume, random breakage into four fragments --> See Leong2023 (10)
    # only for validation with analytical results
    # fragments = 4
    if BREAKFVAL == 1:
        theta = 4.0
        # p = 4.0
    # Conservation of First-Order Moments, random breakage into two fragments --> See Leong2023 (10)
    # only for validation with analytical results
    elif BREAKFVAL == 2:
        theta = 2.0
        # p = 2.0
    ## product function of power law. --> See Diemer_Olson (2002)
    elif BREAKFVAL == 3:     
        euler_beta = beta_func(q,q*(v-1))
        z = x/y
        theta = v * z**(q-1) * (1-z)**(q*(v-1)-1) / euler_beta
        # p = v
    ## simple function of power law. --> See Diemer_Olson (2002)
    elif BREAKFVAL == 4:
        z = x/y
        theta = (v+1) * z ** (v-1)
        # p = (v+1) / v
    ## Parabolic --> See Diemer_Olson (2002)
    elif BREAKFVAL == 5:
        z = x/y
        theta = (v+2)*(v+1)*z**(v-1)*(1-z)
        # p = (v+2) / v
    return theta / y

@njit
def breakage_func_1d_xk(x,y,k,v,q,BREAKFVAL):
    return x ** k * breakage_func_1d(x,y,v,q,BREAKFVAL)

## Note: The counter-intuitive position in the input parameters (x3,x1...) of the 2d breakage function 
## is determined by the characteristics of the integrate function dblquad()
@njit
def breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL):
    if BREAKFVAL == 1:
        theta = 4.0 
        # p = 4.0
    elif BREAKFVAL == 2:
        theta = 2.0 
        # p = 2.0
    elif BREAKFVAL == 3:  
        # euler_beta = beta_func(q,q*(v-1))
        # z = (x1+x3)/(y1+y3)
        # theta = v * z**(q-1) * (1-z)**(q*(v-1)-1) / euler_beta
        # b = theta / (y1+y3)
        raise Exception("Product function of power law not implemented for 2d!")
    elif BREAKFVAL == 4:
        z = x1*x3 / (y1*y3)
        theta = v * (v+1) * z ** (v-1) / (y1*y3)
        # p = (v+1) / v
    elif BREAKFVAL == 5:
        z = x1*x3 / (y1*y3)
        theta = (v+2)*(v+1)*z**(v-1)*(1-z)*v/2 
        # p = (2v+1)*(v+2)/(2*v*(v+1))
        # p should <2 => v should < 1.28!
    return theta / (y1*y3)


@njit
def breakage_func_2d_x1k(x3,x1,y1,y3,k,v,q,BREAKFVAL):
    return x1 ** k * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@njit
def breakage_func_2d_x3k(x3,x1,y1,y3,k, v,q,BREAKFVAL):
    return x3 ** k * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@njit
def breakage_func_2d_x1kx3l(x3,x1,y1,y3,k,l,v,q,BREAKFVAL):
    return x1 ** k * x3 ** l * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@njit
def breakage_func_2d_trunc(x3,x1,y1,y3,k,l,v,q,BREAKFVAL,eta):
    if x1 < eta or x3 < eta:
        return 0.0
    else:
        return x1 ** k * x3 ** l * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@njit
def breakage_func_2d_x1k_trunc(x3,x1,y1,y3,k,v,q,BREAKFVAL, eta):
    if x1 < eta or x3 < eta:
        return 0.0
    else:
        return x1 ** k * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@njit
def breakage_func_2d_x3k_trunc(x3,x1,y1,y3,k,v,q,BREAKFVAL,eta):
    if x1 < eta or x3 < eta:
        return 0.0
    else:
        return x3 ** k * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@njit
def gauss_legendre(f,a,b,xs,ws,args=()):
    int_f = 0.0
    psi = (b - a) * 0.5 * xs + (b + a) * 0.5
    for idx, psi_val in enumerate(psi):
        f_values = f(psi_val,*args)
        int_f += (b - a) * 0.5 * ws[idx] * f_values
    return int_f

@njit
def dblgauss_legendre(f,a1,b1,a2,b2,xs1,ws1,xs2,ws2,args=()):
    int_f = 0.0
    psi1 = (b1 - a1) * 0.5 * xs1 + (b1 + a1) * 0.5
    psi2 = (b2 - a2) * 0.5 * xs2 + (b2 + a2) * 0.5
    for idx, psi1_val in enumerate(psi1):
        for idy, psi2_val in enumerate(psi2):
            f_values = f(psi2_val,psi1_val,*args)
            int_f += (b1 - a1) * (b2 - a2)* 0.25 * ws1[idx] * ws2[idy] * f_values
    return int_f

def calc_init_B_F_1D_quad(solver):
    ## Let the integration range associated with the breakage function start from zero 
    ## to ensure mass conservation  
    V_e_tem = np.copy(solver.V_e)
    V_e_tem[0] = 0.0
    for idx, tmp in np.ndenumerate(solver.int_B_F):
        a = idx[0]; i = idx[1]
        if i != 0 and a <= i:
            args = (solver.V[i],solver.pl_v,solver.pl_q,solver.BREAKFVAL)
            argsk = (solver.V[i],1,solver.pl_v,solver.pl_q,solver.BREAKFVAL)
            if a == i:
                solver.int_B_F[idx],err = quad(breakage_func_1d,V_e_tem[a],solver.V[a],args=args)
                solver.intx_B_F[idx],err = quad(breakage_func_1d_xk,V_e_tem[a],solver.V[a],args=argsk)
            else:
                solver.int_B_F[idx],err = quad(breakage_func_1d,V_e_tem[a],V_e_tem[a+1],args=args)
                solver.intx_B_F[idx],err = quad(breakage_func_1d_xk,V_e_tem[a],V_e_tem[a+1],args=argsk)
    return True

def calc_int_B_F_2D_GL(solver):
    return calc_int_B_F_2D_GL_jit(
        int(solver.NS),
        np.ascontiguousarray(solver.V1,  dtype=np.float64),
        np.ascontiguousarray(solver.V3,  dtype=np.float64),
        np.ascontiguousarray(solver.V_e1,  dtype=np.float64),
        np.ascontiguousarray(solver.V_e3,  dtype=np.float64),
        np.ascontiguousarray(solver.int_B_F,  dtype=np.float64),
        np.ascontiguousarray(solver.intx_B_F,  dtype=np.float64),
        np.ascontiguousarray(solver.inty_B_F,  dtype=np.float64),
        float(solver.pl_v),
        float(solver.pl_q),
        int(solver.BREAKFVAL),
        )
@njit
## integration function scipy.quad and scipy.dblquad are not compatible with jit!
## So a manually implemented integration method(GL: gauss-legendre quadrature) is needed here.
def calc_int_B_F_2D_GL_jit(NS,V1,V3,V_e1,V_e3,int_B_F, intx_B_F, inty_B_F, v,q,BREAKFVAL):
    V_e1_tem = np.copy(V_e1)
    V_e1_tem[0] = 0.0
    V_e3_tem = np.copy(V_e3)
    V_e3_tem[0] = 0.0
    
    ## Get GL points and weights for 2 axis (degree=9)
    # deg1 = 9
    # deg3 = 9
    # xs1, ws1 = np.polynomial.legendre.leggauss(deg1)
    # xs3, ws3 = np.polynomial.legendre.leggauss(deg3)
    xs1 = np.array([-9.681602395076260859e-01,
                -8.360311073266357695e-01,
                -6.133714327005903577e-01,
                -3.242534234038089158e-01,
                0.000000000000000000e+00,
                3.242534234038089158e-01,
                6.133714327005903577e-01,
                8.360311073266357695e-01,
                9.681602395076260859e-01,])
    ws1 = np.array([8.127438836157471758e-02,
                1.806481606948571184e-01,
                2.606106964029356599e-01,
                3.123470770400028074e-01,
                3.302393550012596712e-01,
                3.123470770400028074e-01,
                2.606106964029356599e-01,
                1.806481606948571184e-01,
                8.127438836157471758e-02])

    xs3 = xs1
    ws3 = ws1
    ## for low boundary/x
    for a in range(NS):
        for i in range(a, NS):
            if i == 0:
                continue
            args = (V1[i],v,q,BREAKFVAL)
            argsk = (V1[i],1,v,q,BREAKFVAL)
            if a == i:
                int_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d,V_e1_tem[a],V1[a],xs1,ws1,args=args)
                intx_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d_xk,V_e1_tem[a],V1[a],xs1,ws1,args=argsk) 
            else:
                int_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d,V_e1_tem[a],V_e1_tem[a+1],xs1,ws1,args=args)
                intx_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d_xk,V_e1_tem[a],V_e1_tem[a+1],xs1,ws1,args=argsk)
    ## for left boundary/y            
    for b in range(NS):
        for j in range(b,NS):
            if j == 0:
                continue
            args = (V3[j],v,q,BREAKFVAL)
            argsk = (V3[j],1,v,q,BREAKFVAL)
            if b == j:
                int_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d,V_e3_tem[b],V3[b],xs3,ws3,args=args)
                inty_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d_xk,V_e3_tem[b],V3[b],xs3,ws3,args=argsk)
            else:
                int_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d,V_e3_tem[b],V_e3_tem[b+1],xs3,ws3,args=args)
                inty_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d_xk,V_e3_tem[b],V_e3_tem[b+1],xs3,ws3,args=argsk)
    for idx, tmp in np.ndenumerate(int_B_F):
        a = idx[0] ; b = idx[1]; i = idx[2]; j = idx[3]
        if i == 0 or j == 0:
            continue
        elif a <= i and b <= j:
            args = (V1[i],V3[j],v,q,BREAKFVAL)
            argsk = (V1[i],V3[j],1,v,q,BREAKFVAL)
            ## The contribution of fragments in the same cell
            if a == i and b == j:
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1k,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=argsk)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3k,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=argsk)
            ## The contributions of fragments on the same vertical axis
            elif a == i:
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1k,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=argsk)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3k,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=argsk)
            ## The contributions of fragments on the same horizontal axis
            elif b == j:   
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=argsk)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=argsk)
            ## The contribution from the fragments of large particles on the upper right side 
            else:
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=argsk)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=argsk)
                    
    return True

## integration function scipy.quad and scipy.dblquad are not compatible with jit!
def calc_int_B_F_2D_quad(solver):
    V_e1_tem = np.zeros(solver.NS) 
    V_e1_tem[:] = solver.V_e1[1:]
    V_e1_tem[0] = 0.0
    V_e3_tem = np.zeros(solver.NS) 
    V_e3_tem[:] = solver.V_e3[1:]
    V_e3_tem[0] = 0.0
    
    ## for low boundary/x
    for a in range(solver.NS):
        for i in range(a, solver.NS):
            if i == 0:
                continue
            args = (solver.V1[i],solver.v,solver.q,solver.BREAKFVAL)
            argsk = (solver.V1[i],1,solver.v,solver.q,solver.BREAKFVAL)
            if a == i:
                solver.int_B_F[a,0,i,0],err = quad(breakage_func_1d,V_e1_tem[a],solver.V1[a],args=args)
                solver.intx_B_F[a,0,i,0],err = quad(breakage_func_1d_xk,V_e1_tem[a],solver.V1[a],args=argsk) 
            else:
                solver.int_B_F[a,0,i,0],err = quad(breakage_func_1d,V_e1_tem[a],V_e1_tem[a+1],args=args)
                solver.intx_B_F[a,0,i,0],err = quad(breakage_func_1d_xk,V_e1_tem[a],V_e1_tem[a+1],args=argsk)
    ## for left boundary/y            
    for b in range(solver.NS):
        for j in range(b,solver.NS):
            if j == 0:
                continue
            args = (solver.V3[j],solver.v,solver.q,solver.BREAKFVAL)
            argsk = (solver.V3[j],1,solver.v,solver.q,solver.BREAKFVAL)
            if b == j:
                solver.int_B_F[0,b,0,j],err = quad(breakage_func_1d,V_e3_tem[b],solver.V3[b],args=args)
                solver.inty_B_F[0,b,0,j],err = quad(breakage_func_1d_xk,V_e3_tem[b],solver.V3[b],args=argsk)
            else:
                solver.int_B_F[0,b,0,j],err = gauss_legendre(breakage_func_1d,V_e3_tem[b],V_e3_tem[b+1],args=args)
                solver.inty_B_F[0,b,0,j],err = gauss_legendre(breakage_func_1d_xk,V_e3_tem[b],V_e3_tem[b+1],args=argsk)
    for idx, tmp in np.ndenumerate(solver.int_B_F):
        a = idx[0] ; b = idx[1]; i = idx[2]; j = idx[3]
        if i == 0 or j == 0:
            continue
        elif a <= i and b <= j:
            args = (solver.V1[i],solver.V3[j],solver.v,solver.q,solver.BREAKFVAL)
            argsk = (solver.V1[i],solver.V3[j],1,solver.v,solver.q,solver.BREAKFVAL)
            ## The contributions of fragments on the same vertical axis
            if a == i and b == j:
                solver.int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],solver.V1[a],V_e3_tem[b],solver.V3[b],args=args)
                solver.intx_B_F[idx],err = dblquad(breakage_func_2d_x1k,V_e1_tem[a],solver.V1[a],V_e3_tem[b],solver.V3[b],args=argsk)
                solver.inty_B_F[idx],err = dblquad(breakage_func_2d_x3k,V_e1_tem[a],solver.V1[a],V_e3_tem[b],solver.V3[b],args=argsk)
            elif a == i:
                solver.int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],solver.V1[a],V_e3_tem[b],V_e3_tem[b+1],args=args)
                solver.intx_B_F[idx],err = dblquad(breakage_func_2d_x1k,V_e1_tem[a],solver.V1[a],V_e3_tem[b],V_e3_tem[b+1],args=argsk)
                solver.inty_B_F[idx],err = dblquad(breakage_func_2d_x3k,V_e1_tem[a],solver.V1[a],V_e3_tem[b],V_e3_tem[b+1],args=argsk)
            ## The contributions of fragments on the same horizontal axis
            elif b == j:   
                solver.int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],solver.V3[b],args=args)
                solver.intx_B_F[idx],err = dblquad(breakage_func_2d_x1k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],solver.V3[b],args=argsk)
                solver.inty_B_F[idx],err = dblquad(breakage_func_2d_x3k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],solver.V3[b],args=argsk)
            ## The contribution from the fragments of large particles on the upper right side 
            else:
                solver.int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                solver.intx_B_F[idx],err = dblquad(breakage_func_2d_x1k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=argsk)
                solver.inty_B_F[idx],err = dblquad(breakage_func_2d_x3k,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=argsk)
                                            
    return True

def calc_B_R_1d(solver):
    return calc_B_R_1d_jit(
        np.ascontiguousarray(solver.V,  dtype=np.float64),
        np.ascontiguousarray(solver.B_R,  dtype=np.float64),
        float(solver.pl_P1),
        float(solver.pl_P2),
        float(solver.G),
        int(solver.BREAKRVAL),
    )
    
@njit
def calc_B_R_1d_jit(V, B_R, pl_P1, pl_P2, G, BREAKRVAL):
    num_particles = len(B_R)
    if V[0] == 0:
        for i in range(1, num_particles):
            B_R[i] = calc_break_rate_1d(V, pl_P1, pl_P2, G, BREAKRVAL, i)
    else:
        for i in range(num_particles):
            B_R[i] = calc_break_rate_1d(V, pl_P1, pl_P2, G, BREAKRVAL, i)
    return B_R

@njit
def calc_break_rate_1d(V, pl_P1, pl_P2, G, BREAKRVAL, i):
    if BREAKRVAL == 1:
        # Size independent breakage rate --> See Leong2023 (10)
        # only for validation with analytical results
        B_R = pl_P1
    elif BREAKRVAL == 2:
        # Size dependent breakage rate --> See Leong2023 (10)
        # only for validation with analytical results
        B_R = pl_P1 * V[i]
    elif BREAKRVAL == 3:
        # Power Law Pandy and Spielmann --> See Jeldres2018 (28)
        B_R = pl_P1 * G * V[i] ** pl_P2
    elif BREAKRVAL == 4:
        # Hypothetical formula considering volume fraction
        B_R = pl_P1 * G * V[i] ** pl_P2
    return B_R
    
def calc_B_R_2d(solver):
    return calc_B_R_2d_jit(
        np.ascontiguousarray(solver.V,  dtype=np.float64),
        np.ascontiguousarray(solver.B_R,  dtype=np.float64),
        np.ascontiguousarray(solver.V1,  dtype=np.float64),
        np.ascontiguousarray(solver.V3,  dtype=np.float64),
        float(solver.G),
        float(solver.pl_P1),
        float(solver.pl_P2),
        float(solver.pl_P3),
        float(solver.pl_P4),
        int(solver.BREAKRVAL),
        int(solver.BREAKFVAL),
        )
@njit
def calc_B_R_2d_jit(V, B_R, V1, V3, G, pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL):
    for idx, _ in np.ndenumerate(B_R):
        i = idx[0]; j = idx[1]
        if i == 0 and j == 0:
            continue
        elif i == 0:
            if BREAKRVAL == 3:
                B_R[i,j] = calc_break_rate_1d(V3, pl_P1, pl_P2, G, BREAKRVAL, j)
            else:
                B_R[i,j] = calc_break_rate_1d(V3, pl_P3, pl_P4, G, BREAKRVAL, j)
        elif j == 0:
            B_R[i,j] = calc_break_rate_1d(V1, pl_P1, pl_P2, G, BREAKRVAL, i)
        else:
            B_R[i,j] = calc_break_rate_2d(V, V1, V3, G, 
                            pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL, i, j)

    return B_R

def calc_B_R_2d_flat(solver):
    return calc_B_R_2d_flat_jit(
        np.ascontiguousarray(solver.V,  dtype=np.float64),
        np.ascontiguousarray(solver.B_R,  dtype=np.float64),
        np.ascontiguousarray(solver.V1,  dtype=np.float64),
        np.ascontiguousarray(solver.V3,  dtype=np.float64),
        float(solver.G),
        float(solver.pl_P1),
        float(solver.pl_P2),
        float(solver.pl_P3),
        float(solver.pl_P4),
        int(solver.BREAKRVAL),
        int(solver.BREAKFVAL),
        )

@njit
def calc_B_R_2d_flat_jit(V, B_R, V1, V3, G, pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL):
    ## Normally B_R here should be a flat one-dimensional array, for example in MC-PBE
    ## And there is a one-to-one correspondence between V1, 
    # B_R = np.zeros_like(V)
    num_particles = len(B_R)
    for i in range(num_particles):
        if V1[i] == 0:
            if BREAKRVAL == 3:
                B_R[i] = calc_break_rate_1d(V, pl_P1, pl_P2, G, BREAKRVAL, i)
            else:
                B_R[i] = calc_break_rate_1d(V, pl_P3, pl_P4, G, BREAKRVAL, i)
        elif V3[i] == 0:
            B_R[i] = calc_break_rate_1d(V, pl_P1, pl_P2, G, BREAKRVAL, i)
        else:
            B_R[i] = calc_break_rate_2d_flat(V, V1, V3, G, 
                            pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL, i)

    return B_R

@njit
def calc_break_rate_2d(V, V1, V3, G, 
                pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL, i, j):
    if BREAKRVAL == 1:
        B_R = (pl_P1 + pl_P3) / 2.0
    elif BREAKRVAL == 2:
        if BREAKFVAL == 1:
            B_R = (pl_P1 + pl_P3) / 2.0 * V1[i]*V3[j]
        elif BREAKFVAL == 2:
            B_R = (pl_P1 + pl_P3) / 2.0 * (V1[i] + V3[j])
    elif BREAKRVAL == 3:
        B_R = pl_P1 * G * (V[i, j]) ** pl_P2
    elif BREAKRVAL == 4:
        B_R =  pl_P1 * G * (V1[i])**pl_P2 + pl_P3 * G * (V3[j])**pl_P4
    return B_R

@njit
def calc_break_rate_2d_flat(V, V1, V3, G, 
                pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL, i):
    if BREAKRVAL == 1:
        B_R = (pl_P1 + pl_P3) / 2.0
    elif BREAKRVAL == 2:
        if BREAKFVAL == 1:
            B_R = (pl_P1 + pl_P3) / 2.0 * V1[i]*V3[i]
        elif BREAKFVAL == 2:
            B_R = (pl_P1 + pl_P3) / 2.0 * (V1[i] + V3[i])
    elif BREAKRVAL == 3:
        B_R = pl_P1 * G * (V[i]) ** pl_P2
    elif BREAKRVAL == 4:
        B_R =  pl_P1 * G * (V1[i])**pl_P2 + pl_P3 * G * (V3[i])**pl_P4
    return B_R
