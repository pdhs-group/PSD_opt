# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:31:38 2024

@author: px2030
"""
import numpy as np
import math
from numba import jit, njit, float64, int64
from scipy.integrate import quad, dblquad

@jit(nopython=True)
def beta_func(x, y):
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)

@njit
def breakage_func_1d(x,y,v,q,BREAKFVAL):
    # Conservation of Hypervolume, random breakage into four fragments --> See Leong2023 (10)
    # only for validation with analytical results
    if BREAKFVAL == 1:
        theta = 4.0
    # Conservation of First-Order Moments, random breakage into two fragments --> See Leong2023 (10)
    # only for validation with analytical results
    elif BREAKFVAL == 2:
        theta = 2.0
    ## product function of power law. --> See Diemer_Olson (2002)
    elif BREAKFVAL == 3:     
        euler_beta = beta_func(q,q*(v-1))
        z = x/y
        theta = v * z**(q-1) * (1-z)**(q*(v-1)-1) / euler_beta
    ## simple function of power law. --> See Diemer_Olson (2002)
    elif BREAKFVAL == 4:
        z = x/y
        theta = (v+1) * z ** (v-1)
    ## Parabolic --> See Diemer_Olson (2002)
    elif BREAKFVAL == 5:
        z = x/y
        theta = (v+2)*(v+1)*z**(v-1)*(1-z)
    return theta / y

@jit(nopython=True)
def breakage_func_1d_vol(x,y,v,q,BREAKFVAL):
    return x * breakage_func_1d(x,y,v,q,BREAKFVAL)

## Note: The counter-intuitive position in the input parameters (x3,x1...) of the 2d breakage function 
## is determined by the characteristics of the integrate function dblquad()
@jit(nopython=True)
def breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL):
    if BREAKFVAL == 1:
        theta = 4.0 
    elif BREAKFVAL == 2:
        theta = 2.0 
    elif BREAKFVAL == 3:  
        # euler_beta = beta_func(q,q*(v-1))
        # z = (x1+x3)/(y1+y3)
        # theta = v * z**(q-1) * (1-z)**(q*(v-1)-1) / euler_beta
        # b = theta / (y1+y3)
        raise Exception("Product function of power law not implemented for 2d!")
    elif BREAKFVAL == 4:
        z = x1*x3 / (y1*y3)
        theta = v * (v+1) * z ** (v-1) / (y1*y3)
    elif BREAKFVAL == 5:
        z = x1*x3 / (y1*y3)
        theta = (v+2)*(v+1)*z**(v-1)*(1-z)*v/2 
    return theta / (y1*y3)
@jit(nopython=True)
def breakage_func_2d_x1vol(x3,x1,y3,y1,v,q,BREAKFVAL):
    return x1 * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@jit(nopython=True)
def breakage_func_2d_x3vol(x3,x1,y3,y1,v,q,BREAKFVAL):
    return x3 * breakage_func_2d(x3,x1,y1,y3,v,q,BREAKFVAL)

@jit(nopython=True)
def gauss_legendre(f,a,b,xs,ws,args=()):
    int_f = 0.0
    psi = (b - a) * 0.5 * xs + (b + a) * 0.5
    for idx, psi_val in enumerate(psi):
        f_values = f(psi_val,*args)
        int_f += (b - a) * 0.5 * ws[idx] * f_values
    return int_f

@jit(nopython=True)
def dblgauss_legendre(f,a1,b1,a2,b2,xs1,ws1,xs2,ws2,args=()):
    int_f = 0.0
    psi1 = (b1 - a1) * 0.5 * xs1 + (b1 + a1) * 0.5
    psi2 = (b2 - a2) * 0.5 * xs2 + (b2 + a2) * 0.5
    for idx, psi1_val in enumerate(psi1):
        for idy, psi2_val in enumerate(psi2):
            f_values = f(psi2_val,psi1_val,*args)
            int_f += (b1 - a1) * (b2 - a2)* 0.25 * ws1[idx] * ws2[idy] * f_values
    return int_f

@jit(nopython=True)
## integration function scipy.quad and scipy.dblquad are not compatible with jit!
## So a manually implemented integration method(GL: gauss-legendre quadrature) is needed here.
def calc_int_B_F_2D_GL(NS,V1,V3,V_e1,V_e3,BREAKFVAL,v,q):
    int_B_F = np.zeros((NS, NS, NS, NS))
    intx_B_F = np.zeros((NS, NS, NS, NS))
    inty_B_F = np.zeros((NS, NS, NS, NS))
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
            if a == i:
                int_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d,V_e1_tem[a],V1[a],xs1,ws1,args=args)
                intx_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d_vol,V_e1_tem[a],V1[a],xs1,ws1,args=args) 
            else:
                int_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d,V_e1_tem[a],V_e1_tem[a+1],xs1,ws1,args=args)
                intx_B_F[a,0,i,0] = gauss_legendre(breakage_func_1d_vol,V_e1_tem[a],V_e1_tem[a+1],xs1,ws1,args=args)
    ## for left boundary/y            
    for b in range(NS):
        for j in range(b,NS):
            if j == 0:
                continue
            args = (V3[j],v,q,BREAKFVAL)
            if b == j:
                int_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d,V_e3_tem[b],V3[b],xs3,ws3,args=args)
                inty_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d_vol,V_e3_tem[b],V3[b],xs3,ws3,args=args)
            else:
                int_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d,V_e3_tem[b],V_e3_tem[b+1],xs3,ws3,args=args)
                inty_B_F[0,b,0,j] = gauss_legendre(breakage_func_1d_vol,V_e3_tem[b],V_e3_tem[b+1],xs3,ws3,args=args)
    for idx, tmp in np.ndenumerate(int_B_F):
        a = idx[0] ; b = idx[1]; i = idx[2]; j = idx[3]
        if i == 0 or j == 0:
            continue
        elif a <= i and b <= j:
            args = (V1[i],V3[j],v,q,BREAKFVAL)
            ## The contribution of fragments in the same cell
            if a == i and b == j:
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1vol,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3vol,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
            ## The contributions of fragments on the same vertical axis
            elif a == i:
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1vol,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3vol,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
            ## The contributions of fragments on the same horizontal axis
            elif b == j:   
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],xs1,ws1,xs3,ws3,args=args)
            ## The contribution from the fragments of large particles on the upper right side 
            else:
                int_B_F[idx]  = dblgauss_legendre(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
                intx_B_F[idx] = dblgauss_legendre(breakage_func_2d_x1vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
                inty_B_F[idx] = dblgauss_legendre(breakage_func_2d_x3vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],xs1,ws1,xs3,ws3,args=args)
                    
    return int_B_F, intx_B_F, inty_B_F

## integration function scipy.quad and scipy.dblquad are not compatible with jit!
def calc_int_B_F_2D_quad(NS,V1,V3,V_e1,V_e3,BREAKFVAL,v,q):
    int_B_F = np.zeros((NS-1, NS-1, NS-1, NS-1))
    intx_B_F = np.zeros((NS-1, NS-1, NS-1, NS-1))
    inty_B_F = np.zeros((NS-1, NS-1, NS-1, NS-1))
    V_e1_tem = np.zeros(NS) 
    V_e1_tem[:] = V_e1[1:]
    V_e1_tem[0] = 0.0
    V_e3_tem = np.zeros(NS) 
    V_e3_tem[:] = V_e3[1:]
    V_e3_tem[0] = 0.0
    
    ## for low boundary/x
    for a in range(NS):
        for i in range(a, NS):
            if i == 0:
                continue
            args = (V1[i],v,q,BREAKFVAL)
            if a == i:
                int_B_F[a,0,i,0],err = quad(breakage_func_1d,V_e1_tem[a],V1[a],args=args)
                intx_B_F[a,0,i,0],err = quad(breakage_func_1d_vol,V_e1_tem[a],V1[a],args=args) 
            else:
                int_B_F[a,0,i,0],err = quad(breakage_func_1d,V_e1_tem[a],V_e1_tem[a+1],args=args)
                intx_B_F[a,0,i,0],err = quad(breakage_func_1d_vol,V_e1_tem[a],V_e1_tem[a+1],args=args)
    ## for left boundary/y            
    for b in range(NS):
        for j in range(b,NS):
            if j == 0:
                continue
            args = (V3[j],v,q,BREAKFVAL)
            if b == j:
                int_B_F[0,b,0,j],err = quad(breakage_func_1d,V_e3_tem[b],V3[b],args=args)
                inty_B_F[0,b,0,j],err = quad(breakage_func_1d_vol,V_e3_tem[b],V3[b],args=args)
            else:
                int_B_F[0,b,0,j],err = gauss_legendre(breakage_func_1d,V_e3_tem[b],V_e3_tem[b+1],args=args)
                inty_B_F[0,b,0,j],err = gauss_legendre(breakage_func_1d_vol,V_e3_tem[b],V_e3_tem[b+1],args=args)
    for idx, tmp in np.ndenumerate(int_B_F):
        a = idx[0] ; b = idx[1]; i = idx[2]; j = idx[3]
        if i == 0 or j == 0:
            continue
        elif a <= i and b <= j:
            args = (V1[i],V3[j],v,q,BREAKFVAL)
            ## The contributions of fragments on the same vertical axis
            if a == i and b == j:
                int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],args=args)
                intx_B_F[idx],err = dblquad(breakage_func_2d_x1vol,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],args=args)
                inty_B_F[idx],err = dblquad(breakage_func_2d_x3vol,V_e1_tem[a],V1[a],V_e3_tem[b],V3[b],args=args)
            elif a == i:
                int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],args=args)
                intx_B_F[idx],err = dblquad(breakage_func_2d_x1vol,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],args=args)
                inty_B_F[idx],err = dblquad(breakage_func_2d_x3vol,V_e1_tem[a],V1[a],V_e3_tem[b],V_e3_tem[b+1],args=args)
            ## The contributions of fragments on the same horizontal axis
            elif b == j:   
                int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],args=args)
                intx_B_F[idx],err = dblquad(breakage_func_2d_x1vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],args=args)
                inty_B_F[idx],err = dblquad(breakage_func_2d_x3vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V3[b],args=args)
            ## The contribution from the fragments of large particles on the upper right side 
            else:
                int_B_F[idx],err  = dblquad(breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                intx_B_F[idx],err = dblquad(breakage_func_2d_x1vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                inty_B_F[idx],err = dblquad(breakage_func_2d_x3vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                                            
    return int_B_F, intx_B_F, inty_B_F
