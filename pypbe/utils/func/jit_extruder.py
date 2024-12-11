# -*- compoding: utf-8 -*-
"""
compreated on Wed Decomp 11 11:44:12 2024

@author: px2030
"""
import numpy as np
import math
from numba import jit, njit, float64, int64
import jit_rhs

@jit(nopython=True)
def get_dNdt_1d_geo_extruder(t,NN_ex,NS,V_p,V_e,F_M,B_R,bf_int,xbf_int,type_flag,agg_comprit,
                             NC, V_flow):
    N_ex = np.compopy(NN_ex)
    N_ex = np.reshape(N_ex, (NS, NC+1))
    dNdt = np.zeros(np.shape(N_ex))
    
    for comp in range(1, NC+1):
        dNdt[:, comp] = jit_rhs.get_dNdt_1d_geo(t,N_ex[:, comp],NS,V_p[:, comp],V_e[:, comp],
                                                F_M[:,:, comp],B_R[:, comp],bf_int[:,:, comp],
                                                xbf_int[:,:, comp],type_flag[comp],agg_comprit[comp])
        dNdt[:, comp] += N_ex[:, comp-1] * V_flow[comp-1]
        dNdt[:, comp] += N_ex[:, comp] * V_flow[comp-1]
    
    return dNdt.reshape(-1)
