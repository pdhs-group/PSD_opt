# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
import time
import numpy as np
import multiprocessing
import copy
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
from pypbe.pbe.dpbe_base import DPBESolver as pop

def calc_pbe(p,t_vec,corr_beta,alpha_prim,v,P1,P2,P3,P4,flag):
    p_calc = copy.deepcopy(p)
    p_calc.pl_v= v
    p_calc.pl_P1= P1
    p_calc.pl_P2= P2
    p_calc.pl_P3= P3
    p_calc.pl_P4= P4
    p_calc.alpha_prim = np.array([alpha_prim[0], alpha_prim[1], alpha_prim[1],alpha_prim[2]])
    p_calc.CORR_BETA= corr_beta
    p_calc.full_init(calc_alpha=False)
    if flag == 'N_change_rate':
        results = pbe_N_change_rate(p_calc, t_vec, corr_beta, alpha_prim, v, P1, P2, P3, P4)
    return results

def pbe_N_change_rate(p_calc, t_vec, corr_beta,alpha_prim,v,P1,P2,P3,P4, threshold=0.05):
    start_time = time.time()
    p_calc.solve_PBE(t_vec=t_vec)
    end_time = time.time()
    elapsed_time = end_time - start_time
    N = p_calc.N
    V_p = p_calc.V
    if p_calc.dim == 1:
        N[0,:] = 0
        N_sum = N.sum(axis=0)
        N0 = N[:,0]
        NE = N[:,-1]
    elif p_calc.dim == 2:
        N[0,0,:] = 0
        N_sum = N.sum(axis=0).sum(axis=0)
        N0 = N[:,:,0]
        NE = N[:,:,-1]
        
    rate_of_change = np.diff(N_sum) / N_sum[:-1]
    abs_rate_of_change = np.abs(rate_of_change)
    valid_time_points = np.where(abs_rate_of_change < threshold)[0] + 1
    
    ## Total Volume before and after
    rel_delta_V = abs(np.sum(NE*V_p) - np.sum(N0*V_p)) / np.sum(N0*V_p)
    if p_calc.calc_status and valid_time_points.shape[0] != 0:
        time_point = valid_time_points[0]
        calc_time_crit = t_vec[time_point]
    elif p_calc.calc_status and valid_time_points.shape[0] == 0:
        calc_time_crit = t_vec[-1]
    elif not p_calc.calc_status:
        calc_time_crit = -1
    CORR_AGG = corr_beta*alpha_prim
    results = np.array([CORR_AGG[0],CORR_AGG[1],CORR_AGG[2],v,P1,P2,P3,P4,calc_time_crit,elapsed_time,rel_delta_V])
    return results

if __name__ == '__main__':
    ## flag = 'elapsed_time': 
    ## flag = 'N_change_rate': 
    flag = 'N_change_rate'
    dim=2
    p = pop(dim=dim)
    smoothing = True
    
    ## Set the PBE parameters
    t_vec = np.arange(0, 3601, 100, dtype=float)
    # Note that it m5ust correspond to the settings of MC-Bond-Break.3 üf
    p.NS = 8
    p.S = 4
    
    p.process_type= "mix"
    p.aggl_crit= 100
    p.COLEVAL= 1
    p.EFFEVAL= 1
    p.SIZEEVAL= 1
    p.BREAKRVAL= 4
    p.BREAKFVAL= 5
    ## The original value is the particle size at 1% of the PSD distribution. 
    ## The position of this value in the coordinate system can be adjusted by multiplying by size_scale.
    size_scale = 1e0
    p.R01 = 8.677468940430804e-07*size_scale
    p.R03 = 8.677468940430804e-07*size_scale
    
    ## If you need to read PSD data as initial conditions, set the PSD data path
    if p.process_type == 'breakage':
        p.USE_PSD = False
    else:
        p.USE_PSD = True
    
    ## Use the breakage function calculated by the MC-Bond-Break method
    p.USE_MC_BOND = False
    p.solver = "ivp"
    
    ## Initialize the PBE
    p.V_unit = 1e-15
    # p.full_init(calc_alpha=False)

    ## define the range of corr_beta
    var_corr_beta = np.array([1e-1])
    # var_corr_beta = np.array([1e-2])
    ## define the range of alpha_prim 27x3
    values = np.array([0.1,1.0])
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    ## The case of all zero α is meaningless, that means no Agglomeration occurs
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]
    ## For cases where R01 and R03 have the same size, the elements of alpha_prim mirror symmetry 
    ## are equivalent and can be removed to simplify the calculation.
    unique_alpha_prim = []
    for comp in var_alpha_prim:
        comp_reversed = comp[::-1]
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_alpha_prim):
            unique_alpha_prim.append(comp)
            
    var_alpha_prim = np.array(unique_alpha_prim)

    ## define the range of v(breakage function)
    var_v = np.array([0.7,2.0])
    # var_v = np.array([0.01])    ## define the range of P1, P2 for power law breakage rate
    var_P1 = np.array([1e-3,1e-1])
    var_P2 = np.array([0.5,2.0])
    var_P3 = np.array([1e-3,1e-1])
    var_P4 = np.array([0.5,2.0])
    p.V1_mean = 1e-15
    p.V3_mean = 1e-15

    pbe_list = []
    for j,corr_beta in enumerate(var_corr_beta):
        for k,alpha_prim in enumerate(var_alpha_prim):
            for l,v in enumerate(var_v):
                for m1,P1 in enumerate(var_P1):
                    for m2,P2 in enumerate(var_P2):
                        for m3,P3 in enumerate(var_P3):
                            for m4,P4 in enumerate(var_P4):
                                ## Test with single process
                                # calc_pbe(p, t_vec, corr_beta, alpha_prim, v, P1, P2, P3, P4, flag)
                                pbe_list.append((p,t_vec,corr_beta,alpha_prim,v,P1,P2,P3,P4,flag))
    pool = multiprocessing.Pool(processes=24)
    try:
        results=pool.starmap(calc_pbe, pbe_list) 
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    finally:          
        pool.close()
        pool.join()                        
        results_arr = np.array(results)             