# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:12:46 2025

@author: px2030
"""

import numpy as np
import math
import time
from optframework.pbe.mcpbe_new import population_MC as pop_mc_new
from optframework.pbe.mcpbe import population_MC as pop_mc 

def run_mcpbe(m):
    t_start = time.time()
    mu_tmp = []
    m_save = []
    for l in range(N_MC):
        m.c = np.array([n0*v0,n0*v0])
        m.x = np.array([x,x])
        m.PGV = np.array(['mono','mono'])
        m.COLEVAL = 4
        m.beta0 = beta0
        m.a0 = a0
        m.VERBOSE = VERBOSE
        m.savesteps = len(t)
        m.tA = t[-1]
        
        m.init_calc()
        
        m.solve_MC()
        mu_tmp.append(m.calc_mom_t())
        m_save.append(m)
    
    # Mean and STD of moments for all N_MC loops
    mu_mc = np.mean(mu_tmp,axis=0)
    if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
    t_run = time.time()-t_start
    return mu_mc, std_mu_mc, t_run

if __name__ == "__main__":
    N_MC = 5
    a0 = 200
    VERBOSE = True    
    alpha_pbe = np.array([1,1,1,1])
    alpha_mc = np.reshape(alpha_pbe,(2,2))
    t = np.arange(0, 5, 0.25, dtype=float)     # Time array [s]
    c = 1                # Volume concentration [-]
    x = 2e-2            # Particle diameter [m]
    n0 = 3*c/(4*math.pi*(x/2)**3)   # Total number concentration of primary particles
    v0 = 4*math.pi*(x/2)**3/3
    beta0 = 1e-2                  # Collision frequency parameter [m^3/s]
    
    m = pop_mc(2)
    m_new = pop_mc_new(2)
    
    mu_mc_new, std_mu_mc_new, t_run_new = run_mcpbe(m_new)
    mu_mc, std_mu_mc, t_run = run_mcpbe(m)
    
    error = np.mean(abs(mu_mc - mu_mc_new) / mu_mc)
    print(f"### Runing time of old MC-PBE: {t_run} s ###")
    print(f"### Runing time of new MC-PBE: {t_run_new} s ###")
    print(f"### Mean error of new MC-PBE: {error} ###")
    
    
    