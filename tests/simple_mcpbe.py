# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:12:46 2025

@author: px2030
"""

import numpy as np
import copy
import math
import time
from optframework.pbe.mcpbe_new import MCPBESolver as pop_mc_new
from optframework.pbe.mcpbe import population_MC as pop_mc 

def run_mcpbe(m):
    t_start = time.time()
    mu_tmp = []
    m_save = []
    for l in range(N_MC):
        m_tem = copy.deepcopy(m)
        m_tem.solve_MC()
        mu_tmp.append(m_tem.calc_mom_t())
        m_save.append(m_tem)
    
    # Mean and STD of moments for all N_MC loops
    mu_mc = np.mean(mu_tmp,axis=0)
    if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
    t_run = time.time()-t_start
    return mu_mc, std_mu_mc, t_run

if __name__ == "__main__":
    N_MC = 5
    
    m_new = pop_mc_new(1)
    
    # m = pop_mc(2)
    # m.c = m_new.c
    # m.x = m_new.x
    # m.PGV = m_new.PGV
    # m.BETACALC = 4
    # m.beta0 = m_new.CORR_BETA
    # m.a0 = m_new.a0
    # m.VERBOSE = True
    # m.savesteps = m_new.savesteps
    # m.tA = m_new.tA
    # m.init_calc()
    
    mu_mc_new, std_mu_mc_new, t_run_new = run_mcpbe(m_new)
    # mu_mc, std_mu_mc, t_run = run_mcpbe(m)
    
    # error = np.mean(abs(mu_mc - mu_mc_new) / mu_mc)
    # print(f"### Runing time of old MC-PBE: {t_run} s ###")
    # print(f"### Runing time of new MC-PBE: {t_run_new} s ###")
    # print(f"### Mean error of new MC-PBE: {error} ###")
    
    
    