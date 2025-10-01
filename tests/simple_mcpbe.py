# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:12:46 2025

@author: px2030

MCPBESolver Basic Usage Example
===============================

This script demonstrates the basic usage of the Monte Carlo Population Balance Equation (MCPBE) solver
for simulating particle agglomeration and breakage processes using stochastic methods.

The MCPBE solver uses Monte Carlo techniques to track individual particles/agglomerates explicitly,
providing detailed particle size distributions and statistical information about the evolution process.
Unlike dPBE and PBM, MC-PBE can capture the full complexity of particle interactions and
size distributions without assumptions about distribution shapes.

Key Features:
    
    - Support for 1D (single component) and 2D (multi-component) systems
    - Configurable agglomeration and breakage kernels
    - Statistical analysis through multiple Monte Carlo realizations
    - Various initial particle size distribution shapes
"""

import numpy as np
import copy
import time
from optframework.mcpbe_old_stru import MCPBESolver
from optframework.mcpbe import MCPBESolver as MCPBESolver_new

import cProfile, pstats

import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt
from optframework.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

def run_mcpbe_new(m_new, seed, N_MC):
    """
    Execute multiple Monte Carlo realizations and calculate statistical moments.
    
    This function runs the MCPBE solver N_MC times with different random seeds
    to obtain statistically reliable results. Each realization provides one
    possible evolution pathway of the particle system.
    
    Parameters
    ----------
    m_new : MCPBESolver
        Initialized MCPBE solver instance with configured parameters
        
    Returns
    -------
    mu_mc : numpy.ndarray
        Mean moments across all Monte Carlo realizations
        Shape: (n_moments, n_components, n_time_steps)
    std_mu_mc : numpy.ndarray
        Standard deviation of moments across realizations
    t_run : float
        Total computation time for all realizations [seconds]
    """
    t_start = time.time()
    mu_tmp = []
    # Run Monte Carlo simulation
    results = m_new.solve_repeats(N_MC, base_seed=seed)
    # Execute N_MC independent Monte Carlo realizations
    for l in range(N_MC):
        # Moments provide statistical characterization of the particle distribution
        mu_tmp.append(results[l]['moments'])
    
    # Mean and STD of moments for all realizations
    mu_mc = np.mean(mu_tmp,axis=0)
    if N_MC > 1: 
        std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        # std_mu_mc = 1e-3
    else: 
        std_mu_mc = 0
    t_run = time.time()-t_start
    return mu_mc, std_mu_mc, t_run, mu_tmp
def run_mcpbe(m):
    t_start = time.time()
    mu_tmp = []
    m_save = []
    # Execute N_MC independent Monte Carlo realizations
    for l in range(N_MC):
        m_tem = copy.deepcopy(m)
        # Run Monte Carlo simulation
        # This performs the actual PBE solving through stochastic events
        m_tem.solve_MC()
        # Calculate moments for this realization
        # Moments provide statistical characterization of the particle distribution
        mu_tmp.append(m_tem.calc_mom_t())
        m_save.append(m_tem)
    
    # Mean and STD of moments for all realizations
    mu_mc = np.mean(mu_tmp,axis=0)
    if N_MC > 1: 
        std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
    else: 
        std_mu_mc = 0
    t_run = time.time()-t_start
    return mu_mc, std_mu_mc, t_run

def plot_moment_t(tp, mu_mc, std_mu_mc, mu_mc_new, std_mu_mc_new, i=0, j=0):
    fig=plt.figure()    
    ax=fig.add_subplot(1,1,1) 
    ylbl = 'Moment $\mu_{' + f'{i}{j}' + '}$ / '+'$m^{3\cdot'+str(i+j)+'}$'
    ax, fig = pt.plot_data(tp,mu_mc[i,j,:], err=std_mu_mc[i,j,:], fig=fig, ax=ax,
                           xlbl='time $t_\mathrm{A}$ / $s$',
                           ylbl=ylbl, lbl='MC, $N_{\mathrm{MC}}='+str(N_MC)+'$',
                           clr=c_KIT_red,mrk='s', alpha=1, mrkedgecolor='k')
    ax, fig = pt.plot_data(tp,mu_mc_new[i,j,:], err=std_mu_mc_new[i,j,:], fig=fig, ax=ax,
                           xlbl='time $t_\mathrm{A}$ / $s$',
                           ylbl=ylbl, lbl='MC_new, $N_{\mathrm{MC}}='+str(N_MC)+'$',
                           clr=c_KIT_green,mrk='^', alpha=1, mrkedgecolor='k')
    ax.grid('minor')
    plt.tight_layout()   
    plt.show()
    return 

if __name__ == "__main__":
    plot_results = False
    # Simulation Configuration
    # ========================

    N_MC = 2        # Number of Monte Carlo realizations for statistical reliability
                    # More realizations → better statistics but longer computation time
                    # Recommended: 5-20 for testing, 50-100 for production runs
    dim = 2         # System dimension:
    seed = 41
    # Other key parameters can be modified in MCPBE_config.py

    m = MCPBESolver(dim=dim)
    m_new = MCPBESolver_new(dim=dim, init=False)

    print(f"Running {N_MC} Monte Carlo realizations for {dim}D system...")
    print(f"Initial particle count: {m.a_tot}")
    print(f"Initial particle concentrations: {m.c}")
    print(f"Simulation time: 0 to {m.t_total} seconds")
    print(f"Process type: {m.process_type}")
    
    # mu_mc, std_mu_mc, t_run = run_mcpbe(m)
    # profiler = cProfile.Profile()
    # profiler.enable()
    mu_mc_new, std_mu_mc_new, t_run_new, mu_tmp_new = run_mcpbe_new(m_new, seed, N_MC)
    # profiler.disable()
    # stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    # stats.print_stats(20)
    
    tp = m.t_vec
    # plot_moment_t(tp, mu_mc, std_mu_mc, mu_mc_new, std_mu_mc_new, i=0, j=0)
    # plot_moment_t(tp, mu_mc, std_mu_mc, mu_mc_new, std_mu_mc_new, i=1, j=0)
    # plot_moment_t(tp, mu_mc, std_mu_mc, mu_mc_new, std_mu_mc_new, i=2, j=0)