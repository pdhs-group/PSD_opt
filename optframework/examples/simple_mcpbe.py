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
from optframework.mcpbe import MCPBESolver

def run_mcpbe(m):
    """
    Execute multiple Monte Carlo realizations and calculate statistical moments.
    
    This function runs the MCPBE solver N_MC times with different random seeds
    to obtain statistically reliable results. Each realization provides one
    possible evolution pathway of the particle system.
    
    Parameters
    ----------
    m : MCPBESolver
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

if __name__ == "__main__":
    # Simulation Configuration
    # ========================

    N_MC = 2        # Number of Monte Carlo realizations for statistical reliability
                    # More realizations → better statistics but longer computation time
                    # Recommended: 5-20 for testing, 50-100 for production runs
    dim = 1         # System dimension:
    # Other key parameters can be modified in MCPBE_config.py

    m = MCPBESolver(dim=dim)

    # What happens during MCPBESolver initialization:
    # ==============================================
    # 1. **Parameter Setup**: Loads base parameters and MC-specific settings
    # 2. **Initial Distribution Creation**: 
    #    - Creates volume matrix V_flat where each column represents one particle
    #    - Supports various initial distributions: mono, normal, weibull
    #    - For 1D: V_flat shape = (2, N_particles) [component_volume, total_volume]
    #    - For 2D: V_flat shape = (3, N_particles) [comp1_vol, comp2_vol, total_vol]
    # 3. **Control Volume Calculation**: Sets Vc to achieve target particle count a0
    # 4. **Kernel Initialization**: Pre-calculates collision frequencies if needed
    # 5. **Process Configuration**: Sets up agglomeration/breakage parameters

    # Execute Monte Carlo Simulations
    # ===============================
    print(f"Running {N_MC} Monte Carlo realizations for {dim}D system...")
    print(f"Initial particle count: {m.a_tot}")
    print(f"Initial particle concentrations: {m.c}")
    print(f"Simulation time: 0 to {m.t_total} seconds")
    print(f"Process type: {m.process_type}")
    mu_mc, std_mu_mc, t_run = run_mcpbe(m)

    # Results Analysis
    # ===============
    print(f"\nSimulation completed in {t_run:.2f} seconds")
    print(f"Average computation time per realization: {t_run/N_MC:.2f} seconds")
    
    # Physical Interpretation of Results:
    # ==================================
    if dim == 1:
        print("\n1D System Results:")
        print(f"Initial total number (μ₀₀): {mu_mc[0,0,0]:.2e} [#/m³]")
        print(f"Final total number (μ₀₀): {mu_mc[0,0,-1]:.2e} [#/m³]")
        print(f"Number increase/decrease ratio: {mu_mc[0,0,-1]/mu_mc[0,0,0]:.3f}")
        print(f"Initial total volume (μ₁₀): {mu_mc[1,0,0]:.2e} [m³/m³]")
        print(f"Final total volume (μ₁₀): {mu_mc[1,0,-1]:.2e} [m³/m³]")
        print(f"Volume conservation: {mu_mc[1,0,-1]/mu_mc[1,0,0]:.6f} (should ≈ 1.0)")
        
        # Calculate characteristic sizes
        mean_volume_initial = mu_mc[1,0,0] / mu_mc[0,0,0]
        mean_volume_final = mu_mc[1,0,-1] / mu_mc[0,0,-1]
        mean_diameter_initial = (6*mean_volume_initial/np.pi)**(1/3)
        mean_diameter_final = (6*mean_volume_final/np.pi)**(1/3)
        
        print(f"Mean particle diameter growth: {mean_diameter_initial*1e6:.2f} → {mean_diameter_final*1e6:.2f} μm")
        print(f"Size decrease/increase factor: {mean_diameter_final/mean_diameter_initial:.2f}")
        
    elif dim == 2:
        print("\n2D System Results:")
        print(f"Component 1 - Initial volume (μ₁₀): {mu_mc[1,0,0]:.2e} [m³/m³]")
        print(f"Component 1 - Final volume (μ₁₀): {mu_mc[1,0,-1]:.2e} [m³/m³]")
        print(f"Component 2 - Initial volume (μ₀₁): {mu_mc[0,1,0]:.2e} [m³/m³]")
        print(f"Component 2 - Final volume (μ₀₁): {mu_mc[0,1,-1]:.2e} [m³/m³]")
        print(f"Cross-moment (μ₁₁): {mu_mc[1,1,0]:.2e} → {mu_mc[1,1,-1]:.2e}")
        print(f"Total particle count: {mu_mc[0,0,0]:.2e} → {mu_mc[0,0,-1]:.2e}")
    
    # Statistical Reliability Assessment:
    # ==================================
    if N_MC > 1:
        print("\nStatistical Analysis:")
        # Calculate coefficient of variation for final moment values
        cv_final = std_mu_mc[:,:,-1] / np.abs(mu_mc[:,:,-1]) * 100
        print(f"Coefficient of variation in final moments: {np.nanmean(cv_final):.1f}%")
    
    # Optional: Advanced Analysis and Visualization
    # ============================================
    # Uncomment the following lines for additional analysis:
    
    # # Visualize particle size distributions at different times
    # m.visualize_qQ_t(t_plot=np.array([0, 0.5, 1.0]))  # Start, middle, end
    # 
    # # Plot moment evolution over time
    # m.visualize_mom_t(i=0, j=0)  # Total number concentration
    # m.visualize_mom_t(i=1, j=0)  # Total volume concentration
    # 
    # # Extract detailed size distribution at final time
    # x_final, q3_final, Q3_final, x10, x50, x90 = m.return_distribution(t=-1)
    # print(f"Final size distribution characteristics:")
    # print(f"x₁₀ (10th percentile): {x10*1e6:.2f} μm")
    # print(f"x₅₀ (median): {x50*1e6:.2f} μm") 
    # print(f"x₉₀ (90th percentile): {x90*1e6:.2f} μm")
    
    print("\nSimulation completed successfully!")
    print("Results available in variables: mu_mc (mean moments), std_mu_mc (standard deviation)")