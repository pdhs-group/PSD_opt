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
import warnings
# from optframework.mcpbe.mcpbe_old_stru import MCPBESolver
from mcpbe import MCPBESolver as MCPBESolver_new

import cProfile, pstats

import matplotlib.pyplot as plt
import pbe_core.plotter.plotter as pt
from pbe_core.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, c_KIT_orange, c_KIT_purple

compare_models = []  # 可选："table", "rank", "copula", "flow", 或 ["all"]
dim = 1
N_MC = 2
seed = 42

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
    results, psd_info = m_new.solve_repeats(N_MC, base_seed=seed, workers=1,
                                            psd_enable=True)
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
    return mu_mc, std_mu_mc, t_run, mu_tmp, psd_info
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

def plot_moment_t(tp, curves, i=0, j=0):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ylbl = r"Moment $\mu_{%d%d}$" % (i, j)
    for (lbl, clr, mu, std) in curves:
        ax, fig = pt.plot_data(
            tp,
            mu[i, j, :],
            err=std if np.isscalar(std) else std[i, j, :],
            fig=fig,
            ax=ax,
            xlbl=r"time $t_\mathrm{A}$ / $s$",
            ylbl=ylbl,
            lbl=lbl,
            clr=clr,
            mrk="o",
            alpha=1,
            mrkedgecolor="k",
        )
    ax.grid("minor")
    plt.tight_layout()
    plt.show()
    
def build_solver_with_lmc(dim: int, pre_model: str):
    """
    按指定的预处理模型构建一个 MCPBESolver。
    pre_model: "table" | "rank" | "copula" | "flow"
    """
    m = MCPBESolver_new(dim=dim, init=False)
    m.use_lmc_live = False
    m.use_lmc_pre_model = True
    m.lmc_pre_model = pre_model

    # 初始化 adapter
    m._init_lmc()
    return m

def main():
    # 颜色映射
    color_map = {
        "live": c_KIT_red,
        "table": c_KIT_green,
        "rank": c_KIT_blue,
        "copula": c_KIT_orange,
        "flow": c_KIT_purple,
    }
    # profiler = cProfile.Profile()
    # profiler.enable()
    m_live = MCPBESolver_new(dim=dim, init=False)
    mu_live, std_live, t_live, _, psd_info = run_mcpbe_new(m_live, seed, N_MC)
    # profiler.disable()
    # stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    # stats.print_stats(20)
    
    tp = m_live.t_vec
    results = {"live": (mu_live, std_live)}

    # 对比模型
    if "all" in compare_models:
        models = ["table", "rank", "copula", "flow"]
    else:
        models = compare_models

    for mdl in models:
        print(f"[run] comparing live vs {mdl}")
        try:
            m_other = build_solver_with_lmc(dim, mdl)
        except Exception as e:
            warnings.warn(f"failed to init solver with lmc_pre_model='{mdl}': {e}")
            continue
        mu_o, std_o, t_o, _, _ = run_mcpbe_new(m_other, seed, N_MC)
        results[mdl] = (mu_o, std_o)
        print(f"[ok] {mdl} finished in {t_o:.2f}s")

    # 绘图
    for i in (0, 1, 2):
        curves = [
            (
                "live (baseline)",
                color_map["live"],
                results["live"][0],
                results["live"][1],
            )
        ]
        for mdl in models:
            if mdl in results:
                mu_o, std_o = results[mdl]
                curves.append(
                    (
                        f"{mdl} model",
                        color_map.get(mdl, "gray"),
                        mu_o,
                        std_o,
                    )
                )
        plot_moment_t(tp, curves, i=i, j=0)
    return psd_info
        
if __name__ == "__main__":
    psd_info = main()