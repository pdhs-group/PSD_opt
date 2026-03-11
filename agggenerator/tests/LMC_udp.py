# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:56:09 2025

@author: px2030
"""
import numpy as np
import matplotlib.pyplot as plt
import generator
# import generator_psd as generator
from lmc import LMCSimulator
from lmc import Plotter

def live_generator():
    labels = generator.main()
    sim = LMCSimulator(STR=STR, NO_FRAG=NO_FRAG, gamma=gamma,
                       allow_loops=False, accept_all_cracks=False,
                       use_weighted_start=True, plotter=Plotter())
    F = sim.mc_breakage_udp(mat=labels, N_FRACS=N_FRACS,
                            a_code=0, b_code=1, empty_code=-1,
                            A0=A0,int_bre=int_bre,seed=seed,plot_each=True)
    sim.plotter.plot_F(F)
    plt.show()
    
def from_pool():
    sim = LMCSimulator(STR=STR, NO_FRAG=NO_FRAG, gamma=gamma,
                       allow_loops=False, accept_all_cracks=False,
                       use_weighted_start=True, plotter=Plotter())
    F = sim.mc_breakage_from_pool(
        pool_dir=r"C:\Users\px2030\Code\LMC_ANN\agggenerator",
        Df=1.8,
        MAS=0.40,
        A=A,
        X1=X1,
        N_GRIDS=N_GRIDS,
        N_FRACS=N_FRACS,
        A0=A0,
        int_bre=int_bre,
        seed=seed,
        plot_each=False,
    )
    sim.agg_pool.close_pool_cache()
    sim.plotter.plot_F(F)
    plt.show()
if __name__ == "__main__":
    A = 4000.0
    A0 = 1.0
    X1 = 0.6
    X2 = 1 - X1
    STR = np.array([1.0, 0.1, 1.0], dtype=float)
    NO_FRAG = 4
    aspect_ratio = 2.0
    int_bre = 0.2
    gamma = 1.0
    
    N_GRIDS = 10
    N_FRACS = 5
    seed = 42
    
    live_generator()
    # from_pool()
    