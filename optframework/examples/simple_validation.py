# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:02:26 2025

@author: px2030
"""
import os
import numpy as np
from optframework.validation import PBEValidation
from optframework.utils.general_scripts.generate_psd import full_psd

if __name__ == "__main__":
    # To ensure that PBE remains relatively stable, 
    # the function internally adjusts the particle initialization manually. 
    # As a result, the actual volume concentration may no longer match the value of c below! 
    # However, the initial conditions for MC-PBE and PBM are directly taken from dPBE, 
    # so theoretically, the initial conditions for all three methods remain the same.
    c = 1e-2  # m3/m3
    x = 2e-5  # m
    beta0 = 1e-16 # /m3
    P1 = 1e12
    P2 = 1.0
    use_psd = True
    rel_mom = True
    
    ## generate initial PSD
    pth = os.path.dirname( __file__ )
    output_dir = os.path.join(pth, "PSD_data")
    x50 = x * 1e6  # convert to um  
    resigma = 0.2
    minscale = 0.01
    maxscale = 100
    dist_path = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False, output_dir=output_dir)
    
    dim = 1
    grid = "geo"
    NS1 = 20
    NS2 = None
    S1 = 2
    # S2 = 2
    kernel = "sum"
    process = "mix"
    t = np.arange(0, 1, 0.1, dtype=float)
    
    v = PBEValidation(dim, grid, NS1, S1, kernel, process, t=t, c=c, x=x, 
                      beta0=beta0, use_psd=use_psd, dist_path=dist_path)
    v.P1 = P1
    v.P2 = P2
    v.V_unit = 1
    v.init_all()
    v.calculate_case(calc_pbe=True, calc_mc=True, calc_pbm=True)
    v.init_plot(size = 'half', extra = True, mrksize=6)
    v.plot_all_moments(REL=rel_mom)
    v.add_new_moments(NS=NS2,REL=rel_mom)
    v.show_plot()
    