# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:02:26 2025

@author: px2030
"""
import numpy as np
from optframework.pbe.validation import PBEValidation

if __name__ == "__main__":
    dim = 1
    grid = "geo"
    NS1 = 10
    NS2 = None
    S1 = 4
    # S2 = 2
    kernel = "const"
    process = "breakage"
    t = np.arange(0, 1, 0.05, dtype=float)
    
    v = PBEValidation(dim, grid, NS1, S1, kernel, process, t=t, c=1e-2, x=2e-2, beta0=1e-11)
    v.calculate_case()
    v.init_plot(size = 'half', extra = True, mrksize=6)
    v.plot_all_moments()
    v.add_new_moments(NS=NS2)
    v.show_plot()
    