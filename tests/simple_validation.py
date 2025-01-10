# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:02:26 2025

@author: px2030
"""

from optframework.pbe.validation import PBEValidation

if __name__ == "__main__":
    dim = 1
    grid = "geo"
    NS1 = 20
    NS2 = 30
    S1 = 2
    # S2 = 2
    kernel = "const"
    process = "breakage"
    
    v = PBEValidation(dim, grid, NS1, S1, kernel, process)
    v.calculate_case()
    v.init_plot(size = 'half', extra = True, mrksize=4)
    v.plot_all_moments()
    v.add_new_moments(NS=NS2)
    v.show_plot()
    