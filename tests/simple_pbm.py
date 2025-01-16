# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:13:37 2025

@author: px2030
"""

from optframework.pbm import PBMSolver

if __name__ == "__main__":
    dim = 1
    pbm = PBMSolver(dim)
    # moments, moments_QMOM, moments_GQMOM = pbm.quick_test_QMOM(NDF_shape="normal")
    # moments_n, moments_QMOM_n, moments_GQMOM_n = pbm.quick_test_QMOM_normal(NDF_shape="normal")
    pbm.init_moments(NDF_shape="normal")
    pbm.solve_PBM()