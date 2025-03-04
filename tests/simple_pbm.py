# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:13:37 2025

@author: px2030
"""

from optframework.pbm import PBMSolver

if __name__ == "__main__":
    dim = 2
    pbm = PBMSolver(dim)
    # moments, moments_QMOM, moments_GQMOM = pbm.quick_test.QMOM(NDF_shape="normal")
    # moments_n, moments_QMOM_n, moments_GQMOM_n = pbm.quick_test.QMOM_normal(NDF_shape="normal")
    # moments, moments_chyqmom = pbm.quick_test.CHyQMOM_2d()
    # pbm.quick_test.CQMOM_2d(use_central=True)
    
    # pbm.core.init_moments(NDF_shape="normal",N0=1e3,x_range=(1e-15,1e-12), mean=5e-14, std_dev=1e-13)
    pbm.core.init_moments_2d()
    pbm.core.solve_PBM()
    
    moments = pbm.moments
    moments_norm = moments[0,:] / moments[0,0]