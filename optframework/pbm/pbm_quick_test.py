# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:40:32 2025

"""

import numpy as np
import optframework.utils.plotter.plotter as pt
import optframework.utils.func.jit_pbm_qmom as qmom
import optframework.utils.func.jit_pbm_chyqmom as chyqmom

class PBMQuickTest:
    def __init__(self, solver):
        self.solver = solver

    def QMOM(self, NDF_shape="normal"):
        solver = self.solver
        if NDF_shape == "normal":
            x, NDF = solver.create_ndf(distribution="normal", x_range=(0,1), mean=0.5, std_dev=0.1)
        elif NDF_shape == "gamma":
            x, NDF = solver.create_ndf(distribution="gamma", x_range=(0, 1), shape=5, scale=1)
        elif NDF_shape == "lognormal":
            x, NDF = solver.create_ndf(distribution="lognormal", x_range=(0, 1), mean=0.1, sigma=1)
        elif NDF_shape == "beta":
            x, NDF = solver.create_ndf(distribution="beta", x_range=(0, 1), a=2, b=2)
        n = solver.n_order
        moments = np.array([np.trapz(NDF * (x ** k), x) for k in range(2*n)])
        nodes, weights, n = qmom.calc_qmom_nodes_weights(moments, n)
        bx, mom_c = chyqmom.compute_central_moments_1d(moments)
        nodes_G, weights_G, n = qmom.calc_qmom_nodes_weights(mom_c, n)
        weights_G *= moments[0]
        nodes_G += bx 
        moments_QMOM = np.zeros_like(moments)
        moments_GQMOM = np.zeros_like(moments)  
        for i in range(2*n):
            moments_QMOM[i] = sum(weights * nodes**i)
            moments_GQMOM[i] = sum(weights_G * nodes_G**i)
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        solver.post.plot_nodes_weights_comparision(x, NDF, nodes, weights, nodes_G, weights_G)
        solver.post.plot_moments_comparison(moments, moments_QMOM, moments_GQMOM)
        return moments, moments_QMOM, moments_GQMOM

    def QMOM_normal(self, NDF_shape="normal"):
        solver = self.solver
        if NDF_shape == "normal":
            x, NDF = solver.create_ndf(distribution="normal", x_range=(0, 1e-12), mean=5e-13, std_dev=2e-13)
        elif NDF_shape == "gamma":
            x, NDF = solver.create_ndf(distribution="gamma", x_range=(0, 1e-12), shape=1, scale=1)
        elif NDF_shape == "lognormal":
            x, NDF = solver.create_ndf(distribution="lognormal", x_range=(0, 1e-12), mean=5e-13, sigma=1e-10)
        elif NDF_shape == "beta":
            x, NDF = solver.create_ndf(distribution="beta", x_range=(0, 1), a=2, b=2)
        elif NDF_shape == "mono":
            x, NDF = solver.create_ndf(distribution="mono", x_range=(0, 1e-12), size=5e-13)
        n = solver.n_order
        NDF *= 1e12
        moments = np.array([np.trapz(NDF * (x ** k), x) for k in range(2*n)])
        moments_normal = np.array([moments[k] / x[-1]**k for k in range(2*n)])
        nodes, weights, n = qmom.calc_qmom_nodes_weights(moments_normal, n)
        nodes_G, weights_G, n = qmom.calc_gqmom_nodes_weights(moments_normal, n, solver.n_add, 
                                                           method=solver.GQMOM_method, nu=solver.nu)
        nodes *= x[-1]
        nodes_G *= x[-1]
        moments_QMOM = np.zeros_like(moments)
        moments_GQMOM = np.zeros_like(moments)  
        for i in range(2*n):
            moments_QMOM[i] = sum(weights * nodes**i)
            moments_GQMOM[i] = sum(weights_G * nodes_G**i)
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        solver.post.plot_nodes_weights_comparision(x, NDF, nodes, weights, nodes_G, weights_G)
        solver.post.plot_moments_comparison(moments, moments_QMOM, moments_GQMOM)
        
        return moments, moments_QMOM, moments_GQMOM

    def CHyQMOM_2d(self):
        solver = self.solver
        x1, NDF1 = solver.create_ndf(distribution="normal", x_range=(0,1), mean=0.5, std_dev=0.1)
        x2, NDF2 = solver.create_ndf(distribution="normal", x_range=(-1,10), mean=3, std_dev=1)
        
        solver.moment_2d_indices_chy()
        mu_num = len(solver.indices)
        moments = np.zeros(mu_num)
        for n, _ in enumerate(moments):
            k = solver.indices[n][0]
            l = solver.indices[n][1]
            moments[n] = solver.trapz_2d(NDF1, NDF2, x1, x2, k, l)
        
        if solver.n_order == 2:
            abscissas, weights = chyqmom.chyqmom4(moments, solver.indices)
        elif solver.n_order == 3:
            abscissas, weights = chyqmom.chyqmom9(moments, solver.indices)
            
        moments_chyqmom = np.zeros_like(moments)
        for n, _ in enumerate(moments_chyqmom):
            indice = solver.indices[n]
            moments_chyqmom[n] = chyqmom.quadrature_2d(weights, abscissas, indice)
            
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        solver.post.plot_moments_comparison(moments, moments_chyqmom, moments_chyqmom)
        return moments, moments_chyqmom

    def CQMOM_2d(self, use_central):
        solver = self.solver
        x1, NDF1 = solver.create_ndf(distribution="normal", x_range=(0,1), mean=0.5, std_dev=0.1)
        x2, NDF2 = solver.create_ndf(distribution="normal", x_range=(0,1), mean=0.5, std_dev=0.1)
        
        solver.moment_2d_indices_c()
        mu_num = len(solver.indices)
        moments = np.zeros(mu_num)
        for n, _ in enumerate(moments):
            k = solver.indices[n][0]
            l = solver.indices[n][1]
            moments[n] = solver.trapz_2d(NDF1, NDF2, x1, x2, k, l)
        
        abscissas, weights, _ = qmom.calc_cqmom_2d(moments, solver.n_order, solver.indices, use_central)
            
        moments_cqmom = np.zeros_like(moments)
        for n, _ in enumerate(moments_cqmom):
            indice = solver.indices[n]
            moments_cqmom[n] = chyqmom.quadrature_2d(weights, abscissas, indice)
            
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        solver.post.plot_moments_comparison(moments, moments_cqmom, moments_cqmom)
        return moments, moments_cqmom