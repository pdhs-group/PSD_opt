# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:13:27 2025

"""

import numpy as np
import scipy.integrate as integrate
import optframework.utils.func.jit_pbm_rhs as jit_pbm_rhs
from optframework.utils.func.static_method import interpolate_psd

class PBMCore:
    def __init__(self, solver):
        self.solver = solver

    def init_moments(self, x=None, NDF=None, NDF_shape="normal", N0=1.0,
                     V0=None, x_range=(0,1), mean=0.5, std_dev=0.1, shape=2, scale=1,
                     sigma=1, a=2, b=2, size=0.5):
        """
        Initialize the moments for the PBM solver.

        Parameters:
            x (numpy.ndarray): x-coordinates for the distribution function.
            NDF (numpy.ndarray): Normalized distribution function.
            NDF_shape (str): Shape of the distribution ("normal", "gamma", "lognormal", "beta", "mono").
            N0 (float): Initial number concentration.
            V0 (float): Total volume of particles.
            x_range (tuple): Range of x values.
            mean (float): Mean value for normal/lognormal distribution.
            std_dev (float): Standard deviation for normal distribution.
            shape (float): Shape parameter for gamma distribution.
            scale (float): Scale parameter for gamma distribution.
            sigma (float): Standard deviation for lognormal distribution.
            a (float): Alpha parameter for beta distribution.
            b (float): Beta parameter for beta distribution.
            size (float): Size parameter for mono distribution.
        """
        solver = self.solver
        if x is None or NDF is None:
            if solver.USE_PSD:
                x = np.linspace(0.0, x_range[1], 10000)
                if np.any(x < 0):
                    raise ValueError("Error: Volume (x) cannot be negative!")
                d = np.where(x > 0, (6 * x / np.pi) ** (1/3), 0)
                if V0 is None:
                    raise ValueError("Total volume of particle must be gave to get PSD")
                NDF = np.zeros_like(d)
                NDF[1:] = interpolate_psd(d[1:], solver.DIST1, V0)
                NDF /= x[1]-x[0]
            else:
                if NDF_shape == "normal":
                    x, NDF = solver.create_ndf(distribution="normal", x_range=x_range, mean=mean, std_dev=std_dev)
                elif NDF_shape == "gamma":
                    x, NDF = solver.create_ndf(distribution="gamma", x_range=x_range, shape=shape, scale=scale)
                elif NDF_shape == "lognormal":
                    x, NDF = solver.create_ndf(distribution="lognormal", x_range=x_range, mean=mean, sigma=sigma)
                elif NDF_shape == "beta":
                    x, NDF = solver.create_ndf(distribution="beta", x_range=x_range, a=a, b=b)
                elif NDF_shape == "mono":
                    x, NDF = solver.create_ndf(distribution="mono", x_range=x_range, size=size)
                moment0 = np.trapz(NDF, x)
                NDF /= moment0
            
        solver.x_max = 1.0
        solver.moments = np.zeros((solver.n_order*2, solver.t_num))
        NDF *= N0
        solver.moments[:,0] = np.array([np.trapz(NDF * (x ** k), x) for k in range(2*solver.n_order)]) * solver.V_unit
        solver.normalize_mom()
        solver.set_tol(solver.moments_norm[:,0])

    def init_moments_2d(self, N01=1.0, N02=1.0):
        """
        Initialize the 2D moments for the PBM solver.

        Parameters:
            N01 (float): Initial number concentration for component 1.
            N02 (float): Initial number concentration for component 2.
        """
        solver = self.solver
        x1, NDF1 = solver.create_ndf(distribution="normal", x_range=(1e-2,1e-1), mean=6e-2, std_dev=2e-2)
        x2, NDF2 = solver.create_ndf(distribution="normal", x_range=(1e-2,1e-1), mean=6e-2, std_dev=2e-2)
        NDF1 *= N01
        NDF2 *= N02
        
        solver.moment_2d_indices_c()
        mu_num = len(solver.indices)
        solver.moments = np.zeros((mu_num, solver.t_num))
        for idx in range(mu_num):
            k = solver.indices[idx,0]
            l = solver.indices[idx,1]
            solver.moments[idx,0] = solver.trapz_2d(NDF1, NDF2, x1, x2, k, l) * solver.V_unit
        solver.set_tol(solver.moments[:,0])

    def solve_PBM(self, t_vec=None):
        """
        Solve the Population Balance Model (PBM) using the specified time vector.

        Parameters:
            t_vec (numpy.ndarray): Time vector for the simulation.
        """
        solver = self.solver
        if t_vec is None:
            t_vec = solver.t_vec
            t_max = solver.t_vec[-1]
        else:
            t_max = t_vec[-1]

        if solver.dim == 1:
            solver.alpha_prim = solver.alpha_prim.item() if isinstance(solver.alpha_prim, np.ndarray) else solver.alpha_prim
            rhs = jit_pbm_rhs.get_dMdt_1d
            args = (solver.x_max, solver.GQMOM, solver.GQMOM_method, 
                    solver.moments_norm_factor, solver.n_add, solver.nu, 
                    solver.COLEVAL, solver.CORR_BETA, solver.G, 
                    solver.alpha_prim, solver.EFFEVAL, solver.SIZEEVAL, solver.V_unit,
                    solver.X_SEL, solver.Y_SEL, solver.V1_mean, 
                    solver.pl_P1, solver.pl_P2, solver.BREAKRVAL, 
                    solver.pl_v, solver.pl_q, solver.BREAKFVAL, solver.process_type)

            with np.errstate(divide='raise', over='raise',invalid='raise'):
                try:
                    solver.RES = integrate.solve_ivp(rhs, 
                                                     [0, t_max], 
                                                     solver.moments_norm[:,0], t_eval=t_vec,
                                                     args=args,
                                                     method='RK45',first_step=None,
                                                     atol=solver.atolarray, rtol=solver.rtolarray)
                    t_vec = solver.RES.t
                    y_evaluated = solver.RES.y
                    status = True if solver.RES.status == 0 else False
                except (FloatingPointError, ValueError) as e:
                    print(f"Exception encountered: {e}")
                    y_evaluated = -np.ones((2*solver.n_order,len(t_vec)))
                    status = False

        if solver.dim == 2:
            rhs = jit_pbm_rhs.get_dMdt_2d
            args = (solver.n_order, solver.indices, solver.COLEVAL, solver.CORR_BETA, solver.G, 
                    solver.alpha_prim, solver.EFFEVAL, solver.SIZEEVAL, solver.V_unit,
                    solver.X_SEL, solver.Y_SEL, solver.V1_mean, solver.V3_mean,
                    solver.pl_P1, solver.pl_P2, solver.pl_P3, solver.pl_P4, solver.BREAKRVAL, 
                    solver.pl_v, solver.pl_q, solver.BREAKFVAL, solver.process_type)

            with np.errstate(divide='raise', over='raise',invalid='raise'):
                try:
                    solver.RES = integrate.solve_ivp(rhs, 
                                                     [0, t_max], 
                                                     solver.moments[:,0], t_eval=t_vec,
                                                     args=args,
                                                     method='RK45',first_step=None,max_step=np.inf,
                                                     atol=solver.atolarray, rtol=solver.rtolarray)
                    t_vec = solver.RES.t
                    y_evaluated = solver.RES.y
                    status = True if solver.RES.status == 0 else False
                except (FloatingPointError, ValueError) as e:
                    print(f"Exception encountered: {e}")
                    y_evaluated = -np.ones((2*solver.n_order,len(t_vec)))
                    status = False

        solver.t_vec = t_vec
        if hasattr(solver, "moments_norm_factor") and solver.moments_norm_factor is not None:
            solver.moments = y_evaluated * solver.moments_norm_factor[:, np.newaxis] / solver.V_unit
        else:
            solver.moments = y_evaluated / solver.V_unit
        solver.calc_status = status
        if not solver.calc_status:
            print('Warning: The integral failed to converge!')

