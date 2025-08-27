# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:13:27 2025

"""

import numpy as np
import scipy.integrate as integrate
import optframework.utils.func.jit_pbm_rhs as jit_pbm_rhs
from optframework.utils.func.static_method import interpolate_psd

class PBMCore:
    """
    Core computation module for Population Balance Model (PBM) solver.
    
    This class handles the core computational tasks for the PBM solver, including
    moment initialization and PBE solving. It provides methods for setting up
    initial conditions and integrating the moment equations over time.
    """
    
    def __init__(self, solver):
        """Initialize PBMCore with reference to parent solver."""
        self.solver = solver

    def init_moments(self, x=None, NDF=None, NDF_shape="normal", N0=1.0, N01=1.0, N02=1.0,
                     V0=None, x_range=(0,1), mean=0.5, std_dev=0.1, shape=2, scale=1,
                     sigma=1, a=2, b=2, size=0.5):
        """
        Initialize moments for the PBM solver (supports both 1D and 2D).

        Parameters
        ----------
        x : numpy.ndarray, optional
            x-coordinates for the distribution function (1D only)
        NDF : numpy.ndarray, optional
            (Normalized) Distribution function (1D only)
        NDF_shape : str, optional
            Shape of the distribution ("normal", "gamma", "lognormal", "beta", "mono") (default: "normal")
        N0 : float, optional
            Initial total number concentration (1D only) (default: 1.0)
        N01 : float, optional
            Initial number concentration for component 1 (2D only) (default: 1.0)
        N02 : float, optional
            Initial number concentration for component 2 (2D only) (default: 1.0)
        V0 : float, optional
            Total volume of particles (1D only)
        x_range : tuple, optional
            Range of x values (default: (0,1))
        mean : float, optional
            Mean value for normal/lognormal distribution (default: 0.5)
        std_dev : float, optional
            Standard deviation for normal distribution (default: 0.1)
        shape : float, optional
            Shape parameter for gamma distribution (default: 2)
        scale : float, optional
            Scale parameter for gamma distribution (default: 1)
        sigma : float, optional
            Standard deviation for lognormal distribution (default: 1)
        a : float, optional
            Alpha parameter for beta distribution (default: 2)
        b : float, optional
            Beta parameter for beta distribution (default: 2)
        size : float, optional
            Size parameter for mono distribution (default: 0.5)
        """
        solver = self.solver
        
        if solver.dim == 1:
            self._init_moments_1d(x, NDF, NDF_shape, N0, V0, x_range, mean, std_dev, 
                                 shape, scale, sigma, a, b, size)
        elif solver.dim == 2:
            self._init_moments_2d(N01, N02)
        else:
            raise ValueError(f"Unsupported dimension: {solver.dim}. Only 1D and 2D are supported.")

    def _init_moments_1d(self, x, NDF, NDF_shape, N0, V0, x_range, mean, std_dev, 
                        shape, scale, sigma, a, b, size):
        """
        Initialize 1D moments for the PBM solver.
        
        Creates distribution function and calculates initial moments for 1D systems.
        Handles both PSD file input and analytical distributions.
        """
        solver = self.solver
        
        # Generate NDF if not provided
        if x is None or NDF is None:
            if solver.USE_PSD:
                x = np.linspace(0.0, x_range[1], 10000)
                if np.any(x < 0):
                    raise ValueError("Error: Volume (x) cannot be negative!")
                d = np.where(x > 0, (6 * x / np.pi) ** (1/3), 0)
                if V0 is None:
                    raise ValueError("Total volume of particle must be given to get PSD")
                NDF = np.zeros_like(d)
                NDF[1:] = interpolate_psd(d[1:], solver.DIST1, V0)
                NDF /= x[1] - x[0]
            else:
                x, NDF = self._create_distribution(NDF_shape, x_range, mean, std_dev, 
                                                 shape, scale, sigma, a, b, size)
                # Normalize the distribution
                moment0 = np.trapz(NDF, x)
                NDF /= moment0
        
        # Set maximum x value and initialize moments array
        solver.x_max = 1.0
        solver.moments = np.zeros((solver.n_order * 2, solver.t_num))
        
        # Scale NDF by initial concentration and calculate moments
        NDF *= N0
        solver.moments[:, 0] = np.array([np.trapz(NDF * (x ** k), x) 
                                       for k in range(2 * solver.n_order)]) * solver.V_unit
        
        # Normalize moments and set tolerance
        solver.normalize_mom()
        solver.set_tol(solver.moments_norm[:, 0])

    def _init_moments_2d(self, N01, N02):
        """
        Initialize 2D moments for the PBM solver.
        
        Creates distribution functions for both components and calculates
        initial 2D moments using trapezoidal integration.
        """
        solver = self.solver
        
        # Create normalized distribution functions for both components
        x1, NDF1 = solver.create_ndf(distribution="normal", x_range=(1e-2, 1e-1), 
                                    mean=6e-2, std_dev=2e-2)
        x2, NDF2 = solver.create_ndf(distribution="normal", x_range=(1e-2, 1e-1), 
                                    mean=6e-2, std_dev=2e-2)
        
        # Scale by initial concentrations
        NDF1 *= N01
        NDF2 *= N02
        
        # Generate 2D moment indices and initialize moments array
        solver.moment_2d_indices_c()
        mu_num = len(solver.indices)
        solver.moments = np.zeros((mu_num, solver.t_num))
        
        # Calculate initial moments using 2D integration
        for idx in range(mu_num):
            k, l = solver.indices[idx, 0], solver.indices[idx, 1]
            solver.moments[idx, 0] = solver.trapz_2d(NDF1, NDF2, x1, x2, k, l) * solver.V_unit
        
        # Set tolerance
        solver.set_tol(solver.moments[:, 0])

    def _create_distribution(self, NDF_shape, x_range, mean, std_dev, shape, scale, sigma, a, b, size):
        """
        Create distribution based on shape parameter.
        
        Helper method that delegates to solver's create_ndf method with
        appropriate parameters for different distribution types.
        
        Returns
        -------
        tuple
            (x, NDF) coordinate array and distribution values
        """
        solver = self.solver
        
        if NDF_shape == "normal":
            return solver.create_ndf(distribution="normal", x_range=x_range, mean=mean, std_dev=std_dev)
        elif NDF_shape == "gamma":
            return solver.create_ndf(distribution="gamma", x_range=x_range, shape=shape, scale=scale)
        elif NDF_shape == "lognormal":
            return solver.create_ndf(distribution="lognormal", x_range=x_range, mean=mean, sigma=sigma)
        elif NDF_shape == "beta":
            return solver.create_ndf(distribution="beta", x_range=x_range, a=a, b=b)
        elif NDF_shape == "mono":
            return solver.create_ndf(distribution="mono", x_range=x_range, size=size)
        else:
            raise ValueError(f"Unsupported distribution shape: {NDF_shape}")

    def init_moments_2d(self, N01=1.0, N02=1.0):
        """
        Initialize the 2D moments for the PBM solver.
        
        .. deprecated:: 
            This method is deprecated. Use `init_moments()` instead, which automatically
            handles both 1D and 2D cases based on solver.dim.

        Parameters
        ----------
        N01 : float, optional
            Initial number concentration for component 1 (default: 1.0)
        N02 : float, optional
            Initial number concentration for component 2 (default: 1.0)
        """
        import warnings
        warnings.warn("init_moments_2d() is deprecated. Use init_moments() instead.", 
                     DeprecationWarning, stacklevel=2)
        self.init_moments(N01=N01, N02=N02)

    def solve_PBM(self, t_vec=None):
        """
        Solve the Population Balance Model using moment equations.
        
        Integrates the moment ODEs over time using scipy.integrate.solve_ivp
        with RK45 method. Handles both 1D and 2D systems with appropriate
        right-hand-side functions.

        Parameters
        ----------
        t_vec : numpy.ndarray, optional
            Time vector for the simulation. If None, uses solver.t_vec
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
                    solver.alpha_prim, solver.SIZEEVAL, solver.V_unit,
                    solver.X_SEL, solver.Y_SEL, 
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
                    solver.alpha_prim, solver.SIZEEVAL, solver.V_unit,
                    solver.X_SEL, solver.Y_SEL, 
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

