# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:13:27 2025

@author: px2030
"""

import os
import numpy as np
import math
from pathlib import Path
import scipy.stats as stats
import runpy
from optframework.base.base_solver import BaseSolver
from optframework.pbm.pbm_post import PBMPost
from optframework.pbm.pbm_quick_test import PBMQuickTest
from optframework.pbm.pbm_core import PBMCore

class PBMSolver(BaseSolver):
    """
    Population Balance Model (PBM) solver using the Method of Moments.
    
    This class implements a moment-based approach to solve Population Balance Equations (PBE)
    for particle systems undergoing agglomeration and/or breakage processes. Unlike discrete
    methods that track the full particle size distribution, the Method of Moments tracks
    only the statistical moments of the distribution, making it computationally more efficient
    while providing information about mean sizes, total concentrations, and distribution width.
    
    The solver supports both 1D (single component) and 2D (two-component) systems, making it
    suitable for modeling processes like flocculation, crystallization, magnetic separation,
    and other particle population dynamics.
    
    Key Features:
    - Method of Moments for computational efficiency
    - Support for 1D and 2D particle systems
    - Configurable agglomeration and breakage kernels
    - Integration with visualization and post-processing tools
    - Flexible initial distribution shapes (normal, gamma, lognormal, beta, monodisperse)
    - Advanced quadrature methods (GQMOM) for moment reconstruction
    
    The solver integrates moment equations of the form:
    dμₖ/dt = Birth terms - Death terms
    where μₖ represents the k-th moment of the particle size distribution.
    
    Parameters
    ----------
    dim : int
        Dimension of the PBE problem (1 for single component, 2 for two components)
    t_total : int, optional
        Total simulation time in seconds (default: 601)
    t_write : int, optional
        Output time interval in seconds (default: 100)
    t_vec : array-like, optional
        Custom time vector for simulation output points
    load_attr : bool, optional
        Whether to load attributes from configuration file (default: True)
    config_path : str, optional
        Path to configuration file (default: uses PBM_config.py)
    
    Attributes
    ----------
    n_order : int
        Order parameter where n_order×2 is the total number of moments tracked
    moments : ndarray
        Array storing moment values over time
    indices : ndarray
        Moment indices for 2D systems (k,l pairs for μₖₗ)
    core : PBMCore
        Core computation module for moment initialization and PBE solving
    post : PBMPost
        Post-processing module for analysis and data extraction
    quick_test : PBMQuickTest
        Testing module for convergence and validation studies
    """
    
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, 
                 load_attr=True, config_path=None):
        """
        Initialize the PBM solver with specified parameters.
        
        Sets up moment tracking parameters, loads configuration, initializes
        submodules, and prepares the solver for moment-based PBE solving.
        """
        self._init_base_parameters(dim, t_total, t_write, t_vec)
        
        ## Simulation parameters
        self.n_order = 5                      # n_order*2 is the order of the moments [-]
        self.n_add = 10                       # Number of additional nodes [-] 
        self.GQMOM = False                    # Flag for using GQMOM method
        self.GQMOM_method = "gaussian"        # Method for GQMOM
        self.nu = 1                           # Exponent for the correction in gaussian-GQMOM

        self.atol_min  = 1e-16                # Minimum absolute tolerance
        self.atol_scale = 1e-9                # Scaling factor for absolute tolerance
        self.rtol = 1e-6                      # Relative tolerance
        
        # Load the configuration file, if available
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir,"config","PBM_config.py")

        if load_attr:
            self.load_attributes(config_path)
        self._check_params()
        self._reset_params()

        # Instantiate submodules
        self.post = PBMPost(self)
        self.quick_test = PBMQuickTest(self)
        self.core = PBMCore(self)
 
    def trapz_2d(self, NDF1, NDF2, x1, x2, k, l):
        """
        Perform 2D trapezoidal integration for moment calculation.
        
        Computes the integral ∫∫ x1^k × x2^l × NDF1(x1) × NDF2(x2) dx1 dx2
        using the trapezoidal rule for 2D moment initialization.

        Parameters
        ----------
        NDF1 : numpy.ndarray
            First normalized distribution function
        NDF2 : numpy.ndarray
            Second normalized distribution function
        x1 : numpy.ndarray
            x-coordinates for NDF1
        x2 : numpy.ndarray
            x-coordinates for NDF2
        k : int
            Power for x1 coordinate
        l : int
            Power for x2 coordinate

        Returns
        -------
        float
            Result of the 2D trapezoidal integration
        """
        integrand = np.outer(NDF1 * (x1 ** k), NDF2 * (x2 ** l))
        integral_x2 = np.trapz(integrand, x2, axis=1)
        integral_x1 = np.trapz(integral_x2, x1)
        return integral_x1
    
    def normalize_mom(self):
        """
        Normalize moments by scaling with maximum x-coordinate.
        
        Creates dimensionless normalized moments for numerical stability
        and comparison purposes. Sets moments_norm and moments_norm_factor arrays.
        """
        self.moments_norm = np.copy(self.moments)
        self.moments_norm_factor = np.array([self.x_max**k for k in range(self.n_order*2)])
        self.moments_norm[:,0] = self.moments[:,0] / self.moments_norm_factor
        
    def set_tol(self, moments):
        """
        Set integration tolerance arrays based on initial moment values.
        
        Calculates absolute and relative tolerance arrays for ODE integration
        to ensure numerical stability across different moment magnitudes.

        Parameters
        ----------
        moments : numpy.ndarray
            Initial moment values for tolerance scaling
        """
        self.atolarray = np.maximum(self.atol_min, self.atol_scale * np.abs(moments))
        self.rtolarray = np.full_like(moments, self.rtol)
        
    def create_ndf(self, distribution="normal", x_range=(0, 100), points=1000, **kwargs):
        """
        Create a normalized distribution function (probability density function).
        
        Generates various types of probability density functions for initial
        particle size distribution setup in moment calculations.

        Parameters
        ----------
        distribution : str, optional
            Type of distribution: "normal", "gamma", "lognormal", "beta", "mono" (default: "normal")
        x_range : tuple, optional
            Range of the variable (start, end) (default: (0, 100))
        points : int, optional
            Number of discretization points (default: 1000)
        \*\*kwargs
            Distribution-specific parameters:
            
            - normal: mean, std_dev
            - gamma: shape, scale
            - lognormal: mean, sigma
            - beta: a, b
            - mono: size

        Returns
        -------
        tuple
            (x, ndf) where x is coordinate array and ndf is distribution values

        Raises
        ------
        ValueError
            If distribution type is unsupported or x_range is invalid for distribution
        """
        # Generate the x variable
        x = np.linspace(x_range[0], x_range[1], points)

        # Ensure x_range is valid for gamma and lognormal distributions
        if distribution in ["gamma", "lognormal"] and x_range[0] < 0:
            raise ValueError(f"{distribution.capitalize()} distribution requires x_range[0] >= 0.")

        # Define the distribution based on the input type
        if distribution == "normal":
            mean = kwargs.get("mean", 50)  # Default mean
            std_dev = kwargs.get("std_dev", 10)  # Default standard deviation
            ndf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

        elif distribution == "gamma":
            shape = kwargs.get("shape", 2)  # Default shape parameter
            scale = kwargs.get("scale", 1)  # Default scale parameter
            ndf = stats.gamma.pdf(x, shape, scale=scale)

        elif distribution == "lognormal":
            mean = kwargs.get("mean", 1)  # Default mean of log-space
            sigma = kwargs.get("sigma", 0.5)  # Default standard deviation of log-space
            ndf = stats.lognorm.pdf(x, sigma, scale=np.exp(mean))

        elif distribution == "beta":
            a = kwargs.get("a", 2)  # Default alpha parameter
            b = kwargs.get("b", 2)  # Default beta parameter
            if not (0 <= x_range[0] < x_range[1] <= 1):
                raise ValueError("Beta distribution requires x_range in [0, 1].")
            ndf = stats.beta.pdf(x, a, b)
        elif distribution == "mono":
            mean = kwargs.get("size", (x_range[1]-x_range[0])/2 )
            std_dev = (x_range[1]-x_range[0]) / 1000
            ndf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)  
        else:
            raise ValueError("Unsupported distribution type. Choose from 'normal', 'gamma', 'lognormal', 'beta'.")

        return x, ndf

    def NDF_approx(self, x, nodes, weights, width=1e-1):
        """
        Approximate normalized distribution function using sum of Gaussian kernels.
        
        Creates smooth approximation of delta functions or discrete distributions
        using weighted Gaussian distributions, useful for GQMOM reconstruction.

        Parameters
        ----------
        x : numpy.ndarray
            Points where function is evaluated
        nodes : numpy.ndarray
            Positions of distribution peaks
        weights : numpy.ndarray
            Weights of distribution peaks
        width : float, optional
            Standard deviation of Gaussian kernels (default: 1e-1)

        Returns
        -------
        numpy.ndarray
            Approximated distribution function values
        """
        NDF_ap = np.zeros_like(x)
        for pos, weight in zip(nodes, weights):
            NDF_ap += weight * stats.norm.pdf(x, loc=pos, scale=width)
        norm_factor = np.trapz(NDF_ap, x)
        NDF_ap /= norm_factor
        return NDF_ap
    
    def moment_2d_indices_chy(self):
        """
        Generate 2D moment indices for CHY (Conditional Hyperbolic) method.
        
        Creates specific moment index patterns optimized for CHY quadrature
        method. Only supports n_order = 2 or 3.
        
        Raises
        ------
        ValueError
            If n_order is not 2 or 3
        """
        if self.n_order == 2:
            self.indices = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
            
        elif self.n_order == 3:
            self.indices = np.array([
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [2, 0],
                    [1, 1],
                    [0, 2],
                    [3, 0],
                    [0, 3],
                    [4, 0],
                    [0, 4],
                ])
            
        else:
            raise ValueError(f"Incorrect order of quadrature nodes, which is {self.n_order} (only 2 or 3 is supported!)")
            
    def moment_2d_indices_c(self):
        """
        Generate 2D moment indices for C (Complete) method.
        
        Creates comprehensive moment index patterns for general 2D moment
        calculations. Generates indices for both pure moments (k,0) and (0,l)
        and mixed moments (k,l) up to specified orders.
        """
        self.indices = []
        for i in range(2 * self.n_order):
            self.indices.append([i, 0])
        
        for i in range(self.n_order):
            for j in range(1, 2 * self.n_order):
                self.indices.append([i, j])
        self.indices = np.array(self.indices)
