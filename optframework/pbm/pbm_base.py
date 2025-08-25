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
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, 
                 load_attr=True, config_path=None):
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
        Perform 2D trapezoidal integration.

        Parameters:
            NDF1 (numpy.ndarray): First distribution function.
            NDF2 (numpy.ndarray): Second distribution function.
            x1 (numpy.ndarray): x-coordinates for NDF1.
            x2 (numpy.ndarray): x-coordinates for NDF2.
            k (int): Power for x1.
            l (int): Power for x2.

        Returns:
            float: Result of the 2D trapezoidal integration.
        """
        integrand = np.outer(NDF1 * (x1 ** k), NDF2 * (x2 ** l))
        integral_x2 = np.trapz(integrand, x2, axis=1)
        integral_x1 = np.trapz(integral_x2, x1)
        return integral_x1
    
    def normalize_mom(self):
        """
        Normalize the moments by scaling them with the maximum x-coordinate.
        """
        self.moments_norm = np.copy(self.moments)
        self.moments_norm_factor = np.array([self.x_max**k for k in range(self.n_order*2)])
        self.moments_norm[:,0] = self.moments[:,0] / self.moments_norm_factor
        
    def set_tol(self, moments):
        """
        Set the integration tolerance based on the initial moments.

        Parameters:
            moments (numpy.ndarray): Initial moments.
        """
        self.atolarray = np.maximum(self.atol_min, self.atol_scale * np.abs(moments))
        self.rtolarray = np.full_like(moments, self.rtol)
        
    def create_ndf(self, distribution="normal", x_range=(0, 100), points=1000, **kwargs):
        """
        Create a normalized distribution function (NDF).
        PS: Actually they are Probability Density Function!

        Parameters:
            distribution (str): Type of distribution ("normal", "gamma", "lognormal", "beta").
            x_range (tuple): Range of the variable (start, end). Defaults to (0, 100).
            points (int): Number of points in the range. Defaults to 5000.
            kwargs: Additional parameters for the selected distribution.

        Returns:
            tuple: (x, ndf), where x is the variable range and ndf is the distribution function values.
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
            # size = kwargs.get("size", (x_range[1] - x_range[0]) / 2)
            # ndf = np.zeros_like(x) 
            # ## Set the value of ndf to ensure trapz(ndf, x) = 1
            # closest_idx = np.argmin(np.abs(x - size))
            # dx = x[1] - x[0]
            # ndf[closest_idx] = 1 / dx
            
            mean = kwargs.get("size", (x_range[1]-x_range[0])/2 )
            std_dev = (x_range[1]-x_range[0]) / 1000
            ndf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)  
        else:
            raise ValueError("Unsupported distribution type. Choose from 'normal', 'gamma', 'lognormal', 'beta'.")

        return x, ndf

    def NDF_approx(self, x, nodes, weights, width=1e-1):
        """
        Approximate NDF/Dirac delta function using a sum of Gaussian distributions.

        Parameters:
            x (numpy.ndarray): Points where the function is evaluated.
            nodes (numpy.ndarray): Positions of delta peaks.
            weights (numpy.ndarray): Weights of the delta peaks.
            width (float): Standard deviation of the Gaussian kernel.

        Returns:
            numpy.ndarray: Approximated δ function values.
        """
        NDF_ap = np.zeros_like(x)
        for pos, weight in zip(nodes, weights):
            NDF_ap += weight * stats.norm.pdf(x, loc=pos, scale=width)
        norm_factor = np.trapz(NDF_ap, x)
        NDF_ap /= norm_factor
        return NDF_ap
    
    def moment_2d_indices_chy(self):
        """
        Generate 2D moment indices for CHY method.
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
        Generate 2D moment indices for C method.
        """
        self.indices = []
        for i in range(2 * self.n_order):
            self.indices.append([i, 0])
        
        for i in range(self.n_order):
            for j in range(1, 2 * self.n_order):
                self.indices.append([i, j])
        self.indices = np.array(self.indices)
