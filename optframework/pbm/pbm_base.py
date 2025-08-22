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
from optframework.pbm.pbm_post import PBMPost
from optframework.pbm.pbm_quick_test import PBMQuickTest
from optframework.pbm.pbm_core import PBMCore

class PBMSolver:
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, 
                 load_attr=True, config_path=None):
        # BASELINE PATH
        self.work_dir = Path(os.getcwd()).resolve()
        
        ## Simulation parameters
        self.dim = dim                        # Dimension (1=1D, 2=2D, 3=3D)
        self.t_total = t_total                # Total simulation time [second]
        self.t_write = t_write                # Time interval for writing output
        self.t_vec = t_vec                    # Time vector for simulation
        self.n_order = 5                      # n_order*2 is the order of the moments [-]
        self.n_add = 10                       # Number of additional nodes [-] 
        self.GQMOM = False                    # Flag for using GQMOM method
        self.GQMOM_method = "gaussian"        # Method for GQMOM
        self.nu = 1                           # Exponent for the correction in gaussian-GQMOM

        self.atol_min  = 1e-16                # Minimum absolute tolerance
        self.atol_scale = 1e-9                # Scaling factor for absolute tolerance
        self.rtol = 1e-6                      # Relative tolerance
        ## Parameters in agglomeration kernels
        self.COLEVAL = 1                      # Case for calculation of beta. 1 = Orthokinetic, 2 = Perikinetic
        self.EFFEVAL = 2                      # Case for calculation of alpha. 1 = Full calculation, 2 = Reduced model (only based on primary particle interactions)
        self.SIZEEVAL = 2                     # Case for implementation of size dependency. 1 = No size dependency, 2 = Model from Soos2007 

        self.alpha_prim = np.ones(dim**2)     # Primary particle interaction parameter
        self.CORR_BETA = 1e6*2.5e-5           # Correction Term for collision frequency [-]

        ## Parameters in breakage kernels
        self.BREAKRVAL = 3                    # Case for calculation breakage rate. 1 = constant, 2 = size dependent
        self.BREAKFVAL = 3                    # Case for calculation breakage function. 1 = conservation of Hypervolume, 2 = conservation of 1 Moments 
        self.process_type = "breakage"        # Process type: "agglomeration", "breakage", or "mix"
        self.pl_v = 1                         # Number of fragments in product function of power law
        self.pl_q = 1                         # Parameter describing the breakage type in product function of power law
        self.pl_P1 = 1e-6                     # 1st parameter in power law for breakage rate 1D/2D
        self.pl_P2 = 0.5                      # 2nd parameter in power law for breakage rate 1D/2D
        self.pl_P3 = 1e-6                     # 3rd parameter in power law for breakage rate 2D
        self.pl_P4 = 0.5                      # 4th parameter in power law for breakage rate 2D
        
        self.B_F_type = 'int_func'            # Type for calculating B_F: 'int_func', 'MC_bond', 'ANN_MC'
        self.work_dir_MC_BOND = os.path.join(self.work_dir,'bond_break','int_B_F.npz')
        
        ## MATERIAL parameters:
        # NOTE: component 3 is defined as the magnetic component (both in 2D and 3D case)
        self.USE_PSD = True                   # Flag to initialize PSD (False = monodisperse primary particles)
        
        # Set default initial PSD file paths
        self.DIST1_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST2_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST3_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST1_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        self.DIST2_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        self.DIST3_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy' 
        
        self.V_unit = 1                       # Unit volume used to calculate the total particle concentration
        
        self.X_SEL = 0.310601                 # Size dependency parameter for Selomulya2003 / Soos2006 
        self.Y_SEL = 1.06168                  # Size dependency parameter for Selomulya2003 / Soos2006
        
        ## EXPERIMENTAL / PROCESS parameters:
        self.c_mag_exp = 0.01                 # Volume concentration of magnetic particles [Vol-%] 
        self.Psi_c1_exp = 1                   # Concentration ratio component 1 (V_NM1/V_M) [-] 
        self.Psi_c2_exp = 1                   # Concentration ratio component 2 (V_NM2/V_M) [-] 
        self.G = 1                            # Shear rate [1/s]

        # Load the configuration file, if available
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir,"config","PBM_config.py")

        if load_attr:
            self.load_attributes(config_path)
        self.check_params()
        self.reset_params()

        # Instantiate submodules
        self.post = PBMPost(self)
        self.quick_test = PBMQuickTest(self)
        self.core = PBMCore(self)
        
    def check_params(self):
        """
        Check the validity of dPBE parameters.
        """
        pass

    def load_attributes(self, config_path):
        """
        Load attributes dynamically from a configuration file.
        
        This method dynamically loads attributes from a specified Python configuration file 
        and assigns them to the DPBESolver instance. It checks for certain key attributes like 
        `alpha_prim` to ensure they match the PBE's dimensionality.
        
        Parameters
        ----------
        config_name : str
            The name of the configuration file (without the extension).
        config_path : str
            The file path to the configuration file.
        
        Raises
        ------
        Exception
            If the length of `alpha_prim` does not match the expected dimensionality.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Warning: Config file not found at: {config_path}.")
        print(f"The PBM is using config file at : {config_path}." )
        # Dynamically load the configuration file
        conf = runpy.run_path(config_path)
        config = conf['config']
        
        # Assign attributes from the configuration file to the DPBESolver instance
        for key, value in config.items():
            if value is not None:
                if key == "alpha_prim" and len(value) != self.dim**2:
                    raise Exception(f"The length of the array alpha_prim needs to be {self.dim**2}.")
                setattr(self, key, value)
                
        # Reset parameters, including time-related attributes
        reset_t = self.t_vec is None
        self.reset_params(reset_t=reset_t)
        
    def reset_params(self, reset_t=False):
        """
        This method is used to update the time vector (`t_vec`) if a new time configuration is provided, 
        or to recalculate key attributes related to particle concentrations (such as `V01`, `N01`, etc.) 
        when the volume unit (`V_unit`) or other related parameters are modified.
        
        Parameters
        ----------
        reset_t : bool, optional
            If True, the time vector (`t_vec`) is reset based on `t_total` and `t_write`. Defaults to False.
        
        """
        # Reset the time vector if reset_t is True
        if reset_t:
            self.t_vec = np.arange(0, self.t_total, self.t_write, dtype=float)
            
        # Set the number of time steps based on the time vector
        if self.t_vec is not None:
            self.t_num = len(self.t_vec)  
            
        self.DIST1 = os.path.join(self.DIST1_path,self.DIST1_name)
        self.DIST2 = os.path.join(self.DIST2_path,self.DIST2_name)
        self.DIST3 = os.path.join(self.DIST3_path,self.DIST3_name)  
        
        self.cv_1 = self.c_mag_exp*self.Psi_c1_exp   # Volume concentration of NM1 particles [Vol-%] 
        self.cv_2 = self.c_mag_exp*self.Psi_c2_exp   # Volume concentration of NM2 particles [Vol-%] 
        self.V01 = self.cv_1*self.V_unit             # Total volume concentration of component 1 [unit/unit] - NM1
        self.V02 = self.cv_2*self.V_unit             # Total volume concentration of component 2 [unit/unit] - NM2
        self.V03 = self.c_mag_exp*self.V_unit        # Total volume concentration of component 3 [unit/unit] - M

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
