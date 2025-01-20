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
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt
import optframework.utils.func.jit_pbm_qmom as qmom
import optframework.utils.func.jit_pbm_rhs as jit_pbm_rhs

class PBMSolver:
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, 
                 load_attr=True, config_path=None):
        # BASELINE PATH
        self.work_dir = Path(os.getcwd()).resolve()
        
        ## Simulation parameters
        self.dim = dim                        # Dimension (1=1D, 2=2D, 3=3D)
        self.t_total = t_total                       # total simulation time [second]
        self.t_write = t_write
        self.t_vec = t_vec
        self.n_order = 5                          # Order of the moments [-]
        self.n_add = 10                          # Number of additional nodes [-] 
        self.GQMOM = False
        self.GQMOM_method = "gaussian"
        self.nu = 1                           # Exponent for the correction in gaussian-GQMOM

        ## Parameters in agglomeration kernels
        self.COLEVAL = 1                      # Case for calculation of beta. 1 = Orthokinetic, 2 = Perikinetic
        self.EFFEVAL = 2                      # Case for calculation of alpha. 1 = Full calculation, 2 = Reduced model (only based on primary particle interactions)
        self.SIZEEVAL = 2                     # Case for implementation of size dependency. 1 = No size dependency, 2 = Model from Soos2007 

        self.alpha_prim = np.ones(dim**2)
        self.CORR_BETA = 1e6*2.5e-5           # Correction Term for collision frequency [-]. Can be defined
                                            # dependent on rotary speed, e.g. ((corr_beta250-corr_beta100)*(n_exp-100)/(250-100)+corr_beta100)

        ## Parameters in breakage kernels
        self.BREAKRVAL = 3                    # Case for calculation breakage rate. 1 = constant, 2 = size dependent
        self.BREAKFVAL = 3                    # Case for calculation breakage function. 1 = conservation of Hypervolume, 2 = conservation of 0 Moments 
        self.process_type = "breakage"    # "agglomeration": only calculate agglomeration, "breakage": only calculate breakage, "mix": calculate both agglomeration and breakage
        self.pl_v = 1                         # number of fragments in product function of power law
                                            # or (v+1)/v: number of fragments in simple power law  
        self.pl_q = 1                         # parameter describes the breakage type(in product function of power law) 
        self.pl_P1 = 1e-6                     # 1. parameter in power law for breakage rate  1d/2d
        self.pl_P2 = 0.5                      # 2. parameter in power law for breakage rate  1d/2d
        self.pl_P3 = 1e-6                     # 3. parameter in power law for breakage rate  2d
        self.pl_P4 = 0.5                      # 4. parameter in power law for breakage rate  2d
        self.V1_mean = 4.37*1e-14
        self.V3_mean = 4.37*1e-14
        self.B_F_type = 'int_func'            # 'int_func': calculate B_F with breakage function
                                            # 'MC_bond': Obtain B_F directly from the result of MC_bond
                                            # 'ANN_MC': Calculate MC results using ANN model and convert to B_F
        self.work_dir_MC_BOND = os.path.join(self.work_dir,'bond_break','int_B_F.npz')
        
        ## MATERIAL parameters:
        # NOTE: component 3 is defined as the magnetic component (both in 2D and 3D case)
        self.R01 = 2.9e-7                     # Radius primary particle component 1 [m] - NM1
        self.R02 = 2.9e-7                     # Radius primary particle component 2 [m] - NM2
        self.R03 = 2.9e-7                     # Radius primary particle component 3 [m] - M3
        self.USE_PSD = True                   # Define wheter or not the PSD should be initializes (False = monodisperse primary particles)
        
        # Set default initial PSD file paths
        self.DIST1_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST2_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST3_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST1_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        self.DIST2_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        self.DIST3_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy' 
        
        self.V_unit = 1                  # The unit volume used to calculate the total particle concentration. 
                                            # It is essentially a parameter used to scale the variabel.
                                            
        self.X_SEL = 0.310601                 # Size dependency parameter for Selomulya2003 / Soos2006 
        self.Y_SEL = 1.06168                  # Size dependency parameter for Selomulya2003 / Soos2006
        
        ## EXPERIMENTAL / PROCESS parameters:
        self.c_mag_exp = 0.01                 # Volume concentration of magnetic particles [Vol-%] 
        self.Psi_c1_exp = 1                   # Concentration ratio component 1 (V_NM1/V_M) [-] 
        self.Psi_c2_exp = 1                   # Concentration ratio component 2 (V_NM2/V_M) [-] 
        self.G = 1                            # Shear rate [1/s]. Can be defined dependent on rotary speed, 
                                            # e.g. G=(1400-354)*(n_exp-100)/(250-100)+354

        # Load the configuration file, if available
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir,"config","PBM_config.py")

        if load_attr:
            self.load_attributes(config_path)
        self.check_params()
        
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
        self.N01 = 3*self.V01/(4*math.pi*self.R01**3)     # Total number concentration of primary particles component 1 [1/m³] - NM1 (if no PSD)
        self.V02 = self.cv_2*self.V_unit         # Total volume concentration of component 2 [unit/unit] - NM2
        self.N02 = 3*self.V02/(4*math.pi*self.R02**3)     # Total number concentration of primary particles component 2 [1/m³] - NM2 (if no PSD)
        self.V03 = self.c_mag_exp*self.V_unit        # Total volume concentration of component 3 [unit/unit] - M
        self.N03 = 3*self.V03/(4*math.pi*self.R03**3)     # Total number concentration of primary particles component 1 [1/m³] - M (if no PSD) 
          
    def quick_test_QMOM(self, NDF_shape="normal"):
        if NDF_shape == "normal":
            x, NDF = self.create_ndf(distribution="normal", x_range=(0, 100), mean=50, std_dev=20)
        elif NDF_shape == "gamma":
            x, NDF = self.create_ndf(distribution="gamma", x_range=(0, 1), shape=5, scale=1)
        elif NDF_shape == "lognormal":
            x, NDF = self.create_ndf(distribution="lognormal", x_range=(0, 1), mean=0.1, sigma=1)
        elif NDF_shape == "beta":
            x, NDF = self.create_ndf(distribution="beta", x_range=(0, 1), a=2, b=2)
        n = self.n_order
        moments = np.array([np.trapz(NDF * (x ** k), x) for k in range(2*n)])
        nodes, weights = qmom.calc_qmom_nodes_weights(moments, n)
        nodes_G, weights_G = qmom.calc_gqmom_nodes_weights(moments, n, self.n_add, 
                                                           method=self.GQMOM_method, nu=self.nu)
        moments_QMOM = np.zeros_like(moments)
        moments_GQMOM = np.zeros_like(moments)  
        for i in range(2*n):
            moments_QMOM[i] = sum(weights * nodes**i)
            moments_GQMOM[i] = sum(weights_G * nodes_G**i)
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        self.plot_nodes_weights_comparision(x, NDF, nodes, weights, nodes_G, weights_G)
        self.plot_moments_comparison(moments, moments_QMOM, moments_GQMOM)
        return moments, moments_QMOM, moments_GQMOM
    
    def quick_test_QMOM_normal(self, NDF_shape="normal"):
        if NDF_shape == "normal":
            x, NDF = self.create_ndf(distribution="normal", x_range=(0, 1e-12), mean=5e-13, std_dev=2e-13)
        elif NDF_shape == "gamma":
            x, NDF = self.create_ndf(distribution="gamma", x_range=(0, 1e-12), shape=1, scale=1)
        elif NDF_shape == "lognormal":
            x, NDF = self.create_ndf(distribution="lognormal", x_range=(0, 1e-12), mean=5e-13, sigma=1e-10)
        elif NDF_shape == "beta":
            x, NDF = self.create_ndf(distribution="beta", x_range=(0, 1), a=2, b=2)
        elif NDF_shape == "mono":
            x, NDF = self.create_ndf(distribution="mono", x_range=(0, 1e-12), size=5e-13)
        n = self.n_order
        NDF *= 1e12
        # x_normal = x / x[-1]
        # NDF_normal = NDF / x[-1]
        moments = np.array([np.trapz(NDF * (x ** k), x) for k in range(2*n)])
        moments_normal = np.array([moments[k] / x[-1]**k for k in range(2*n)])
        nodes, weights = qmom.calc_qmom_nodes_weights(moments_normal, n)
        nodes_G, weights_G = qmom.calc_gqmom_nodes_weights(moments_normal, n, self.n_add, 
                                                           method=self.GQMOM_method, nu=self.nu)
        nodes *= x[-1]
        # weights *= x[-1]
        nodes_G *= x[-1]
        # weights_G *= x[-1]
        moments_QMOM = np.zeros_like(moments)
        moments_GQMOM = np.zeros_like(moments)  
        for i in range(2*n):
            moments_QMOM[i] = sum(weights * nodes**i)
            moments_GQMOM[i] = sum(weights_G * nodes_G**i)
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        self.plot_nodes_weights_comparision(x, NDF, nodes, weights, nodes_G, weights_G)
        self.plot_moments_comparison(moments, moments_QMOM, moments_GQMOM)
        
        return moments, moments_QMOM, moments_GQMOM
    def init_moments(self, NDF_shape="normal"):
        if self.USE_PSD:
            return
        else:
            if NDF_shape == "normal":
                x, NDF = self.create_ndf(distribution="normal", x_range=(0, 1e-12), mean=5e-13, std_dev=2e-13)
            elif NDF_shape == "gamma":
                x, NDF = self.create_ndf(distribution="gamma", x_range=(0, 1), shape=2, scale=1)
            elif NDF_shape == "lognormal":
                x, NDF = self.create_ndf(distribution="lognormal", x_range=(0, 1e-12), mean=5e-13, sigma=1)
            elif NDF_shape == "beta":
                x, NDF = self.create_ndf(distribution="beta", x_range=(0, 1), a=2, b=2)
            # NDF *= 1e6
            self.x_max = x[-1]
            self.moments = np.zeros((self.n_order*2,self.t_num))
            self.moments[:,0] = np.array([np.trapz(NDF * (x ** k), x) for k in range(2*self.n_order)])
            self.moments_norm = np.copy(self.moments)
            self.moments_norm_factor = np.array([self.x_max**k for k in range(self.n_order*2)])
            self.moments_norm[:,0] = self.moments[:,0] / self.moments_norm_factor
            
        self.m = 2*self.n_order
    def solve_PBM(self, t_vec=None):
        if t_vec is None:
            t_vec = self.t_vec
            t_max = self.t_vec[-1]
        else:
            t_max = t_vec[-1]
            
        if self.dim == 1:
            rhs = jit_pbm_rhs.get_dMdt_1d
            args = (self.x_max, self.GQMOM, self.GQMOM_method, 
                    self.moments_norm_factor, self.n_add, self.nu, 
                    self.COLEVAL, self.CORR_BETA, self.G, 
                    self.alpha_prim, self.EFFEVAL, self.SIZEEVAL, 
                    self.X_SEL, self.Y_SEL, self.V1_mean, 
                    self.pl_P1, self.pl_P2, self.BREAKRVAL, 
                    self.pl_v, self.pl_q, self.BREAKFVAL, self.process_type)
            
            with np.errstate(divide='raise', over='raise',invalid='raise'):
                try:
                    self.RES = integrate.solve_ivp(rhs, 
                                                    [0, t_max], 
                                                    self.moments_norm[:,0], t_eval=t_vec,
                                                    args=args,
                                                    ## If `rtol` is set too small, it may cause the results to diverge, 
                                                    ## leading to the termination of the calculation.
                                                    method='Radau',first_step=1e-3,rtol=1e-1)
                    
                    # Reshape and save result to N and t_vec
                    t_vec = self.RES.t
                    y_evaluated = self.RES.y
                    status = True if self.RES.status == 0 else False
                except (FloatingPointError, ValueError) as e:
                    print(f"Exception encountered: {e}")
                    y_evaluated = -np.ones((self.m,len(t_vec)))
                    status = False
        # Monitor whether integration are completed  
        self.t_vec = t_vec 
        # self.N = y_evaluated / eva_N_scale
        self.moments = y_evaluated 
        self.calc_status = status   
        if not self.calc_status:
            print('Warning: The integral failed to converge!')
        
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
            # size = kwargs.get("size", (x_range[1]-x_range[0])/2 )
            # ndf = np.zeros_like(x)
            # closest_idx = np.argmin(np.abs(x - size))
            # ndf[closest_idx] = 1.0
            mean = kwargs.get("size", (x_range[1]-x_range[0])/2 )
            std_dev = (x_range[1]-x_range[0]) / points
            ndf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)  
        else:
            raise ValueError("Unsupported distribution type. Choose from 'normal', 'gamma', 'lognormal', 'beta'.")

        return x, ndf

    def plot_moments_comparison(self, moments, moments_QMOM, moments_GQMOM):
        """
        Plot a visual comparison of QMOM and GQMOM moments against original moments.

        Parameters:
            moments (array-like): Original moments (true values).
            moments_QMOM (array-like): Moments calculated using QMOM.
            moments_GQMOM (array-like): Moments calculated using GQMOM.
        """
        fig=plt.figure()
        ori_ax = fig.add_subplot(1,2,1)   
        rel_ax = fig.add_subplot(1,2,2)  
        # Calculate relative errors
        relative_error_QMOM = np.abs((moments_QMOM - moments) / moments)
        relative_error_GQMOM = np.abs((moments_GQMOM - moments) / moments)

        # Define the orders of moments
        orders = np.arange(len(moments))

        # Plot 1: Original values comparison
        ori_ax, fig = pt.plot_data(orders, moments, fig=fig, ax=ori_ax,
                                xlbl='Order of Moment',
                                ylbl='Moment Value',
                                lbl='Original Moments (True)',
                                clr='k',mrk='o')
        ori_ax, fig = pt.plot_data(orders, moments_QMOM, fig=fig, ax=ori_ax,
                                lbl='QMOM Moments',
                                clr='b',mrk='o')
        ori_ax, fig = pt.plot_data(orders, moments_GQMOM, fig=fig, ax=ori_ax,
                                lbl='GQMOM Moments',
                                clr='r',mrk='o')
        
        rel_ax, fig = pt.plot_data(orders, relative_error_QMOM, fig=fig, ax=rel_ax,
                                xlbl='Order of Moment',
                                ylbl='Relative Error',
                                lbl='Relative Error (QMOM)',
                                clr='b',mrk='o')
        rel_ax, fig = pt.plot_data(orders, relative_error_GQMOM, fig=fig, ax=rel_ax,
                                lbl='Relative Error (GQMOM)',
                                clr='r',mrk='o')
        ori_ax.grid('minor')
        ori_ax.set_yscale('log')
        rel_ax.grid('minor')
        plt.title('Comparison of Moments')
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_NDF_comparison(self, x, NDF, NDF_QMOM, NDF_GQMOM):
        """
        Plot a visual comparison of QMOM and GQMOM NDF against original NDF.

        Parameters:
            NDF (array-like): Original NDF (true values).
            NDF_QMOM (array-like): NDF calculated using QMOM.
            NDF_GQMOM (array-like): NDF calculated using GQMOM.
        """
        fig=plt.figure()

        # Plot 1: Original values comparison
        ax, fig = pt.plot_data(x, NDF, fig=fig, ax=None,
                                xlbl='x',
                                ylbl='NDF',
                                lbl='Original(True)',
                                clr='k',mrk='o')
        ax, fig = pt.plot_data(x, NDF_QMOM, fig=fig, ax=ax,
                                lbl='QMOM',
                                clr='b',mrk='o')
        ax, fig = pt.plot_data(x, NDF_GQMOM, fig=fig, ax=ax,
                                lbl='GQMOM',
                                clr='r',mrk='o')
        
        ax.grid('minor')
        plt.title('Comparison of NDF')
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_nodes_weights_comparision(self, x, NDF, nodes, weights, nodes_G, weights_G):
        fig=plt.figure()
        # ax1 = fig.add_subplot(1,3,1)   
        # ax2 = fig.add_subplot(1,3,2)
        # ax3 = fig.add_subplot(1,3,3)
        # Plot 1: Original values comparison
        ax1, fig = pt.plot_data(x, NDF, fig=fig, ax=None,
                                xlbl='x',
                                ylbl='NDF',
                                lbl='Original(True)',
                                clr='k',mrk='o')
        
        ax2 = ax1.twinx()
        ax2, fig = pt.plot_data(nodes, weights, fig=fig, ax=ax2,
                                lbl='QMOM',
                                clr='b',mrk='o')
        ax2, fig = pt.plot_data(nodes_G, weights_G, fig=fig, ax=ax2,
                                lbl='GQMOM',
                                clr='r',mrk='o')
        
        ax1.grid('minor')
        ax2.grid('minor')
        # ax3.grid('minor')
        plt.title('Comparison of nodes and weights')
        plt.tight_layout()
        plt.legend()
        plt.show()

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