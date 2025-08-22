# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:53:09 2024

@author: Administrator
"""

import os
from pathlib import Path
import runpy
import math
import numpy as np
from optframework.dpbe.dpbe_core import DPBECore
from optframework.dpbe.dpbe_visualization import DPBEVisual
from optframework.dpbe.dpbe_post import DPBEPost

        
class DPBESolver():
    """
    A discrete population balance equation (dPBE) solver for 1D and 2D particle systems.
    
    This class is responsible for initializing ,solving and post-processing population balance equations (PBE) 
    in 1D or 2D, depending on the specified dimensionality. It integrates core PBE functionality 
    with visualization, post-processing, and magnetic separation capabilities by dynamically 
    binding methods from external modules.
    
    Note
    ----
    This class uses the `bind_methods_from_module` function to dynamically bind methods from external
    modules. Some methods in this class are not explicitly defined here, but instead are imported from
    other files. To fully understand or modify those methods, please refer to the corresponding external
    file.
    
    Parameters
    ----------
    dim : int
        The dimensionality of the PBE problem (1 for 1D, 2 for 2D).
    t_total : int, optional
        The total process time in second. Defaults to 601.
    t_write : int, optional
        The frequency (per second) for writing output data. Defaults to 100.
    t_vec : array-like, optional
        A time vector directly specifying output time points for the simulation.
    load_attr : bool, optional
        If True, loads attributes from a configuration file. Defaults to True.
    config_path : str, optional
        The file path to the configuration file. If None, the default config path is used.
    disc : str, optional
        The discretization scheme to use for the PBE. Defaults to 'geo'.
    attr : dict, optional
        Additional attributes for PBE initialization.
    """
    def __init__(self, dim, t_total=601, t_write=100, t_vec=None, load_attr=True, config_path=None, disc='geo', **attr):
        # Check if given dimension and discretization is valid
        if not (dim in [1,2,3] and disc in ['geo','uni']):
            print('Given dimension and/or discretization are not valid. Exiting..')
            
        # BASELINE PATH
        # self.work_dir = os.path.dirname( __file__ )
        self.work_dir = Path(os.getcwd()).resolve()
        
        ## Simulation parameters
        self.dim = dim                        # Dimension (1=1D, 2=2D, 3=3D)
        self.disc = disc                      # 'geo': geometric grid, 'uni': uniform grid
        self.NS = 12                          # Grid parameter [-]
        self.S = 2                            # Geometric grid ratio (V_e[i] = S*V_e[i-1])      
        self.t_total = t_total                       # total simulation time [second]
        self.t_write = t_write
        self.t_vec = t_vec
        self.solve_algo = "ivp"                   # "ivp": use integrate.solve_ivp
                                              # "radau": use RK.radau_ii_a, only for debug, not recommended  
        
        ## Parameters in agglomeration kernels
        self.COLEVAL = 1                      # Case for calculation of beta. 1 = Orthokinetic, 2 = Perikinetic
        self.EFFEVAL = 2                      # Case for calculation of alpha. 1 = Full calculation, 2 = Reduced model (only based on primary particle interactions)
        self.SIZEEVAL = 2                     # Case for implementation of size dependency. 1 = No size dependency, 2 = Model from Soos2007 
        self.POTEVAL = 1                      # Case for the set of used interaction potentials. See int_fun_Xd for infos.
                                            # Case 2 massively faster and legit acc. to Kusters1997 and Bäbler2008
                                            # Case 3 to use pre-defines alphas (e.g. from ANN) --> alphas need to be provided at some point
        self.alpha_prim = np.ones(dim**2)
        self.CORR_BETA = 1e6*2.5e-5           # Correction Term for collision frequency [-]. Can be defined
                                            # dependent on rotary speed, e.g. ((corr_beta250-corr_beta100)*(n_exp-100)/(250-100)+corr_beta100)
        self.aggl_crit = 1e3                  # relative maximum aggregate volume(to primary particle) allowed to further agglomeration
        ## Parameters in breakage kernels
        self.BREAKRVAL = 3                    # Case for calculation breakage rate. 1 = constant, 2 = size dependent
        self.BREAKFVAL = 3                    # Case for calculation breakage function. 1 = conservation of Hypervolume, 2 = conservation of 0 Moments 
        self.process_type = "breakage"    # "agglomeration": only calculate agglomeration, "breakage": only calculate breakage, "mix": calculate both agglomeration and breakage
        self.pl_v = 2                         # number of fragments in product function of power law
                                              # or (v+1)/v: number of fragments in simple power law  
        self.pl_q = 1                         # parameter describes the breakage type(in product function of power law) 
        self.pl_P1 = 1e-2                     # 1. parameter in power law for breakage rate  1d/2d
        self.pl_P2 = 1                      # 2. parameter in power law for breakage rate  1d/2d
        self.pl_P3 = 1e-2                     # 3. parameter in power law for breakage rate  2d
        self.pl_P4 = 1                      # 4. parameter in power law for breakage rate  2d
        
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
        # self.DIST1_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        # self.DIST2_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        # self.DIST3_name = 'PSD_x50_1.0E-6_r01_2.9E-7.npy' 
        
        self.V_unit = 1                  # The unit volume used to calculate the total particle concentration. 
                                            # It is essentially a parameter used to scale the variabel.
                                            
        ## Parameters used to analytical calculate alpha_prim
        self.PSI1 = 1*1e-3                   # Surface potential component 1 [V] - NM1
        self.PSI2 = 1*1e-3                    # Surface potential component 2 [V] - NM2
        self.PSI3 = -40*1e-3                  # Surface potential component 3 [V] - M
        self.A_NM1NM1 = 10e-21                # Hamaker constant for interaction NM1-NM1 [J] 
        self.A_NM2NM2 = 10e-21                # Hamaker constant for interaction NM2-NM2 [J]
        self.A_MM = 80e-21                    # Hamaker constant for interaction M-M [J]
        # Definition of hydrophobic interaction parameters according to bi-exponential empiric 
        # equation from Christenson2001 [N/m]. Naming is as follows: C{i}_{j}{k}, where
        # i element of [1,2], 1 = Short ranged interaction, 2 = long ranged interaction
        # j element of [NM1, NM2, M] = Interaction partner 1
        # k element of [NM1, NM2, M] = Interaction partner 2
        # "Good" default values are C1_jk=5e-3, C2_ij=50e-3
        self.C1_NM1NM1 = 0    
        self.C2_NM1NM1 = 0    
        self.C1_MNM1 = 0      
        self.C2_MNM1 = 0      
        self.C1_MM = 0
        self.C2_MM = 0
        self.C1_NM2NM2 = 0
        self.C2_NM2NM2 = 0
        self.C1_MNM2 = 0
        self.C2_MNM2 = 0
        self.C1_NM1NM2 = 0
        self.C2_NM1NM2 = 0
        self.LAM1 = 1.2*10**-9                # Range of short ranged hydrophobic interations [m]
        self.LAM2 = 10*10**-9                 # Range of long ranged hydrophobic interations [m]
        self.X_CR = 2*10**-9                  # Alternative range criterion hydrophobic interactions 
        self.X_SEL = 0.310601                 # Size dependency parameter for Selomulya2003 / Soos2006 
        self.Y_SEL = 1.06168                  # Size dependency parameter for Selomulya2003 / Soos2006
            
        ## GENERAL constants
        self.KT = 1.38*(10**-23)*293          # k*T in [J]
        self.MU0 = 4*math.pi*10**-7           # Permeability constant vacuum [N/A²]
        self.EPS0 = 8.854*10**-12             # Permettivity constant vacuum [F/m]
        self.EPSR = 80                        # Permettivity material factor [-]
        self.E = 1.602*10**-19                # Electron charge [C]    
        self.NA = 6.022*10**23                # Avogadro number [1/mol]
        self.MU_W = 10**-3                    # Viscosity water [Pa*s]
        
        ## EXPERIMENTAL / PROCESS parameters:
        self.I = 1e-3*1e3                     # Ionic strength [mol/m³] - CARE: Standard unit is mol/L
        self.c_mag_exp = 0.01                 # Volume concentration of magnetic particles [Vol-%] 
        self.Psi_c1_exp = 1                   # Concentration ratio component 1 (V_NM1/V_M) [-] 
        self.Psi_c2_exp = 1                   # Concentration ratio component 2 (V_NM2/V_M) [-] 
        self.G = 1                            # Shear rate [1/s]. Can be defined dependent on rotary speed, 
                                            # e.g. G=(1400-354)*(n_exp-100)/(250-100)+354
        self.G_agg_corr = 1
        self.G_break_corr = 1                                    
        
        self.JIT_DN = True                    # Define wheter or not the DN calculation (timeloop) should be precompiled
        self.JIT_FM = True                    # Define wheter or not the FM calculation should be precompiled
        self.JIT_BF = True
        # Initialize **attr
        for key, value in attr.items():
            setattr(self, key, value)
            
            
        # Initialize PBE core, visualization, post-processing, and magnetic separation parameters
        # dpbe_core.init_pbe_params(self, dim, t_total, t_write, t_vec, disc, **attr)
        # dpbe_visualization.init_visual_params(self)
        # dpbe_post.init_post_params(self)
        # dpbe_mag_sep.init_mag_sep_params(self)
        self.core = DPBECore(self)
        self.post = DPBEPost(self)
        self.visualization = DPBEVisual(self)
        
        # Load the configuration file, if available
        if config_path is None and load_attr:
            config_path = os.path.join(self.work_dir,"config","PBE_config.py")

        if load_attr:
            self.load_attributes(config_path)
        self.check_params()
        self.reset_params()
        
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
        print(f"The dPBE is using config file at : {config_path}." )
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
        
        if getattr(self, "DIST1_name", None):
            self.DIST1 = os.path.join(self.DIST1_path,self.DIST1_name)
        if getattr(self, "DIST2_name", None):
            self.DIST2 = os.path.join(self.DIST2_path,self.DIST2_name)
        if getattr(self, "DIST3_name", None):
            self.DIST3 = os.path.join(self.DIST3_path,self.DIST3_name)  
        # Recalculate physical constants and particle concentrations
        self.EPS = self.EPSR*self.EPS0
        
        self.cv_1 = self.c_mag_exp*self.Psi_c1_exp   # Volume concentration of NM1 particles [Vol-%] 
        self.cv_2 = self.c_mag_exp*self.Psi_c2_exp   # Volume concentration of NM2 particles [Vol-%] 
        self.V01 = self.cv_1*self.V_unit             # Total volume concentration of component 1 [unit/unit] - NM1
        self.N01 = 3*self.V01/(4*math.pi*self.R01**3)     # Total number concentration of primary particles component 1 [1/m³] - NM1 (if no PSD)
        self.V02 = self.cv_2*self.V_unit         # Total volume concentration of component 2 [unit/unit] - NM2
        self.N02 = 3*self.V02/(4*math.pi*self.R02**3)     # Total number concentration of primary particles component 2 [1/m³] - NM2 (if no PSD)
        self.V03 = self.c_mag_exp*self.V_unit        # Total volume concentration of component 3 [unit/unit] - M
        self.N03 = 3*self.V03/(4*math.pi*self.R03**3)     # Total number concentration of primary particles component 1 [1/m³] - M (if no PSD) 
        
