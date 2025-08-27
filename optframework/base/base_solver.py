# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:52:38 2025

@author: px2030
"""
import os
import math
import runpy
import numpy as np
from pathlib import Path

class BaseSolver():
    def _init_base_parameters(self, dim, t_total=601, t_write=100, t_vec=None):
        # BASELINE PATH
        self.work_dir = Path(os.getcwd()).resolve()
        
        ## Simulation parameters
        self.dim = dim                        # Dimension (1=1D, 2=2D, 3=3D). The 3D case has not been fully adapted yet.
        self.t_vec = t_vec                      # Simulation time vector.
        self.t_total = t_total                       # total simulation time [second]
        self.t_write = t_write                      # Output time interval.
                                                # If t_vec is not specified, then use t_total and t_write to construct t_vec.

        ## Parameters in agglomeration kernels
        self.COLEVAL = 1                      # Case for calculation of beta (collision frequency). The currently applied kernel models are:
                                            # 1: Chin 1998 – shear-induced flocculation in stirred tanks
                                            # 2: Tsouris 1995 – Brownian, diffusion as the controlling mechanism
                                            # 3: Constant kernel
                                            # 4: Volume-sum kernel
        self.SIZEEVAL = 1                     # Case for implementation of size dependency for agglomeration kernels. 1 = No size dependency, 2 = Model from Soos2007 
        self.X_SEL = 0.310601                 # Size dependency parameter for Selomulya2003 / Soos2006 
        self.Y_SEL = 1.06168                  # Size dependency parameter for Selomulya2003 / Soos2006
        self.POTEVAL = 1                      # Case for the set of used interaction potentials in DLOV theory. See int_fun_Xd for infos.
                                            # Case 2 massively faster and legit acc. to Kusters1997 and Bäbler2008
                                            # Case 3 to use pre-defines alphas (e.g. from ANN) --> alphas need to be provided at some point
                                            
        self.alpha_prim = np.ones(dim**2)   # Collision efficiency
        self.CORR_BETA = 1           # Correction Term for collision frequency [-]. The meaning may vary across different kernel models. 
                                                # It describes the influence of external factors such as the environment and particle shape on the model.
                                                
        ## Parameters in breakage kernels
        self.BREAKRVAL = 3                    # Case for calculation breakage rate. The currently applied kernel models are:
                                            # 1: Constant kernel, 
                                            # 2: Volume-sum kernel
                                            # 3: Jeldres 2018 - Power Law Pandy and Spielmann
                                            # 4: Jeldres 2018 - Hypothetical formula considering volume fraction
        self.BREAKFVAL = 3                  # Case for calculation breakage function. The currently applied kernel models are:
                                            # 1: Leong2023 - Random breakage into four fragments, conservation of Hypervolume, 
                                            # 2: Leong2023 - Random breakage into two fragments, conservation of 0. Moments 
                                            # 3: Diemer Olson 2002 - Product function of power law
                                            # 4: Diemer Olson 2002 - Simple function of power law
                                            # 5: Diemer Olson 2002 - Parabolic function
        self.process_type = "breakage"    # "agglomeration": only calculate agglomeration, "breakage": only calculate breakage, "mix": calculate both agglomeration and breakage
        self.pl_v = 2                         # number of fragments in product function of power law
                                              # or (v+1)/v: number of fragments in simple power law  
        self.pl_q = 1                         # parameter describes the breakage type(in product function of power law) 
        self.pl_P1 = 1e10                     # 1. parameter in power law for breakage rate  1d/2d
        self.pl_P2 = 1                      # 2. parameter in power law for breakage rate  1d/2d
        self.pl_P3 = 1e10                     # 3. parameter in power law for breakage rate  2d
        self.pl_P4 = 1                      # 4. parameter in power law for breakage rate  2d
        
        self.B_F_type = 'int_func'            # 'int_func': calculate B_F with breakage function
                                              # 'MC_bond': Obtain B_F directly from the result of MC_bond (experimental)
                                              # 'ANN_MC': Calculate MC results using ANN model and convert to B_F (experimental)
        self.work_dir_MC_BOND = os.path.join(self.work_dir,'bond_break','int_B_F.npz')
        
        self.USE_PSD = True                   # Defines whether to use a .npy file to initialize the particle distribution. (False = monodisperse primary particles)
        # Set default initial PSD file paths
        self.DIST1_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST2_path = os.path.join(self.work_dir,'data','PSD_data')
        self.DIST3_path = os.path.join(self.work_dir,'data','PSD_data')
        # self.DIST1_name = 'your_PSD.npy'
        # self.DIST2_name = 'your_PSD.npy'
        # self.DIST3_name = 'your_PSD.npy' 
        
        ## Parameters used to analytical calculate alpha_prim
        self.PSI1 = 1*1e-3                   # Surface potential component 1 [V] - NM1 (Actual primary particle size R[1] = ((1+S)/2)**(1/3)*R01 in geo grid)
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
        
        ## EXPERIMENTAL / PROCESS parameters:
        self.c_mag_exp = 0.01                 # Volume concentration of magnetic particles [Vol-%] 
        self.Psi_c1_exp = 1                   # Concentration ratio component 1 (V_NM1/V_M) [-] 
        self.Psi_c2_exp = 1                   # Concentration ratio component 2 (V_NM2/V_M) [-] 
        self.G = 1                            # Shear rate [1/s].          
        self.V_unit = 1                       # Unit volume used to calculate the total particle concentration
        

    def _check_params(self):
        """
        Check the validity of dPBE parameters.
        """
        pass    

    def _load_attributes(self, config_path):
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
        print(f"The Solver is using config file at : {config_path}." )
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
        self._reset_params(reset_t=reset_t)    
        
    def _reset_params(self, reset_t=False):
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
            self.t_total = self.t_vec[-1]
        
        if getattr(self, "DIST1_name", None) is not None:
            self.DIST1 = os.path.join(self.DIST1_path,self.DIST1_name)
        if getattr(self, "DIST2_name", None) is not None:
            self.DIST2 = os.path.join(self.DIST2_path,self.DIST2_name)
        if getattr(self, "DIST3_name", None) is not None:
            self.DIST3 = os.path.join(self.DIST3_path,self.DIST3_name)  
        
        self.cv_1 = self.c_mag_exp*self.Psi_c1_exp   # Volume concentration of NM1 particles [Vol-%] 
        self.cv_2 = self.c_mag_exp*self.Psi_c2_exp   # Volume concentration of NM2 particles [Vol-%] 
        self.V01 = self.cv_1*self.V_unit             # Total volume concentration of component 1 [unit/unit] - NM1
        if getattr(self, "R01", None):
            self.N01 = 3*self.V01/(4*math.pi*self.R01**3)     # Total number concentration of primary particles component 1 [1/m³] - NM1 (if no PSD)
        self.V02 = self.cv_2*self.V_unit         # Total volume concentration of component 2 [unit/unit] - NM2
        if getattr(self, "R02", None):
            self.N02 = 3*self.V02/(4*math.pi*self.R02**3)     # Total number concentration of primary particles component 2 [1/m³] - NM2 (if no PSD)
        self.V03 = self.c_mag_exp*self.V_unit        # Total volume concentration of component 3 [unit/unit] - M
        if getattr(self, "R03", None):
            self.N03 = 3*self.V03/(4*math.pi*self.R03**3)     # Total number concentration of primary particles component 1 [1/m³] - M (if no PSD) 
        
        
        