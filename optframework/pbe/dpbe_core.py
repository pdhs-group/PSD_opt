""" Solving 1D, 2D and 3D discrete population balance equations for agglomerating systems. """

### ------ IMPORTS ------ ###
## General
import os
from pathlib import Path
import numpy as np
import math
import scipy.integrate as integrate
## jit function
from  optframework.utils.func import jit_rhs, jit_kernel_agg, jit_kernel_break
from  optframework.utils.func.static_method import interpolate_psd
## For math
from  optframework.utils.func import RK_Radau as RK

### ------ POPULATION CLASS DEFINITION ------ ###

def init_pbe_params(self, dim, t_total, t_write, t_vec, disc, **attr):
    """
    Initialize all necessary settings and physical parameters of dPBE-solver 
    
    Note
    ----
    The solver for 3D PBE problem is not yet implemented.
    
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
    **attr : dict, optional
        Additional attributes for PBE initialization.
    """
    
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
    self.solver = "ivp"                   # "ivp": use integrate.solve_ivp
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
    self.pl_v = 4                         # number of fragments in product function of power law
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
    
    self.JIT_DN = True                    # Define wheter or not the DN calculation (timeloop) should be precompiled
    self.JIT_FM = True                    # Define wheter or not the FM calculation should be precompiled
    self.JIT_BF = True
    # Initialize **attr
    for key, value in attr.items():
        setattr(self, key, value)
        
    self.reset_params()

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
    
def full_init(self, calc_alpha=True, init_N=True):
    """Fully initialize a population instance.
    
    This method calls     
        * ``pop.calc_R( )`` 
        * ``pop.init_N( )``   
        * ``pop.calc_alpha_prim( )`` (optional)
        * ``pop.calc_F_M( )`` 
        * ``pop.calc_B_M( )`` 
    
    Parameters
    ----------
    calc_alpha : `bool`, optional
        If ``True``, calculate collision efficiency values from provided material data.     
        If ``False``, use pop.alpha_prim (initialize beforehand!)
    """
    
    self.calc_R()
    # self.calc_R_new()
    if init_N: self.init_N() 
    
    if calc_alpha: self.calc_alpha_prim()
    self.calc_F_M()
    self.calc_B_R()
    self.calc_int_B_F()

## Calculate R, V and X matrices (radii, total and partial volumes and volume fractions)
def calc_R(self):
    """
    Initialize the discrete calculation grid for particle radii, volumes, and volume fractions.
    
    This method calculates and initializes the radii (R), total volumes (V), and material 
    volume fractions (Xi_vol) for each class of particles. The results are stored in the 
    class attributes `self.V`, `self.R`, and `self.Xi_vol`.
    
    - For the 1D case, this method generates a grid for total volumes and radii.
    - For the 2D case, this method generates a grid for combined volumes of two particle 
      types (V1 and V3) and calculates the corresponding radii and volume fractions.
    
    Parameters
    ----------
    None
    
    Notes
    -----
    - For uniform (`uni`) grids, the total volumes are directly proportional to the 
      particle class index.
    - For geometric (`geo`) grids, the volume edges (`V_e`) are calculated first using 
      parameters `NS` (number of grid points) and `S` (scaling factor). The actual grid 
      nodes (`V`) are then calculated as the midpoints between these volume edges.
    - Agglomeration criteria are also handled to prevent integration issues for large 
      agglomerate sizes by limiting the agglomeration process at critical points.
    """
    # 1-D case
    if self.dim == 1:
        
        # Initialize V and R
        self.V = np.zeros(self.NS)
        self.R = np.zeros(self.NS)
        
        # Uniform grid: volumes are proportional to the class index
        if self.disc == 'uni':
            for i in range(len(self.V)): #range(0,self.NS):
                self.V[i] = i*4*math.pi*self.R01**3/3
                
        # Geometric grid: volumes are calculated as midpoints between volume edges (V_e)    
        if self.disc == 'geo':
            self.V_e = np.zeros(self.NS+1)
            self.V_e[0] = -4*math.pi*self.R01**3/3
            for i in range(self.NS):         
                self.V_e[i+1] = self.S**(i)*4*math.pi*self.R01**3/3
                self.V[i] = (self.V_e[i] + self.V_e[i+1]) / 2  
                  
        # Calculate radii and initialize volume fraction matrices
        self.R[1:] = (self.V[1:]*3/(4*math.pi))**(1/3)
        self.X1_vol = np.ones(self.NS) 
        self.X1_a = np.ones(self.NS) 
        
        # Handle agglomeration criteria to limit agglomeration process
        aggl_crit_ids = self.aggl_crit + 1
        if (aggl_crit_ids > 0 and aggl_crit_ids < len(self.V)):
            self.aggl_crit_id = aggl_crit_ids 
        else: 
            self.aggl_crit_id = (len(self.V) -1)
                    
    # 2-D case
    elif self.dim == 2:
        
        # Initialize volumes: V1 (NM1) and V3 (M)
        self.V1 = np.zeros(self.NS)
        self.V3 = np.copy(self.V1) 
        
        # Uniform grid: volumes are proportional to the class index
        if self.disc == 'uni':
            for i in range(len(self.V1)): #range(0,self.NS):
                self.V1[i] = i*4*math.pi*self.R01**3/3
                self.V3[i] = i*4*math.pi*self.R03**3/3
                
        # Geometric grid: volumes are calculated as midpoints between volume edges (V_e1 and V_e3)
        if self.disc == 'geo': 
            self.V_e1 = np.zeros(self.NS+1)
            self.V_e3 = np.zeros(self.NS+1)
            self.V_e1[0] = -4*math.pi*self.R01**3/3
            self.V_e3[0] = -4*math.pi*self.R03**3/3
            for i in range(self.NS):
                self.V_e1[i+1] = self.S**(i)*4*math.pi*self.R01**3/3
                self.V_e3[i+1] = self.S**(i)*4*math.pi*self.R03**3/3
                self.V1[i] = (self.V_e1[i] + self.V_e1[i+1]) / 2
                self.V3[i] = (self.V_e3[i] + self.V_e3[i+1]) / 2

        A1 = 3*self.V1/self.R01
        A3 = 3*self.V3/self.R03
        
        # Calculate radii and initialize volume fraction matrices
        self.V = np.zeros((self.NS,self.NS))
        self.R = np.copy(self.V)
        self.X1_vol = np.copy(self.V); self.X1_a=np.copy(self.V) 
        self.X3_vol = np.copy(self.V); self.X3_a=np.copy(self.V)
        
        # Write V1 and V3 in respective "column" of V
        self.V[:,0] = self.V1 
        self.V[0,:] = self.V3 
        
        # Calculate the remaining entries of V, R, and other matrices
        for i in range(self.NS):
            for j in range(self.NS):
                self.V[i,j] = self.V1[i]+self.V3[j]
                self.R[i,j] = (self.V[i,j]*3/(4*math.pi))**(1/3)
                if i==0 and j==0:
                    self.X1_vol[i,j] = 0
                    self.X3_vol[i,j] = 0
                    self.X1_a[i,j] = 0
                    self.X3_a[i,j] = 0
                else:
                    self.X1_vol[i,j] = self.V1[i]/self.V[i,j]
                    self.X3_vol[i,j] = self.V3[j]/self.V[i,j]
                    self.X1_a[i,j] = A1[i]/(A1[i]+A3[j])
                    self.X3_a[i,j] = A3[j]/(A1[i]+A3[j])
                    
        # Set a limit on particle size for NM1 and M to avoid issues with large particle agglomeration
        aggl_crit_ids1 = self.aggl_crit + 1
        aggl_crit_ids2 = self.aggl_crit + 1
        self.aggl_crit_id = np.zeros(2, dtype=int)
        if (aggl_crit_ids1 > 0 and aggl_crit_ids1 < len(self.V1)):
            self.aggl_crit_id[0] = aggl_crit_ids1  
        else: 
            self.aggl_crit_id[0] = (len(self.V1) -1)
        if (aggl_crit_ids2 > 0 and aggl_crit_ids2 < len(self.V3)):
            self.aggl_crit_id[1] = aggl_crit_ids2 
        else: 
            self.aggl_crit_id[1] = (len(self.V3) -1)
    # # 3-D case                
    # elif self.dim == 3:
        
    #     # Initialize V1-V3 
    #     self.V1 = np.zeros(self.NS+3)-1 
    #     self.V1[1] = 0
    #     self.V2 = np.copy(self.V1)
    #     self.V3 = np.copy(self.V1) 
        
    #     for i in range(0,self.NS+1): 
    #         # Geometric grid
    #         if self.disc == 'geo': 
    #             self.V1[i+2] = self.S**(i)*4*math.pi*self.R01**3/3
    #             self.V2[i+2] = self.S**(i)*4*math.pi*self.R02**3/3
    #             self.V3[i+2] = self.S**(i)*4*math.pi*self.R03**3/3
            
    #         # Uniform grid
    #         elif self.disc == 'uni':
    #             self.V1[i+2] = (i+1)*4*math.pi*self.R01**3/3
    #             self.V2[i+2] = (i+1)*4*math.pi*self.R02**3/3
    #             self.V3[i+2] = (i+1)*4*math.pi*self.R03**3/3
        
    #     A1 = 3*self.V1/self.R01
    #     A2 = 3*self.V2/self.R02
    #     A3 = 3*self.V3/self.R03
        
    #     # Initialize V, R and ratio matrices
    #     self.V = np.zeros((self.NS+3,self.NS+3,self.NS+3))-1
    #     self.R = np.copy(self.V)
    #     self.X1_vol = np.copy(self.V); self.X1_a=np.copy(self.V) 
    #     self.X2_vol = np.copy(self.V); self.X2_a=np.copy(self.V)
    #     self.X3_vol = np.copy(self.V); self.X3_a=np.copy(self.V)
        
    #     # Write V1 and V3 in respective "column" of V
    #     self.V[:,1,1] = self.V1
    #     self.V[1,:,1] = self.V2 
    #     self.V[1,1,:] = self.V3 
        
    #     # Calculate remaining entries of V and other matrices
    #     # range(1,X) excludes X itself -> self.NS+3
    #     for i in range(1,self.NS+3):
    #         for j in range(1,self.NS+3):
    #             for k in range(1,self.NS+3):
    #                 self.V[i,j,k] = self.V1[i]+self.V2[j]+self.V3[k]
    #                 self.R[i,j,k] = (self.V[i,j,k]*3/(4*math.pi))**(1/3)
    #                 if i==1 and j==1 and k==1:
    #                     self.X1_vol[i,j,k] = 0
    #                     self.X2_vol[i,j,k] = 0
    #                     self.X3_vol[i,j,k] = 0
    #                     self.X1_a[i,j,k] = 0
    #                     self.X2_a[i,j,k] = 0
    #                     self.X3_a[i,j,k] = 0
    #                 else:
    #                     self.X1_vol[i,j,k] = self.V1[i]/self.V[i,j,k]
    #                     self.X2_vol[i,j,k] = self.V2[j]/self.V[i,j,k]
    #                     self.X3_vol[i,j,k] = self.V3[k]/self.V[i,j,k]
    #                     self.X1_a[i,j,k] = A1[i]/(A1[i]+A2[j]+A3[k])
    #                     self.X2_a[i,j,k] = A2[j]/(A1[i]+A2[j]+A3[k])
    #                     self.X3_a[i,j,k] = A3[k]/(A1[i]+A2[j]+A3[k])

## Initialize concentration matrix N
def init_N(self, reset_N=True, N01=None, N02=None, N03=None): 
    """Initialize discrete number concentration array. 
    
    Creates the following class attributes: 
        * ``pop.N``: Number concentration of each class 
    """
    if reset_N:
        ## Reset EXPERIMENTAL / PROCESS parameters
        self.cv_1 = self.c_mag_exp*self.Psi_c1_exp   # Volume concentration of NM1 particles [Vol-%] 
        self.cv_2 = self.c_mag_exp*self.Psi_c2_exp   # Volume concentration of NM2 particles [Vol-%] 
        self.V01 = self.cv_1*self.V_unit             # Total volume concentration of component 1 [unit/unit] - NM1
        self.V02 = self.cv_2*self.V_unit         # Total volume concentration of component 2 [unit/unit] - NM2
        self.V03 = self.c_mag_exp*self.V_unit        # Total volume concentration of component 3 [unit/unit] - M
        if N01 is None:
            self.N01 = 3*self.V01/(4*math.pi*self.R01**3)     # Total number concentration of primary particles component 1 [1/m³] - NM1 (if no PSD)
        else:
            self.N01 = N01 * self.V_unit 
        if N02 is None:
            self.N02 = 3*self.V02/(4*math.pi*self.R02**3)     # Total number concentration of primary particles component 2 [1/m³] - NM2 (if no PSD)
        else:
            self.N02 = N02 * self.V_unit 
        if N03 is None:
            self.N03 = 3*self.V03/(4*math.pi*self.R03**3)     # Total number concentration of primary particles component 1 [1/m³] - M (if no PSD) 
        else:
            self.N03 = N03 * self.V_unit 
            
    if self.t_vec is not None:
        self.t_num = len(self.t_vec) 
    # 1-D case
    if self.dim == 1:
        self.N = np.zeros((self.NS,self.t_num))
        if self.USE_PSD:
            self.N[1:,0] = interpolate_psd(2*self.R[1:],self.DIST1,self.V01)
        else:
            if self.process_type == "agglomeration":
                self.N[1,0] = self.N01
            elif self.process_type == "breakage":
                self.N[-1,0] = self.N01
            elif self.process_type == "mix":
                self.N[1,0] = self.N01
                # self.N[-1,0] = self.N01
            else:
                raise Exception("Current process_type not allowed!")
    
    # 2-D case
    elif self.dim == 2:
        self.N = np.zeros((self.NS,self.NS,self.t_num))
        if self.USE_PSD:
            self.N[1:,0,0] = interpolate_psd(2*self.R[1:,0],self.DIST1,self.V01)
            self.N[0,1:,0] = interpolate_psd(2*self.R[0,1:],self.DIST3,self.V03)
        else:
            if self.process_type == "agglomeration":
                self.N[1,0,0] = self.N01
                self.N[0,1,0] = self.N03
            elif self.process_type == "breakage":
                self.N[-1,-1,0] = self.N01
            elif self.process_type == "mix":
                self.N[1,0,0] = self.N01
                self.N[0,1,0] = self.N03  
                # self.N[-1,-1,0] = self.N01
            
    
    # 3-D case
    elif self.dim == 3:
        self.N = np.zeros((self.NS+3,self.NS+3,self.NS+3,self.t_num))
        if self.USE_PSD:
            self.N[2:-1,1,1,0] = interpolate_psd(2*self.R[2:-1,1,1],self.DIST1,self.V01)
            self.N[1,2:-1,1,0] = interpolate_psd(2*self.R[1,2:-1,1],self.DIST2,self.V02)
            self.N[1,1,2:-1,0] = interpolate_psd(2*self.R[1,1,2:-1],self.DIST3,self.V03)
        else:
            self.N[2,1,1,0] = self.N01
            self.N[1,2,1,0] = self.N02
            self.N[1,1,2,0] = self.N03

## Calculate agglomeration rate matrix.
## JIT_FM controls whether the pre-compiled function is used or not. 
def calc_F_M(self):
    """Initialize agglomeration frequency array. 
    
    Creates the following class attributes: 
        * ``pop.F_M``: (2D)Agglomeration frequency between two classes ij and ab is stored in ``F_M[i,j,a,b]`` 
    """
    # 1-D case
    if self.dim == 1:
        self.alpha_prim = self.alpha_prim.item() if isinstance(self.alpha_prim, np.ndarray) else self.alpha_prim
        # To avoid mass leakage at the boundary in CAT, boundary cells are not directly involved in the calculation. 
        # So there is no need to define the corresponding F_M at boundary. F_M is (NS-1)^2 instead (NS)^2
        self.F_M = np.zeros((self.NS-1,self.NS-1))
        if self.process_type == 'breakage':
            return  
        # Go through all agglomeration partners 1 [a] and 2 [i]
        # The current index tuple idx stores them as (a,i)
        for idx, tmp in np.ndenumerate(self.F_M):
            # Indices [a]=[0] and [i]=[0] not allowed!
            if idx[0]==0 or idx[1]==0:
                continue
            
            # Calculate the corresponding agglomeration efficiency
            # Add one to indices to account for borders
            a = idx[0] ; i = idx[1]
            
            # Calculate collision frequency beta depending on COLEVAL
            if self.COLEVAL == 1:
                # Chin 1998 (shear induced flocculation in stirred tanks)
                # Optional reduction factor.
                # corr_beta=1;
                beta_ai = self.CORR_BETA*self.G*2.3*(self.R[a]+self.R[i])**3 # [m^3/s]
            elif self.COLEVAL == 2:
                # Tsouris 1995 Brownian diffusion as controlling mechanism
                # Optional reduction factor
                # corr_beta=1;
                beta_ai = self.CORR_BETA*2*self.KT*(self.R[a]+self.R[i])**2/(3*self.MU_W*(self.R[a]*self.R[i])) #[m^3/s]
            elif self.COLEVAL == 3:
                # Use a constant collision frequency given by CORR_BETA
                beta_ai = self.CORR_BETA
            elif self.COLEVAL == 4:
                # Sum-Kernal (for validation) scaled by CORR_BETA
                beta_ai = self.CORR_BETA*4*math.pi*(self.R[a]**3+self.R[i]**3)/3
                            
            # Calculate collision effiecieny depending on EFFEVAL. 
            # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun_2d
            # Case(2): Reduced model. Calculation only based on primary particles
            # Case(3): Alphas are pre-fed from ANN or other source.
            if self.EFFEVAL == 1:
                # Not coded here
                alpha_ai = self.alpha_prim
            elif self.EFFEVAL == 2:
                alpha_ai = self.alpha_prim
            
            # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
            # Calculate lam
            if self.R[a]<=self.R[i]:
                lam = self.R[a]/self.R[i]
            else:
                lam = self.R[i]/self.R[a]
                
            if self.SIZEEVAL == 1:
                # No size dependency of alpha
                corr_size = 1
            if self.SIZEEVAL == 2:
                # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
                # with model parameters x and y. corr_size is lowered with lowered
                # value of lambda (numerator) and with increasing particles size (denominator)
                corr_size = np.exp(-self.X_SEL*(1-lam)**2)/((self.R[a]*self.R[i]/(self.R01**2))**self.Y_SEL)
            
            # Store result
            # self.alpha[idx] = alpha_ai
            # self.beta[idx] = beta_ai
            self.F_M[idx] = beta_ai*alpha_ai*corr_size/self.V_unit
            
    # 2-D case.
    elif self.dim == 2:
        # To avoid mass leakage at the boundary in CAT, boundary cells are not directly involved in the calculation. 
        # So there is no need to define the corresponding F_M at boundary. F_M is (NS-1)^4 instead (NS)^4
        # calc_beta = jit_kernel_agg.prepare_calc_beta(self.COLEVAL)
        self.F_M = np.zeros((self.NS-1,self.NS-1,self.NS-1,self.NS-1))
        if self.process_type == 'breakage':
            return
        if self.JIT_FM:
            self.F_M = jit_kernel_agg.calc_F_M_2D(self.NS,self.disc,self.COLEVAL,self.CORR_BETA,
                                       self.G,self.R,self.X1_vol,self.X3_vol,
                                       self.EFFEVAL,self.alpha_prim,self.SIZEEVAL,
                                       self.X_SEL,self.Y_SEL)/self.V_unit
        
        else:
            # Go through all agglomeration partners 1 [a,b] and 2 [i,j]
            # The current index tuple idx stores them as (a,b,i,j)
            for idx, tmp in np.ndenumerate(self.F_M):
                # # Indices [a,b]=[0,0] and [i,j]=[0,0] not allowed!
                if idx[0]+idx[1]==0 or idx[2]+idx[3]==0:
                    continue
                
                # Calculate the corresponding agglomeration efficiency
                # Add one to indices to account for borders
                a = idx[0]; b = idx[1]; i = idx[2]; j = idx[3]
                
                # Calculate collision frequency beta depending on COLEVAL
                if self.COLEVAL == 1:
                    # Chin 1998 (shear induced flocculation in stirred tanks)
                    # Optional reduction factor.
                    # corr_beta=1;
                    beta_ai = self.CORR_BETA*self.G*2.3*(self.R[a,b]+self.R[i,j])**3 # [m^3/s]
                if self.COLEVAL == 2:
                    # Tsouris 1995 Brownian diffusion as controlling mechanism
                    # Optional reduction factor
                    # corr_beta=1;
                    beta_ai = self.CORR_BETA*2*self.KT*(self.R[a,b]+self.R[i,j])**2/(3*self.MU_W*(self.R[a,b]*self.R[i,j])) #[m^3/s]
                if self.COLEVAL == 3:
                    # Use a constant collision frequency given by CORR_BETA
                    beta_ai = self.CORR_BETA
                if self.COLEVAL == 4:
                    # Sum-Kernal (for validation) scaled by CORR_BETA
                    beta_ai = self.CORR_BETA*4*math.pi*(self.R[a,b]**3+self.R[i,j]**3)/3
                
                # Calculate probabilities, that particle 1 [a,b] is colliding as
                # nonmagnetic 1 (NM1) or magnetic (M). Repeat for
                # particle 2 [i,j]. Use area weighted composition.
                # Calculate probability vector for all combinations. 
                # Indices: 
                # 1) a:N1 <-> i:N1  -> X1[a,b]*X1[i,j]
                # 2) a:N1 <-> i:M   -> X1[a,b]*X3[i,j]
                # 3) a:M  <-> i:N1  -> X3[a,b]*X1[i,j]
                # 4) a:M  <-> i:M   -> X3[a,b]*X3[i,j]
                # Use volume fraction (May be changed to surface fraction)
                X1 = self.X1_vol; X3 = self.X3_vol
                
                p=np.array([X1[a,b]*X1[i,j],\
                            X1[a,b]*X3[i,j],\
                            X3[a,b]*X1[i,j],\
                            X3[a,b]*X3[i,j]])
                
                # Calculate collision effiecieny depending on EFFEVAL. 
                # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun_2d
                # Case(2): Reduced model. Calculation only based on primary particles
                # Case(3): Alphas are pre-fed from ANN or other source.
                if self.EFFEVAL == 1:
                    # Not coded here
                    alpha_ai = np.sum(p*self.alpha_prim)
                if self.EFFEVAL == 2:
                    alpha_ai = np.sum(p*self.alpha_prim)
                
                # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
                # Calculate lam
                if self.R[a,b]<=self.R[i,j]:
                    lam = self.R[a,b]/self.R[i,j]
                else:
                    lam = self.R[i,j]/self.R[a,b]
                    
                if self.SIZEEVAL == 1:
                    # No size dependency of alpha
                    corr_size = 1
                if self.SIZEEVAL == 2:
                    # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
                    # with model parameters x and y. corr_size is lowered with lowered
                    # value of lambda (numerator) and with increasing particles size (denominator)
                    corr_size = np.exp(-self.X_SEL*(1-lam)**2)/((self.R[a,b]*self.R[i,j]/(np.min(np.array([self.R01,self.R03]))**2))**self.Y_SEL)
                
                # Store result
                self.F_M[idx] = beta_ai*alpha_ai*corr_size/self.V_unit
            
    # 3-D case. 
    # elif self.dim == 3:
    #     if self.process_type == 'breakage':
    #         return
    #     if self.JIT_FM: 
    #         self.F_M = jit.calc_F_M_3D(self.NS,self.disc,self.COLEVAL,self.CORR_BETA,
    #                                    self.G,self.R,self.X1_vol,self.X2_vol,self.X3_vol,
    #                                    self.EFFEVAL,self.alpha_prim,self.SIZEEVAL,
    #                                    self.X_SEL,self.Y_SEL)/self.V_unit
        
    #     else:
    #         # Initialize F_M Matrix. NOTE: F_M is defined without the border around the calculation grid
    #         # as e.g. N or V are (saving memory and calculations). 
    #         # Thus, F_M is (NS+1)^6 instead of (NS+3)^6. As reference, V is (NS+3)^3.
    #         self.F_M = np.zeros((self.NS+1,self.NS+1,self.NS+1,self.NS+1,self.NS+1,self.NS+1))
            
    #         # Go through all agglomeration partners 1 [a,b] and 2 [i,j]
    #         # The current index tuple idx stores them as (a,b,i,j)
    #         for idx, tmp in np.ndenumerate(self.F_M):
    #             # # Indices [a,b,c]=[0,0,0] and [i,j,k]=[0,0,0] not allowed!
    #             if idx[0]+idx[1]+idx[2]==0 or idx[3]+idx[4]+idx[5]==0:
    #                 continue
                
    #             # Calculate the corresponding agglomeration efficiency
    #             # Add one to indices to account for borders
    #             a = idx[0]+1; b = idx[1]+1; c = idx[2]+1;
    #             i = idx[3]+1; j = idx[4]+1; k = idx[5]+1;
                
    #             # Calculate collision frequency beta depending on COLEVAL
    #             if self.COLEVAL == 1:
    #                 # Chin 1998 (shear induced flocculation in stirred tanks)
    #                 # Optional reduction factor.
    #                 # corr_beta=1;
    #                 beta_ai = self.CORR_BETA*self.G*2.3*(self.R[a,b,c]+self.R[i,j,k])**3 # [m^3/s]
    #             if self.COLEVAL == 2:
    #                 # Tsouris 1995 Brownian diffusion as controlling mechanism
    #                 # Optional reduction factor
    #                 # corr_beta=1;
    #                 beta_ai = self.CORR_BETA*2*self.KT*(self.R[a,b,c]+self.R[i,j,k])**2/(3*self.MU_W*(self.R[a,b,c]*self.R[i,j,k])) #[m^3/s]
    #             if self.COLEVAL == 3:
    #                 # Use a constant collision frequency given by CORR_BETA
    #                 beta_ai = self.CORR_BETA
    #             if self.COLEVAL == 4:
    #                 # Sum-Kernal (for validation) scaled by CORR_BETA
    #                 beta_ai = self.CORR_BETA*4*math.pi*(self.R[a,b,c]**3+self.R[i,j,k]**3)/3
                
    #             # Calculate probabilities, that particle 1 [a,b,c] is colliding as
    #             # nonmagnetic 1 (NM1), nonmagnetic 2 (NM2) or magnetic (M). Repeat for
    #             # particle 2 [i,j,k]. Use area weighted composition.
    #             # Calculate probability vector for all combinations. 
    #             # Indices: 
    #             # 1) a:N1 <-> i:N1  -> X1[a,b,c]*X1[i,j,k]
    #             # 2) a:N1 <-> i:N2  -> X1[a,b,c]*X2[i,j,k]
    #             # 3) a:N1 <-> i:M   -> X1[a,b,c]*X3[i,j,k]
    #             # 4) a:N2 <-> i:N1  -> X2[a,b,c]*X1[i,j,k] 
    #             # 5) a:N2 <-> i:N2  -> X2[a,b,c]*X2[i,j,k]
    #             # 6) a:N2 <-> i:M   -> X2[a,b,c]*X3[i,j,k]
    #             # 7) a:M  <-> i:N1  -> X3[a,b,c]*X1[i,j,k]
    #             # 8) a:M  <-> i:N2  -> X3[a,b,c]*X2[i,j,k]
    #             # 9) a:M  <-> i:M   -> X3[a,b,c]*X3[i,j,k]
    #             # Use volume fraction (May be changed to surface fraction)
    #             X1 = self.X1_vol; X2 = self.X2_vol; X3 = self.X3_vol
                
    #             p=np.array([X1[a,b,c]*X1[i,j,k],\
    #                         X1[a,b,c]*X2[i,j,k],\
    #                         X1[a,b,c]*X3[i,j,k],\
    #                         X2[a,b,c]*X1[i,j,k],\
    #                         X2[a,b,c]*X2[i,j,k],\
    #                         X2[a,b,c]*X3[i,j,k],\
    #                         X3[a,b,c]*X1[i,j,k],\
    #                         X3[a,b,c]*X2[i,j,k],\
    #                         X3[a,b,c]*X3[i,j,k]])
                
    #             # Calculate collision effiecieny depending on EFFEVAL. 
    #             # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun
    #             # Case(2): Reduced model. Calculation only based on primary particles
    #             # Case(3): Alphas are pre-fed from ANN or other source.
    #             if self.EFFEVAL == 1:
    #                 # Not coded here
    #                 alpha_ai = np.sum(p*self.alpha_prim)
    #             if self.EFFEVAL == 2:
    #                 alpha_ai = np.sum(p*self.alpha_prim)
                
    #             # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
    #             # Calculate lam
    #             if self.R[a,b,c]<=self.R[i,j,k]:
    #                 lam = self.R[a,b,c]/self.R[i,j,k]
    #             else:
    #                 lam = self.R[i,j,k]/self.R[a,b,c]
                    
    #             if self.SIZEEVAL == 1:
    #                 # No size dependency of alpha
    #                 corr_size = 1
    #             if self.SIZEEVAL == 2:
    #                 # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
    #                 # with model parameters x and y. corr_size is lowered with lowered
    #                 # value of lambda (numerator) and with increasing particles size (denominator)
    #                 corr_size = np.exp(-self.X_SEL*(1-lam)**2)/((self.R[a,b,c]*self.R[i,j,k]/(np.min(np.array([self.R01,self.R02,self.R03]))**2))**self.Y_SEL)
                
    #             # Store result
    #             self.F_M[idx] = beta_ai*alpha_ai*corr_size/self.V_unit

## Calculate breakage rate matrix. 
def calc_B_R(self):
    """Initialize breakage rate array. 
    
    Creates the following class attributes: 
        * ``pop.B_R``: (2D)Breakage rate for class ab. The result is stored in ``B_R[a,b]`` 
    """
    self.B_R = np.zeros_like(self.V)
    # 1-D case
    if self.dim == 1:
        ## Note: The breakage rate of the smallest particle is 0. 
        ## Note: Because particles with a volume of zero are skipped, 
        ##       calculation with V requires (index+1)
        if self.process_type == 'agglomeration':
            return
        self.B_R = jit_kernel_break.breakage_rate_1d(self.V, self.V1_mean, self.G, self.pl_P1, self.pl_P2, self.BREAKRVAL)          
    # 2-D case            
    if self.dim == 2:
        if self.process_type == 'agglomeration':
            return
        self.B_R = jit_kernel_break.breakage_rate_2d(self.V, self.V1, self.V3, self.V1_mean, self.V3_mean, self.G, self.pl_P1, self.pl_P2, self.pl_P3, self.pl_P4, self.BREAKRVAL, self.BREAKFVAL)
                    
## Calculate integrated breakage function matrix.         
def calc_int_B_F(self):
    """Initialize integrated breakage function array. 
    
    Creates the following class attributes: 
        * ``pop.int_B_F``: (2D)The integral of the breakage function from class ab to class ij. Result is stored in ``int_B_F[a,b,i,j]`` 
        * ``pop.intx_B_F``: (2D)The integral of the (breakage function*x) from class ab to class ij. Result is stored in ``intx_B_F[a,b,i,j]`` 
        * ``pop.inty_B_F``: (2D)The integral of the (breakage function*y) from class ab to class ij. Result is stored in ``inty_B_F[a,b,i,j]`` 
    """
    if self.BREAKFVAL == 4:
        if self.pl_v <= 0 or self.pl_v > 1:
            raise Exception("Value of pl_v is out of range (0,1] for simple Power law.")
    # 1-D case
    if self.dim == 1:
        if self.disc == 'uni':
            self.B_F = np.zeros((self.NS,self.NS))
            V = np.copy(self.V)
            V[:-1] = self.V[1:]
            V[-1] = self.V[-1] + self.V[1]
            for idx, tep in np.ndenumerate(self.B_F):
                a = idx[0]; i = idx[1]
                if i != 0 and a <= i:
                    self.B_F[idx] = jit_kernel_break.breakage_func_1d(V[a],V[i],self.pl_v,self.pl_q,self.BREAKFVAL) * V[0]
        else:
            self.int_B_F = np.zeros((self.NS, self.NS))
            self.intx_B_F = np.zeros((self.NS, self.NS))
            if self.process_type == 'agglomeration':
                return
            
            if self.B_F_type == 'MC_bond':
                mc_bond = np.load(self.work_dir_MC_BOND, allow_pickle=True)
                self.int_B_F = mc_bond['int_B_F']
                self.intx_B_F = mc_bond['intx_B_F']
            elif self.B_F_type == 'int_func':
                ## Let the integration range associated with the breakage function start from zero 
                ## to ensure mass conservation  
                V_e_tem = np.copy(self.V_e)
                V_e_tem[0] = 0.0
                for idx, tmp in np.ndenumerate(self.int_B_F):
                    a = idx[0]; i = idx[1]
                    if i != 0 and a <= i:
                        args = (self.V[i],self.pl_v,self.pl_q,self.BREAKFVAL)
                        argsk = (self.V[i],self.pl_v,self.pl_q,self.BREAKFVAL,1)
                        if a == i:
                            self.int_B_F[idx],err = integrate.quad(jit_kernel_break.breakage_func_1d,V_e_tem[a],self.V[a],args=args)
                            self.intx_B_F[idx],err = integrate.quad(jit_kernel_break.breakage_func_1d_xk,V_e_tem[a],self.V[a],args=argsk)
                        else:
                            self.int_B_F[idx],err = integrate.quad(jit_kernel_break.breakage_func_1d,V_e_tem[a],V_e_tem[a+1],args=args)
                            self.intx_B_F[idx],err = integrate.quad(jit_kernel_break.breakage_func_1d_xk,V_e_tem[a],V_e_tem[a+1],args=argsk)
                
    # 2-D case
    elif self.dim == 2:
        self.int_B_F = np.zeros((self.NS, self.NS, self.NS, self.NS))
        self.intx_B_F = np.zeros((self.NS, self.NS, self.NS, self.NS))
        self.inty_B_F = np.zeros((self.NS, self.NS, self.NS, self.NS)) 
        if self.process_type == 'agglomeration':
            return
        
        if self.B_F_type == 'MC_bond':
            mc_bond = np.load(self.work_dir_MC_BOND, allow_pickle=True)
            self.int_B_F = mc_bond['int_B_F']
            self.intx_B_F = mc_bond['intx_B_F']
            self.inty_B_F = mc_bond['inty_B_F']

        elif self.B_F_type == 'int_func':
            if self.JIT_BF:
                self.int_B_F, self.intx_B_F, self.inty_B_F = jit_kernel_break.calc_int_B_F_2D_GL(
                    self.NS,self.V1,self.V3,self.V_e1,self.V_e3,self.BREAKFVAL,self.pl_v,self.pl_q)
            else:
                self.int_B_F, self.intx_B_F,self.inty_B_F = jit_kernel_break.calc_int_B_F_2D_quad(
                    self.NS, self.V1, self.V3, self.V_e1, self.V_e3, self.BREAKFVAL, self.pl_v, self.pl_q)
        elif self.B_F_type == 'ANN_MC': 
            return
        
## Calculate alphas of primary particles
def calc_alpha_prim(self):
    """Calculate collision efficiency between primary particles based on material data."""
    
    # Use reduced model if EFFEVAL==2. Only primary agglomeration efficiencies are calculated. 
    # Due to numerical issues it may occur that the integral is 0, thus dividing by zero
    # This appears to be the case in fully destabilized systems --> set the integral to 1
    # See 3-D case or int_fun for definition of comb_flag order

    # Define integration range
    maxint = np.inf
    minint = 2
    
    # 1-D case
    if self.dim == 1:
        
        # NM1 NM1
        tmp=integrate.quad(lambda s: self.int_fun(s,2,2,comb_flag=0), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim = (2*tmp)**(-1)
        
        
    # 2-D case
    elif self.dim == 2:

        self.alpha_prim = np.zeros(4)
        
        tmp=integrate.quad(lambda s: self.int_fun(s,2,2,1,1,comb_flag=0), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[0] = (2*tmp)**(-1)
        
        # NM1 M
        tmp=integrate.quad(lambda s: self.int_fun(s,2,1,1,2,comb_flag=2), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[1] = (2*tmp)**(-1)
        
        # M NM1
        self.alpha_prim[2] = self.alpha_prim[1]
        
        # M M
        tmp=integrate.quad(lambda s: self.int_fun(s,1,1,2,2,comb_flag=8), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[3] = (2*tmp)**(-1)
        
    # 3-D case
    elif self.dim == 3:

        self.alpha_prim = np.zeros(9)
        
        # NM1 NM1 (0)
        tmp=integrate.quad(lambda s: self.int_fun(s,2,2,1,1,1,1,comb_flag=0), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[0] = (2*tmp)**(-1)
        
        # NM1 NM2 (1)
        tmp=integrate.quad(lambda s: self.int_fun(s,2,1,1,2,1,1,comb_flag=1), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[1] = (2*tmp)**(-1)
        
        # NM1 M (2)
        tmp=integrate.quad(lambda s: self.int_fun(s,2,1,1,1,1,2,comb_flag=2), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[2] = (2*tmp)**(-1)
        
        # NM2 NM1 (3)
        self.alpha_prim[3] = self.alpha_prim[1]
        
        # NM2 NM2 (4)
        tmp=integrate.quad(lambda s: self.int_fun(s,1,1,2,2,1,1,comb_flag=4), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[4] = (2*tmp)**(-1)
        
        # NM2 M (5)
        tmp=integrate.quad(lambda s: self.int_fun(s,1,1,2,1,1,2,comb_flag=5), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[5] = (2*tmp)**(-1)
        
        # M NM1 (6)
        self.alpha_prim[6] = self.alpha_prim[2]
        
        # M NM2 (7)
        self.alpha_prim[7] = self.alpha_prim[5]
        
        # M M (8)
        tmp=integrate.quad(lambda s: self.int_fun(s,1,1,1,1,2,2,comb_flag=8), minint, maxint)[0]
        if tmp<1: tmp = 1
        self.alpha_prim[8] = (2*tmp)**(-1)    
## Integral function of interaction potential
def int_fun(self,s,a,i,b=None,j=None,c=None,k=None,comb_flag=0):
    
    # Generate corresponding psi, A, c1 and c2 values for given case.
    # Hamaker combination equation from Butt(2018) equation 3.86
    # comb_flag order same as in alpha_prim: NM1 NM1 (0) | NM1 NM2 (1) | 
    # NM1 M (2) | NM2 NM1 (3)=(1) | NM2 NM2 (4) | NM2 M (5) | 
    # M NM1 (6)=(2) | M NM2 (7)=(5) | M M (8)
    if comb_flag==0:     # a:NM1 i:NM1
        psi_a = self.PSI1
        psi_i = self.PSI1
        c1 = self.C1_NM1NM1
        c2 = self.C2_NM1NM1
        A = self.A_NM1NM1
    if comb_flag==1 or comb_flag==3:     # a:NM1 i:NM2
        psi_a = self.PSI1
        psi_i = self.PSI2
        c1 = self.C1_NM1NM2
        c2 = self.C2_NM1NM2
        A = np.sqrt(self.A_NM1NM1*self.A_NM2NM2)
    if comb_flag==2 or comb_flag==6:     # a:NM1 i:M
        psi_a = self.PSI1
        psi_i = self.PSI3
        c1 = self.C1_MNM1
        c2 = self.C2_MNM1
        A = np.sqrt(self.A_NM1NM1*self.A_MM)
    if comb_flag==4:     # a:NM2 i:NM2
        psi_a = self.PSI2
        psi_i = self.PSI2
        c1 = self.C1_NM2NM2
        c2 = self.C2_NM2NM2
        A = self.A_NM2NM2
    if comb_flag==5 or comb_flag==7:     # a:NM2 i:M
        psi_a = self.PSI2
        psi_i = self.PSI3
        c1 = self.C1_MNM2
        c2 = self.C2_MNM2
        A = np.sqrt(self.A_NM2NM2*self.A_MM)
    if comb_flag==8:     # a:M i:M
        psi_a = self.PSI3
        psi_i = self.PSI3
        c1 = self.C1_MM
        c2 = self.C2_MM
        A = self.A_MM
    
    gam1 = np.tanh(self.E*psi_a/(self.KT*4))
    gam2 = np.tanh(self.E*psi_i/(self.KT*4))
    kappa = np.sqrt((2*self.I*self.NA*self.E**2)/(self.EPS*self.KT))
    
    if self.dim == 1:
        r1 = self.R[a]
        r2 = self.R[i]
    elif self.dim == 2:
        r1 = self.R[a,b]
        r2 = self.R[i,j]
    elif self.dim == 3:
        r1 = self.R[a,b,c]
        r2 = self.R[i,j,k]
        
    a_dist = (r1+r2)/2
    h = (s*a_dist-2*a_dist)

    if self.POTEVAL == 1:
        # Electric potential Gregory 1975 (Elimelech book Particle deposition 
        # and aggregation), vdW potential sphere - sphere 3.23, hydrophobic 
        # interaction / polar potential Christenson 2001 (bi-exponential model)
        # NOTE: Potentials are already divided by kT (For electrostatic
        # potential missing factor kT in comparison to formula)
        Psi_el = 128*np.pi*(r1*r2)*self.I*self.NA*gam1*gam2*np.exp(-kappa*(s*a_dist-2*a_dist))/(kappa**2*(r1+r2))
        Psi_vdw = -A*((2*r1*r2/(h**2+2*r1*h+2*r2*h))+(2*r1*r2/(h**2+2*r1*h+2*r2*h+4*r1*r2))+np.log((h**2+2*r1*h+2*r2*h)/(h**2+2*r1*h+2*r2*h+4*r1*r2)))/(6*self.KT)
        a_sqr = np.sqrt(r1*r2)
        Psi_pol = -a_sqr*(c1*self.LAM1*np.exp(-h/self.LAM1)+c2*self.LAM2*np.exp(-h/self.LAM2))/(self.KT)
        y = np.exp(Psi_el+Psi_vdw+Psi_pol)/(s**2)
        # Prevent potential overflow from ruining the results
        # When np.exp overflows, the result is inf
        if y==np.inf:
            y = 1e90

    # Elimelech page 174 aka. Honig(1971) correction factor
    if s!=2: bet = (6*(s-2)**2+13*(s-2)+2)/(6*(s-2)**2+4*(s-2))
    else: bet = 1e10
    
    y = y*bet
        
    return y

## Solve ODE with Scipy methods:
def solve_PBE(self, t_vec=None):
    """Method for solving the (previously initialized) population with scipy.solve_ivp.
    
    Parameters
    ----------
    t_max : `float`, optional
        Final agglomeration time in seconds.
    t_vec : `array_like`, optional
        Points in time (in seconds) at which to export the numerical solution.
    """
    if t_vec is None:
        t_vec = self.t_vec
        t_max = self.t_vec[-1]
    else:
        t_max = t_vec[-1]

    N = self.N
    # 1-D case
    if self.dim == 1:
        # Define right-hand-side function depending on discretization
        if self.disc == 'geo':
            rhs = jit_rhs.get_dNdt_1d_geo
            args=(self.NS,self.V,self.V_e,self.F_M,self.B_R,self.int_B_F,
                  self.intx_B_F,self.process_type,self.aggl_crit_id)
        elif self.disc == 'uni':
            rhs = jit_rhs.get_dNdt_1d_uni                
            args=(self.V,self.B_R,self.B_F,self.F_M,self.NS,self.aggl_crit_id,self.process_type)
        if self.solver == "ivp":    
            with np.errstate(divide='raise', over='raise',invalid='raise'):
                try:
                    self.RES = integrate.solve_ivp(rhs, 
                                                    [0, t_max], 
                                                    N[:,0], t_eval=t_vec,
                                                    args=args,
                                                    ## If `rtol` is set too small, it may cause the results to diverge, 
                                                    ## leading to the termination of the calculation.
                                                    method='Radau',first_step=0.1,rtol=1e-1)
                    
                    # Reshape and save result to N and t_vec
                    t_vec = self.RES.t
                    y_evaluated = self.RES.y
                    status = True if self.RES.status == 0 else False
                except (FloatingPointError, ValueError) as e:
                    print(f"Exception encountered: {e}")
                    y_evaluated = -np.ones((self.NS,len(t_vec)))
                    status = False
                
        elif self.solver == "radau":
            ode_sys = RK.radau_ii_a(rhs, N[:,0], t_eval=t_vec,
                                    args = args,
                                    dt_first=0.1)
            y_evaluated, y_res_tem, t_res_tem, rate_res_tem, error_res_tem = ode_sys.solve_ode()
            status = not ode_sys.dt_is_too_small  

    elif self.dim == 2:
        # Define right-hand-side function depending on discretization
        if self.disc == 'geo':
            rhs = jit_rhs.get_dNdt_2d_geo
            args=(self.NS,self.V,self.V_e1,self.V_e3,self.F_M,self.B_R,self.int_B_F,
                  self.intx_B_F,self.inty_B_F,self.process_type,self.aggl_crit_id)
        elif self.disc == 'uni':
            rhs = jit_rhs.get_dNdt_2d_uni   
            args=(self.V,self.V1,self.V3,self.F_M,self.NS,self.THR_DN)
        if self.solver == "ivp":  
            with np.errstate(divide='raise', over='raise',invalid='raise'):
                try:
                    self.RES = integrate.solve_ivp(rhs, 
                                                    [0, t_max], 
                                                    np.reshape(N[:,:,0],-1), t_eval=t_vec,
                                                    args=args,
                                                    ## If `rtol` is set too small, it may cause the results to diverge, 
                                                    ## leading to the termination of the calculation.
                                                    method='Radau',first_step=0.1,rtol=1e-1)
                    
                    # Reshape and save result to N and t_vec
                    t_vec = self.RES.t
                    y_evaluated = self.RES.y.reshape((self.NS,self.NS,len(t_vec)))
                    status = True if self.RES.status == 0 else False
                except (FloatingPointError, ValueError) as e:
                    print(f"Exception encountered: {e}")
                    y_evaluated = -np.ones((self.NS,self.NS,len(t_vec)))
                    status = False
            
        elif self.solver == "radau":
            ode_sys = RK.radau_ii_a(rhs, np.reshape(N[:,:,0],-1), t_eval=t_vec,
                                    args = args,
                                    dt_first=0.1)
            y_evaluated, y_res_tem, t_res_tem, rate_res_tem, error_res_tem = ode_sys.solve_ode()
            y_evaluated = y_evaluated.reshape((self.NS,self.NS,len(t_vec)))
            y_res_tem = y_res_tem.reshape((self.NS,self.NS,len(t_res_tem)))
            status = not ode_sys.dt_is_too_small

    elif self.dim == 3:
        # Define right-hand-side function depending on discretization
        if self.disc == 'geo':
            rhs = jit_rhs.get_dNdt_3d_geo
        elif self.disc == 'uni':
            rhs = jit_rhs.get_dNdt_3d_uni   
            
        self.RES = integrate.solve_ivp(rhs, 
                                       [0, t_max], 
                                       np.reshape(self.N[:,:,:,0],-1), t_eval=t_vec,
                                       args=(self.V,self.V1,self.V2,self.V3,self.F_M,self.NS,self.THR_DN),
                                       ## If `rtol` is set too small, it may cause the results to diverge, 
                                       ## leading to the termination of the calculation.
                                       method='Radau',first_step=0.1,rtol=1e-1)
        
        # Reshape and save result to N and t_vec
        self.N = self.RES.y.reshape((self.NS+3,self.NS+3,self.NS+3,len(self.RES.t)))
        self.t_vec = self.RES.t
    # Monitor whether integration are completed  
    self.t_vec = t_vec 
    # self.N = y_evaluated / eva_N_scale
    self.N = y_evaluated
    self.calc_status = status   
    if not self.calc_status:
        print('Warning: The integral failed to converge!')
    if self.solver == "radau":
        # self.N_res_tem = y_res_tem / res_N_scale
        self.N_res_tem = y_res_tem
        self.t_res_tem = t_res_tem
        self.rate_res_tem = rate_res_tem
        self.error_res_tem = error_res_tem

