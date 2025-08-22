""" Solving 1D, 2D and 3D discrete population balance equations for agglomerating systems. """

### ------ IMPORTS ------ ###
## General
import os
from pathlib import Path
import numpy as np
import math
import scipy.integrate as integrate
## jit function
from  optframework.utils.func import jit_dpbe_rhs, jit_kernel_agg, jit_kernel_break
from  optframework.utils.func.static_method import interpolate_psd
## For math
from  optframework.utils.func import RK_Radau as RK

### ------ POPULATION CLASS DEFINITION ------ ###
class DPBECore:
    def __init__(self, base):
        self.base = base
    
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
        base = self.base
        # 1-D case
        if base.dim == 1:
            
            # Initialize V and R
            base.V = np.zeros(base.NS)
            base.R = np.zeros(base.NS)
            
            # Uniform grid: volumes are proportional to the class index
            if base.disc == 'uni':
                for i in range(len(base.V)): #range(0,base.NS):
                    base.V[i] = i*4*math.pi*base.R01**3/3
                    
            # Geometric grid: volumes are calculated as midpoints between volume edges (V_e)    
            if base.disc == 'geo':
                base.V_e = np.zeros(base.NS+1)
                base.V_e[0] = -4*math.pi*base.R01**3/3
                for i in range(base.NS):         
                    base.V_e[i+1] = base.S**(i)*4*math.pi*base.R01**3/3
                    base.V[i] = (base.V_e[i] + base.V_e[i+1]) / 2  
                      
            # Calculate radii and initialize volume fraction matrices
            base.R[1:] = (base.V[1:]*3/(4*math.pi))**(1/3)
            base.X1_vol = np.ones(base.NS) 
            base.X1_a = np.ones(base.NS) 
            
            # Handle agglomeration criteria to limit agglomeration process
            aggl_crit_ids = base.aggl_crit + 1
            if (aggl_crit_ids > 0 and aggl_crit_ids < len(base.V)):
                base.aggl_crit_id = aggl_crit_ids 
            else: 
                base.aggl_crit_id = (len(base.V) -1)
                        
        # 2-D case
        elif base.dim == 2:
            
            # Initialize volumes: V1 (NM1) and V3 (M)
            base.V1 = np.zeros(base.NS)
            base.V3 = np.copy(base.V1) 
            
            # Uniform grid: volumes are proportional to the class index
            if base.disc == 'uni':
                for i in range(len(base.V1)): #range(0,base.NS):
                    base.V1[i] = i*4*math.pi*base.R01**3/3
                    base.V3[i] = i*4*math.pi*base.R03**3/3
                    
            # Geometric grid: volumes are calculated as midpoints between volume edges (V_e1 and V_e3)
            if base.disc == 'geo': 
                base.V_e1 = np.zeros(base.NS+1)
                base.V_e3 = np.zeros(base.NS+1)
                base.V_e1[0] = -4*math.pi*base.R01**3/3
                base.V_e3[0] = -4*math.pi*base.R03**3/3
                for i in range(base.NS):
                    base.V_e1[i+1] = base.S**(i)*4*math.pi*base.R01**3/3
                    base.V_e3[i+1] = base.S**(i)*4*math.pi*base.R03**3/3
                    base.V1[i] = (base.V_e1[i] + base.V_e1[i+1]) / 2
                    base.V3[i] = (base.V_e3[i] + base.V_e3[i+1]) / 2
    
            A1 = 3*base.V1/base.R01
            A3 = 3*base.V3/base.R03
            
            # Calculate radii and initialize volume fraction matrices
            base.V = np.zeros((base.NS,base.NS))
            base.R = np.copy(base.V)
            base.X1_vol = np.copy(base.V); base.X1_a=np.copy(base.V) 
            base.X3_vol = np.copy(base.V); base.X3_a=np.copy(base.V)
            
            # Write V1 and V3 in respective "column" of V
            base.V[:,0] = base.V1 
            base.V[0,:] = base.V3 
            
            # Calculate the remaining entries of V, R, and other matrices
            for i in range(base.NS):
                for j in range(base.NS):
                    base.V[i,j] = base.V1[i]+base.V3[j]
                    base.R[i,j] = (base.V[i,j]*3/(4*math.pi))**(1/3)
                    if i==0 and j==0:
                        base.X1_vol[i,j] = 0
                        base.X3_vol[i,j] = 0
                        base.X1_a[i,j] = 0
                        base.X3_a[i,j] = 0
                    else:
                        base.X1_vol[i,j] = base.V1[i]/base.V[i,j]
                        base.X3_vol[i,j] = base.V3[j]/base.V[i,j]
                        base.X1_a[i,j] = A1[i]/(A1[i]+A3[j])
                        base.X3_a[i,j] = A3[j]/(A1[i]+A3[j])
                        
            # Set a limit on particle size for NM1 and M to avoid issues with large particle agglomeration
            aggl_crit_ids1 = base.aggl_crit + 1
            aggl_crit_ids2 = base.aggl_crit + 1
            base.aggl_crit_id = np.zeros(2, dtype=int)
            if (aggl_crit_ids1 > 0 and aggl_crit_ids1 < len(base.V1)):
                base.aggl_crit_id[0] = aggl_crit_ids1  
            else: 
                base.aggl_crit_id[0] = (len(base.V1) -1)
            if (aggl_crit_ids2 > 0 and aggl_crit_ids2 < len(base.V3)):
                base.aggl_crit_id[1] = aggl_crit_ids2 
            else: 
                base.aggl_crit_id[1] = (len(base.V3) -1)
        # # 3-D case                
        # elif base.dim == 3:
            
        #     # Initialize V1-V3 
        #     base.V1 = np.zeros(base.NS+3)-1 
        #     base.V1[1] = 0
        #     base.V2 = np.copy(base.V1)
        #     base.V3 = np.copy(base.V1) 
            
        #     for i in range(0,base.NS+1): 
        #         # Geometric grid
        #         if base.disc == 'geo': 
        #             base.V1[i+2] = base.S**(i)*4*math.pi*base.R01**3/3
        #             base.V2[i+2] = base.S**(i)*4*math.pi*base.R02**3/3
        #             base.V3[i+2] = base.S**(i)*4*math.pi*base.R03**3/3
                
        #         # Uniform grid
        #         elif base.disc == 'uni':
        #             base.V1[i+2] = (i+1)*4*math.pi*base.R01**3/3
        #             base.V2[i+2] = (i+1)*4*math.pi*base.R02**3/3
        #             base.V3[i+2] = (i+1)*4*math.pi*base.R03**3/3
            
        #     A1 = 3*base.V1/base.R01
        #     A2 = 3*base.V2/base.R02
        #     A3 = 3*base.V3/base.R03
            
        #     # Initialize V, R and ratio matrices
        #     base.V = np.zeros((base.NS+3,base.NS+3,base.NS+3))-1
        #     base.R = np.copy(base.V)
        #     base.X1_vol = np.copy(base.V); base.X1_a=np.copy(base.V) 
        #     base.X2_vol = np.copy(base.V); base.X2_a=np.copy(base.V)
        #     base.X3_vol = np.copy(base.V); base.X3_a=np.copy(base.V)
            
        #     # Write V1 and V3 in respective "column" of V
        #     base.V[:,1,1] = base.V1
        #     base.V[1,:,1] = base.V2 
        #     base.V[1,1,:] = base.V3 
            
        #     # Calculate remaining entries of V and other matrices
        #     # range(1,X) excludes X itbase -> base.NS+3
        #     for i in range(1,base.NS+3):
        #         for j in range(1,base.NS+3):
        #             for k in range(1,base.NS+3):
        #                 base.V[i,j,k] = base.V1[i]+base.V2[j]+base.V3[k]
        #                 base.R[i,j,k] = (base.V[i,j,k]*3/(4*math.pi))**(1/3)
        #                 if i==1 and j==1 and k==1:
        #                     base.X1_vol[i,j,k] = 0
        #                     base.X2_vol[i,j,k] = 0
        #                     base.X3_vol[i,j,k] = 0
        #                     base.X1_a[i,j,k] = 0
        #                     base.X2_a[i,j,k] = 0
        #                     base.X3_a[i,j,k] = 0
        #                 else:
        #                     base.X1_vol[i,j,k] = base.V1[i]/base.V[i,j,k]
        #                     base.X2_vol[i,j,k] = base.V2[j]/base.V[i,j,k]
        #                     base.X3_vol[i,j,k] = base.V3[k]/base.V[i,j,k]
        #                     base.X1_a[i,j,k] = A1[i]/(A1[i]+A2[j]+A3[k])
        #                     base.X2_a[i,j,k] = A2[j]/(A1[i]+A2[j]+A3[k])
        #                     base.X3_a[i,j,k] = A3[k]/(A1[i]+A2[j]+A3[k])
    
    ## Initialize concentration matrix N
    def init_N(self, reset_N=True, reset_path=True, N01=None, N02=None, N03=None): 
        """Initialize discrete number concentration array. 
        
        Creates the following class attributes: 
            * ``pop.N``: Number concentration of each class 
        """
        base = self.base
        if reset_N:
            ## Reset EXPERIMENTAL / PROCESS parameters
            base.cv_1 = base.c_mag_exp*base.Psi_c1_exp   # Volume concentration of NM1 particles [Vol-%] 
            base.cv_2 = base.c_mag_exp*base.Psi_c2_exp   # Volume concentration of NM2 particles [Vol-%] 
            base.V01 = base.cv_1*base.V_unit             # Total volume concentration of component 1 [unit/unit] - NM1
            base.V02 = base.cv_2*base.V_unit         # Total volume concentration of component 2 [unit/unit] - NM2
            base.V03 = base.c_mag_exp*base.V_unit        # Total volume concentration of component 3 [unit/unit] - M
            if N01 is None:
                base.N01 = 3*base.V01/(4*math.pi*base.R01**3)     # Total number concentration of primary particles component 1 [1/m³] - NM1 (if no PSD)
            else:
                base.N01 = N01 * base.V_unit 
            if N02 is None:
                base.N02 = 3*base.V02/(4*math.pi*base.R02**3)     # Total number concentration of primary particles component 2 [1/m³] - NM2 (if no PSD)
            else:
                base.N02 = N02 * base.V_unit 
            if N03 is None:
                base.N03 = 3*base.V03/(4*math.pi*base.R03**3)     # Total number concentration of primary particles component 1 [1/m³] - M (if no PSD) 
            else:
                base.N03 = N03 * base.V_unit 
        if reset_path and getattr(base, "DIST1_name", None):
            base.DIST1 = os.path.join(base.DIST1_path,base.DIST1_name)
        if reset_path and getattr(base, "DIST2_name", None):
            base.DIST2 = os.path.join(base.DIST2_path,base.DIST2_name)
        if reset_path and getattr(base, "DIST3_name", None):
            base.DIST3 = os.path.join(base.DIST3_path,base.DIST3_name)  
            
        if base.t_vec is not None:
            base.t_num = len(base.t_vec) 
        # 1-D case
        if base.dim == 1:
            base.N = np.zeros((base.NS,base.t_num))
            if base.USE_PSD:
                base.N[1:,0] = interpolate_psd(2*base.R[1:],base.DIST1,base.V01)
            else:
                if base.process_type == "agglomeration":
                    base.N[1,0] = base.N01
                elif base.process_type == "breakage":
                    base.N[-1,0] = base.N01
                elif base.process_type == "mix":
                    mid = 1 + (base.NS - 1) // 2
                    base.N[mid, 0] = base.N01
                else:
                    raise Exception("Current process_type not allowed!")
        
        # 2-D case
        elif base.dim == 2:
            base.N = np.zeros((base.NS,base.NS,base.t_num))
            if base.USE_PSD:
                base.N[1:,0,0] = interpolate_psd(2*base.R[1:,0],base.DIST1,base.V01)
                base.N[0,1:,0] = interpolate_psd(2*base.R[0,1:],base.DIST3,base.V03)
            else:
                if base.process_type == "agglomeration":
                    base.N[1,0,0] = base.N01
                    base.N[0,1,0] = base.N03
                elif base.process_type == "breakage":
                    base.N[-1,-1,0] = base.N01
                elif base.process_type == "mix":
                    mid = 1 + (base.NS - 1) // 2
                    base.N[mid, mid, 0] = base.N01
                
        
        # 3-D case
        elif base.dim == 3:
            base.N = np.zeros((base.NS+3,base.NS+3,base.NS+3,base.t_num))
            if base.USE_PSD:
                base.N[2:-1,1,1,0] = interpolate_psd(2*base.R[2:-1,1,1],base.DIST1,base.V01)
                base.N[1,2:-1,1,0] = interpolate_psd(2*base.R[1,2:-1,1],base.DIST2,base.V02)
                base.N[1,1,2:-1,0] = interpolate_psd(2*base.R[1,1,2:-1],base.DIST3,base.V03)
            else:
                base.N[2,1,1,0] = base.N01
                base.N[1,2,1,0] = base.N02
                base.N[1,1,2,0] = base.N03
    
    ## Calculate agglomeration rate matrix.
    ## JIT_FM controls whether the pre-compiled function is used or not. 
    def calc_F_M(self):
        """Initialize agglomeration frequency array. 
        
        Creates the following class attributes: 
            * ``pop.F_M``: (2D)Agglomeration frequency between two classes ij and ab is stored in ``F_M[i,j,a,b]`` 
        """
        base = self.base
        # 1-D case
        if base.dim == 1:
            base.alpha_prim = base.alpha_prim.item() if isinstance(base.alpha_prim, np.ndarray) else base.alpha_prim
            # To avoid mass leakage at the boundary in CAT, boundary cells are not directly involved in the calculation. 
            # So there is no need to define the corresponding F_M at boundary. F_M is (NS-1)^2 instead (NS)^2
            base.F_M = np.zeros((base.NS-1,base.NS-1))
            if base.process_type == 'breakage':
                return  
            if base.JIT_FM:
                jit_kernel_agg.calc_F_M_1D(base)
                base.F_M /= base.V_unit
            else:
                # Go through all agglomeration partners 1 [a] and 2 [i]
                # The current index tuple idx stores them as (a,i)
                for idx, tmp in np.ndenumerate(base.F_M):
                    # Indices [a]=[0] and [i]=[0] not allowed!
                    if idx[0]==0 or idx[1]==0:
                        continue
                    
                    # Calculate the corresponding agglomeration efficiency
                    # Add one to indices to account for borders
                    a = idx[0] ; i = idx[1]
                    
                    # Calculate collision frequency beta depending on COLEVAL
                    if base.COLEVAL == 1:
                        # Chin 1998 (shear induced flocculation in stirred tanks)
                        # Optional reduction factor.
                        # corr_beta=1;
                        beta_ai = base.CORR_BETA*base.G*2.3*(base.R[a]+base.R[i])**3 # [m^3/s]
                    elif base.COLEVAL == 2:
                        # Tsouris 1995 Brownian diffusion as controlling mechanism
                        # Optional reduction factor
                        # corr_beta=1;
                        beta_ai = base.CORR_BETA*2*base.KT*(base.R[a]+base.R[i])**2/(3*base.MU_W*(base.R[a]*base.R[i])) #[m^3/s]
                    elif base.COLEVAL == 3:
                        # Use a constant collision frequency given by CORR_BETA
                        beta_ai = base.CORR_BETA
                    elif base.COLEVAL == 4:
                        # Sum-Kernal (for validation) scaled by CORR_BETA
                        beta_ai = base.CORR_BETA*4*math.pi*(base.R[a]**3+base.R[i]**3)/3
                                    
                    # Calculate collision effiecieny depending on EFFEVAL. 
                    # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun_2d
                    # Case(2): Reduced model. Calculation only based on primary particles
                    # Case(3): Alphas are pre-fed from ANN or other source.
                    if base.EFFEVAL == 1:
                        # Not coded here
                        alpha_ai = base.alpha_prim
                    elif base.EFFEVAL == 2:
                        alpha_ai = base.alpha_prim
                    
                    # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
                    # Calculate lam
                    if base.R[a]<=base.R[i]:
                        lam = base.R[a]/base.R[i]
                    else:
                        lam = base.R[i]/base.R[a]
                        
                    if base.SIZEEVAL == 1:
                        # No size dependency of alpha
                        corr_size = 1
                    if base.SIZEEVAL == 2:
                        # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
                        # with model parameters x and y. corr_size is lowered with lowered
                        # value of lambda (numerator) and with increasing particles size (denominator)
                        corr_size = np.exp(-base.X_SEL*(1-lam)**2)/((base.R[a]*base.R[i]/(base.R01**2))**base.Y_SEL)
                    
                    # Store result
                    # base.alpha[idx] = alpha_ai
                    # base.beta[idx] = beta_ai
                    base.F_M[idx] = beta_ai*alpha_ai*corr_size/base.V_unit
                
        # 2-D case.
        elif base.dim == 2:
            # To avoid mass leakage at the boundary in CAT, boundary cells are not directly involved in the calculation. 
            # So there is no need to define the corresponding F_M at boundary. F_M is (NS-1)^4 instead (NS)^4
            # calc_beta = jit_kernel_agg.prepare_calc_beta(base.COLEVAL)
            base.F_M = np.zeros((base.NS-1,base.NS-1,base.NS-1,base.NS-1))
            if base.process_type == 'breakage':
                return
            if base.JIT_FM:
                jit_kernel_agg.calc_F_M_2D(base)
                base.F_M /= base.V_unit
            
            else:
                # Go through all agglomeration partners 1 [a,b] and 2 [i,j]
                # The current index tuple idx stores them as (a,b,i,j)
                for idx, tmp in np.ndenumerate(base.F_M):
                    # # Indices [a,b]=[0,0] and [i,j]=[0,0] not allowed!
                    if idx[0]+idx[1]==0 or idx[2]+idx[3]==0:
                        continue
                    
                    # Calculate the corresponding agglomeration efficiency
                    # Add one to indices to account for borders
                    a = idx[0]; b = idx[1]; i = idx[2]; j = idx[3]
                    
                    # Calculate collision frequency beta depending on COLEVAL
                    if base.COLEVAL == 1:
                        # Chin 1998 (shear induced flocculation in stirred tanks)
                        # Optional reduction factor.
                        # corr_beta=1;
                        beta_ai = base.CORR_BETA*base.G*2.3*(base.R[a,b]+base.R[i,j])**3 # [m^3/s]
                    if base.COLEVAL == 2:
                        # Tsouris 1995 Brownian diffusion as controlling mechanism
                        # Optional reduction factor
                        # corr_beta=1;
                        beta_ai = base.CORR_BETA*2*base.KT*(base.R[a,b]+base.R[i,j])**2/(3*base.MU_W*(base.R[a,b]*base.R[i,j])) #[m^3/s]
                    if base.COLEVAL == 3:
                        # Use a constant collision frequency given by CORR_BETA
                        beta_ai = base.CORR_BETA
                    if base.COLEVAL == 4:
                        # Sum-Kernal (for validation) scaled by CORR_BETA
                        beta_ai = base.CORR_BETA*4*math.pi*(base.R[a,b]**3+base.R[i,j]**3)/3
                    
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
                    X1 = base.X1_vol; X3 = base.X3_vol
                    
                    p=np.array([X1[a,b]*X1[i,j],\
                                X1[a,b]*X3[i,j],\
                                X3[a,b]*X1[i,j],\
                                X3[a,b]*X3[i,j]])
                    
                    # Calculate collision effiecieny depending on EFFEVAL. 
                    # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun_2d
                    # Case(2): Reduced model. Calculation only based on primary particles
                    # Case(3): Alphas are pre-fed from ANN or other source.
                    if base.EFFEVAL == 1:
                        # Not coded here
                        alpha_ai = np.sum(p*base.alpha_prim)
                    if base.EFFEVAL == 2:
                        alpha_ai = np.sum(p*base.alpha_prim)
                    
                    # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
                    # Calculate lam
                    if base.R[a,b]<=base.R[i,j]:
                        lam = base.R[a,b]/base.R[i,j]
                    else:
                        lam = base.R[i,j]/base.R[a,b]
                        
                    if base.SIZEEVAL == 1:
                        # No size dependency of alpha
                        corr_size = 1
                    if base.SIZEEVAL == 2:
                        # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
                        # with model parameters x and y. corr_size is lowered with lowered
                        # value of lambda (numerator) and with increasing particles size (denominator)
                        corr_size = np.exp(-base.X_SEL*(1-lam)**2)/((base.R[a,b]*base.R[i,j]/(np.min(np.array([base.R01,base.R03]))**2))**base.Y_SEL)
                    
                    # Store result
                    base.F_M[idx] = beta_ai*alpha_ai*corr_size/base.V_unit
                
        # 3-D case. 
        # elif base.dim == 3:
        #     if base.process_type == 'breakage':
        #         return
        #     if base.JIT_FM: 
        #         base.F_M = jit.calc_F_M_3D(base.NS,base.disc,base.COLEVAL,base.CORR_BETA,
        #                                    base.G,base.R,base.X1_vol,base.X2_vol,base.X3_vol,
        #                                    base.EFFEVAL,base.alpha_prim,base.SIZEEVAL,
        #                                    base.X_SEL,base.Y_SEL)/base.V_unit
            
        #     else:
        #         # Initialize F_M Matrix. NOTE: F_M is defined without the border around the calculation grid
        #         # as e.g. N or V are (saving memory and calculations). 
        #         # Thus, F_M is (NS+1)^6 instead of (NS+3)^6. As reference, V is (NS+3)^3.
        #         base.F_M = np.zeros((base.NS+1,base.NS+1,base.NS+1,base.NS+1,base.NS+1,base.NS+1))
                
        #         # Go through all agglomeration partners 1 [a,b] and 2 [i,j]
        #         # The current index tuple idx stores them as (a,b,i,j)
        #         for idx, tmp in np.ndenumerate(base.F_M):
        #             # # Indices [a,b,c]=[0,0,0] and [i,j,k]=[0,0,0] not allowed!
        #             if idx[0]+idx[1]+idx[2]==0 or idx[3]+idx[4]+idx[5]==0:
        #                 continue
                    
        #             # Calculate the corresponding agglomeration efficiency
        #             # Add one to indices to account for borders
        #             a = idx[0]+1; b = idx[1]+1; c = idx[2]+1;
        #             i = idx[3]+1; j = idx[4]+1; k = idx[5]+1;
                    
        #             # Calculate collision frequency beta depending on COLEVAL
        #             if base.COLEVAL == 1:
        #                 # Chin 1998 (shear induced flocculation in stirred tanks)
        #                 # Optional reduction factor.
        #                 # corr_beta=1;
        #                 beta_ai = base.CORR_BETA*base.G*2.3*(base.R[a,b,c]+base.R[i,j,k])**3 # [m^3/s]
        #             if base.COLEVAL == 2:
        #                 # Tsouris 1995 Brownian diffusion as controlling mechanism
        #                 # Optional reduction factor
        #                 # corr_beta=1;
        #                 beta_ai = base.CORR_BETA*2*base.KT*(base.R[a,b,c]+base.R[i,j,k])**2/(3*base.MU_W*(base.R[a,b,c]*base.R[i,j,k])) #[m^3/s]
        #             if base.COLEVAL == 3:
        #                 # Use a constant collision frequency given by CORR_BETA
        #                 beta_ai = base.CORR_BETA
        #             if base.COLEVAL == 4:
        #                 # Sum-Kernal (for validation) scaled by CORR_BETA
        #                 beta_ai = base.CORR_BETA*4*math.pi*(base.R[a,b,c]**3+base.R[i,j,k]**3)/3
                    
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
        #             X1 = base.X1_vol; X2 = base.X2_vol; X3 = base.X3_vol
                    
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
        #             if base.EFFEVAL == 1:
        #                 # Not coded here
        #                 alpha_ai = np.sum(p*base.alpha_prim)
        #             if base.EFFEVAL == 2:
        #                 alpha_ai = np.sum(p*base.alpha_prim)
                    
        #             # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
        #             # Calculate lam
        #             if base.R[a,b,c]<=base.R[i,j,k]:
        #                 lam = base.R[a,b,c]/base.R[i,j,k]
        #             else:
        #                 lam = base.R[i,j,k]/base.R[a,b,c]
                        
        #             if base.SIZEEVAL == 1:
        #                 # No size dependency of alpha
        #                 corr_size = 1
        #             if base.SIZEEVAL == 2:
        #                 # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
        #                 # with model parameters x and y. corr_size is lowered with lowered
        #                 # value of lambda (numerator) and with increasing particles size (denominator)
        #                 corr_size = np.exp(-base.X_SEL*(1-lam)**2)/((base.R[a,b,c]*base.R[i,j,k]/(np.min(np.array([base.R01,base.R02,base.R03]))**2))**base.Y_SEL)
                    
        #             # Store result
        #             base.F_M[idx] = beta_ai*alpha_ai*corr_size/base.V_unit
    
    ## Calculate breakage rate matrix. 
    def calc_B_R(self):
        """Initialize breakage rate array. 
        
        Creates the following class attributes: 
            * ``pop.B_R``: (2D)Breakage rate for class ab. The result is stored in ``B_R[a,b]`` 
        """
        base = self.base
        base.B_R = np.zeros_like(base.V)
        # 1-D case
        if base.dim == 1:
            ## Note: The breakage rate of the smallest particle is 0. 
            ## Note: Because particles with a volume of zero are skipped, 
            ##       calculation with V requires (index+1)
            if base.process_type == 'agglomeration':
                return
            jit_kernel_break.breakage_rate_1d(base)          
        # 2-D case            
        if base.dim == 2:
            if base.process_type == 'agglomeration':
                return
            jit_kernel_break.breakage_rate_2d(base)
                        
    ## Calculate integrated breakage function matrix.         
    def calc_int_B_F(self):
        """Initialize integrated breakage function array. 
        
        Creates the following class attributes: 
            * ``pop.int_B_F``: (2D)The integral of the breakage function from class ab to class ij. Result is stored in ``int_B_F[a,b,i,j]`` 
            * ``pop.intx_B_F``: (2D)The integral of the (breakage function*x) from class ab to class ij. Result is stored in ``intx_B_F[a,b,i,j]`` 
            * ``pop.inty_B_F``: (2D)The integral of the (breakage function*y) from class ab to class ij. Result is stored in ``inty_B_F[a,b,i,j]`` 
        """
        base = self.base
        if base.BREAKFVAL == 4:
            if base.pl_v <= 0 or base.pl_v > 1:
                raise Exception("Value of pl_v is out of range (0,1] for simple Power law.")
        # 1-D case
        if base.dim == 1:
            if base.disc == 'uni':
                base.B_F = np.zeros((base.NS,base.NS))
                V = np.copy(base.V)
                V[:-1] = base.V[1:]
                V[-1] = base.V[-1] + base.V[1]
                for idx, tep in np.ndenumerate(base.B_F):
                    a = idx[0]; i = idx[1]
                    if i != 0 and a <= i:
                        base.B_F[idx] = jit_kernel_break.breakage_func_1d(V[a],V[i],base.pl_v,base.pl_q,base.BREAKFVAL) * V[0]
            else:
                base.int_B_F = np.zeros((base.NS, base.NS))
                base.intx_B_F = np.zeros((base.NS, base.NS))
                if base.process_type == 'agglomeration':
                    return
                
                if base.B_F_type == 'MC_bond':
                    mc_bond = np.load(base.work_dir_MC_BOND, allow_pickle=True)
                    base.int_B_F = mc_bond['int_B_F']
                    base.intx_B_F = mc_bond['intx_B_F']
                elif base.B_F_type == 'int_func':
                    jit_kernel_break.calc_init_B_F_1D_quad(base)
                    
        # 2-D case
        elif base.dim == 2:
            base.int_B_F = np.zeros((base.NS, base.NS, base.NS, base.NS))
            base.intx_B_F = np.zeros((base.NS, base.NS, base.NS, base.NS))
            base.inty_B_F = np.zeros((base.NS, base.NS, base.NS, base.NS)) 
            if base.process_type == 'agglomeration':
                return
            
            if base.B_F_type == 'MC_bond':
                mc_bond = np.load(base.work_dir_MC_BOND, allow_pickle=True)
                base.int_B_F = mc_bond['int_B_F']
                base.intx_B_F = mc_bond['intx_B_F']
                base.inty_B_F = mc_bond['inty_B_F']
    
            elif base.B_F_type == 'int_func':
                if base.JIT_BF:
                    jit_kernel_break.calc_int_B_F_2D_GL(base)
                else:
                    jit_kernel_break.calc_int_B_F_2D_quad(base)
            elif base.B_F_type == 'ANN_MC': 
                return
            
    ## Calculate alphas of primary particles
    def calc_alpha_prim(self):
        """Calculate collision efficiency between primary particles based on material data."""
        
        base = self.base
        # Use reduced model if EFFEVAL==2. Only primary agglomeration efficiencies are calculated. 
        # Due to numerical issues it may occur that the integral is 0, thus dividing by zero
        # This appears to be the case in fully destabilized systems --> set the integral to 1
        # See 3-D case or int_fun for definition of comb_flag order
    
        # Define integration range
        maxint = np.inf
        minint = 2
        
        # 1-D case
        if base.dim == 1:
            
            # NM1 NM1
            tmp=integrate.quad(lambda s: self.int_fun(s,2,2,comb_flag=0), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim = (2*tmp)**(-1)
            
            
        # 2-D case
        elif base.dim == 2:
    
            base.alpha_prim = np.zeros(4)
            
            tmp=integrate.quad(lambda s: self.int_fun(s,2,2,1,1,comb_flag=0), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[0] = (2*tmp)**(-1)
            
            # NM1 M
            tmp=integrate.quad(lambda s: self.int_fun(s,2,1,1,2,comb_flag=2), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[1] = (2*tmp)**(-1)
            
            # M NM1
            base.alpha_prim[2] = base.alpha_prim[1]
            
            # M M
            tmp=integrate.quad(lambda s: self.int_fun(s,1,1,2,2,comb_flag=8), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[3] = (2*tmp)**(-1)
            
        # 3-D case
        elif base.dim == 3:
    
            base.alpha_prim = np.zeros(9)
            
            # NM1 NM1 (0)
            tmp=integrate.quad(lambda s: self.int_fun(s,2,2,1,1,1,1,comb_flag=0), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[0] = (2*tmp)**(-1)
            
            # NM1 NM2 (1)
            tmp=integrate.quad(lambda s: self.int_fun(s,2,1,1,2,1,1,comb_flag=1), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[1] = (2*tmp)**(-1)
            
            # NM1 M (2)
            tmp=integrate.quad(lambda s: self.int_fun(s,2,1,1,1,1,2,comb_flag=2), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[2] = (2*tmp)**(-1)
            
            # NM2 NM1 (3)
            base.alpha_prim[3] = base.alpha_prim[1]
            
            # NM2 NM2 (4)
            tmp=integrate.quad(lambda s: self.int_fun(s,1,1,2,2,1,1,comb_flag=4), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[4] = (2*tmp)**(-1)
            
            # NM2 M (5)
            tmp=integrate.quad(lambda s: self.int_fun(s,1,1,2,1,1,2,comb_flag=5), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[5] = (2*tmp)**(-1)
            
            # M NM1 (6)
            base.alpha_prim[6] = base.alpha_prim[2]
            
            # M NM2 (7)
            base.alpha_prim[7] = base.alpha_prim[5]
            
            # M M (8)
            tmp=integrate.quad(lambda s: self.int_fun(s,1,1,1,1,2,2,comb_flag=8), minint, maxint)[0]
            if tmp<1: tmp = 1
            base.alpha_prim[8] = (2*tmp)**(-1)    
    ## Integral function of interaction potential
    def int_fun(self,s,a,i,b=None,j=None,c=None,k=None,comb_flag=0):
        base = self.base
        # Generate corresponding psi, A, c1 and c2 values for given case.
        # Hamaker combination equation from Butt(2018) equation 3.86
        # comb_flag order same as in alpha_prim: NM1 NM1 (0) | NM1 NM2 (1) | 
        # NM1 M (2) | NM2 NM1 (3)=(1) | NM2 NM2 (4) | NM2 M (5) | 
        # M NM1 (6)=(2) | M NM2 (7)=(5) | M M (8)
        if comb_flag==0:     # a:NM1 i:NM1
            psi_a = base.PSI1
            psi_i = base.PSI1
            c1 = base.C1_NM1NM1
            c2 = base.C2_NM1NM1
            A = base.A_NM1NM1
        if comb_flag==1 or comb_flag==3:     # a:NM1 i:NM2
            psi_a = base.PSI1
            psi_i = base.PSI2
            c1 = base.C1_NM1NM2
            c2 = base.C2_NM1NM2
            A = np.sqrt(base.A_NM1NM1*base.A_NM2NM2)
        if comb_flag==2 or comb_flag==6:     # a:NM1 i:M
            psi_a = base.PSI1
            psi_i = base.PSI3
            c1 = base.C1_MNM1
            c2 = base.C2_MNM1
            A = np.sqrt(base.A_NM1NM1*base.A_MM)
        if comb_flag==4:     # a:NM2 i:NM2
            psi_a = base.PSI2
            psi_i = base.PSI2
            c1 = base.C1_NM2NM2
            c2 = base.C2_NM2NM2
            A = base.A_NM2NM2
        if comb_flag==5 or comb_flag==7:     # a:NM2 i:M
            psi_a = base.PSI2
            psi_i = base.PSI3
            c1 = base.C1_MNM2
            c2 = base.C2_MNM2
            A = np.sqrt(base.A_NM2NM2*base.A_MM)
        if comb_flag==8:     # a:M i:M
            psi_a = base.PSI3
            psi_i = base.PSI3
            c1 = base.C1_MM
            c2 = base.C2_MM
            A = base.A_MM
        
        gam1 = np.tanh(base.E*psi_a/(base.KT*4))
        gam2 = np.tanh(base.E*psi_i/(base.KT*4))
        kappa = np.sqrt((2*base.I*base.NA*base.E**2)/(base.EPS*base.KT))
        
        if base.dim == 1:
            r1 = base.R[a]
            r2 = base.R[i]
        elif base.dim == 2:
            r1 = base.R[a,b]
            r2 = base.R[i,j]
        elif base.dim == 3:
            r1 = base.R[a,b,c]
            r2 = base.R[i,j,k]
            
        a_dist = (r1+r2)/2
        h = (s*a_dist-2*a_dist)
    
        if base.POTEVAL == 1:
            # Electric potential Gregory 1975 (Elimelech book Particle deposition 
            # and aggregation), vdW potential sphere - sphere 3.23, hydrophobic 
            # interaction / polar potential Christenson 2001 (bi-exponential model)
            # NOTE: Potentials are already divided by kT (For electrostatic
            # potential missing factor kT in comparison to formula)
            Psi_el = 128*np.pi*(r1*r2)*base.I*base.NA*gam1*gam2*np.exp(-kappa*(s*a_dist-2*a_dist))/(kappa**2*(r1+r2))
            Psi_vdw = -A*((2*r1*r2/(h**2+2*r1*h+2*r2*h))+(2*r1*r2/(h**2+2*r1*h+2*r2*h+4*r1*r2))+np.log((h**2+2*r1*h+2*r2*h)/(h**2+2*r1*h+2*r2*h+4*r1*r2)))/(6*base.KT)
            a_sqr = np.sqrt(r1*r2)
            Psi_pol = -a_sqr*(c1*base.LAM1*np.exp(-h/base.LAM1)+c2*base.LAM2*np.exp(-h/base.LAM2))/(base.KT)
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
        base = self.base
        if t_vec is None:
            t_vec = base.t_vec
            t_max = base.t_vec[-1]
        else:
            t_max = t_vec[-1]
    
        N = base.N
        # 1-D case
        if base.dim == 1:
            # Define right-hand-side function depending on discretization
            if base.disc == 'geo':
                rhs = jit_dpbe_rhs.get_dNdt_1d_geo
                args=(base.NS,base.V,base.V_e,base.F_M,base.B_R,base.int_B_F,
                      base.intx_B_F,base.process_type,base.aggl_crit_id)
            elif base.disc == 'uni':
                rhs = jit_dpbe_rhs.get_dNdt_1d_uni                
                args=(base.V,base.B_R,base.B_F,base.F_M,base.NS,base.aggl_crit_id,base.process_type)
            if base.solve_algo == "ivp":    
                with np.errstate(divide='raise', over='raise',invalid='raise'):
                    try:
                        base.RES = integrate.solve_ivp(rhs, 
                                                        [0, t_max], 
                                                        N[:,0], t_eval=t_vec,
                                                        args=args,
                                                        ## If `rtol` is set too small, it may cause the results to diverge, 
                                                        ## leading to the termination of the calculation.
                                                        method='Radau',first_step=0.1,rtol=1e-1)
                        
                        # Reshape and save result to N and t_vec
                        t_vec = base.RES.t
                        y_evaluated = base.RES.y
                        status = True if base.RES.status == 0 else False
                    except (FloatingPointError, ValueError) as e:
                        print(f"Exception encountered: {e}")
                        y_evaluated = -np.ones((base.NS,len(t_vec)))
                        status = False
                    
            elif base.solve_algo == "radau":
                ode_sys = RK.radau_ii_a(rhs, N[:,0], t_eval=t_vec,
                                        args = args,
                                        dt_first=0.1)
                y_evaluated, y_res_tem, t_res_tem, rate_res_tem, error_res_tem = ode_sys.solve_ode()
                status = not ode_sys.dt_is_too_small  
    
        elif base.dim == 2:
            # Define right-hand-side function depending on discretization
            if base.disc == 'geo':
                rhs = jit_dpbe_rhs.get_dNdt_2d_geo
                args=(base.NS,base.V,base.V_e1,base.V_e3,base.F_M,base.B_R,base.int_B_F,
                      base.intx_B_F,base.inty_B_F,base.process_type,base.aggl_crit_id)
                base.agrs = args
            elif base.disc == 'uni':
                rhs = jit_dpbe_rhs.get_dNdt_2d_uni   
                args=(base.V,base.V1,base.V3,base.F_M,base.NS,base.THR_DN)
            if base.solve_algo == "ivp":  
                with np.errstate(divide='raise', over='raise',invalid='raise'):
                    try:
                        base.RES = integrate.solve_ivp(rhs, 
                                                        [0, t_max], 
                                                        np.reshape(N[:,:,0],-1), t_eval=t_vec,
                                                        args=args,
                                                        ## If `rtol` is set too small, it may cause the results to diverge, 
                                                        ## leading to the termination of the calculation.
                                                        method='Radau',first_step=0.1,rtol=1e-1)
                        
                        # Reshape and save result to N and t_vec
                        t_vec = base.RES.t
                        y_evaluated = base.RES.y.reshape((base.NS,base.NS,len(t_vec)))
                        status = True if base.RES.status == 0 else False
                    except (FloatingPointError, ValueError) as e:
                        print(f"Exception encountered: {e}")
                        y_evaluated = -np.ones((base.NS,base.NS,len(t_vec)))
                        status = False
                
            elif base.solve_algo == "radau":
                ode_sys = RK.radau_ii_a(rhs, np.reshape(N[:,:,0],-1), t_eval=t_vec,
                                        args = args,
                                        dt_first=0.1)
                y_evaluated, y_res_tem, t_res_tem, rate_res_tem, error_res_tem = ode_sys.solve_ode()
                y_evaluated = y_evaluated.reshape((base.NS,base.NS,len(t_vec)))
                y_res_tem = y_res_tem.reshape((base.NS,base.NS,len(t_res_tem)))
                status = not ode_sys.dt_is_too_small
    
        elif base.dim == 3:
            # Define right-hand-side function depending on discretization
            if base.disc == 'geo':
                rhs = jit_dpbe_rhs.get_dNdt_3d_geo
            elif base.disc == 'uni':
                rhs = jit_dpbe_rhs.get_dNdt_3d_uni   
                
            base.RES = integrate.solve_ivp(rhs, 
                                           [0, t_max], 
                                           np.reshape(base.N[:,:,:,0],-1), t_eval=t_vec,
                                           args=(base.V,base.V1,base.V2,base.V3,base.F_M,base.NS,base.THR_DN),
                                           ## If `rtol` is set too small, it may cause the results to diverge, 
                                           ## leading to the termination of the calculation.
                                           method='Radau',first_step=0.1,rtol=1e-1)
            
            # Reshape and save result to N and t_vec
            base.N = base.RES.y.reshape((base.NS+3,base.NS+3,base.NS+3,len(base.RES.t)))
            base.t_vec = base.RES.t
        # Monitor whether integration are completed  
        base.t_vec = t_vec 
        # base.N = y_evaluated / eva_N_scale
        base.N = y_evaluated
        base.calc_status = status   
        if not base.calc_status:
            print('Warning: The integral failed to converge!')
        if base.solve_algo == "radau":
            # base.N_res_tem = y_res_tem / res_N_scale
            base.N_res_tem = y_res_tem
            base.t_res_tem = t_res_tem
            base.rate_res_tem = rate_res_tem
            base.error_res_tem = error_res_tem

    def _close(self, gc_clean=True):
        big_attrs = ("N", "F_M", "B_R", "intx_B_F", "inty_B_F", "int_B_F", "RES")
        for name in big_attrs:
                setattr(self.base, name, None)
        if gc_clean:
            import gc
            gc.collect()
