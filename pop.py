""" Solving 1D, 2D and 3D discrete population balance equations for agglomerating systems. """

### ------ IMPORTS ------ ###
## General
import os
import numpy as np
import math
import scipy.integrate as integrate
## jit function
import func.jit_pop as jit
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt          
from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue
## For math
from func.func_math import float_in_list, float_equal, isZero

### ------ POPULATION CLASS DEFINITION ------ ###
class population():
    """Class definition for (discrete) population class
    
    Parameters
    ----------
    dim : `int`
        Dimension of PBE in [1, 2, 3].
    disc : `str`, optional
        Discretization strategy in ['uni', 'geo'].
    file : `str`, optional
        Path to initialization file.
    t_exp : `float`, optional
        Agglomeration time in minutes.
    attr : 
        Additional attributes to initialize. Can be any of the class attributes.
        
    Methods
    -------
    """
    
    ## Solve ODE with Scipy methods:
    def solve_PBE(self, t_max=None, t_vec=None):
        """Method for solving the (previously initialized) population with scipy.solve_ivp.
        
        Parameters
        ----------
        t_max : `float`, optional
            Final agglomeration time in seconds.
        t_vec : `array_like`, optional
            Points in time (in seconds) at which to export the numerical solution.
        """
        
        # If t_vec is not given (=None), let solver decide where to export time data
        if t_max is None and t_vec is None:
            t_max = self.NUM_T*self.DEL_T
        elif t_vec is not None:
            t_max = max(t_vec)
        
        # 1-D case
        if self.dim == 1:
            # Define right-hand-side function depending on discretization
            if self.disc == 'geo':
                rhs = jit.get_dNdt_1d_geo
                args=(self.NS,self.V,self.V_e,self.F_M,self.B_R,self.int_B_F,
                      self.intx_B_F,self.process_type,self.aggl_crit_id)
            elif self.disc == 'uni':
                rhs = jit.get_dNdt_1d_uni                
                args=(self.V,self.F_M,self.NS,self.THR_DN)
            self.RES = integrate.solve_ivp(rhs, 
                                           [0, t_max], 
                                           self.N[:,0], t_eval=t_vec,
                                           args=args,
                                           method='Radau',first_step=0.1,rtol=1e-1)
            
            # Reshape and save result to N and t_vec
            self.N = self.RES.y
            self.t_vec = self.RES.t
            
        elif self.dim == 2:
            # Define right-hand-side function depending on discretization
            if self.disc == 'geo':
                rhs = jit.get_dNdt_2d_geo
                args=(self.NS,self.V,self.V_e1,self.V_e3,self.F_M,self.B_R,self.int_B_F,
                      self.intx_B_F,self.inty_B_F,self.process_type,self.aggl_crit_id)
            elif self.disc == 'uni':
                rhs = jit.get_dNdt_2d_uni   
                args=(self.V,self.V1,self.V3,self.F_M,self.NS,self.THR_DN)
            self.RES = integrate.solve_ivp(rhs, 
                                           [0, t_max], 
                                           np.reshape(self.N[:,:,0],-1), t_eval=t_vec,
                                           args=args,
                                           method='RK23',first_step=0.1,rtol=1e-3)
            
            # Reshape and save result to N and t_vec
            self.N = self.RES.y.reshape((self.NS,self.NS,len(self.RES.t)))
            self.t_vec = self.RES.t
        
        elif self.dim == 3:
            # Define right-hand-side function depending on discretization
            if self.disc == 'geo':
                rhs = jit.get_dNdt_3d_geo
            elif self.disc == 'uni':
                rhs = jit.get_dNdt_3d_uni   
                
            self.RES = integrate.solve_ivp(rhs, 
                                           [0, t_max], 
                                           np.reshape(self.N[:,:,:,0],-1), t_eval=t_vec,
                                           args=(self.V,self.V1,self.V2,self.V3,self.F_M,self.NS,self.THR_DN),
                                           method='RK23',first_step=0.1,rtol=1e-1)
            
            # Reshape and save result to N and t_vec
            self.N = self.RES.y.reshape((self.NS+3,self.NS+3,self.NS+3,len(self.RES.t)))
            self.t_vec = self.RES.t
             
    ## Solve ODE (forward Euler scheme):
    def solve_PBE_Euler(self):
        """ `(Legacy)` Simple solver with forward Euler. Use ``solve_PBE( )`` instead. """
        
        # 2-D case
        if self.dim == 2:
            for tt in range(0,self.NUM_T): 
                # Calculate concentration change matrix depending on given timestep
                DN = jit.get_dNdt_2d_geo(tt,np.reshape(self.N[:,:,tt],-1),self.V,self.V1,self.V3,self.F_M,self.NS,self.THR_DN)
                DN = DN.reshape((self.NS+3,self.NS+3))       
                
                # If timestep is too large, negative concentrations can occur!
                tmp_N = self.N[:,:,tt]+DN*self.DEL_T
                
                # if any(tmp_N[tmp_N<0]):
                #     #print("! Negative values at t=",t,". Adjust time stepsize! Setting value of DN so that N(:,t+1)=0. ")
                #     zeroflag = True
                #     # Adjust DN of the indices that would lead to negative values
                #     DN[tmp_N<0] = -gv.N[tmp_N<0,t]/gc.DEL_T;          
                #     # Calculate new tmp_N
                #     tmp_N = gv.N[:,:,t]+DN*gc.DEL_T
                
                # Update concentration matrix
                self.N[:,:,tt+1] = tmp_N
                
        else:
            print('Euler solver is currently only implemented for the 2-D case.')
    
    ## Full initialization of population instance (calc_R, init_N, calc_alpha_prim,
    ##        calc_F_M, calc_B_M)
    def full_init(self, calc_alpha=True):
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
        self.init_N()        
        if calc_alpha: self.calc_alpha_prim()
        self.calc_F_M()
        self.calc_B_R()
        self.calc_int_B_F()
     
    ## Calculate R, V and X matrices (radii, total and partial volumes and volume fractions)
    def calc_R(self):
        """Initialize discrete calculation grid. 
        
        Creates the following class attributes: 
            * ``pop.V``: Total Volume of each class 
            * ``pop.R``: Radius of each class
            * ``pop.Xi_vol``: Volume fraction material i of each class
        """
        # 1-D case
        if self.dim == 1:
            
            # Initialize V and R
            self.V = np.zeros(self.NS)
            self.R = np.zeros(self.NS)
            
            if self.disc == 'uni':
                for i in range(len(self.V)): #range(0,self.NS):
                    self.V[i] = i*4*math.pi*self.R01**3/3
                
            if self.disc == 'geo':
                self.V_e = np.zeros(self.NS+1)
                self.V_e[0] = -4*math.pi*self.R01**3/3
                for i in range(self.NS):         
                    self.V_e[i+1] = self.S**(i)*4*math.pi*self.R01**3/3
                    self.V[i] = (self.V_e[i] + self.V_e[i+1]) / 2                    
            # Initialize V, R and ratio matrices
            self.R[1:] = (self.V[1:]*3/(4*math.pi))**(1/3)
            self.X1_vol = np.ones(self.NS) 
            self.X1_a = np.ones(self.NS) 
            ## Large particle agglomeration may cause the integration to not converge. 
            ## A Limit can be placed on the particle size.
            aggl_crit_ids = np.where(self.V < self.aggl_crit*self.V[1])[0]
            if (aggl_crit_ids.size > 0 and aggl_crit_ids.size < len(self.V)):
                self.aggl_crit_id = aggl_crit_ids[-1]  
            else: 
                self.aggl_crit_id = (len(self.V) -1)
                        
        # 2-D case
        elif self.dim == 2:
            
            # Initialize V1-V2 
            self.V1 = np.zeros(self.NS)
            # self.V1[1] = 0
            self.V3 = np.copy(self.V1) 
            
            if self.disc == 'uni':
                for i in range(len(self.V1)): #range(0,self.NS):
                    self.V1[i] = i*4*math.pi*self.R01**3/3
                    self.V3[i] = i*4*math.pi*self.R03**3/3
            
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
            
            # Initialize V, R and ratio matrices
            self.V = np.zeros((self.NS,self.NS))
            self.R = np.copy(self.V)
            self.X1_vol = np.copy(self.V); self.X1_a=np.copy(self.V) 
            self.X3_vol = np.copy(self.V); self.X3_a=np.copy(self.V)
            
            # Write V1 and V3 in respective "column" of V
            self.V[:,0] = self.V1 
            self.V[0,:] = self.V3 
            
            # Calculate remaining entries of V and other matrices
            # range(1,X) excludes X itself -> self.NS+2
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
            ## Large particle agglomeration may cause the integration to not converge. 
            ## A Limit can be placed on the particle size.
            aggl_crit_ids1 = np.where(self.V1 < self.aggl_crit*self.V1[1])[0]
            aggl_crit_ids2 = np.where(self.V3 < self.aggl_crit*self.V3[1])[0]
            self.aggl_crit_id = np.zeros(2, dtype=int)
            if (aggl_crit_ids1.size > 0 and aggl_crit_ids1.size < len(self.V1)):
                self.aggl_crit_id[0] = aggl_crit_ids1[-1]  
            else: 
                self.aggl_crit_id[0] = (len(self.V1) -1)
            if (aggl_crit_ids2.size > 0 and aggl_crit_ids2.size < len(self.V3)):
                self.aggl_crit_id[1] = aggl_crit_ids2[-1]  
            else: 
                self.aggl_crit_id[1] = (len(self.V3) -1)
                
        # 3-D case                
        elif self.dim == 3:
            
            # Initialize V1-V3 
            self.V1 = np.zeros(self.NS+3)-1 
            self.V1[1] = 0
            self.V2 = np.copy(self.V1)
            self.V3 = np.copy(self.V1) 
            
            for i in range(0,self.NS+1): 
                # Geometric grid
                if self.disc == 'geo': 
                    self.V1[i+2] = self.S**(i)*4*math.pi*self.R01**3/3
                    self.V2[i+2] = self.S**(i)*4*math.pi*self.R02**3/3
                    self.V3[i+2] = self.S**(i)*4*math.pi*self.R03**3/3
                
                # Uniform grid
                elif self.disc == 'uni':
                    self.V1[i+2] = (i+1)*4*math.pi*self.R01**3/3
                    self.V2[i+2] = (i+1)*4*math.pi*self.R02**3/3
                    self.V3[i+2] = (i+1)*4*math.pi*self.R03**3/3
            
            A1 = 3*self.V1/self.R01
            A2 = 3*self.V2/self.R02
            A3 = 3*self.V3/self.R03
            
            # Initialize V, R and ratio matrices
            self.V = np.zeros((self.NS+3,self.NS+3,self.NS+3))-1
            self.R = np.copy(self.V)
            self.X1_vol = np.copy(self.V); self.X1_a=np.copy(self.V) 
            self.X2_vol = np.copy(self.V); self.X2_a=np.copy(self.V)
            self.X3_vol = np.copy(self.V); self.X3_a=np.copy(self.V)
            
            # Write V1 and V3 in respective "column" of V
            self.V[:,1,1] = self.V1
            self.V[1,:,1] = self.V2 
            self.V[1,1,:] = self.V3 
            
            # Calculate remaining entries of V and other matrices
            # range(1,X) excludes X itself -> self.NS+3
            for i in range(1,self.NS+3):
                for j in range(1,self.NS+3):
                    for k in range(1,self.NS+3):
                        self.V[i,j,k] = self.V1[i]+self.V2[j]+self.V3[k]
                        self.R[i,j,k] = (self.V[i,j,k]*3/(4*math.pi))**(1/3)
                        if i==1 and j==1 and k==1:
                            self.X1_vol[i,j,k] = 0
                            self.X2_vol[i,j,k] = 0
                            self.X3_vol[i,j,k] = 0
                            self.X1_a[i,j,k] = 0
                            self.X2_a[i,j,k] = 0
                            self.X3_a[i,j,k] = 0
                        else:
                            self.X1_vol[i,j,k] = self.V1[i]/self.V[i,j,k]
                            self.X2_vol[i,j,k] = self.V2[j]/self.V[i,j,k]
                            self.X3_vol[i,j,k] = self.V3[k]/self.V[i,j,k]
                            self.X1_a[i,j,k] = A1[i]/(A1[i]+A2[j]+A3[k])
                            self.X2_a[i,j,k] = A2[j]/(A1[i]+A2[j]+A3[k])
                            self.X3_a[i,j,k] = A3[k]/(A1[i]+A2[j]+A3[k])
    
    ## Initialize concentration matrix N
    def init_N(self): 
        """Initialize discrete number concentration array. 
        
        Creates the following class attributes: 
            * ``pop.N``: Number concentration of each class 
        """
        
        # 1-D case
        if self.dim == 1:
            self.N = np.zeros((self.NS,self.NUM_T+1))
            if self.USE_PSD:
                self.N[1:,0] = self.new_initialize_psd(2*self.R[1:],self.DIST1,self.V01)
            else:
                if self.process_type == "agglomeration":
                    self.N[1,0] = self.N01
                elif self.process_type == "breakage":
                    self.N[-1,0] = self.N01
                elif self.process_type == "mix":
                    self.N[1,0] = self.N01
                    self.N[-1,0] = self.N01
                else:
                    raise Exception("Current process_type not allowed!")
        
        # 2-D case
        elif self.dim == 2:
            self.N = np.zeros((self.NS,self.NS,self.NUM_T+1))
            if self.USE_PSD:
                self.N[1:,1,0] = self.new_initialize_psd(2*self.R[1:,0],self.DIST1,self.V01)
                self.N[1,1:,0] = self.new_initialize_psd(2*self.R[0,1:],self.DIST3,self.V03)
            else:
                if self.process_type == "agglomeration":
                    self.N[1,0,0] = self.N01
                    self.N[0,1,0] = self.N03
                elif self.process_type == "breakage":
                    self.N[-1,-1,0] = self.N01
                elif self.process_type == "mix":
                    self.N[1,0,0] = self.N01
                    self.N[0,1,0] = self.N03  
                    self.N[-1,-1,0] = self.N01
                
        
        # 3-D case
        elif self.dim == 3:
            self.N = np.zeros((self.NS+3,self.NS+3,self.NS+3,self.NUM_T+1))
            if self.USE_PSD:
                self.N[2:-1,1,1,0] = self.new_initialize_psd(2*self.R[2:-1,1,1],self.DIST1,self.V01)
                self.N[1,2:-1,1,0] = self.new_initialize_psd(2*self.R[1,2:-1,1],self.DIST2,self.V02)
                self.N[1,1,2:-1,0] = self.new_initialize_psd(2*self.R[1,1,2:-1],self.DIST3,self.V03)
            else:
                self.N[2,1,1,0] = self.N01
                self.N[1,2,1,0] = self.N02
                self.N[1,1,2,0] = self.N03
    
    ## Calculate agglomeration rate matrix.
    ## JIT_FM controls whether the pre-compiled function is used or not. 
    def calc_F_M(self):
        """Initialize agglomeration frequency array. 
        
        Creates the following class attributes: 
            * ``pop.F_M``: Agglomeration frequency between two classes ij and ab is stored in ``F_M[i,j,a,b]`` 
        """
        # 1-D case
        if self.dim == 1:
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
                self.F_M[idx] = beta_ai*alpha_ai*corr_size
                
        # 2-D case.
        elif self.dim == 2:
            # To avoid mass leakage at the boundary in CAT, boundary cells are not directly involved in the calculation. 
            # So there is no need to define the corresponding F_M at boundary. F_M is (NS-1)^4 instead (NS)^4
            self.F_M = np.zeros((self.NS-1,self.NS-1,self.NS-1,self.NS-1))
            if self.process_type == 'breakage':
                return
            if self.JIT_FM:
                self.F_M = jit.calc_F_M_2D(self.NS,self.disc,self.COLEVAL,self.CORR_BETA,
                                           self.G,self.R,self.X1_vol,self.X3_vol,
                                           self.EFFEVAL,self.alpha_prim,self.SIZEEVAL,
                                           self.X_SEL,self.Y_SEL)
            
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
                    self.F_M[idx] = beta_ai*alpha_ai*corr_size
                
        # 3-D case. 
        elif self.dim == 3:
            if self.process_type == 'breakage':
                return
            if self.JIT_FM: 
                self.F_M = jit.calc_F_M_3D(self.NS,self.disc,self.COLEVAL,self.CORR_BETA,
                                           self.G,self.R,self.X1_vol,self.X2_vol,self.X3_vol,
                                           self.EFFEVAL,self.alpha_prim,self.SIZEEVAL,
                                           self.X_SEL,self.Y_SEL)
            
            else:
                # Initialize F_M Matrix. NOTE: F_M is defined without the border around the calculation grid
                # as e.g. N or V are (saving memory and calculations). 
                # Thus, F_M is (NS+1)^6 instead of (NS+3)^6. As reference, V is (NS+3)^3.
                self.F_M = np.zeros((self.NS+1,self.NS+1,self.NS+1,self.NS+1,self.NS+1,self.NS+1))
                
                # Go through all agglomeration partners 1 [a,b] and 2 [i,j]
                # The current index tuple idx stores them as (a,b,i,j)
                for idx, tmp in np.ndenumerate(self.F_M):
                    # # Indices [a,b,c]=[0,0,0] and [i,j,k]=[0,0,0] not allowed!
                    if idx[0]+idx[1]+idx[2]==0 or idx[3]+idx[4]+idx[5]==0:
                        continue
                    
                    # Calculate the corresponding agglomeration efficiency
                    # Add one to indices to account for borders
                    a = idx[0]+1; b = idx[1]+1; c = idx[2]+1;
                    i = idx[3]+1; j = idx[4]+1; k = idx[5]+1;
                    
                    # Calculate collision frequency beta depending on COLEVAL
                    if self.COLEVAL == 1:
                        # Chin 1998 (shear induced flocculation in stirred tanks)
                        # Optional reduction factor.
                        # corr_beta=1;
                        beta_ai = self.CORR_BETA*self.G*2.3*(self.R[a,b,c]+self.R[i,j,k])**3 # [m^3/s]
                    if self.COLEVAL == 2:
                        # Tsouris 1995 Brownian diffusion as controlling mechanism
                        # Optional reduction factor
                        # corr_beta=1;
                        beta_ai = self.CORR_BETA*2*self.KT*(self.R[a,b,c]+self.R[i,j,k])**2/(3*self.MU_W*(self.R[a,b,c]*self.R[i,j,k])) #[m^3/s]
                    if self.COLEVAL == 3:
                        # Use a constant collision frequency given by CORR_BETA
                        beta_ai = self.CORR_BETA
                    if self.COLEVAL == 4:
                        # Sum-Kernal (for validation) scaled by CORR_BETA
                        beta_ai = self.CORR_BETA*4*math.pi*(self.R[a,b,c]**3+self.R[i,j,k]**3)/3
                    
                    # Calculate probabilities, that particle 1 [a,b,c] is colliding as
                    # nonmagnetic 1 (NM1), nonmagnetic 2 (NM2) or magnetic (M). Repeat for
                    # particle 2 [i,j,k]. Use area weighted composition.
                    # Calculate probability vector for all combinations. 
                    # Indices: 
                    # 1) a:N1 <-> i:N1  -> X1[a,b,c]*X1[i,j,k]
                    # 2) a:N1 <-> i:N2  -> X1[a,b,c]*X2[i,j,k]
                    # 3) a:N1 <-> i:M   -> X1[a,b,c]*X3[i,j,k]
                    # 4) a:N2 <-> i:N1  -> X2[a,b,c]*X1[i,j,k] 
                    # 5) a:N2 <-> i:N2  -> X2[a,b,c]*X2[i,j,k]
                    # 6) a:N2 <-> i:M   -> X2[a,b,c]*X3[i,j,k]
                    # 7) a:M  <-> i:N1  -> X3[a,b,c]*X1[i,j,k]
                    # 8) a:M  <-> i:N2  -> X3[a,b,c]*X2[i,j,k]
                    # 9) a:M  <-> i:M   -> X3[a,b,c]*X3[i,j,k]
                    # Use volume fraction (May be changed to surface fraction)
                    X1 = self.X1_vol; X2 = self.X2_vol; X3 = self.X3_vol
                    
                    p=np.array([X1[a,b,c]*X1[i,j,k],\
                                X1[a,b,c]*X2[i,j,k],\
                                X1[a,b,c]*X3[i,j,k],\
                                X2[a,b,c]*X1[i,j,k],\
                                X2[a,b,c]*X2[i,j,k],\
                                X2[a,b,c]*X3[i,j,k],\
                                X3[a,b,c]*X1[i,j,k],\
                                X3[a,b,c]*X2[i,j,k],\
                                X3[a,b,c]*X3[i,j,k]])
                    
                    # Calculate collision effiecieny depending on EFFEVAL. 
                    # Case(1): "Correct" calculation for given indices. Accounts for size effects in int_fun
                    # Case(2): Reduced model. Calculation only based on primary particles
                    # Case(3): Alphas are pre-fed from ANN or other source.
                    if self.EFFEVAL == 1:
                        # Not coded here
                        alpha_ai = np.sum(p*self.alpha_prim)
                    if self.EFFEVAL == 2:
                        alpha_ai = np.sum(p*self.alpha_prim)
                    
                    # Calculate a correction factor to account for size dependency of alpha, depending on SIZEEVAL
                    # Calculate lam
                    if self.R[a,b,c]<=self.R[i,j,k]:
                        lam = self.R[a,b,c]/self.R[i,j,k]
                    else:
                        lam = self.R[i,j,k]/self.R[a,b,c]
                        
                    if self.SIZEEVAL == 1:
                        # No size dependency of alpha
                        corr_size = 1
                    if self.SIZEEVAL == 2:
                        # Case 3: Soos2007 (developed from Selomuya 2003). Empirical Equation
                        # with model parameters x and y. corr_size is lowered with lowered
                        # value of lambda (numerator) and with increasing particles size (denominator)
                        corr_size = np.exp(-self.X_SEL*(1-lam)**2)/((self.R[a,b,c]*self.R[i,j,k]/(np.min(np.array([self.R01,self.R02,self.R03]))**2))**self.Y_SEL)
                    
                    # Store result
                    self.F_M[idx] = beta_ai*alpha_ai*corr_size
    
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
    
    ## Calculate breakage rate matrix. 
    def calc_B_R(self):
        ## In breakage is not allowed to define parameter of particle without volume
        ## So B_R is (NS-1) instead (NS)
        # 1-D case
        if self.dim == 1:
            ## Note: The breakage rate of the smallest particle is 0. 
            ## Note: Because particles with a volume of zero are skipped, 
            ##       calculation with V requires (index+1)
            self.B_R = np.zeros(self.NS-1)
            if self.process_type == 'agglomeration':
                return
            # Size independent breakage rate --> See Leong2023 (10)
            # only for validation with analytical results
            if self.BREAKRVAL == 1:
                self.B_R[1:] = 1
                
            # Size dependent breakage rate --> See Leong2023 (10)
            # only for validation with analytical results
            elif self.BREAKRVAL == 2:
                for idx, tmp in np.ndenumerate(self.B_R):
                    a = idx[0]
                    if a != 0:
                        self.B_R[a] = self.V[a+1]
                        
            # Power Law Pandy and Spielmann --> See Jeldres2018 (28)
            elif self.BREAKRVAL == 3:
                for idx, tmp in np.ndenumerate(self.B_R):
                    a = idx[0]
                    if a != 0:
                        self.B_R[a] = self.pl_P1*self.G*(self.V[a+1]/self.V[1])**self.pl_P2          
        
        # 2-D case            
        if self.dim == 2:
            self.B_R = np.zeros((self.NS-1, self.NS-1))
            if self.process_type == 'agglomeration':
                return
            # Size independent breakage rate --> See Leong2023 (10)
            # only for validation with analytical results
            if self.BREAKRVAL == 1:
                self.B_R[:,:] = 1
                self.B_R[0,0] = 0
                
            # Size dependent breakage rate --> See Leong2023 (10)
            elif self.BREAKRVAL == 2:
                for idx, tmp in np.ndenumerate(self.B_R):
                    a = idx[0]; b = idx[1]
                    ## Note: Because of the conditional restrictions of breakage on the boundary, 
                    ##       1d calculation needs to be performed on the boundary.
                    if a == 0 and b == 0:
                        continue
                    elif a == 0:
                        self.B_R[idx] = self.V3[b+1]
                    elif b == 0:
                        self.B_R[idx] = self.V1[a+1]
                    else:
                        if self.BREAKFVAL == 1:
                            self.B_R[idx] = self.V1[a+1]*self.V3[b+1]
                        elif self.BREAKFVAL == 2:
                            self.B_R[idx] = self.V1[a+1] + self.V3[b+1]
            elif self.BREAKRVAL == 3:
                for idx, tmp in np.ndenumerate(self.B_R):
                    a = idx[0]; b = idx[1]
                    if a == 0 and b == 0:
                        continue
                    elif a == 0:
                        self.B_R[idx] = self.pl_P1 * self.G * (self.V3[b+1]/self.V3[1])**self.pl_P2
                    elif b == 0:
                        self.B_R[idx] = self.pl_P1 * self.G * (self.V1[a+1]/self.V1[1])**self.pl_P2
                    else:
                        self.B_R[idx] = self.pl_P1 * self.G * (self.V[a+1,b+1]/self.V[1,1])**self.pl_P2
                        
            
    ## Calculate integrated breakage function matrix.         
    def calc_int_B_F(self):
        if self.BREAKFVAL == 4:
            if self.pl_v <= 0 or self.pl_v > 1:
                raise Exception("Value of pl_v is out of range (0,1] for simple Power law.")
        ## In breakage is not allowed to define parameter of particle without volume
        ## So int_B_F and intx_B_F is (NS-1)^2 instead (NS)^2
        # 1-D case
        if self.dim == 1:
            ## Note: The breakage function of the smallest particle is 0. 
            ##       And small particle can not break into large one. 
            ## Note: Because particles with a volume of zero are skipped, 
            ##       calculation with V requires (index+1)
            self.int_B_F = np.zeros((self.NS-1, self.NS-1))
            self.intx_B_F = np.zeros((self.NS-1, self.NS-1))
            if self.process_type == 'agglomeration':
                return
            ## Let the integration range associated with the breakage function start from zero 
            ## to ensure mass conservation  
            V_e_tem = np.zeros(self.NS) 
            V_e_tem[:] = self.V_e[1:]
            V_e_tem[0] = 0.0
            for idx, tmp in np.ndenumerate(self.int_B_F):
                a = idx[0]; i = idx[1]
                if i != 0 and a <= i:
                    args = (self.V[i+1],self.pl_v,self.pl_q,self.BREAKFVAL)
                    if a == i:
                        self.int_B_F[idx],err = integrate.quad(jit.breakage_func_1d,V_e_tem[a],self.V[a+1],args=args)
                        self.intx_B_F[idx],err = integrate.quad(jit.breakage_func_1d_vol,V_e_tem[a],self.V[a+1],args=args)
                    else:
                        self.int_B_F[idx],err = integrate.quad(jit.breakage_func_1d,V_e_tem[a],V_e_tem[a+1],args=args)
                        self.intx_B_F[idx],err = integrate.quad(jit.breakage_func_1d_vol,V_e_tem[a],V_e_tem[a+1],args=args)
                    
        # 2-D case
        elif self.dim == 2:
            self.int_B_F = np.zeros((self.NS-1, self.NS-1, self.NS-1, self.NS-1))
            self.intx_B_F = np.zeros((self.NS-1, self.NS-1, self.NS-1, self.NS-1))
            self.inty_B_F = np.zeros((self.NS-1, self.NS-1, self.NS-1, self.NS-1))
            if self.process_type == 'agglomeration':
                return
            if self.JIT_BF == True:
                self.int_B_F, self.intx_B_F, self.inty_B_F = jit.calc_int_B_F_2D_quad(
                    self.NS,self.V1,self.V3,self.V_e1,self.V_e3,self.BREAKFVAL,self.pl_v,self.pl_q)
                # self.int_B_F, self.intx_B_F, self.inty_B_F = jit.calc_int_B_F_2D(
                #     self.NS,self.V1,self.V3,self.V_e1,self.V_e3,self.BREAKFVAL,self.pl_v,self.pl_q)
            else:
                V_e1_tem = np.zeros(self.NS) 
                V_e1_tem[:] = self.V_e1[1:]
                V_e1_tem[0] = 0.0
                V_e3_tem = np.zeros(self.NS) 
                V_e3_tem[:] = self.V_e3[1:]
                V_e3_tem[0] = 0.0
                for idx, tmp in np.ndenumerate(self.int_B_F):
                    a = idx[0] ; b = idx[1]; i = idx[2]; j = idx[3]
                    if i + j != 0 and a<=i or b <= j:
                        if i == 0:
                            ## for left boundary/y
                            args = (self.V3[j+1],self.pl_v,self.pl_q,self.BREAKFVAL)
                            if j == 0:
                                continue
                            elif b == j:
                                self.int_B_F[idx],err  = integrate.quad(jit.breakage_func_1d,V_e3_tem[b],self.V3[b+1],args=args)
                                self.inty_B_F[idx],err  = integrate.quad(jit.breakage_func_1d_vol,V_e3_tem[b],self.V3[b+1],args=args)
                            else:
                                self.int_B_F[idx],err  = integrate.quad(jit.breakage_func_1d,V_e3_tem[b],V_e3_tem[b+1],args=args)
                                self.inty_B_F[idx],err  = integrate.quad(jit.breakage_func_1d_vol,V_e3_tem[b],V_e3_tem[b+1],args=args)
                        elif j == 0:
                            ## for low boundary/x
                            args = (self.V1[i+1],self.pl_v,self.pl_q,self.BREAKFVAL)
                            if a == i:
                                self.int_B_F[idx],err = integrate.quad(jit.breakage_func_1d,V_e1_tem[a],self.V1[a+1],args=args)
                                self.intx_B_F[idx],err = integrate.quad(jit.breakage_func_1d_vol,V_e1_tem[a],self.V1[a+1],args=args)
                            else:
                                self.int_B_F[idx],err = integrate.quad(jit.breakage_func_1d,V_e1_tem[a],V_e1_tem[a+1],args=args)
                                self.intx_B_F[idx],err = integrate.quad(jit.breakage_func_1d_vol,V_e1_tem[a],V_e1_tem[a+1],args=args)
                        else:
                            args = (self.V1[i+1],self.V3[j+1],self.pl_v,self.pl_q,self.BREAKFVAL)
                            ## The contributions of fragments on the same vertical axis
                            if a == i and b == j:
                                self.int_B_F[idx],err  = integrate.dblquad(jit.breakage_func_2d,V_e1_tem[a],self.V1[a+1],V_e3_tem[b],self.V3[b+1],args=args)
                                self.intx_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x1vol,V_e1_tem[a],self.V1[a+1],V_e3_tem[b],self.V3[b+1],args=args)
                                self.inty_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x3vol,V_e1_tem[a],self.V1[a+1],V_e3_tem[b],self.V3[b+1],args=args)
                            elif a == i:
                                self.int_B_F[idx],err  = integrate.dblquad(jit.breakage_func_2d,V_e1_tem[a],self.V1[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                                self.intx_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x1vol,V_e1_tem[a],self.V1[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                                self.inty_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x3vol,V_e1_tem[a],self.V1[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                            ## The contributions of fragments on the same horizontal axis
                            elif b == j:   
                                self.int_B_F[idx],err  = integrate.dblquad(jit.breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],self.V3[b+1],args=args)
                                self.intx_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x1vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],self.V3[b+1],args=args)
                                self.inty_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x3vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],self.V3[b+1],args=args)
                            ## The contribution from the fragments of large particles on the upper right side 
                            else:
                                self.int_B_F[idx],err  = integrate.dblquad(jit.breakage_func_2d,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                                self.intx_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x1vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
                                self.inty_B_F[idx],err = integrate.dblquad(jit.breakage_func_2d_x3vol,V_e1_tem[a],V_e1_tem[a+1],V_e3_tem[b],V_e3_tem[b+1],args=args)
            
    ## Visualize / plot population:
    def visualize_distN_t(self,t_plot=None,t_pause=0.5,close_all=False,scl_a4=1,figsze=[12.8,6.4*1.5]):
        # Definition of t_plot:
        # None: Plot all available times
        # Numpy array in range [0,1] --> Relative values of time indices
        # E.g. t_plot=np.array([0,0.5,1]) plots start, half-time and end
        
        # Double figsize in 3-D case
        if self.dim == 1 or self.dim == 2: 
            pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        else: 
            pt.plot_init(scl_a4=4,frac_lnewdth=2,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
            
        if close_all:
            plt.close('all')
        
        fig=plt.figure()    
        
        if t_plot is None:
            tmp = None
            t_plot = np.arange(len(self.t_vec))
        else:
            t_plot = np.round(t_plot*(len(self.t_vec)-1))
            
        # 1-D case: Plot PSD over time        
        if self.dim == 1:
            print('For 1-D case executing visualize_qQ_t instead.')
            ax1, ax2, fig = self.visualize_qQ_t(t_plot=tmp,t_pause=t_pause,close_all=close_all,
                                                scl_a4=scl_a4,figsze=figsze,
                                                show_x10=False, show_x50=True, show_x90=False)
            return [ax1, ax2], fig
        
        # 2-D case: Plot distribution over time
        elif self.dim == 2:
            ax=fig.add_subplot(1,1,1)
            
            for t in t_plot:
                
                if 'cb' in locals(): cb.remove()                
                ax,cb,fig = self.plot_N2D(self.N[1:,1:,t],self.V[1:,1:],np.sum(self.N[1:,1:,0]*self.V[1:,1:]),
                                          ax=ax,fig=fig,t_stamp=f'{np.round(self.t_vec[t])}s')
            
                plt.pause(t_pause)
        
            plt.show()
            return ax, fig
            
        # 3-D case: Plot distributions over time   
        elif self.dim ==3:
            ax1 = fig.add_subplot(2,2,1)
            ax2 = fig.add_subplot(2,2,2)
            ax3 = fig.add_subplot(2,2,3)
            ax4 = fig.add_subplot(2,2,4)
            
            #Calculate date for distribution plot:
            Xt=np.zeros((3,len(t_plot)))
    
            for t in t_plot:
                Ntmp=self.N[:,:,:,t]
                Nagg=np.sum(Ntmp)-np.sum(Ntmp[:,1,1])-np.sum(Ntmp[1,:,1])-np.sum(Ntmp[1,1,:])
                if not Nagg == 0:
                    Xt[0,t]=np.sum(Ntmp[self.X1_vol!=1]*self.X1_vol[self.X1_vol!=1])/Nagg
                    Xt[1,t]=np.sum(Ntmp[self.X2_vol!=1]*self.X2_vol[self.X2_vol!=1])/Nagg
                    Xt[2,t]=np.sum(Ntmp[self.X3_vol!=1]*self.X3_vol[self.X3_vol!=1])/Nagg
                else:
                    Xt[0,t] = Xt[1,t] = Xt[2,t] = 0 
            
            Xt[:,0]=Xt[:,1]
            
            for t in t_plot:
                
                if 'cb1' in locals(): cb1.remove()
                if 'cb2' in locals(): cb2.remove()    
                if 'cb3' in locals(): cb3.remove()
                ax1,cb1,fig = self.plot_N2D(self.N[1:,1:,1,t],self.V[1:,1:,1],np.sum(self.N[:,:,:,0]*self.V),
                                            ax=ax1,fig=fig)
                ax2,cb2,fig = self.plot_N2D(self.N[1:,1,1:,t],self.V[1:,1,1:],np.sum(self.N[:,:,:,0]*self.V),
                                            ax=ax2,fig=fig)
                ax3,cb3,fig = self.plot_N2D(self.N[1,1:,1:,t],self.V[1,1:,1:],np.sum(self.N[:,:,:,0]*self.V),
                                            ax=ax3,fig=fig)
                
                
                ax2.set_xlabel('Partial volume comp. 3 $V_{3}$ ($k$) / $-$')  # Add a y-label to the axes.
                ax3.set_ylabel('Partial volume comp. 2 $V_{2}$ ($k$) / $-$')  # Add a y-label to the axes.
                ax3.set_xlabel('Partial volume comp. 3 $V_{3}$ ($k$) / $-$')  # Add a y-label to the axes.
                
                ax4.cla()
                ax4, fig = pt.plot_data(self.t_vec[:t+1],Xt[0,:t+1], fig=fig, ax=ax4,
                                        xlbl='Agglomeration time $t_\mathrm{A}$ / $-$',
                                        ylbl='Agglomerate composition / $-$',
                                        lbl=None,clr='k',plt_type='line',leg=False)
                ax4, fig = pt.plot_data(self.t_vec[:t+1],Xt[1,:t+1]+Xt[0,:t+1], 
                                        fig=fig, ax=ax4, lbl=None,clr='k',plt_type='line',leg=False)
                ax4, fig = pt.plot_data(self.t_vec[:t+1],Xt[2,:t+1]+Xt[1,:t+1]+Xt[0,:t+1], 
                                        fig=fig, ax=ax4, lbl=None,clr='k',plt_type='line',leg=False)
                ax4.stackplot(self.t_vec[:t+1],Xt[:,:t+1],colors=[c_KIT_green,c_KIT_red,c_KIT_blue],
                              labels=['Comp. 1','Comp. 2','Comp. 3'])
                ax4.legend(loc='upper right')
                ax4.text(0.05, 0.95, f'{np.round(self.t_vec[t])}s', transform=ax4.transAxes, fontsize=10*1.6,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='w', alpha=1))
                
                ax4.set_xlim([0,self.t_vec[-1]])
                ax4.set_ylim([0,1])
                plt.tight_layout()
                plt.pause(t_pause)
        
            plt.show()
        
            return [ax1, ax2, ax3, ax4], fig
    
    ## Visualize / plot density and sum distribution:
    def visualize_qQ_t(self,t_plot=None,t_pause=0.5,close_all=False,scl_a4=1,figsze=[12.8,6.4*1.5],
                       show_x10=False, show_x50=True, show_x90=False):
        # Definition of t_plot:
        # None: Plot all available times
        # Numpy array in range [0,1] --> Relative values of time indices
        # E.g. t_plot=np.array([0,0.5,1]) plots start, half-time and end
        
        import seaborn as sns
        import pandas as pd
        
        # Initialize plot
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
            
        if close_all:
            plt.close('all')
        
        fig=plt.figure()    
        ax1=fig.add_subplot(1,2,1) 
        ax2=fig.add_subplot(1,2,2)
        
        if t_plot is None:
            t_plot = np.arange(len(self.t_vec))
        else:
            t_plot = np.round(t_plot*(len(self.t_vec)-1)).astype(int)
        
        xmin = min(self.return_distribution(t=t_plot[0])[0])*1e6
        xmax = max(self.return_distribution(t=t_plot[-1])[0])*1e6
        
        for t in t_plot:

            # Calculate distribution
            x_uni, q3, Q3, x_10, x_50, x_90 = self.return_distribution(t=t)
            
            ax1.cla()
            ax2.cla()
            
            ax1, fig = pt.plot_data(x_uni*1e6, q3, ax=ax1, fig=fig, plt_type='scatter',
                                xlbl='Particle Diameter $x$ / $\mu$m',
                                ylbl='Volume density distribution $q_3$ / $-$',
                                clr='k',mrk='o',leg=False,zorder=2)
            
            if len(x_uni) > 1:
                sns.histplot(data=pd.DataFrame({'x':x_uni[q3>=0]*1e6,'q':q3[q3>=0]}), x='x',weights='q', 
                              log_scale=True, bins=15, ax=ax1, kde=True, color=c_KIT_green)
            
            ax2, fig = pt.plot_data(x_uni*1e6, Q3, ax=ax2, fig=fig,
                                xlbl='Particle Diameter $x$ / $\mu$m',
                                ylbl='Volume sum distribution $Q_3$ / $-$',
                                clr='k',mrk='o',leg=False)
            
            #ax1.set_ylim([0,0.25])
            ax1.grid('minor')
            ax2.grid('minor')
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            
            if show_x10: ax1.axvline(x_10*1e6, color=c_KIT_green)
            if show_x10: ax2.axvline(x_10*1e6, color=c_KIT_green)
            if show_x50: ax1.axvline(x_50*1e6, color=c_KIT_red)
            if show_x50: ax2.axvline(x_50*1e6, color=c_KIT_red)
            if show_x90: ax1.axvline(x_90*1e6, color=c_KIT_blue)
            if show_x90: ax2.axvline(x_90*1e6, color=c_KIT_blue)
            plt.tight_layout() 
            plt.pause(t_pause)
    
        plt.show()        
        
        return ax1, ax2, fig

    ## Visualize / plot population:
    def visualize_sumN_t(self,ax=None,fig=None,close_all=False,lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]):
        
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig=plt.figure()    
            ax=fig.add_subplot(1,1,1)   
        
        ax, fig = pt.plot_data(self.t_vec,self.return_N_t(), fig=fig, ax=ax,
                               xlbl='Agglomeration time $t_\mathrm{A}$ / $-$',
                               ylbl='Total number of agglomerates $\Sigma N$ / $-$',
                               lbl=lbl,clr=clr,mrk=mrk)
            
        ax.grid('minor')
        plt.tight_layout()   
        
        return ax, fig
    
    def visualize_sumvol_t(self, sumvol, ax=None,fig=None,close_all=False,lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]):
        
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig=plt.figure()    
            ax=fig.add_subplot(1,1,1)   
        
        ax, fig = pt.plot_data(self.t_vec,sumvol, fig=fig, ax=ax,
                               xlbl='Agglomeration time $t_\mathrm{A}$ / $-$',
                               ylbl='Total volume of agglomerates $\Sigma N$ / $-$',
                               lbl=lbl,clr=clr,mrk=mrk)
        
        if self.dim == 1:
            mu = self.calc_mom_t()
            mu_10 = mu[1, 0, :]
            ax, fig = pt.plot_data(self.t_vec,mu_10, fig=fig, ax=ax,
                                   lbl=lbl,clr=clr,mrk='^')
            
        if self.dim == 2:
            
            mu = self.calc_mom_t()
            mu_10 = mu[1, 0, :] + mu[0, 1, :]
            ax, fig = pt.plot_data(self.t_vec,mu_10, fig=fig, ax=ax,
                                   lbl=lbl,clr=clr,mrk='v')
        
        ax.grid('minor')
        plt.tight_layout()   
        
        return ax, fig

    def visualize_distribution(self, x_uni, q3, Q3, ax=None,fig=None,close_all=False,lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]):
        
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig=plt.figure()    
            axq3=fig.add_subplot(1,2,1)   
            axQ3=fig.add_subplot(1,2,2)   
            
        
        axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $q3$ / $-$',
                               lbl=lbl,clr=clr,mrk=mrk)
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                               ylbl='volume distribution of agglomerates $Q3$ / $-$',
                               lbl=lbl,clr=clr,mrk='^')

        axq3.grid('minor')
        axQ3.grid('minor')
        plt.tight_layout()   
        
        return axq3, axQ3, fig
    
    ## Return particle size distribution on fixed grid 
    def return_distribution(self, comp='all', t=0, N=None, flag='all'):
        def unique_with_tolerance(V, tol=1e-3):
            V_sorted = np.sort(V)
            V_unique = [V_sorted[0]]
            
            for V_val in V_sorted[1:]:
                if not np.isclose(V_val, V_unique[-1], atol=tol*V_sorted[0], rtol=0):
                    V_unique.append(V_val)
            return np.array(V_unique)
        # If no N is provided use the one from the class instance
        if N is None:
            N = self.N
        
        # Extract unique values that are NOT -1 or 0 (border)
        v_uni = np.setdiff1d(self.V,[-1])
        # v_uni = unique_with_tolerance(v_uni)
        q3 = np.zeros(len(v_uni))
        x_uni = np.zeros(len(v_uni))
        sumvol_uni = np.zeros(len(v_uni))
        
        if comp == 'all':
            # Loop through all entries in V and add volume concentration to specific entry in sumvol_uni
            if self.dim == 1:
                for i in range(self.NS):
                    # if self.V[i] in v_uni:
                    sumvol_uni[v_uni == self.V[i]] += self.V[i]*N[i,t] 
                        
            if self.dim == 2:
                for i in range(self.NS):
                    for j in range(self.NS):
                        # if self.V[i,j] in v_uni:
                        sumvol_uni[v_uni == self.V[i,j]] += self.V[i,j]*N[i,j,t]

            if self.dim == 3:
                for i in range(self.NS):
                    for j in range(self.NS):
                        for k in range(self.NS):
                            # if self.V[i,j,k] in v_uni:
                            sumvol_uni[v_uni == self.V[i,j,k]] += self.V[i,j,k]*N[i,j,k,t]
            ## convert unit m into um
            sumvol_uni *= 1e18
            sumV = np.sum(sumvol_uni)
            # Calculate diameter array
            x_uni[1:]=(6*v_uni[1:]/np.pi)**(1/3)*1e6
            
            # Calculate sum and density distribution
            Q3 = np.cumsum(sumvol_uni)/sumV
            for i in range(1,len(x_uni)):
                q3[i] = (Q3[i] - Q3[i-1]) / (x_uni[i]-x_uni[i-1])
            
            # Retrieve x10, x50 and x90 through interpolation
            x_10=np.interp(0.1, Q3, x_uni)
            x_50=np.interp(0.5, Q3, x_uni)
            x_90=np.interp(0.9, Q3, x_uni)   
        else:
            print('Case for comp not coded yet. Exiting')
            return
        
        outputs = {
        'x_uni': x_uni,
        'q3': q3,
        'Q3': Q3,
        'x_10': x_10,
        'x_50': x_50,
        'x_90': x_90,
        'sumvol_uni': sumvol_uni,
        }
        
        if flag == 'all':
            return outputs.values()
        else:
            flags = flag.split(',')
            return tuple(outputs[f.strip()] for f in flags if f.strip() in outputs)
    
    def return_num_distribution(self, comp='all', t=0, N=None, flag='all'):
        # If no N is provided use the one from the class instance
        if N is None:
            N = self.N
        
        # Extract unique values that are NOT -1 or 0 (border)
        # At the same time, v_uni will be rearranged according to size.
        v_uni = np.setdiff1d(self.V,[-1])

        q3 = np.zeros(len(v_uni))
        x_uni = np.zeros(len(v_uni))
        sumN_uni = np.zeros(len(v_uni))
        
        if comp == 'all':
            # Loop through all entries in V and add volume concentration to specific entry in sumN_uni
            if self.dim == 1:
                for i in range(self.NS):
                    if float_in_list(self.V[i], v_uni) and (not N[i,t] < 0):
                        sumN_uni[v_uni == self.V[i]] += N[i,t] 
                        
            if self.dim == 2:
                for i in range(self.NS):
                    for j in range(self.NS):
                        if float_in_list(self.V[i,j], v_uni) and (not N[i,j,t] < 0):
                            sumN_uni[v_uni == self.V[i,j]] += N[i,j,t]

            if self.dim == 3:
                for i in range(self.NS):
                    for j in range(self.NS):
                        for k in range(self.NS):
                            if float_in_list(self.V[i,j,k], v_uni) and (not N[i,j,t] < 0):
                                sumN_uni[v_uni == self.V[i,j,k]] += N[i,j,k,t]
                                
            sumN = np.sum(sumN_uni)
            # Calculate diameter array and convert into mm
            x_uni[1:]=(6*v_uni[1:]/np.pi)**(1/3)*1e6
            
            # Calculate sum and density distribution
            Q3 = np.cumsum(sumN_uni)/sumN
            for i in range(1,len(x_uni)):
                q3[i] = (Q3[i] - Q3[i-1]) / (x_uni[i]-x_uni[i-1])
            
            # Retrieve x10, x50 and x90 through interpolation
            x_10=np.interp(0.1, Q3, x_uni)
            x_50=np.interp(0.5, Q3, x_uni)
            x_90=np.interp(0.9, Q3, x_uni)
            
        else:
            print('Case for comp not coded yet. Exiting')
            return
    
        outputs = {
        'x_uni': x_uni,
        'q3': q3,
        'Q3': Q3,
        'x_10': x_10,
        'x_50': x_50,
        'x_90': x_90,
        'sumN_uni': sumN_uni,
        }
        
        if flag == 'all':
            return outputs.values()
        else:
            flags = flag.split(',')
            return tuple(outputs[f.strip()] for f in flags if f.strip() in outputs)
        
    ## Return total number. For t=None return full array, else return total number at time index t 
    def return_N_t(self,t=None):
        
        # 1-D case
        if self.dim == 1:
            if t is None:
                return np.sum(self.N,axis=0)
            else:
                return np.sum(self.N[:,t])  
            
        # 2-D case    
        elif self.dim == 2:
            if t is None:
                return np.sum(self.N,axis=(0,1))
            else:
                return np.sum(self.N[:,:,t])
        
        # 3-D case    
        elif self.dim == 3:
            if t is None:
                return np.sum(self.N,axis=(0,1,2))
            else:
                return np.sum(self.N[:,:,:,t])
    
    # Calculate distribution moments mu(i,j,t)
    def calc_mom_t(self):
        mu = np.zeros((3,3,len(self.t_vec)))
        
        # Time loop
        for t in range(len(self.t_vec)):
            for i in range(3):
                if self.dim == 1:
                    mu[i,0,t] = np.sum(self.V**i*self.N[:,t])
                    
                # The following only applies for 2D and 3D case
                # Moment between component 1 and 3
                else:
                    for j in range(3):
                        if self.dim == 2:
                            mu[i,j,t] = np.sum((self.X1_vol*self.V)**i*(self.X3_vol*self.V)**j*self.N[:,:,t])
                        if self.dim == 3:
                            mu[i,j,t] = np.sum((self.X1_vol*self.V)**i*(self.X3_vol*self.V)**j*self.N[:,:,:,t])
                        
        return mu
    
    ## Perform magnetic separation and return separation efficiencies
    def mag_sep(self):
        
        # Initialize separation matrix and result array
        T_m = np.zeros(np.shape(self.R))
        T = np.zeros(4) # T[0]: overall separation efficiency, T[1-3]: component 1-3       
        
        # Calculate model constants
        c2=self.R03**2*(1-np.log((1-self.TC2)/self.TC2)/np.log((1-self.TC1)/self.TC1))**(-1)
        c1=np.log((1-self.TC1)/self.TC1)*9*self.MU_W/(c2*2*self.MU0*self.M_SAT_M)
        
        # 1-D case not available
        if self.dim == 1:
            print('Magnetic separation not possible in 1-D case.')
        
        elif self.dim == 2:
            # Calculate T_m (Separation efficiency matrix)
            for idx, tmp in np.ndenumerate(self.R[1:-1,1:-1]):
                i=idx[0]+1
                j=idx[1]+1
                
                if not (i == 1 and j == 1):                
                    T_m[i,j]=1/(1+np.exp(-2*self.MU0*self.M_SAT_M*c1*(self.R[i,j]**2*self.X3_vol[i,j]-c2)/(9*self.MU_W)))
            
            # Calculate overall separation efficiency
            T[0]=100*np.sum(self.N[:,:,-1]*self.V*T_m)/np.sum(self.N[:,:,0]*self.V)
            # Calculate separation efficiency of component 1
            T[1]=100*np.sum(self.N[:,:,-1]*self.X1_vol*self.V*T_m)/np.sum(self.N[:,:,0]*self.X1_vol*self.V)
            # Calculate separation efficiency of component 3
            T[3]=100*np.sum(self.N[:,:,-1]*self.X3_vol*self.V*T_m)/np.sum(self.N[:,:,0]*self.X3_vol*self.V)
            
        else:
            # Calculate T_m (Separation efficiency matrix)
            for idx, tmp in np.ndenumerate(self.R[1:-1,1:-1,1:-1]):
                i=idx[0]+1
                j=idx[1]+1
                k=idx[2]+1
                
                if not (i == 1 and j == 1 and k == 1):                
                    T_m[i,j,k]=1/(1+np.exp(-2*self.MU0*self.M_SAT_M*c1*(self.R[i,j,k]**2*self.X3_vol[i,j,k]-c2)/(9*self.MU_W)))
            
            # Calculate overall separation efficiency
            T[0]=100*np.sum(self.N[:,:,:,-1]*self.V*T_m)/np.sum(self.N[:,:,:,0]*self.V)
            # Calculate separation efficiency of component 1
            T[1]=100*np.sum(self.N[:,:,:,-1]*self.X1_vol*self.V*T_m)/np.sum(self.N[:,:,:,0]*self.X1_vol*self.V)
            # Calculate separation efficiency of component 2
            T[2]=100*np.sum(self.N[:,:,:,-1]*self.X2_vol*self.V*T_m)/np.sum(self.N[:,:,:,0]*self.X2_vol*self.V)
            # Calculate separation efficiency of component 3
            T[3]=100*np.sum(self.N[:,:,:,-1]*self.X3_vol*self.V*T_m)/np.sum(self.N[:,:,:,0]*self.X3_vol*self.V)
        
        return T, T_m

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
    
    ## Save Variables to file:
    def save_vars(self,file):
        np.save(file,vars(self))
            
    ## Initialize:
    def __init__(self, dim, t_exp=10, file=None, disc='geo', **attr):
        
        # Check if given dimension and discretization is valid
        if not (dim in [1,2,3] and disc in ['geo','uni']):
            print('Given dimension and/or discretization are not valid. Exiting..')
            
        # BASELINE PATH
        self.pth = os.path.dirname( __file__ )
        
        ## MODEL parameters
        self.dim = dim                        # Dimension (1=1D, 2=2D, 3=3D)
        self.disc = disc                      # 'geo': geometric grid, 'uni': uniform grid
        self.DEL_T = 1                        # Time stepsize [s]
        self.NS = 12                          # Grid parameter [-]
        self.S = 2                            # Geometric grid ratio (V[i] = S*V[i-1])
        self.JIT_DN = True                    # Define wheter or not the DN calculation (timeloop) should be precompiled
        self.JIT_FM = True                    # Define wheter or not the FM calculation should be precompiled
        self.JIT_BF = True
        self.COLEVAL = 1                      # Case for calculation of beta. 1 = Orthokinetic, 2 = Perikinetic
        self.EFFEVAL = 2                      # Case for calculation of alpha. 1 = Full calculation, 2 = Reduced model (only based on primary particle interactions)
                                            # Case 2 massively faster and legit acc. to Kusters1997 and Bäbler2008
                                            # Case 3 to use pre-defines alphas (e.g. from ANN) --> alphas need to be provided at some point
        self.BREAKRVAL = 3                    # Case for calculation breakage rate. 1 = constant, 2 = size dependent
        self.BREAKFVAL = 3                    # Case for calculation breakage function. 1 = conservation of Hypervolume, 2 = conservation of 0 Moments 
        self.aggl_crit = 1e3                  # relative maximum aggregate volume(to primary particle) allowed to further agglomeration
        self.process_type = "breakage"    # "agglomeration": only calculate agglomeration, "breakage": only calculate breakage, "mix": calculate both agglomeration and breakage
        self.pl_v = 4                         # number of fragments in product function of power law
                                              # or (v+1)/v: number of fragments in simple power law  
        self.pl_q = 1                         # parameter describes the breakage type(in product function of power law) 
        self.pl_P1 = 1e-6                     # 1. parameter in power law for breakage rate  
        self.pl_P2 = 0.5                      # 2. parameter in power law for breakage rate  
                       
        self.SIZEEVAL = 2                     # Case for implementation of size dependency. 1 = No size dependency, 2 = Model from Soos2007 
        self.POTEVAL = 1                      # Case for the set of used interaction potentials. See int_fun_Xd for infos.
        self.USE_PSD = True                   # Define wheter or not the PSD should be initializes (False = monodisperse primary particles)
        self.FOLDERNAME = '210414_default'    # Foldername in ../export/
        self.EXPORTNAME = 'default'           # Filename extension for export
        self.GC_EXPSTR = ''                   # Full exportstring of GC (only initialization, gets real value upon export)
        self.GV_EXPSTR = ''                   # Full exportstring of GV (only initialization, gets real value upon export)
        self.REPORT = True                    # When true, population informs about current calculation and reports results
        
        
        # NOTE: The following two parameters define the magnetic separation step.
        self.TC1 = 1e-3                       # Separation efficiency of nonmagnetic particles. 0 and 1 not allowed!    
        self.TC2 = 1-1e-3                     # Separation efficiency of magnetic particles. 0 and 1 not allowed!
        self.THR_DN = 1e-10                    # Threshold for concentration change matrix (DN[DN<THR]=0)
        self.CORR_PSI = 1                     # Correction factor for zeta potentials in general
        self.CORR_PSI_SIO2 = 1                # Correction factor for SiO2 zeta potential
        self.CORR_PSI_ZNO = 1                 # Correction factor for ZnO zeta potential
        self.CORR_PSI_MAG = 1                 # Correction factor for MAG zeta potential
        self.CORR_A = 1                       # Correction factor for Hamaker constants in general
        self.CORR_A_SIO2 = 1                  # Correction factor for SiO2 Hamaker constant
        self.CORR_A_ZNO = 1                   # Correction factor for ZnO Hamaker constant
        self.CORR_A_MAG = 1                   # Correction factor for MAG Hamaker constant
        
        ## MATERIAL parameters:
        # NOTE: component 3 is defined as the magnetic component (both in 2D and 3D case)
        self.R01 = 2.9e-7                     # Radius primary particle component 1 [m] - NM1
        self.R02 = 2.9e-7                     # Radius primary particle component 2 [m] - NM2
        self.R03 = 2.9e-7                     # Radius primary particle component 3 [m] - M3
        self.DIST1 = os.path.join(self.pth,"data\\PSD_data\\")+'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        self.DIST2 = os.path.join(self.pth,"data\\PSD_data\\")+'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        self.DIST3 = os.path.join(self.pth,"data\\PSD_data\\")+'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        self.alpha_prim = np.ones(dim**2)
        self.PSI1 = 1*1e-3                   # Surface potential component 1 [V] - NM1
        self.PSI2 = 1*1e-3                    # Surface potential component 2 [V] - NM2
        self.PSI3 = -40*1e-3                  # Surface potential component 3 [V] - M
        self.A_NM1NM1 = 10e-21                # Hamaker constant for interaction NM1-NM1 [J] 
        self.A_NM2NM2 = 10e-21                # Hamaker constant for interaction NM2-NM2 [J]
        self.A_MM = 80e-21                    # Hamaker constant for interaction M-M [J]
        self.M_SAT_N1 = 505.09                # Saturization magnetization component 1 [A/m] - NM1
        self.M_SAT_N2 = 505.09                # Saturization magnetization component 2 [A/m] - NM2
        self.M_SAT_M = 20.11*10**5            # Saturization magnetization component 3 [A/m] - M
        self.H_CRIT_N1 = 300*10**3            # Critical magnetic field strength where NM1 is saturated [A/m]
        self.H_CRIT_N2 = 300*10**3            # Critical magnetic field strength where NM2 is saturated [A/m]
        self.H_CRIT_M = 200*10**3             # Critical magnetic field strength where M is saturated [A/m]
        self.XI_N1 = self.M_SAT_N1/self.H_CRIT_N1    # Magnetic susceptibility component 1 [-] (linear approximation)
        self.XI_N2 = self.M_SAT_N2/self.H_CRIT_N2    # Magnetic susceptibility component 2 [-] (linear approximation)  
        self.XI_M = self.M_SAT_M/self.H_CRIT_M       # Magnetic susceptibility component 3 [-] (linear approximation)
        
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
        self.P1 = 0                           # Power law breakage parameter Pandya, Spielmann 
        self.P2 = 1                           # Power law breakage parameter Pandya, Spielmann 
            
        ## GENERAL constants
        self.KT = 1.38*(10**-23)*293          # k*T in [J]
        self.MU0 = 4*math.pi*10**-7           # Permeability constant vacuum [N/A²]
        self.EPS0 = 8.854*10**-12             # Permettivity constant vacuum [F/m]
        self.EPSR = 80                        # Permettivity material factor [-]
        self.EPS = self.EPSR*self.EPS0
        self.E = 1.602*10**-19                # Electron charge [C]    
        self.NA = 6.022*10**23                # Avogadro number [1/mol]
        self.MU_W = 10**-3                    # Viscosity water [Pa*s]
        
        ## EXPERIMENTAL / PROCESS parameters:
        self.t_exp = t_exp                       # Agglomeration time [min]
        self.I = 1e-3*1e3                     # Ionic strength [mol/m³] - CARE: Standard unit is mol/L
        self.c_mag_exp = 0.01                 # Volume concentration of magnetic particles [Vol-%] 
        self.Psi_c1_exp = 1                   # Concentration ratio component 1 (V_NM1/V_M) [-] 
        self.Psi_c2_exp = 1                   # Concentration ratio component 2 (V_NM2/V_M) [-] 
        self.cv_1 = self.c_mag_exp*self.Psi_c1_exp   # Volume concentration of NM1 particles [Vol-%] 
        self.cv_2 = self.c_mag_exp*self.Psi_c2_exp   # Volume concentration of NM2 particles [Vol-%] 
        self.n_exp = 250                      # Rotary speed [1/min]
        self.PROCESS_VOL = 0.025              # OLD, had influence! Process volume [L] 
        self.B0 = 0                           # Magnetic Field strength [T]
        self.G = 1                            # Shear rate [1/s]. Can be defined dependent on rotary speed, 
                                            # e.g. G=(1400-354)*(n_exp-100)/(250-100)+354
        self.CORR_BETA = 1e6*2.5e-5           # Correction Term for collision frequency [-]. Can be defined
                                            # dependent on rotary speed, e.g. ((corr_beta250-corr_beta100)*(n_exp-100)/(250-100)+corr_beta100)

        self.V01 = self.cv_1*1e-2             # Total volume concentration of component 1 [m³/m³] - NM1
        self.N01 = 3*self.V01/(4*math.pi*self.R01**3)     # Total number concentration of primary particles component 1 [1/m³] - NM1 (if no PSD)
        self.V02 = self.cv_2*1e-2             # Total volume concentration of component 2 [m³/m³] - NM2
        self.N02 = 3*self.V02/(4*math.pi*self.R02**3)     # Total number concentration of primary particles component 2 [1/m³] - NM2 (if no PSD)
        self.V03 = self.c_mag_exp*1e-2        # Total volume concentration of component 3 [m³/m³] - M
        self.N03 = 3*self.V03/(4*math.pi*self.R03**3)     # Total number concentration of primary particles component 1 [1/m³] - M (if no PSD)   
        self.N_INF = self.NA*self.I           # Total number concentration of ions [1/m³] 
        self.NUM_T = round(self.t_exp*60/self.DEL_T)      # Number of timesteps for calculation [-]
        self.t_vec = np.arange(self.NUM_T+1)*self.DEL_T        
        
        # Initialize **attr
        for key, value in attr.items():
            setattr(self, key, value)
            
        # Initialize from file
        if file is not None:
            
            params = np.load(file, allow_pickle=True).item()        
            for i in params.keys(): 
                setattr(self,i,params[i])

            # Reset dimension
            self.dim = dim
            
    ### ------ Static Methods ------ ### 
    
    ## Initialize PSD
    @staticmethod
    def initialize_psd(r,psd_data,v0,x_init=None,Q_init=None):
        
        from scipy.interpolate import interp1d
        import sys
        
        ## OUTPUT-parameters:
        # n: NUMBER concentration vector corresponding to r (for direct usage in N
        # vector of population balance calculation
        
        ## INPUT-parameters:
        # r: Particle size grid on which the PSD should be initialized. NOTE: This
        #    vector contains RADII (and NOT diameters)
        # PSD_data: Complete path (including filename) to datafile in which the PSD is saved. 
        #           This file should only contain 2 variables: Q_PSD and x_PSD.
        #           Here, x_PSD contains diameters (standard format of PSD)
        # v0: Total VOLUME the distribution should be scaled to
                
        
        # If x and Q are not directly given import from file
        if x_init is None and Q_init is None:
            # Import x values from dictionary save in psd_data
            psd_dict = np.load(psd_data,allow_pickle=True).item()
            x = psd_dict['x_PSD']
            
            ## Initializing the variables
            n = np.zeros(len(r)) 
            q = np.zeros(len(x)) 
            
            if 'Q_PSD' not in psd_dict and 'q_PSD' not in psd_dict:
                sys.exit("ERROR: Neither Q_PSD nor q_PSD given in distribution file. Exiting..")
            
            if 'Q_PSD' in psd_dict:
                # Load Q if given
                Q = psd_dict['Q_PSD']
            if 'q_PSD' in psd_dict:
                # Load q if given
                q = psd_dict['q_PSD']
            else:
                # If q is not given: Calculate from Q. Note that q[0] will always be zero
                for i in range(1,len(x)):
                    q[i] = (Q[i]-Q[i-1])/(x[i]-x[i-1]) 
                    
        else:
            ## Initializing the variables
            n = np.zeros(len(r)) 
            q = np.zeros(len(x_init)) 
            
            x = x_init
            
            for i in range(1,len(x)):
                q[i] = (Q_init[i]-Q_init[i-1])/(x[i]-x[i-1])
            
        # Transform x from diameter to radius information. Also transform q
        x = x/2
        #q = q/2
        
        # Interpolate q on r grid and normalize it to 1 (account for numerical error)
        # If the ranges don't match well, insert 0. This is legit since it is the density
        # distribution
        f_q = interp1d(x,q,bounds_error=False,fill_value=0)
        q_r = f_q(r)
                
        #q_r(math.isnan(q_r)) = 0
        q_r = q_r/np.trapz(q_r,r)
        
        # Temporary r vector. Add lower and upper boarder (given by x_PSD) 
        # This allows calculation of the first and last entry of q / r.
        rt = np.zeros(len(r)+2) 
        qt = np.zeros(len(r)+2)  
        rt[0] = min(min(x),min(r))
        rt[-1] = max(max(x),max(r))
        rt[1:-1] = r 
        qt[1:-1] = q_r;
        
        # Calculate concentration vector
        for i in range(1,len(rt)-1):
            v_total_tmp = v0*qt[i]*((rt[i+1]-rt[i])/2+(rt[i]-rt[i-1])/2) # Calculated with DIFFERENCE (i+1), (i-1)
            #v_one_tmp = (4/3)*pi*((rt[i+1]+rt[i-1])*1e-6/2)^3; # Calculated with MEAN (i+1), (i-1) 
            v_one_tmp = (4/3)*np.pi*r[i-1]**3; # Calculated with MEAN (i+1), (i-1) 
            n[i-1] = v_total_tmp/v_one_tmp;
        
        # Eliminate sub and near zero values (sub-thrshold)
        thr = 1e-5
        n[n<thr*np.mean(n)] = 0
        
        return n   

    ## Initialize PSD
    @staticmethod
    def new_initialize_psd(d,psd_data,v0,x_init=None,Q_init=None):
            
        from scipy.interpolate import interp1d
        import sys
        
        ## OUTPUT-parameters:
        # n: NUMBER concentration vector corresponding to 2*r (for direct usage in N
        # vector of population balance calculation
        
        ## INPUT-parameters:
        # d: Particle size grid on which the PSD should be initialized. NOTE: This
        #    vector contains diameters
        # PSD_data: Complete path (including filename) to datafile in which the PSD is saved. 
        #           This file should only contain 2 variables: Q_PSD and x_PSD.
        #           Here, x_PSD contains diameters (standard format of PSD)
        # v0: Total VOLUME the distribution should be scaled to
                
        
        # If x and Q are not directly given import from file
        if x_init is None and Q_init is None:
            # Import x values from dictionary save in psd_data
            psd_dict = np.load(psd_data,allow_pickle=True).item()
            x = psd_dict['x_PSD']
            
            ## Initializing the variables
            n = np.zeros(len(d)) 
            
            if 'Q_PSD' not in psd_dict and 'q_PSD' not in psd_dict:
                sys.exit("ERROR: Neither Q_PSD nor q_PSD given in distribution file. Exiting..")
            
            if 'Q_PSD' in psd_dict:
                # Load Q if given
                Q = psd_dict['Q_PSD']
                    
        else:
            ## Initializing the variables
            n = np.zeros(len(d)) 
            x = x_init
            Q = Q_init
        
        # Interpolate Q on d grid and normalize it to 1 (account for numerical error)
        # If the ranges don't match well, insert 0. This is legit since it is the density
        # distribution
        f_Q = interp1d(x,Q,bounds_error=False,fill_value=0)
        Q_d = np.zeros(len(d)+1)
        Q_d[1:] = f_Q(d)
                
        #Q_d(math.isnan(Q_d)) = 0
        Q_d = Q_d/Q_d.max()
        
        for i in range(1, len(d)+1):
            v_total_tmp = max(v0 * (Q_d[i] - Q_d[i-1]), 0)
            v_one_tmp = (1/6)*np.pi*d[i-1]**3
            n[i-1] = v_total_tmp/v_one_tmp
        
        # # Eliminate sub and near zero values (sub-thrshold)
        # thr = 1e-5
        # n[n<thr*np.mean(n)] = 0
        
        return n    
    ## Plot 2D-distribution:
    @staticmethod
    def plot_N2D(N,V,V0_tot,ax=None,fig=None,close_all=False,scl_a4=1,figsze=[12.8*1.05,12.8],THR_N=1e-4,
                 exp_file=None,t_stamp=None):
                    
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import plotter.plotter as pt
        from plotter.KIT_cmap import KIT_black_green_white
        
        if close_all:
            plt.close('all')
            
        if fig is None:
            pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
            fig = plt.figure()    
            ax = fig.add_subplot(1,1,1)
        
        # Clear axes
        ax.cla()
        
        # Calculate meshgrid for plot
        _ii, _jj = np.meshgrid(np.arange(len(N)),np.arange(len(N)))
                        
        # Calculate relative Volume and apply threshold
        Nr = 100*(N*V)/V0_tot
        Nr[Nr<THR_N]=THR_N
        
        # Color plot
        cp = ax.pcolor(_ii,_jj,Nr,norm=LogNorm(vmin=1e3*THR_N, vmax=100),edgecolors=[0.5,0.5,0.5],
                       shading='auto',cmap=KIT_black_green_white.reversed()) 
        
        # Colorbar / legend
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(cp,cax)
        c = cb.set_label('Total volume $V$ / $\%$')
       
        # Format the plot
        ax.set_ylabel('Partial volume comp. 1 $V_{1}$ ($i$) / $-$')  # Add an x-label to the axes.
        ax.set_xlabel('Partial volume comp. 2 $V_{2}$ ($j$) / $-$')  # Add a y-label to the axes.
        
        # Plot time textbox if t_stamp is provided
        if t_stamp is not None:
            ax.text(0.05, 0.95, f"$t={t_stamp}$", transform=ax.transAxes, fontsize=10*1.6,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='w', alpha=1))
        
        #Remove whitespace around plot
        plt.tight_layout()
        
        # Plot frame
        if exp_file is not None: pt.plot_export(exp_file)
        
        return ax, cb, fig
    
### ------ PRE-COMPILED FUNCTIONS (NUMBA JIT) ------ ### 
 
# Define np.heaviside for JIT compilation
    

   