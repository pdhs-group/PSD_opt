# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:02:27 2024

@author: Administrator
"""
import numpy as np

def init_mag_sep_params(self):
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
    self.M_SAT_N1 = 505.09                # Saturization magnetization component 1 [A/m] - NM1
    self.M_SAT_N2 = 505.09                # Saturization magnetization component 2 [A/m] - NM2
    self.M_SAT_M = 20.11*10**5            # Saturization magnetization component 3 [A/m] - M
    self.H_CRIT_N1 = 300*10**3            # Critical magnetic field strength where NM1 is saturated [A/m]
    self.H_CRIT_N2 = 300*10**3            # Critical magnetic field strength where NM2 is saturated [A/m]
    self.H_CRIT_M = 200*10**3             # Critical magnetic field strength where M is saturated [A/m]
    self.XI_N1 = self.M_SAT_N1/self.H_CRIT_N1    # Magnetic susceptibility component 1 [-] (linear approximation)
    self.XI_N2 = self.M_SAT_N2/self.H_CRIT_N2    # Magnetic susceptibility component 2 [-] (linear approximation)  
    self.XI_M = self.M_SAT_M/self.H_CRIT_M       # Magnetic susceptibility component 3 [-] (linear approximation)
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