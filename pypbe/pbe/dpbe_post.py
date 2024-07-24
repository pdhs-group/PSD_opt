# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:11:33 2024

@author: Administrator
"""
import numpy as np
from ..utils.func.func_math import float_in_list, float_equal, isZero

class PBEPost():
    def __init__(self, pop):
        self.pop = pop
    ## Return particle size distribution on fixed grid 
    def return_distribution(self, comp='all', t=0, N=None, flag='all'):
        """
        Returns the results of Volume-based PSD(Particle density distribution) of a time step.
        
        Parameters
        ----------
        comp : `str`, optional
            Which particles are counted. The case for a specific component has not yet been coded.
        t : `int`, optional
            Time step to return.
        N : `array_like`, optional
            Array holding the calculated particle number distribution. 
            If is not provided, use the one from the class instance * ``pop.N( )``.
        flag: `str`, optional
            Which data form the PSD is returned. Default is 'all'. Options include:
            - 'x_uni': Unique particle diameters(unique particle size class)
            - 'q3': Volumen density distribution
            - 'Q3': Cumulative distribution
            - 'x_10': Particle size corresponding to 10% cumulative distribution
            - 'x_50': Particle size corresponding to 50% cumulative distribution
            - 'x_90': Particle size corresponding to 90% cumulative distribution
            - 'sumvol_uni': Cumulative particle volumen on each particle size class
            - 'all': Returns all the above data
        
        """
        def unique_with_tolerance(V, tol=1e-3):
            V_sorted = np.sort(V)
            V_unique = [V_sorted[0]]
            
            for V_val in V_sorted[1:]:
                if not np.isclose(V_val, V_unique[-1], atol=tol*V_sorted[0], rtol=0):
                    V_unique.append(V_val)
            return np.array(V_unique)
        # If no N is provided use the one from the class instance
        if N is None:
            N = self.pop.N
        
        # Extract unique values that are NOT -1 (border)
        v_uni = np.setdiff1d(self.pop.V,[-1])
        # v_uni = unique_with_tolerance(v_uni)
        q3 = np.zeros(len(v_uni))
        x_uni = np.zeros(len(v_uni))
        sumvol_uni = np.zeros(len(v_uni))
        
        if comp == 'all':
            # Loop through all entries in V and add volume concentration to specific entry in sumvol_uni
            if self.pop.dim == 1:
                for i in range(self.pop.NS):
                    # if self.pop.V[i] in v_uni:
                    sumvol_uni[v_uni == self.pop.V[i]] += self.pop.V[i]*N[i,t] 
                        
            if self.pop.dim == 2:
                for i in range(self.pop.NS):
                    for j in range(self.pop.NS):
                        # if self.pop.V[i,j] in v_uni:
                        sumvol_uni[v_uni == self.pop.V[i,j]] += self.pop.V[i,j]*N[i,j,t]

            if self.pop.dim == 3:
                for i in range(self.pop.NS):
                    for j in range(self.pop.NS):
                        for k in range(self.pop.NS):
                            # if self.pop.V[i,j,k] in v_uni:
                            sumvol_uni[v_uni == self.pop.V[i,j,k]] += self.pop.V[i,j,k]*N[i,j,k,t]
            ## Preventing division by zero
            sumvol_uni[sumvol_uni<0] = v_uni[1] * 1e-3
            ## Convert unit m into um
            sumvol_uni *= 1e18
            sumV = np.sum(sumvol_uni)
            # Calculate diameter array
            x_uni[1:]=(6*v_uni[1:]/np.pi)**(1/3)*1e6
            
            # Calculate sum and density distribution
            Q3 = np.cumsum(sumvol_uni)/sumV
            q3[1:] = np.diff(Q3) / np.diff(x_uni)
            
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
        """
        Returns the results of Number-based PSD(Particle density distribution) of a time step.
        
        Parameters
        ----------
        comp : `str`, optional
            Which particles are counted. The case for a specific component has not yet been coded.
        t : `int`, optional
            Time step to return.
        N : `array_like`, optional
            Array holding the calculated particle number distribution. 
            If is not provided, use the one from the class instance * ``pop.N( )``.
        flag: `str`, optional
            Which data form the PSD is returned. Default is 'all'. Options include:
            - 'x_uni': Unique particle diameters(unique particle size class)
            - 'q0': Number density distribution
            - 'Q0': Cumulative distribution
            - 'x_10': Particle size corresponding to 10% cumulative distribution
            - 'x_50': Particle size corresponding to 50% cumulative distribution
            - 'x_90': Particle size corresponding to 90% cumulative distribution
            - 'sumN_uni': Cumulative particle concentration on each particle size class
            - 'all': Returns all the above data
        
        """
        # If no N is provided use the one from the class instance
        if N is None:
            N = self.pop.N
        
        # Extract unique values that are NOT -1(border)
        # At the same time, v_uni will be rearranged according to size.
        v_uni = np.setdiff1d(self.pop.V,[-1])

        q0 = np.zeros(len(v_uni))
        x_uni = np.zeros(len(v_uni))
        sumN_uni = np.zeros(len(v_uni))
        
        if comp == 'all':
            # Loop through all entries in V and add number concentration to specific entry in sumN_uni
            if self.pop.dim == 1:
                for i in range(self.pop.NS):
                    if float_in_list(self.pop.V[i], v_uni) and (not N[i,t] < 0):
                        sumN_uni[v_uni == self.pop.V[i]] += N[i,t] 
                        
            if self.pop.dim == 2:
                for i in range(self.pop.NS):
                    for j in range(self.pop.NS):
                        if float_in_list(self.pop.V[i,j], v_uni) and (not N[i,j,t] < 0):
                            sumN_uni[v_uni == self.pop.V[i,j]] += N[i,j,t]

            if self.pop.dim == 3:
                for i in range(self.pop.NS):
                    for j in range(self.pop.NS):
                        for k in range(self.pop.NS):
                            if float_in_list(self.pop.V[i,j,k], v_uni) and (not N[i,j,t] < 0):
                                sumN_uni[v_uni == self.pop.V[i,j,k]] += N[i,j,k,t]
            sumN_uni[sumN_uni<0] = 0                    
            sumN = np.sum(sumN_uni)
            # Calculate diameter array and convert into mm
            x_uni[1:]=(6*v_uni[1:]/np.pi)**(1/3)*1e6
            
            # Calculate sum and density distribution
            Q0 = np.cumsum(sumN_uni)/sumN
            q0[1:] = np.diff(Q0) / np.diff(x_uni)
            
            # Retrieve x10, x50 and x90 through interpolation
            x_10=np.interp(0.1, Q0, x_uni)
            x_50=np.interp(0.5, Q0, x_uni)
            x_90=np.interp(0.9, Q0, x_uni)
            
        else:
            print('Case for comp not coded yet. Exiting')
            return
    
        outputs = {
        'x_uni': x_uni,
        'q0': q0,
        'Q0': Q0,
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
        if self.pop.dim == 1:
            if t is None:
                return np.sum(self.pop.N,axis=0)
            else:
                return np.sum(self.pop.N[:,t])  
            
        # 2-D case    
        elif self.pop.dim == 2:
            if t is None:
                return np.sum(self.pop.N,axis=(0,1))
            else:
                return np.sum(self.pop.N[:,:,t])
        
        # 3-D case    
        elif self.pop.dim == 3:
            if t is None:
                return np.sum(self.pop.N,axis=(0,1,2))
            else:
                return np.sum(self.pop.N[:,:,:,t])
    
    # Calculate distribution moments mu(i,j,t)
    def calc_mom_t(self):
        mu = np.zeros((3,3,len(self.pop.t_vec)))
        
        # Time loop
        for t in range(len(self.pop.t_vec)):
            for i in range(3):
                if self.pop.dim == 1:
                    self.pop.N[0,:] = 0.0
                    mu[i,0,t] = np.sum(self.pop.V**i*self.pop.N[:,t])
                    
                # The following only applies for 2D and 3D case
                # Moment between component 1 and 3
                else:
                    for j in range(3):
                        if self.pop.dim == 2:
                            self.pop.N[0,0,:] = 0.0
                            mu[i,j,t] = np.sum((self.pop.X1_vol*self.pop.V)**i*(self.pop.X3_vol*self.pop.V)**j*self.pop.N[:,:,t])
                        if self.pop.dim == 3:
                            self.pop.N[0,0,0,:] = 0.0
                            mu[i,j,t] = np.sum((self.pop.X1_vol*self.pop.V)**i*(self.pop.X3_vol*self.pop.V)**j*self.pop.N[:,:,:,t])
                        
        return mu
    
    ## Save Variables to file:
    def save_vars(self,file):
        np.save(file,vars(self))