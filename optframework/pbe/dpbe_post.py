# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:11:33 2024

@author: Administrator
"""
import numpy as np
import math
from ..utils.func.func_math import float_in_list, float_equal, isZero

def init_post_params(self):
    pass

def calc_v_uni(self, unit_trans=True):
    """
    Calculate unique volume values for a given DPBESolver.
    """
    if unit_trans:
        ## Converting cubic meters to cubic micrometers 
        unit_scale = 1e18
    else:
        unit_scale = 1
    return np.setdiff1d(self.V, [-1])*unit_scale

def calc_x_uni(self, unit_trans=True):
    """
    Calculate unique particle diameters from volume values for a given DPBESolver.
    """
    v_uni = self.calc_v_uni(unit_trans)
    # Because the length unit in the experimental data is millimeters 
    # and in the simulation it is meters, so it needs to be converted 
    # before use.
    # x_uni = np.zeros(len(v_uni))
    x_uni = (6*v_uni/np.pi)**(1/3)
    return x_uni

def calc_Q3(self, x_uni, q3=None, sum_uni=None):
    """
    Calculate the cumulative distribution Q3 from q3 or sum_uni distribution data.
    """
    Q3 = np.zeros_like(q3) if q3 is not None else np.zeros_like(sum_uni)
    if q3 is None:
        Q3 = np.cumsum(sum_uni)/sum_uni.sum()
    else:
        for i in range(1, len(Q3)):
                # Q3[i] = np.trapz(q3[:i+1], x_uni[:i+1])
                ## Euler backward
                Q3[i] = Q3[i-1] + q3[i] * (x_uni[i] - x_uni[i-1])
    return Q3

def calc_sum_uni(self, Q3, sum_total):
    """
    Calculate the sum_uni distribution from the Q3 cumulative distribution and total sum.
    """
    sum_uni = np.zeros_like(Q3)
    # sum_uni[0] = sum_total * Q3[0]
    for i in range(1, len(Q3)):
        sum_uni[i] = sum_total * max((Q3[i] -Q3[i-1] ), 0)
    return sum_uni

def calc_q3(self, Q3, x_uni):
    """
    Calculate the q3 distribution from the Q3 cumulative distribution.
    """
    q3 = np.zeros_like(Q3)
    q3[1:] = np.diff(Q3) / np.diff(x_uni)
    return q3
    
def re_calc_distribution(self, x_uni, q3=None, sum_uni=None, flag='all'):
    """
    Recalculate distribution metrics for a given DPBESolver and distribution data.
    
    Can operate on either q3 or sum_uni distribution data to calculate Q3, q3, and particle
    diameters corresponding to specific percentiles (x_10, x_50, x_90).
    
    Parameters
    ----------
    x_uni : `array`
        Unique particle diameters.
    q3 : `array`, optional
        q3 distribution data.
    sum_uni : `array`, optional
        sum_uni distribution data.
    flag : `str`, optional
        Specifies which metrics to return. Defaults to 'all', can be a comma-separated list
        of 'q3', 'Q3', 'x_10', 'x_50', 'x_90'.
    
    Returns
    -------
    `tuple`
        Selected distribution metrics based on the `flag`. Can include q3, Q3, x_10, x_50,
        and x_90 values.
    """
    if q3 is not None:
        q3_new = q3
        Q3_new = np.apply_along_axis(lambda q3_slice: 
                                 self.calc_Q3(x_uni, q3=q3_slice), 0, q3)

    else:
        Q3_new = np.apply_along_axis(lambda sum_uni_slice: 
                                 self.calc_Q3(x_uni, sum_uni=sum_uni_slice), 0, sum_uni)
        q3_new = np.apply_along_axis(lambda Q3_slice: 
                                      self.calc_q3(Q3_slice, x_uni), 0, Q3_new)
    
    dim = q3_new.shape[1]
    x_10_new = np.zeros(dim)
    x_50_new = np.zeros(dim)
    x_90_new = np.zeros(dim)
    for idx in range(dim):
        x_10_new[idx] = np.interp(0.1, Q3_new[:, idx], x_uni)
        x_50_new[idx] = np.interp(0.5, Q3_new[:, idx], x_uni)
        x_90_new[idx] = np.interp(0.9, Q3_new[:, idx], x_uni)
    outputs = {
    'q3': q3_new,
    'Q3': Q3_new,
    'x_10': x_10_new,
    'x_50': x_50_new,
    'x_90': x_90_new,
    }
    
    if flag == 'all':
        return outputs.values()
    else:
        flags = flag.split(',')
        return tuple(outputs[f.strip()] for f in flags if f.strip() in outputs)
    
## Return particle size distribution on fixed grid 
def return_distribution(self, comp='all', t=0, N=None, flag='all', rel_q=False):
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
        ## When using `uni_grid`, it was found that some particles with the same volume are 
        ## treated as having different volumes due to floating-point precision issues. 
        ## Therefore, `np.isclose` needs to be used for comparing floating-point numbers.
        V_sorted = np.sort(V)
        V_unique = [V_sorted[0]]
        
        for V_val in V_sorted[1:]:
            if not np.isclose(V_val, V_unique[-1], atol=tol*V_sorted[0], rtol=0):
                V_unique.append(V_val)
        return np.array(V_unique)
    # If no N is provided use the one from the class instance
    if N is None:
        N = self.N
    
    # Extract unique values that are NOT -1 (border)
    if self.disc == 'geo':
        v_uni = np.setdiff1d(self.V,[-1])
    else:
        v_uni = unique_with_tolerance(v_uni)
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
    if rel_q:
        max_q3 = max(q3)
        if max_q3 != 0:
            q3 = q3/max_q3
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

def return_num_distribution(self, comp='all', t=0, N=None, flag='all', rel_q=False):
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
    def unique_with_tolerance(V, tol=1e-3):
        ## When using `uni_grid`, it was found that some particles with the same volume are 
        ## treated as having different volumes due to floating-point precision issues. 
        ## Therefore, `np.isclose` needs to be used for comparing floating-point numbers.
        V_sorted = np.sort(V)
        V_unique = [V_sorted[0]]
        
        for V_val in V_sorted[1:]:
            if not np.isclose(V_val, V_unique[-1], atol=tol*V_sorted[0], rtol=0):
                V_unique.append(V_val)
        return np.array(V_unique)
    # If no N is provided use the one from the class instance
    if N is None:
        N = self.N
    
    # Extract unique values that are NOT -1 (border)
    if self.disc == 'geo':
        v_uni = np.setdiff1d(self.V,[-1])
    else:
        v_uni = unique_with_tolerance(v_uni)

    q0 = np.zeros(len(v_uni))
    x_uni = np.zeros(len(v_uni))
    sumN_uni = np.zeros(len(v_uni))
    
    if comp == 'all':
        # Loop through all entries in V and add number concentration to specific entry in sumN_uni
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
    if rel_q:
        max_q0 = max(q0)
        if max_q0 != 0:
            q0 = q0/max_q0
            
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

# Calculate distribution moments related to particle volume mu(i,j,t)
def calc_mom_t(self):
    mu = np.zeros((3,3,len(self.t_vec)))
    
    # Time loop
    for t in range(len(self.t_vec)):
        for i in range(3):
            if self.dim == 1:
                self.N[0,:] = 0.0
                mu[i,0,t] = np.sum(self.V**i*self.N[:,t])
                
            # The following only applies for 2D and 3D case
            # Moment between component 1 and 3
            else:
                for j in range(3):
                    if self.dim == 2:
                        self.N[0,0,:] = 0.0
                        mu[i,j,t] = np.sum((self.X1_vol*self.V)**i*(self.X3_vol*self.V)**j*self.N[:,:,t])
                    if self.dim == 3:
                        self.N[0,0,0,:] = 0.0
                        mu[i,j,t] = np.sum((self.X1_vol*self.V)**i*(self.X3_vol*self.V)**j*self.N[:,:,:,t])
                    
    return mu

# Calculate distribution moments related to particle radii mu_r(i,j,t)
def calc_mom_r_t(self):
    mu_r = np.zeros((3,3,len(self.t_vec)))
    
    # Time loop
    for t in range(len(self.t_vec)):
        for i in range(3):
            if self.dim == 1:
                self.N[0,:] = 0.0
                mu_r[i,0,t] = np.sum(self.R**i*self.N[:,t])
                
            # The following only applies for 2D and 3D case
            # Moment between component 1 and 3
            else:
                R1 = (self.X1_vol*self.V(4*math.pi))**(1/3)
                R3 = (self.X3_vol*self.V(4*math.pi))**(1/3)
                for j in range(3):
                    if self.dim == 2:
                        self.N[0,0,:] = 0.0
                        mu_r[i,j,t] = np.sum((R1)**i*(R3)**j*self.N[:,:,t])
                    if self.dim == 3:
                        self.N[0,0,0,:] = 0.0
                        mu_r[i,j,t] = np.sum((R1)**i*(R3)**j*self.N[:,:,:,t])
                    
    return mu_r

## Save Variables to file:
def save_vars(self,file):
    np.save(file,vars(self))