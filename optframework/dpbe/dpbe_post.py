# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:11:33 2024

@author: Administrator
"""
import numpy as np
import math
from optframework.utils.func.func_math import float_in_list, float_equal, isZero

class DPBEPost:
    def __init__(self, base):
        self.base = base
        
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
        return np.setdiff1d(self.base.V, [-1])*unit_scale
    
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
    
    def calc_Qx(self, x_uni, qx=None, sum_uni=None):
        """
        Calculate the cumulative distribution Qx from qx or sum_uni distribution data.
        """
        Qx = np.zeros_like(qx) if qx is not None else np.zeros_like(sum_uni)
        if qx is None:
            Qx = np.cumsum(sum_uni)/sum_uni.sum()
        else:
            for i in range(1, len(Qx)):
                    # Qx[i] = np.trapz(qx[:i+1], x_uni[:i+1])
                    ## Euler backward
                    Qx[i] = Qx[i-1] + qx[i] * (x_uni[i] - x_uni[i-1])
        return Qx
    
    def calc_sum_uni(self, Qx, sum_total):
        """
        Calculate the sum_uni distribution from the Qx cumulative distribution and total sum.
        """
        sum_uni = np.zeros_like(Qx)
        # sum_uni[0] = sum_total * Qx[0]
        for i in range(1, len(Qx)):
            sum_uni[i] = sum_total * max((Qx[i] -Qx[i-1] ), 0)
        return sum_uni
    
    def calc_qx(self, Qx, x_uni):
        """
        Calculate the qx distribution from the Qx cumulative distribution.
        """
        qx = np.zeros_like(Qx)
        qx[1:] = np.diff(Qx) / np.diff(x_uni)
        return qx
        
    def re_calc_distribution(self, x_uni, qx=None, Qx=None, sum_uni=None, flag='all'):
        """
        Recalculate distribution metrics for a given DPBESolver and distribution data.
        
        Can operate on either qx or sum_uni distribution data to calculate Qx, qx, and particle
        diameters corresponding to specific percentiles (x_10, x_50, x_90).
        
        Parameters
        ----------
        x_uni : `array`
            Unique particle diameters.
        qx : `array`, optional
            qx distribution data.
        sum_uni : `array`, optional
            sum_uni distribution data.
        flag : `str`, optional
            Specifies which metrics to return. Defaults to 'all', can be a comma-separated list
            of 'qx', 'Qx', 'x_10', 'x_50', 'x_90'.
        
        Returns
        -------
        `tuple`
            Selected distribution metrics based on the `flag`. Can include qx, Qx, x_10, x_50,
            and x_90 values.
        """
        if qx is not None:
            qx_new = qx
            Qx_new = np.apply_along_axis(lambda qx_slice: 
                                     self.calc_Qx(x_uni, qx=qx_slice), 0, qx)
    
        else:
            if Qx is None:
                Qx_new = np.apply_along_axis(lambda sum_uni_slice: 
                                         self.calc_Qx(x_uni, sum_uni=sum_uni_slice), 0, sum_uni)
            else:
                Qx_new = Qx
            qx_new = np.apply_along_axis(lambda Qx_slice: 
                                          self.calc_qx(Qx_slice, x_uni), 0, Qx_new)
                
        x_weibull, y_weibull = np.apply_along_axis(lambda Qx_slice: 
                                      self.calc_weibull(Qx_slice, x_uni), 0, Qx_new)
        dim = qx_new.shape[1]
        x_10_new = np.zeros(dim)
        x_50_new = np.zeros(dim)
        x_90_new = np.zeros(dim)
        for idx in range(dim):
            x_10_new[idx] = np.interp(0.1, Qx_new[:, idx], x_uni)
            x_50_new[idx] = np.interp(0.5, Qx_new[:, idx], x_uni)
            x_90_new[idx] = np.interp(0.9, Qx_new[:, idx], x_uni)
            
        outputs = {
        'qx': qx_new,
        'Qx': Qx_new,
        'x_10': x_10_new,
        'x_50': x_50_new,
        'x_90': x_90_new,
        'x_weibull': x_weibull,
        'y_weibull': y_weibull,
        }
        
        if flag == 'all':
            return outputs.values()
        else:
            flags = flag.split(',')
            return tuple(outputs[f.strip()] for f in flags if f.strip() in outputs)
    
    def return_distribution(self, comp='all', t=0, N=None, flag='all', rel_q=False, q_type='q3'):
        """
        Returns the particle size distribution (PSD) on a fixed grid.
        
        This function computes the PSD based on a chosen quantity:
          - 'q0': Number-based PSD (weight = N, i.e., V^0 × N)
          - 'q3': Volume-based PSD (weight = V * N, i.e., V^1 × N)
          - 'q6': Square-volume PSD (weight = V^2 * N)
          
        The returned values include:
          - 'x_uni': Unique particle diameters (in µm)
          - 'qx': The density distribution corresponding to the chosen q_type
          - 'Qx': Cumulative distribution (0–1)
          - 'x_10', 'x_50', 'x_90': Particle diameters corresponding to 10%, 50%, and 90% cumulative distribution, respectively (in µm)
          - 'sum_uni': The cumulative weight at each particle size class (number, volume, or squared-volume)
        
        Note on unit conversion:
          - For q3, original volumes (in m³) are converted to µm³ by multiplying by 1e18.
          - For q6, squared volumes (in m⁶) are converted to µm⁶ by multiplying by 1e36.
          - For q0, no conversion is necessary.
        
        Parameters
        ----------
        comp : str, optional
            Which particles are counted. Currently, only 'all' is implemented.
        t : int, optional
            Time step index to return.
        N : array_like, optional
            The particle number distribution. If not provided, the class instance self.N is used.
        flag : str, optional
            Specifies which data to return. Options include:
               - 'x_uni': Unique particle diameters
               - 'qx': Density distribution (according to q_type)
               - 'Qx': Cumulative distribution
               - 'x_10': Particle diameter for 10% cumulative distribution
               - 'x_50': Particle diameter for 50% cumulative distribution
               - 'x_90': Particle diameter for 90% cumulative distribution
               - 'sum_uni': Cumulative weight in each particle size class
               - 'all': Returns all of the above data.
        rel_q : bool, optional
            If True, the density distribution qx will be normalized by its maximum value.
        q_type : str, optional
            The type of distribution to compute. Should be one of:
                - 'q0' for number-based distribution,
                - 'q3' for volume-based distribution,
                - 'q6' for square-volume distribution.
        
            The default is 'q3'.
        
        Returns
        -------
        A tuple containing the requested arrays. If flag == 'all' then the order of return is: 
        (x_uni, qx, Qx, x_10, x_50, x_90, sum_uni)
        Otherwise, only the items corresponding to the keys specified in flag (comma-separated) are returned.
        """
        base = self.base
        def unique_with_tolerance(V, tol=1e-3):
            # Sort the flattened array and remove duplicate values within a tolerance,
            # using np.isclose to compare floating point numbers.
            V_sorted = np.sort(V.flatten())
            V_unique = [V_sorted[0]]
            for V_val in V_sorted[1:]:
                if not np.isclose(V_val, V_unique[-1], atol=tol * V_sorted[0], rtol=0):
                    V_unique.append(V_val)
            return np.array(V_unique)
        
        # If no N is provided, use the one from the class instance.
        if N is None:
            N = base.N
    
        # Extract unique V values from base.V (ignoring boundary value -1)
        if base.disc == 'geo':
            v_uni = np.setdiff1d(base.V.flatten(), np.array([-1.0]))
        else:
            v_uni = unique_with_tolerance(base.V)
            
        num_classes = len(v_uni)
        qx = np.zeros(num_classes)
        x_uni = np.zeros(num_classes)
        sum_uni = np.zeros(num_classes)
        ATOL = v_uni[1] * 1e-3
        
        # Determine the exponent for V based on q_type.
        # q0: exponent 0; q3: exponent 1; q6: exponent 2.
        if q_type not in ['q0', 'q3', 'q6']:
            raise ValueError("Unsupported q_type. Options are 'q0', 'q3', or 'q6'.")
        exp = {'q0': 0, 'q3': 1, 'q6': 2}[q_type]
        
        # Precompute V_power = base.V**exp so that each weight is given by V_power * N.
        V_power = base.V ** exp
    
        # Loop through all grid cells and accumulate weight
        if comp == 'all':
            if base.dim == 1:
                for i in range(base.NS):
                    # For each cell in 1D, weight = V_power[i] * N[i,t]
                    weight = V_power[i] * N[i, t]
                    # Find the index in unique v_uni (based on base.V)
                    idx = np.where(np.isclose(v_uni, base.V[i], atol=ATOL))[0]
                    if idx.size > 0:
                        sum_uni[idx[0]] += weight
            elif base.dim == 2:
                for i in range(base.NS):
                    for j in range(base.NS):
                        weight = V_power[i, j] * N[i, j, t]
                        idx = np.where(np.isclose(v_uni, base.V[i, j], atol=ATOL))[0]
                        if idx.size > 0:
                            sum_uni[idx[0]] += weight
            elif base.dim == 3:
                for i in range(base.NS):
                    for j in range(base.NS):
                        for k in range(base.NS):
                            weight = V_power[i, j, k] * N[i, j, k, t]
                            idx = np.where(np.isclose(v_uni, base.V[i, j, k], atol=ATOL))[0]
                            if idx.size > 0:
                                sum_uni[idx[0]] += weight
            else:
                raise ValueError("Unsupported dimension: {}".format(base.dim))
        else:
            print('Case for comp not coded yet. Exiting')
            return
        
        # Unit conversion: For q3 and q6, convert V units accordingly.
        if q_type == 'q3':
            # Prevent extremely small (or negative) values, assign a small positive value based on v_uni[1]
            if num_classes > 1:
                sum_uni[sum_uni < 0] = v_uni[1] * 1e-3
            sum_uni *= 1e18  # Convert from m³ to µm³.
        elif q_type == 'q6':
            if num_classes > 1:
                sum_uni[sum_uni < 0] = (v_uni[1] ** 2) * 1e-3
            sum_uni *= 1e36  # Convert from m⁶ to µm⁶.
        # For q0, no conversion is needed.
        sum_uni[0] = 0.0
        total_weight = np.sum(sum_uni)
        Qx = np.cumsum(sum_uni) / total_weight if total_weight != 0 else np.zeros(num_classes)
        
        # Calculate the diameter array based on particle volume.
        # Using the spherical volume formula: d = (6V/π)^(1/3), converting meters to micrometers.
        x_uni[0] = 0
        if num_classes > 1:
            x_uni[1:] = (6 * v_uni[1:] / np.pi) ** (1/3) * 1e6
        
        # Compute the density distribution from the cumulative distribution.
        if num_classes > 1:
            qx[1:] = np.diff(Qx) / np.diff(x_uni)
        
        # Obtain the x10, x50, and x90 values by interpolation.
        x_10 = np.interp(0.1, Qx, x_uni)
        x_50 = np.interp(0.5, Qx, x_uni)
        x_90 = np.interp(0.9, Qx, x_uni)
        
        x_weibull, y_weibull = self.calc_weibull(Qx, x_uni)
        
        if rel_q:
            max_qx = np.max(qx)
            if max_qx != 0:
                qx = qx / max_qx
        
        outputs = {
            'x_uni': x_uni,
            'qx': qx,
            'Qx': Qx,
            'x_10': x_10,
            'x_50': x_50,
            'x_90': x_90,
            'sum_uni': sum_uni,
            'x_weibull': x_weibull,
            'y_weibull': y_weibull
        }
        
        if flag == 'all':
            return outputs.values()
        else:
            flags = flag.split(',')
            return tuple(outputs[f.strip()] for f in flags if f.strip() in outputs)
        
    ## Return total number. For t=None return full array, else return total number at time index t 
    def return_N_t(self,t=None):
        base = self.base
        # 1-D case
        if base.dim == 1:
            base.N[0,:] = 0.0
            if t is None:
                return np.sum(base.N,axis=0)
            else:
                return np.sum(base.N[:,t])  
            
        # 2-D case    
        elif base.dim == 2:
            base.N[0,0,:] = 0.0
            if t is None:
                return np.sum(base.N,axis=(0,1))
            else:
                return np.sum(base.N[:,:,t])
        
        # 3-D case    
        elif base.dim == 3:
            base.N[0,0,0,:] = 0.0
            if t is None:
                return np.sum(base.N,axis=(0,1,2))
            else:
                return np.sum(base.N[:,:,:,t])
    
    # Calculate distribution moments related to particle volume mu(i,j,t)
    def calc_mom_t(self):
        base = self.base
        mu = np.zeros((3,3,len(base.t_vec)))
        
        # Time loop
        for t in range(len(base.t_vec)):
            for i in range(3):
                if base.dim == 1:
                    base.N[0,:] = 0.0
                    mu[i,0,t] = np.sum(base.V**i*base.N[:,t])
                    
                # The following only applies for 2D and 3D case
                # Moment between component 1 and 3
                else:
                    for j in range(3):
                        if base.dim == 2:
                            base.N[0,0,:] = 0.0
                            mu[i,j,t] = np.sum((base.X1_vol*base.V)**i*(base.X3_vol*base.V)**j*base.N[:,:,t])
                        if base.dim == 3:
                            base.N[0,0,0,:] = 0.0
                            mu[i,j,t] = np.sum((base.X1_vol*base.V)**i*(base.X3_vol*base.V)**j*base.N[:,:,:,t])
                        
        return mu
    
    # Calculate distribution moments related to particle radii mu_r(i,j,t)
    def calc_mom_r_t(self):
        base = self.base
        mu_r = np.zeros((3,3,len(base.t_vec)))
        
        # Time loop
        for t in range(len(base.t_vec)):
            for i in range(3):
                if base.dim == 1:
                    base.N[0,:] = 0.0
                    mu_r[i,0,t] = np.sum(base.R**i*base.N[:,t])
                    
                # The following only applies for 2D and 3D case
                # Moment between component 1 and 3
                else:
                    R1 = (base.X1_vol*base.V(4*math.pi))**(1/3)
                    R3 = (base.X3_vol*base.V(4*math.pi))**(1/3)
                    for j in range(3):
                        if base.dim == 2:
                            base.N[0,0,:] = 0.0
                            mu_r[i,j,t] = np.sum((R1)**i*(R3)**j*base.N[:,:,t])
                        if base.dim == 3:
                            base.N[0,0,0,:] = 0.0
                            mu_r[i,j,t] = np.sum((R1)**i*(R3)**j*base.N[:,:,:,t])
                        
        return mu_r
    
    ## Save Variables to file:
    def save_vars(self,file):
        np.save(file,vars(self.base))
        
    ## Caculate Weibull distribution
    def calc_weibull(self, Q=None, x=None, ATOL=1e-10):
        x_ = y_ = None
    
        if Q is not None:
            y_ = np.zeros_like(Q)
            valid_Q = (Q > ATOL) & (Q < 1 - ATOL)
            y_[valid_Q] = np.log(-np.log(1 - Q[valid_Q]))
    
        if x is not None:
            x_ = np.zeros_like(x)
            valid_x = x > 0
            x_[valid_x] = np.log(x[valid_x])
    
        if x is not None and Q is not None:
            valid = (x > 0) & (Q > ATOL) & (Q < 1 - ATOL)
            x_ = np.zeros_like(x)
            y_ = np.zeros_like(Q)
            x_[valid] = np.log(x[valid])
            y_[valid] = np.log(-np.log(1 - Q[valid]))
    
        return x_, y_