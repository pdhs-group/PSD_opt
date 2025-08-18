# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:05:42 2023

@author: px2030
"""

from .opt_core import OptCore   

class OptCoreMulti(OptCore):
    """
    A specialized optimization core class that handles the use of 1D data to assist in 
    the optimization of 2D data. Inherits from the OptCore class and extends its functionality 
    to manage and calculate population balance equations (PBE) for both 1D and 2D systems.
    """

    def calc_delta(self, params_in,x_uni_exp, data_exp): 
        """
        Calculate the total delta (error) for both 2D and 1D populations (NM and M). The final delta is a weighted 
        sum of the 2D delta and the deltas from the two 1D cases. If any of the population 
        calculations fail to converge, a large delta value (10) is returned to indicate failure.
        
        Parameters
        ----------
        params_in : dict
            The input parameters for PBE.
        x_uni_exp : list of arrays
            A list containing the unique particle sizes from the experimental data for 2D, NM, and M populations.
        data_exp : list of arrays
            A list containing the experimental PSD data for 2D, NM, and M populations.
        
        Returns
        -------
        float
            The total delta value representing the error between the experimental and simulated data.
        """
        params = self.check_corr_agg(params_in)
        
        # Calculate population data for all 2D and 1D populations
        self.calc_all_pop(params, self.t_vec)
        ## for DEBUG
        # if len(self.t_vec) != len(self.p.t_vec):
        #     print(params)
        #     print(self.p.t_vec)
        
        if self.p.calc_status:
            delta = self.calc_delta_tem(x_uni_exp[0], data_exp[0], self.p)
        else:
            print('p not converged')
            delta = 10
            
        if self.p_NM.calc_status:
            delta_NM = self.calc_delta_tem(x_uni_exp[1], data_exp[1], self.p_NM)
        else:
            print('p_NM not converged')
            delta_NM = 10
            
        if self.p_M.calc_status:    
            delta_M = self.calc_delta_tem(x_uni_exp[2], data_exp[2], self.p_M)
        else:
            print('p_M not converged')
            delta_M = 10
            
        # Sum the deltas, with a weight applied to the 2D case
        delta_sum = delta * self.weight_2d + delta_NM + delta_M
        return delta_sum        

        
    def calc_all_pop(self, params=None, t_vec=None):
        """
        Calculate the population balance equations (PBE) for both 1D and 2D populations.
        
        Parameters
        ----------
        params : dict, optional
            The parameters used to calculate the PBE.
        t_vec : array-like, optional
            The time vector for the PBE calculations.
        """
        self.opt_pbe.calc_pop(self.p_NM, params, t_vec, self.init_N_NM)
        self.opt_pbe.calc_pop(self.p_M, params, t_vec, self.init_N_M)
        self.opt_pbe.calc_pop(self.p, params, t_vec, self.init_N_2D)           
        
