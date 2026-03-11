# -*- coding: utf-8 -*-
"""
PBE-related calculations during optimization
"""
# import os
import gc
import numpy as np

class OptPBE():
    def __init__(self, base):
        self.base = base
           
    def close_pbe(self):
        base = self.base
        base.p.close()
        del base.p
        if base.dim == 2:
            base.p_NM.close()
            base.p_M.close()
            del base.p_NM
            del base.p_M
        gc.collect()
            
        
    def calc_pop(self, pop, params=None, t_vec=None, init_N=None):
        """
        Configure and calculate the PBE.
    
        If `calc_init_N` is set to False, the full initialization is performed 
        without calculating alpha. Otherwise, it calculates various terms such 
        as F_M, B_R, and int_B_F before solving the PBE.
    
        Parameters
        ----------
        pop : object
            The population instance for which the PBE will be calculated.
        params : dict, optional
            The population parameters. If not provided, it uses the existing parameters of the population.
        t_vec : array-like, optional
            The time vector for which the PBE will be solved. If not provided, the default time vector is used.
    
        Returns
        -------
        None
        """
        self.set_pop_para(pop, params)
        pop.calc_matrix(init_N)
        pop.solve(t_vec)      
    
    def set_init_pop_para(self,pop_params):
        """
        Initialize population parameters for all PBEs.
    
        This method sets the population parameters for the main population (`p`) 
        as well as for the auxiliary populations (`p_NM` and `p_M`, if they exist).
    
        Parameters
        ----------
        pop_params : dict
            The parameters to be applied to the populations.
    
        Returns
        -------
        None
        """
        self.set_pop_para(self.base.p, pop_params)
        
        if hasattr(self.base, 'p_NM'):
            self.set_pop_para(self.base.p_NM, pop_params)
        if hasattr(self.base, 'p_M'):
            self.set_pop_para(self.base.p_M, pop_params)
        
        self.base.set_init_pop_para_flag = True
    
    def set_pop_para(self, pop, params_in):
        """
        Set the population parameters for a given population instance.
    
        This method configures the population attributes based on the provided parameters. 
        It handles both 1D and 2D populations and adjusts specific parameters such as 
        `alpha_prim` and `CORR_BETA` depending on the dimensionality of the population.
    
        Parameters
        ----------
        pop : object
            The population instance whose parameters are being set.
        params_in : dict
            The dictionary of population parameters to be applied.
    
        Returns
        -------
        None
        """
        base = self.base
        params = params_in.copy()
        if params is None:
            return
        # Set population attributes based on the parameters
        params = base.check_corr_agg(params)
        self.set_pop_attributes(pop, params)

    def set_pop_attributes(self, pop, params):
        """
        Set attributes for a population instance from the provided parameters.
    
        Parameters
        ----------
        pop : object
            The population instance whose attributes are being set.
        params : dict
            A dictionary containing the population parameters. Each key-value pair 
            corresponds to an attribute name and its value.
    
        Returns
        -------
        None
        """
        for key, value in params.items():
            setattr(pop, key, value)
        
