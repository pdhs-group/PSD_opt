# -*- coding: utf-8 -*-
"""
PBE-related calculations during optimization
"""
import os
import gc
import numpy as np
from scipy.interpolate import interp1d
from optframework.dpbe import DPBESolver
from optframework.dpbe.dpbe_post import PBEPost

class OptPBE(PBEPost):
    def __init__(self, base):
        PBEPost.__init__(self, base)
        self.base = base
        
    def create_1d_pop(self, t_vec, disc='geo'):
        """
        Instantiate one-dimensional DPBESolvers for two different types of particles, 
        labeled as 'NM' and 'M' for historical reasons.
    
        Parameters
        ----------
        t_vec : array-like
            The time vector over which the population balance equations will be calculated.
        disc : str, optional
            The discretization method for the solver. Default is 'geo' (geometric discretization).
        
        Returns
        -------
        None
        """
        
        self.base.p_NM = DPBESolver(dim=1,disc=disc, t_vec=t_vec, load_attr=False)
        self.base.p_M = DPBESolver(dim=1,disc=disc, t_vec=t_vec, load_attr=False)
           
    def close_pbe(self):
        base = self.base
        base.p.core._close()
        del base.p
        if base.dim == 2:
            base.p_NM.core._close()
            base.p_M.core._close()
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
        
        if not self.base.calc_init_N:
            pop.core.full_init(calc_alpha=False)
        else:
            if init_N is None:
                raise Exception("initial N is not provided")
            pop.core.calc_R()
            pop.N = init_N
            pop.core.calc_F_M()
            pop.core.calc_B_R()
            pop.core.calc_int_B_F()
        pop.core.solve_PBE(t_vec=t_vec)      
    
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
        
        if hasattr(self, 'p_NM'):
            self.set_pop_para(self.base.p_NM, pop_params)
        if hasattr(self, 'p_M'):
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
        self.set_pop_attributes(pop, params)
        # Handle alpha_prim separately for 1D populations
        if base.dim == 1:
            if 'corr_agg' in params:
                params['CORR_BETA'] = base.return_syth_beta(params['corr_agg'])
                params['alpha_prim'] = params['corr_agg'] / params['CORR_BETA']
                del params["corr_agg"]
            if 'alpha_prim' in params:
                # Set alpha_prim based on its dimensions
                if params['alpha_prim'].ndim != 0:
                    pop.alpha_prim = params['alpha_prim'][0]
                else:
                    pop.alpha_prim = params['alpha_prim']
        # Handle alpha_prim and corr_agg for 2D populations
        elif base.dim == 2:
            if 'corr_agg' in params:
                params['CORR_BETA'] = base.return_syth_beta(params['corr_agg'])
                params['alpha_prim'] = params['corr_agg'] / params['CORR_BETA']
                del params["corr_agg"]
            if 'alpha_prim' in params:
                alpha_prim_value = params['alpha_prim']
                # Set alpha_prim for main, NM, and M populations
                if pop is base.p:
                    alpha_prim_temp = np.zeros(4)
                    alpha_prim_temp[0] = alpha_prim_value[0]
                    alpha_prim_temp[1] = alpha_prim_temp[2] = alpha_prim_value[1]
                    alpha_prim_temp[3] = alpha_prim_value[2]
                    pop.alpha_prim = alpha_prim_temp
                elif pop is base.p_NM:
                    pop.alpha_prim = alpha_prim_value[0]
                elif pop is base.p_M:
                    pop.alpha_prim = alpha_prim_value[2]
            if 'pl_P3' in params and 'pl_P4' in params:
                if pop is base.p_M:
                    pop.pl_P1 = params['pl_P3']
                    pop.pl_P2 = params['pl_P4']
        if 'CORR_BETA' in params:
            pop.CORR_BETA = params['CORR_BETA']
    
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
            if key != 'alpha_prim':
                setattr(pop, key, value)
        
    def set_comp_para(self, data_path):
        """
        Set component parameters for two types of particles (labeled as NM and M).
    
        This method configures the particle size distribution (PSD) parameters for both particles. 
        The PSD can either be loaded from specified file paths or 
        default values can be used if the files are not available.
    
        Parameters
        ----------
        data_path : str
            The base path where PSD data files are located.
    
        Raises
        ------
        Exception
            If the PSD data file for NM or M particles is not found.
    
        Returns
        -------
        None
        """
        base = self.base
        base.p.USE_PSD = base.USE_PSD
        # If PSD is used, load PSD data from the provided file paths
        if base.p.USE_PSD:
            DIST1_path = os.path.join(data_path, 'PSD_data')
            DIST3_path = os.path.join(data_path, 'PSD_data')
            dist_path_R01 = os.path.join(DIST1_path, base.PSD_R01)
            dist_path_R03 = os.path.join(DIST3_path, base.PSD_R03)
            # Raise exceptions if the PSD data files for NM or M particles are missing
            if not os.path.exists(dist_path_R01):
                raise Exception(f"initial PSD data in path: {dist_path_R01} not found!")
            if not os.path.exists(dist_path_R03):
                raise Exception(f"initial PSD data in path: {dist_path_R03} not found!")    
            # Load PSD data for NM and M particles
            psd_dict_R01 = np.load(dist_path_R01,allow_pickle=True).item()
            psd_dict_R03 = np.load(dist_path_R03,allow_pickle=True).item()
            # Set PSD paths and radii for NM and M particles
            base.p.DIST1_path = DIST1_path
            base.p.DIST3_path = DIST3_path
            base.p.DIST1_name = base.PSD_R01
            base.p.DIST3_name = base.PSD_R03
        if base.USE_PSD_R:
            base.p.R01 = psd_dict_R01[base.R01_0] * base.R01_0_scl
            base.p.R03 = psd_dict_R03[base.R03_0] * base.R03_0_scl
        else:
            # If PSD is not used, set radii for NM and M particles manually
            base.p.R01 = base.R_01 * base.R01_0_scl
            base.p.R03 = base.R_03 * base.R03_0_scl
        if base.dim > 1:
            ## Set particle parameter for 1D PBE
            base.p_NM.USE_PSD = base.p_M.USE_PSD = base.p.USE_PSD
            # parameter for particle component 1 - NM
            base.p_NM.R01 = base.p.R01
            base.p_NM.DIST1_path = DIST1_path
            base.p_NM.DIST1_name = base.PSD_R01
            
            # parameter for particle component 2 - M
            base.p_M.R01 = base.p.R03
            base.p_M.DIST1_path = DIST3_path
            base.p_M.DIST1_name = base.PSD_R03
        base.set_comp_para_flag = True
        
    def calc_all_R(self):
        """
        Calculate the radius for particles in all PBEs (Population Balance Equations).
        
        Returns
        -------
        None
        """
        self.base.p.core.calc_R()
        self.base.p_NM.core.calc_R()
        self.base.p_M.core.calc_R()
    
    def set_init_N(self, exp_data_paths, init_flag):
        """
        Initialize the number concentration N for PBE(s) based on experimental data.
        
        This method initializes the N for both 1D and 2D PBE instances. For 2D PBE systems, the initialization 
        assumes that the system initially contains only pure materials (i.e., no mixed particles have formed yet). 
        As a result, the initialization of 2D PBE is effectively equivalent to performing two 1D initializations: 
        one for the NM particles and one for the M particles.
        
        Parameters
        ----------
        sample_num : `int`
            The number of sets of experimental data used for initialization.
        exp_data_paths : `list of str`
            Paths to the experimental data for initialization.
        init_flag : `str`
            The method to use for initialization: 'int' for interpolation or 'mean' for averaging
            the initial sets.
        """
        base = self.base
        if base.dim ==1:
            base.p.calc_R()
            base.p.N = np.zeros((base.p.NS, len(base.p.t_vec)))
            base.init_N = self.set_init_N_1D(base.p, exp_data_paths, init_flag)
        elif base.dim == 2:
            self.calc_all_R()
            base.init_N_NM = self.set_init_N_1D(base.p_NM, exp_data_paths[1], init_flag)
            base.init_N_M = self.set_init_N_1D(base.p_M, exp_data_paths[2], init_flag)
            base.p.N = np.zeros((base.p.NS, base.p.NS, len(base.p.t_vec)))
            base.p_NM.N = np.zeros((base.p.NS, len(base.p.t_vec)))
            base.p_M.N = np.zeros((base.p.NS, len(base.p.t_vec)))
            # Set the number concentration for NM and M populations at the initial time step
            # This assumes the system initially contains only pure materials, so no mixed particles exist
            base.p.N[1:, 1, 0] = base.p_NM.N[1:, 0]
            base.p.N[1, 1:, 0] = base.p_M.N[1:, 0]
            base.init_N_2D = base.p.N.copy()
    
    def set_init_N_1D(self, pop, exp_data_path, init_flag):
        """
        Initialize the number concentration N for 1D PBE based on experimental data.
    
        This method initializes the number concentration (N) for 1D PBE using experimental data. 
        It supports two initialization methods: interpolation of the initial time points ('int') or 
        averaging the initial data sets ('mean').
    
        Parameters
        ----------
        pop : object
            The population instance (PBE solver) for which the number concentration is being initialized.
        exp_data_path : str
            Path to the experimental data file for initialization.
        init_flag : str
            The initialization method. Options are:
                - 'int': Use interpolation for initialization.
                - 'mean': Use the mean of the initial data sets for initialization.
    
        Returns
        -------
        None
        """
        base = self.base
        x_uni = pop.post.calc_x_uni()
        if not base.exp_data:
            # If only one sample exists, initialize N based on the first few time points
            if base.sample_num == 1:
                # Exclude the zero point and extrapolate the initial conditions
                x_uni_exp, sumN_uni_init_sets = base.opt_data.read_exp(exp_data_path, base.t_init[1:])
            else:
                # For multiple samples, average the initial data values
                exp_data_path=base.opt_data.traverse_path(0, exp_data_path)
                x_uni_exp, sumN_uni_tem = base.opt_data.read_exp(exp_data_path, base.t_init[1:])
                sumN_uni_all_samples = np.zeros((len(x_uni_exp), len(base.t_init[1:]), base.sample_num))
                sumN_uni_all_samples[:, :, 0] = sumN_uni_tem
                # Loop through remaining samples and average the data sets
                for i in range(1, base.sample_num):
                    exp_data_path=base.opt_data.traverse_path(i, exp_data_path)
                    _, sumN_uni_tem = base.opt_data.read_exp(exp_data_path, base.t_init[1:])
                    sumN_uni_all_samples[:, :, i] = sumN_uni_tem
                sumN_uni_init_sets = sumN_uni_all_samples.mean(axis=2)
                
            sumN_uni_init = np.zeros(len(x_uni))
            
            # Initialize based on interpolation of the time points    
            if init_flag == 'int':
                for idx in range(len(x_uni_exp)):
                    interp_time = interp1d(base.t_init[1:], sumN_uni_init_sets[idx, :], kind='linear', fill_value="extrapolate")
                    sumN_uni_init[idx] = interp_time(0.0)
            # Initialize based on the mean of the initial data sets
            elif init_flag == 'mean':
                sumN_uni_init = sumN_uni_init_sets.mean(axis=1)
            # Interpolate the experimental data onto the dPBE grid 
            inter_grid = interp1d(x_uni_exp, sumN_uni_init, kind='linear', fill_value="extrapolate")
            sumN_uni_init = inter_grid(x_uni)
        else:
            Q3_init_exp, x_uni_exp = base.opt_data.read_exp(exp_data_path, base.t_init[1:])
            inter_grid = interp1d(x_uni_exp.flatten(), Q3_init_exp, kind='linear', fill_value="extrapolate")
            Q3_init_mod = inter_grid(x_uni)
            Q3_init_mod[np.where(Q3_init_mod < 0)] = 0.0
            Q3_init_mod = Q3_init_mod / Q3_init_mod.max()
            sumN_uni_init = np.zeros(x_uni.shape)
            sumN_uni_init[1:] = np.diff(Q3_init_mod) * base.p.V01 *1e18 / (np.pi * x_uni[1:]**3 /6)
        
        N_init = np.zeros((pop.NS, len(pop.t_vec)))
        N_init[:, 0]= sumN_uni_init
        # Set very small N values to zero
        thr = 1e-5
        N_init[N_init < (thr * N_init[1:, 0].max())]=0   
        return N_init