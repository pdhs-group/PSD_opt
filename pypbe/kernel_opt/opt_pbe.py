# -*- coding: utf-8 -*-
"""
PBE-related calculations during optimization
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from ..pbe.dpbe_base import DPBESolver

def create_1d_pop(self, t_vec, disc='geo'):
    """
    Instantiate one-dimensional DPBESolvers for both non-magnetic (NM) and magnetic (M) particles.
    """
    self.p_NM = DPBESolver(dim=1,disc=disc, t_vec=t_vec, load_attr=False)
    self.p_M = DPBESolver(dim=1,disc=disc, t_vec=t_vec, load_attr=False)
        
def calc_pop(self, pop, params=None, t_vec=None):
    """
    Configure and calculate the PBE.
    """
    self.set_pop_para(pop, params)
    
    if not self.calc_init_N:
        pop.full_init(calc_alpha=False)
    else:
        pop.calc_F_M()
        pop.calc_B_R()
        pop.calc_int_B_F()
    pop.solve_PBE(t_vec=t_vec)      

def set_init_pop_para(self,pop_params):
    
    self.set_pop_para(self.p, pop_params)
    
    if hasattr(self, 'p_NM'):
        self.set_pop_para(self.p_NM, pop_params)
    if hasattr(self, 'p_M'):
        self.set_pop_para(self.p_M, pop_params)
    
    self.set_init_pop_para_flag = True

def set_pop_para(self, pop, params_in):
    params = params_in.copy()
    if params is None:
        return
    self.set_pop_attributes(pop, params)
    ## Because alpha_prim can be an arry, it needs to be handled separatedly 
    if self.dim == 1:
        if 'corr_agg' in params:
            params['CORR_BETA'] = self.return_syth_beta(params['corr_agg'])
            params['alpha_prim'] = params['corr_agg'] / params['CORR_BETA']
            del params["corr_agg"]
        if 'alpha_prim' in params:
            if params['alpha_prim'].ndim != 0:
                pop.alpha_prim = params['alpha_prim'][0]
            else:
                pop.alpha_prim = params['alpha_prim']
    elif self.dim == 2:
        if 'corr_agg' in params:
            params['CORR_BETA'] = self.return_syth_beta(params['corr_agg'])
            params['alpha_prim'] = params['corr_agg'] / params['CORR_BETA']
            del params["corr_agg"]
        if 'alpha_prim' in params:
            alpha_prim_value = params['alpha_prim']
            if pop is self.p:
                alpha_prim_temp = np.zeros(4)
                alpha_prim_temp[0] = alpha_prim_value[0]
                alpha_prim_temp[1] = alpha_prim_temp[2] = alpha_prim_value[1]
                alpha_prim_temp[3] = alpha_prim_value[2]
                pop.alpha_prim = alpha_prim_temp
            elif pop is self.p_NM:
                pop.alpha_prim = alpha_prim_value[0]
            elif pop is self.p_M:
                pop.alpha_prim = alpha_prim_value[2]
        if 'pl_P3' and 'pl_P4' in params:
            if pop is self.p_M:
                pop.pl_P1 = params['pl_P3']
                pop.pl_P2 = params['pl_P4']
    if 'CORR_BETA' in params:
        pop.CORR_BETA = params['CORR_BETA']

def set_pop_attributes(self, pop, params):
    for key, value in params.items():
        if key != 'alpha_prim':
            setattr(pop, key, value)
    
def set_comp_para(self, data_path):
    """
    Set component parameters for non-magnetic and magnetic particle.
    
    Configures the particle size distribution (PSD) parameters from provided paths or sets
    default values.
    
    Parameters
    ----------
    R01_0 : `str`, optional
        Key for accessing the initial radius of NM particles from the PSD dictionary. Defaults to 'r0_005'.
    R03_0 : `str`, optional
        Key for accessing the initial radius of M particles from the PSD dictionary. Defaults to 'r0_005'.
    dist_path_NM : `str`, optional
        Path to the file containing the PSD dictionary for NM particles. If None, default radii are used.
    dist_path_M : `str`, optional
        Path to the file containing the PSD dictionary for M particles. If None, default radii are used.
    R_NM : `float`, optional
        Default radius for NM particles if `dist_path_NM` is not provided. Defaults to 2.9e-7.
    R_M : `float`, optional
        Default radius for M particles if `dist_path_M` is not provided. Defaults to 2.9e-7.
    """
    self.p.USE_PSD = self.USE_PSD
    if self.p.USE_PSD:
        dist_path_R01 = os.path.join(data_path, self.PSD_R01)
        dist_path_R03 = os.path.join(data_path, self.PSD_R03)
        if not os.path.exists(dist_path_R01) or not os.path.exists(dist_path_R03):
            raise Exception("Please give the name to PSD data!")
        psd_dict_R01 = np.load(dist_path_R01,allow_pickle=True).item()
        psd_dict_R03 = np.load(dist_path_R03,allow_pickle=True).item()
        self.p.DIST1 = dist_path_R01
        self.p.DIST3 = dist_path_R03
        self.p.R01 = psd_dict_R01[self.R01_0] * self.R01_0_scl
        self.p.R03 = psd_dict_R03[self.R03_0] * self.R03_0_scl
    else:
        self.p.R01 = self.R_01 * self.R01_0_scl
        self.p.R03 = self.R_03 * self.R03_0_scl
    if self.dim > 1:
        ## Set particle parameter for 1D PBE
        self.p_NM.USE_PSD = self.p_M.USE_PSD = self.p.USE_PSD
        # parameter for particle component 1 - NM
        self.p_NM.R01 = self.p.R01
        self.p_NM.DIST1 = self.p.DIST1
        
        # parameter for particle component 2 - M
        self.p_M.R01 = self.p.R03
        self.p_M.DIST1 = self.p.DIST3
    self.set_comp_para_flag = True
    
def calc_all_R(self):
    """
    Calculate the radius for particles in all PBEs.
    """
    self.p.calc_R()
    self.p_NM.calc_R()
    self.p_M.calc_R()

## only for 1D-pop, 
def set_init_N(self, exp_data_paths, init_flag):
    """
    Initialize the number concentration N for 1D DPBESolvers based on experimental data.
    
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
    if self.dim ==1:
        self.p.calc_R()
        self.set_init_N_1D(self.p, exp_data_paths, init_flag)
    elif self.dim == 2:
        self.calc_all_R()
        self.set_init_N_1D(self.p_NM, exp_data_paths[1], init_flag)
        self.set_init_N_1D(self.p_M, exp_data_paths[2], init_flag)
        self.p.N = np.zeros((self.p.NS, self.p.NS, len(self.p.t_vec)))
        self.p.N[1:, 1, 0] = self.p_NM.N[1:, 0]
        self.p.N[1, 1:, 0] = self.p_M.N[1:, 0]

def set_init_N_1D(self, pop, exp_data_path, init_flag):
    """
    Initialize the number concentration N for a single 1D DPBESolver using experimental data.
    
    It processes the experimental data to align with the DPBESolver's discrete size grid,
    using either interpolation or averaging based on the `init_flag`. Supports processing
    multiple samples of experimental data for averaging purposes.
    
    Parameters
    ----------
    pop : :class:`pop.DPBESolver`
        The DPBESolver instance (either NM or M) to initialize.
    sample_num : `int`
        Number of experimental data sets used for initialization.
    exp_data_path : `str`
        Path to the experimental data file.
    init_flag : `str`
        Initialization method: 'int' for interpolation, 'mean' for averaging.
    """
    x_uni = pop.calc_x_uni(pop)
    if self.sample_num == 1:
        x_uni_exp, sumN_uni_init_sets = self.read_exp(exp_data_path, self.t_init[1:])
    else:
        exp_data_path=self.traverse_path(0, exp_data_path)
        x_uni_exp, sumN_uni_tem = self.read_exp(exp_data_path, self.t_init[1:])
        sumN_uni_all_samples = np.zeros((len(x_uni_exp), len(self.t_init[1:]), self.sample_num))
        sumN_uni_all_samples[:, :, 0] = sumN_uni_tem
        for i in range(1, self.sample_num):
            exp_data_path=self.traverse_path(i, exp_data_path)
            _, sumN_uni_tem = self.read_exp(exp_data_path, self.t_init[1:])
            sumN_uni_all_samples[:, :, i] = sumN_uni_tem
        sumN_uni_init_sets = sumN_uni_all_samples.mean(axis=2)
        
    sumN_uni_init = np.zeros(len(x_uni))
        
    if init_flag == 'int':
        for idx in range(len(x_uni_exp)):
            interp_time = interp1d(self.t_init[1:], sumN_uni_init_sets[idx, :], kind='linear', fill_value="extrapolate")
            sumN_uni_init[idx] = interp_time(0.0)

    elif init_flag == 'mean':
        sumN_uni_init = sumN_uni_init_sets.mean(axis=1)
            
    ## Remap q3 corresponding to the x value of the experimental data to x of the PBE
    # kde = self.KDE_fit(x_uni_exp, q3_init)
    # sumV_uni = self.KDE_score(kde, x_uni)
    # q3_init = sumV_uni / sumV_uni.sum()
    inter_grid = interp1d(x_uni_exp, sumN_uni_init, kind='linear', fill_value="extrapolate")
    sumN_uni_init = inter_grid(x_uni)
            
    pop.N = np.zeros((pop.NS, len(pop.t_vec)))
    ## Because sumN_uni_init[0] = 0
    pop.N[:, 0]= sumN_uni_init
    thr = 1e-5
    pop.N[pop.N < (thr * pop.N[1:, 0].max())]=0   
    pop.N[:, 0] *= pop.V_unit
    
    

