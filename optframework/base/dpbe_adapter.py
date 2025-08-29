# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:54:06 2025

@author: px2030
"""
import os
from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .adapters_api_basics import WriteThroughAdapter
from optframework.dpbe.dpbe_base import DPBESolver
from optframework.utils.func.func_read_exp import write_read_exp

class DPBEAdapter(WriteThroughAdapter):
    """
    Adapter for DPBESolver with role-aware handling.

    Roles:
      - "main": 2D main solver; alpha_prim [a0, a1, a2] -> [a0, a1, a1, a2]
      - "NM"  : 1D auxiliary solver; alpha_prim takes a0
      - "M"   : 1D auxiliary solver; alpha_prim takes a2
                For dim==2:
                  * pl_P3 -> impl.pl_P1 (written immediately)
                  * pl_P4 -> impl.pl_P2 (written immediately)
    """

    def __init__(self, *, opt, role: Literal["main","NM","M"]="main", **kw: Any):
        impl = DPBESolver(**kw)
        super().__init__(impl)
        
        # Optional: name mappings
        # self._map.update({
        #     "grid_x": "V1",
        #     "grid_y": "V3",
        # })
        
        # Optional: Adapter-only field (won't write-through)
        self._skip.update({"role", "opt"})
        self.role = role
        self.opt = opt
        
        # Write-through attributes
        self.calc_status = True

        # ---------- alpha_prim  ----------
        def set_alpha_prim(impl, value):
            arr = np.asarray(value)
            dim = impl.dim  
            if dim is None:
                raise ValueError("DPBEAdapter: 'dim' must be set on impl before alpha_prim.")
        
            r = self.role  
            if dim == 1:
                if arr.ndim == 0:
                    impl.alpha_prim = float(arr)
                else:
                    if arr.size < 1:
                        raise ValueError("alpha_prim must contain at least one value for dim=1.")
                    impl.alpha_prim = float(arr.ravel()[0])
            elif dim == 2:
                flat = arr.ravel()
                if r == "main":
                    if flat.size == 3:
                        a0, a1, a2 = map(float, flat.tolist())
                        impl.alpha_prim = np.array([a0, a1, a1, a2], dtype=float)
                    elif flat.size == 4:
                        impl.alpha_prim = np.array(flat, dtype=float).reshape(4,)
                    else:
                        raise ValueError(
                            f"alpha_prim for main 2D should have length 3 (a0,a1,a2)"
                            f" or 4 (a0,a1,a1,a2); got {flat.size}."
                        )
                elif r == "NM":
                    if flat.size < 1:
                        raise ValueError("alpha_prim must contain at least a0 for NM in 2D.")
                    impl.alpha_prim = float(flat[0])
                elif r == "M":
                    if flat.size < 3:
                        raise ValueError("alpha_prim must contain at least a2 (index 2) for M in 2D.")
                    impl.alpha_prim = float(flat[2])
                else:
                    raise ValueError(f"Unknown role '{r}'.")
            else:
                raise ValueError(f"Unsupported dim={dim} for DPBEAdapter alpha_prim handling.")
        
        self._setters["alpha_prim"] = set_alpha_prim
        
        if self.role == "M" and impl.dim == 2:
            def set_pl_P3(impl, value):
                impl.pl_P1 = value
        
            def set_pl_P4(impl, value):
                impl.pl_P2 = value
        
            self._setters["pl_P3"] = set_pl_P3
            self._setters["pl_P4"] = set_pl_P4


    # %% ESSENTIAL METHOD INTERFACE
    def set_comp_para(self, data_path: str) -> None:
        """
        Wrap component-parameter setup and write-through into solvers (opt.p, opt.p_NM, opt.p_M).
    
        Parameters
        ----------
        opt : object
            A coordinator object that holds optimizer-wide parameters and the solver instances:
            - opt.p      : the main solver (may be a DPBEAdapter or a raw solver)
            - opt.p_NM   : auxiliary solver for NM (dim==1) when opt.dim > 1
            - opt.p_M    : auxiliary solver for M  (dim==1) when opt.dim > 1
            - opt.dim    : problem dimension (1 or 2)
            - opt.USE_PSD_R, opt.R01_0, opt.R03_0, opt.R01_0_scl, opt.R03_0_scl, etc.
    
        data_path : str
            opt path where PSD data folder resides (expects a subfolder 'PSD_data').
    
        Returns
        -------
        bool
            True on success. Also sets `opt.set_comp_para_flag = True`.
        """
        opt = self.opt
        if self.role != "main":
            raise TypeError(f"set_comp_para is only available on main solver adapters, not role={self.role!r}")
        # Always define PSD root paths so later usage (dim>1) won't hit UnboundLocalError
        DIST1_path = os.path.join(data_path, "PSD_data")
        DIST3_path = os.path.join(data_path, "PSD_data")
        opt.p.DIST1_path = DIST1_path
        opt.p.DIST3_path = DIST3_path
        opt.p.DIST1_name = opt.DIST1_name
        opt.p.DIST3_name = opt.DIST3_name
        # Optionally load PSD dicts if the main solver (opt.p) uses PSD files
        psd_dict_R01 = None
        psd_dict_R03 = None
        if opt.p.USE_PSD:
            dist_path_R01 = os.path.join(DIST1_path, opt.DIST1_name)
            dist_path_R03 = os.path.join(DIST3_path, opt.DIST3_name)
    
            if not os.path.exists(dist_path_R01):
                raise Exception(f"initial PSD data in path: {dist_path_R01} not found!")
            if not os.path.exists(dist_path_R03):
                raise Exception(f"initial PSD data in path: {dist_path_R03} not found!")
    
            # Load PSD dicts (expects numpy .npy with dict payload)
            psd_dict_R01 = np.load(dist_path_R01, allow_pickle=True).item()
            psd_dict_R03 = np.load(dist_path_R03, allow_pickle=True).item()
    
        # Decide radii for the main solver (opt.p)
        if opt.USE_PSD_R:
            if psd_dict_R01 is None or psd_dict_R03 is None:
                raise Exception("USE_PSD_R is True but PSD dictionaries were not loaded. "
                                "Ensure opt.p.USE_PSD is True and files exist.")
            opt.p.R01 = psd_dict_R01[opt.R01_0] * opt.R01_0_scl
            opt.p.R03 = psd_dict_R03[opt.R03_0] * opt.R03_0_scl
        else:
            # Manual radii
            opt.p.R01 = opt.R_01 * opt.R01_0_scl
            opt.p.R03 = opt.R_03 * opt.R03_0_scl
    
        # If 2D, also configure the auxiliary 1D solvers (NM and M)
        if opt.dim > 1:
            # Component 1 - NM
            opt.p_NM.R01 = opt.p.R01
            opt.p_NM.DIST1_path = DIST1_path
            opt.p_NM.DIST1_name = opt.p.DIST1_name
    
            # Component 2 - M
            opt.p_M.R01 = opt.p.R03
            opt.p_M.DIST1_path = DIST3_path
            # Mirror the original behavior: use DIST3_name as DIST1_name for M
            opt.p_M.DIST1_name = opt.p.DIST3_name
    
        opt.set_comp_para_flag = True
        
    def reset_params(self) -> None:
        self.impl._reset_params()
    
    def calc_init_from_data(self, exp_data_paths, init_flag) -> None:
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
        opt = self.opt
        if opt.dim ==1:
            opt.p.core.calc_R()
            opt.p.N = np.zeros((opt.p.NS, len(opt.p.t_vec)))
            opt.init_N = self._set_init_N_1D(opt, opt.p, exp_data_paths, init_flag)
        elif opt.dim == 2:
            opt.p.core.calc_R()
            opt.p_NM.core.calc_R()
            opt.p_M.core.calc_R()
            opt.init_N_NM = self._set_init_N_1D(opt, opt.p_NM, exp_data_paths[1], init_flag)
            opt.init_N_M = self._set_init_N_1D(opt, opt.p_M, exp_data_paths[2], init_flag)
            opt.p.N = np.zeros((opt.p.NS, opt.p.NS, len(opt.p.t_vec)))
            opt.p_NM.N = np.zeros((opt.p.NS, len(opt.p.t_vec)))
            opt.p_M.N = np.zeros((opt.p.NS, len(opt.p.t_vec)))
            # Set the number concentration for NM and M populations at the initial time step
            # This assumes the system initially contains only pure materials, so no mixed particles exist
            opt.p.N[1:, 1, 0] = opt.p_NM.N[1:, 0]
            opt.p.N[1, 1:, 0] = opt.p_M.N[1:, 0]
            opt.init_N_2D = opt.p.N.copy()
            
    def calc_matrix(self, init_N) -> None:
        if not self.opt.calc_init_N:
            self.impl.core.full_init(calc_alpha=False)
        else:
            if init_N is None:
                raise Exception("initial N is not provided")
            self.impl.core.calc_R()
            self.impl.N = init_N
            self.impl.core.calc_F_M()
            self.impl.core.calc_B_R()
            self.impl.core.calc_int_B_F()
            
    def solve(self, t_vec) -> None:
        self.impl.core.solve_PBE(t_vec=t_vec)
        
    def get_all_data(self, exp_data_path) -> tuple[np.ndarray, np.ndarray]:
        if self.opt.exp_data:
            x_uni, data_exp = self._get_all_synth_data(exp_data_path)
        else:
            x_uni, data_exp = self._get_all_exp_data(exp_data_path)
        return x_uni, data_exp
    
    def calc_delta_pop(self, x_uni_exp, data_exp) -> float:
        """
        Calculate the average differences (delta) between experimental and simulated PSDs.
 
        This method loops through the experimental data and computes the average difference 
        (delta) between the experimental PSD and the simulated PSD. It supports both single and 
        multiple sample sets and handles optional smoothing via KDE (Kernel Density Estimation).
 
        Parameters
        ----------
        x_uni_exp : array-like
            The unique particle diameters in experimental data.
        data_exp : array-like
            The experimental PSD data that will be compared with the simulated PSD.
        pop : :class:
            An instance of the PBE solver, which generates the simulated PSD.
 
        Returns
        -------
        float
            The average difference (delta) between the experimental PSD and the simulated PSD, 
            normalized by the number of particle sizes in the experimental data (`x_uni_exp`).
        """
        opt = self.opt
        # If smoothing is enabled, initialize a list to store KDE objects
        if opt.smoothing:
            kde_list = []
            
        # Get the unique particle size in PBE
        x_uni = self.impl.post.calc_x_uni()
        
        # Loop through time steps to collect the simulation results and convert to PSD
        for idt in range(opt.delta_t_start_step, opt.num_t_steps):
            if opt.smoothing:
                sum_uni = self.impl.post.return_distribution(t=idt, flag='sum_uni', q_type=opt.dist_type)[0]
                # Volume of particles with index=0 is 0; in theory, such particles do not exist
                kde = opt.opt_data.KDE_fit(x_uni[1:], sum_uni[1:])
                # The qx distribution measured by the Lumisizer is typically matched using the 
                # average values of two measurement nodes, so the corresponding conversion has also been performed here.
                # x_uni_m = (x_uni[:-1]+x_uni[1:]) / 2
                # kde = opt.KDE_fit(x_uni_m, sum_uni[1:])
                kde_list.append(kde)
                
        delta_sum = 0 
        # Single sample case
        if opt.sample_num == 1:
            qx_mod = np.zeros((len(x_uni_exp), opt.num_t_steps-opt.delta_t_start_step))
            for idt in range(opt.num_t_steps-opt.delta_t_start_step):
                if opt.smoothing:
                    qx_mod_tem = opt.opt_data.KDE_score(kde_list[idt], x_uni_exp[1:])
                    qx_mod[1:, idt] = qx_mod_tem
                else:
                    qx_mod[:, idt] = self.impl.post.return_distribution(t=idt+opt.delta_t_start_step, flag='qx', q_type=opt.dist_type)[0]
                Qx = self.impl.post.calc_Qx(x_uni_exp, qx_mod[:, idt]) 
                qx_mod[:, idt] = qx_mod[:, idt] / Qx.max() 
            # Calculate the delta for each cost function type, if is defined.
            for flag, cost_func_type in opt.delta_flag:
                data_mod = self.impl.post.re_calc_distribution(x_uni_exp, qx=qx_mod, flag=flag)[0]
                delta = opt.cost_fun(data_exp, data_mod, cost_func_type, flag)
                delta_sum += delta 
            
            return delta
        
        # Multiple sample case
        else:
            for i in range (0, opt.sample_num):
                qx_mod = np.zeros((len(x_uni_exp[i]), opt.num_t_steps-opt.delta_t_start_step))
                for idt in range(opt.num_t_steps-opt.delta_t_start_step):
                    if opt.smoothing:
                        qx_mod_tem = opt.opt_data.KDE_score(kde_list[idt], x_uni_exp[i][1:])
                        qx_mod[1:, idt] = qx_mod_tem
                    else:
                        qx_mod[:, idt] = self.impl.post.return_distribution(t=idt+opt.delta_t_start_step, flag='qx')[0]
                    Qx = self.impl.post.calc_Qx(x_uni_exp[i], qx_mod[:, idt]) 
                    qx_mod[:, idt] = qx_mod[:, idt] / Qx.max()
                # Calculate delta for each cost function type, if is defined.    
                for flag, cost_func_type in opt.delta_flag:
                    data_mod = self.impl.post.re_calc_distribution(x_uni_exp[i], qx=qx_mod, flag=flag)[0]
                    delta = opt.cost_fun(data_exp[i], data_mod, cost_func_type, flag)
                    delta_sum += delta 
                
            delta_sum /= opt.sample_num
            return delta_sum 
        
    def close(self) -> None:
        self.impl.core._close()
            
    
    # %% OPTIONAL METHOD INTERFACE      
    def generate_data(self, data_path=None, multi_flag=False, pop_params=None, add_info=""):
        """
        Generates synthetic data based on simulation results, with optional noise.

        This method generates synthetic data based on the population parameters and simulation results. 
        If noise is enabled, it modifies the file name to reflect the noise type and strength. The data 
        is saved to an Excel file. For multi-dimensional optimizations, separate files for different 
        dimensions are created.

        Parameters
        ----------
        pop_params : dict, optional
            Parameters for the population model. If not provided, uses `self.pop_params`.
        add_info : str, optional
            Additional information to append to the file name. Default is an empty string.

        Returns
        -------
        None
        """
        if self.role != "main":
            raise TypeError(f"set_comp_para is only available on main solver adapters, not role={self.role!r}")
        opt = self.opt
        # Modify the file name if noise is enabled
        if opt.add_noise:
            # Modify the file name to include noise type and strength
            filename = f"Sim_{opt.noise_type}_{opt.noise_strength}"+add_info+".xlsx"
        else:
            filename = "Sim"+add_info+".xlsx"

        # Construct the full path for the output data file
        exp_data_path = os.path.join(data_path, filename)
        
        # Check if multi_flag is False, indicating no auxiliary 1D data
        if not multi_flag:
            # Calculate the population data based on the given parameters
            opt.calc_pop(opt.p, pop_params, opt.t_all)
            # Only proceed if the calculation status is valid
            if opt.core.p.calc_status:
                for i in range(0, opt.sample_num):
                    # Traverse the file path if multiple samples are generated
                    if opt.sample_num != 1:
                        exp_data_path=opt.traverse_path(i, exp_data_path)
                    # print(opt.exp_data_path)
                    ## Write new data to file
                    self._write_new_data(opt.p, exp_data_path)
            else:
                return
        else:
            # For multi-dimensional data, create multiple file paths for different datasets
            exp_data_paths = [
                exp_data_path,
                exp_data_path.replace(".xlsx", "_NM.xlsx"),
                exp_data_path.replace(".xlsx", "_M.xlsx")
            ]
            # Calculate the population data for all dimensions
            opt.calc_all_pop(pop_params, opt.t_all)
            if opt.p.calc_status and opt.p_NM.calc_status and opt.p_M.calc_status:
                for i in range(0, opt.sample_num):
                    if opt.sample_num != 1:
                        # Traverse the file paths for multiple samples
                        exp_data_paths = opt.core.opt_data.traverse_path(i, exp_data_paths)
                    # Write data for each dimension to separate files
                    self._write_new_data(opt.p, exp_data_paths[0])
                    self._write_new_data(opt.p_NM, exp_data_paths[1])
                    self._write_new_data(opt.p_M, exp_data_paths[2])
            else:
                return
            
    # %% INTERNAL METHOD INTERFACE  
    def _set_init_N_1D(self, pop, exp_data_path, init_flag) -> np.ndarray:
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
        opt = self.opt
        x_uni = pop.post.calc_x_uni()
        if not opt.exp_data:
            # If only one sample exists, initialize N based on the first few time points
            if opt.sample_num == 1:
                # Exclude the zero point and extrapolate the initial conditions
                x_uni_exp, sumN_uni_init_sets = opt.opt_data.read_exp(exp_data_path, opt.t_init[1:])
            else:
                # For multiple samples, average the initial data values
                exp_data_path=opt.opt_data.traverse_path(0, exp_data_path)
                x_uni_exp, sumN_uni_tem = opt.opt_data.read_exp(exp_data_path, opt.t_init[1:])
                sumN_uni_all_samples = np.zeros((len(x_uni_exp), len(opt.t_init[1:]), opt.sample_num))
                sumN_uni_all_samples[:, :, 0] = sumN_uni_tem
                # Loop through remaining samples and average the data sets
                for i in range(1, opt.sample_num):
                    exp_data_path=opt.opt_data.traverse_path(i, exp_data_path)
                    _, sumN_uni_tem = opt.opt_data.read_exp(exp_data_path, opt.t_init[1:])
                    sumN_uni_all_samples[:, :, i] = sumN_uni_tem
                sumN_uni_init_sets = sumN_uni_all_samples.mean(axis=2)
                
            sumN_uni_init = np.zeros(len(x_uni))
            
            # Initialize based on interpolation of the time points    
            if init_flag == 'int':
                for idx in range(len(x_uni_exp)):
                    interp_time = interp1d(opt.t_init[1:], sumN_uni_init_sets[idx, :], kind='linear', fill_value="extrapolate")
                    sumN_uni_init[idx] = interp_time(0.0)
            # Initialize based on the mean of the initial data sets
            elif init_flag == 'mean':
                sumN_uni_init = sumN_uni_init_sets.mean(axis=1)
            # Interpolate the experimental data onto the dPBE grid 
            inter_grid = interp1d(x_uni_exp, sumN_uni_init, kind='linear', fill_value="extrapolate")
            sumN_uni_init = inter_grid(x_uni)
        else:
            Q3_init_exp, x_uni_exp = opt.opt_data.read_exp(exp_data_path, opt.t_init[1:])
            inter_grid = interp1d(x_uni_exp.flatten(), Q3_init_exp, kind='linear', fill_value="extrapolate")
            Q3_init_mod = inter_grid(x_uni)
            Q3_init_mod[np.where(Q3_init_mod < 0)] = 0.0
            Q3_init_mod = Q3_init_mod / Q3_init_mod.max()
            sumN_uni_init = np.zeros(x_uni.shape)
            sumN_uni_init[1:] = np.diff(Q3_init_mod) * opt.p.V01 *1e18 / (np.pi * x_uni[1:]**3 /6)
        
        N_init = np.zeros((pop.NS, len(pop.t_vec)))
        N_init[:, 0]= sumN_uni_init
        # Set very small N values to zero
        thr = 1e-5
        N_init[N_init < (thr * N_init[1:, 0].max())]=0   
        return N_init
    
    def _write_new_data(self, pop, exp_data_path):
        """
        Saves the calculation results in the format of experimental data.

        This method saves the population distribution data into an Excel file in the format used for
        experimental data. It supports both smoothed and non-smoothed distributions, and can apply noise 
        to the data if required.

        Parameters
        ----------
        pop : :class:`optframework.dpbe`
            The population instance for which data is being generated.
        exp_data_path : str
            The file path where the experimental data will be saved.

        Returns
        -------
        None
        """
        opt = self.opt
        # Return if the population data is invalid (calculation status is False)
        if not pop.calc_status:
            return
        # Get particle size and volume in the dPBE grid
        x_uni = pop.post.calc_x_uni()
        v_uni = pop.post.calc_v_uni()
        # Format the simulation times for the output file
        formatted_times = write_read_exp.convert_seconds_to_time(opt.t_all)
        # Initialize the sumN_uni array to store particle count distributions
        sumN_uni = np.zeros((len(x_uni)-1, len(opt.t_all)))
        
        idt_vec = [np.where(opt.t_all == t_time)[0][0] for t_time in opt.t_vec]
        for idt in idt_vec[1:]:
            if opt.smoothing:
                # Get the volume distribution at the current time step
                sumvol_uni = pop.post.return_distribution(t=idt, flag='sum_uni')[0]
                # Skip index=0, as particles with volume 0 theoretically do not exist
                kde = opt.opt_data.KDE_fit(x_uni[1:], sumvol_uni[1:])
                # Smooth the distribution using KDE and insert a zero for the first entry
                q3 = opt.opt_data.KDE_score(kde, x_uni[1:])
                q3 = np.insert(q3, 0, 0.0)
                # Calculate and normalize Q3 values
                Q3 = self.impl.post.calc_Qx(x_uni, q3)
                Q3 = Q3 / Q3.max()
                # Calculate the final smoothed particle volume distribution
                sumvol_uni = self.impl.post.calc_sum_uni(Q3, sumvol_uni.sum())
                # Store the particle count distribution
                sumN_uni[:, idt] = sumvol_uni[1:] / v_uni[1:]
            else:
                # Use the unsmoothed distribution for this time step
                sumN_uni[:, idt] = pop.post.return_distribution(t=idt, flag='sum_uni', q_type='q0')[0][1:]
        # For initialization data, do not apply smoothing
        for idt in opt.idt_init:
            sumN_uni[:, idt] = pop.post.return_distribution(t=idt, flag='sum_uni', q_type='q0')[0][1:]
        # Apply noise to the data if noise is enabled
        if opt.add_noise:
            sumN_uni = opt.opt_data.function_noise(sumN_uni)
            
        # Create a DataFrame for the distribution data and set the index name
        df = pd.DataFrame(data=sumN_uni, index=x_uni[1:], columns=formatted_times)
        df.index.name = 'Circular Equivalent Diameter'
        # Save the DataFrame as an Excel file at the specified path
        df.to_excel(exp_data_path)
        return        
    
    def _get_all_synth_data(self, exp_data_path):
        """
        Process synthetic data by reading the data and converting it to volume-based PSD.
    
        This method processes synthetic experimental data for one or multiple samples. For each sample, 
        it reads the experimental data, converts the number-based particle size distribution (PSD) into 
        a volume-based distribution, and then recalculates the final distribution (including q3, Q3, and 
        x_50) based on the specified flags. The results are stored in `data_exp` for use in subsequent 
        optimization steps.
    
        Parameters
        ----------
        exp_data_path : str
            Path to the experimental data file.
    
        Returns
        -------
        tuple of arrays
            - x_uni_exp: List of arrays containing unique particle sizes for each sample.
            - data_exp: List of arrays containing processed experimental PSD data for each sample.
        """
        opt = self.opt
        # If only one sample exists, read and process the experimental data
        if opt.sample_num == 1:
            x_uni_exp, sumN_uni_exp = opt.opt_data.read_exp(exp_data_path, opt.t_vec[opt.delta_t_start_step:]) 
            x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
            sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
            
            # Convert number-based PSD to volume-based PSD
            vol_uni = np.tile((1/6)*np.pi*x_uni_exp**3, (opt.num_t_steps-opt.delta_t_start_step, 1)).T
            sumvol_uni_exp = sumN_uni_exp * vol_uni
            
            
            # Recalculate the distribution
            for flag, _ in opt.delta_flag:
                data_exp = self.impl.post.re_calc_distribution(x_uni_exp, sum_uni=sumvol_uni_exp, flag=flag)[0]
    
        # If multiple samples exist, process each one
        else:
            x_uni_exp = []
            data_exp = []
            for i in range (0, opt.sample_num):
                # Read and process experimental data for each sample
                exp_data_path = opt.opt_data.traverse_path(i, exp_data_path)
                x_uni_exp_tem, sumN_uni_exp = opt.opt_data.read_exp(exp_data_path, opt.t_vec[opt.delta_t_start_step:])
                x_uni_exp_tem = np.insert(x_uni_exp_tem, 0, 0.0)
                sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
                
                # Convert number-based PSD to volume-based PSD
                vol_uni = np.tile((1/6)*np.pi*x_uni_exp_tem**3, (opt.num_t_steps-opt.delta_t_start_step, 1)).T
                sumvol_uni_exp = sumN_uni_exp * vol_uni
                
                # Recalculate the distribution
                for flag, _ in opt.delta_flag:
                    data_exp_tem = self.impl.post.re_calc_distribution(x_uni_exp_tem, sum_uni=sumvol_uni_exp, flag=flag)[0] 
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
                
        return x_uni_exp, data_exp
    
    ## test only for 1d batch exp data
    def _get_all_exp_data(self, exp_data_path):
        opt = self.opt
        (flag, cost_func_type) = opt.delta_flag[0]
        if opt.sample_num == 1:
            x_uni_exp, data_exp = opt.opt_data.read_exp(exp_data_path, opt.t_vec[opt.delta_t_start_step:]) 
            x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
            zero_row = np.zeros((1, data_exp.shape[1]))
            data_exp = np.insert(data_exp, 0, zero_row, axis=0)
            if flag == 'x_50' or flag == 'y_weibull':
                if opt.sheet_name != 'Q_x_int':
                    ## Input is qx
                    data_exp = self.impl.post.re_calc_distribution(x_uni_exp, qx=data_exp, flag=flag)[0]
                else:
                    ## Input is Qx
                    data_exp = self.impl.post.re_calc_distribution(x_uni_exp, Qx=data_exp, flag=flag)[0]
            
        else:
            x_uni_exp = []
            data_exp = []
            zero_row = np.zeros((1, data_exp.shape[1]))
            for i in range (0, opt.sample_num):
                # Read and process experimental data for each sample
                exp_data_path = opt.opt_data.traverse_path(i, exp_data_path)
                x_uni_exp_tem, data_exp_tem = opt.opt_data.read_exp(exp_data_path, opt.t_vec[opt.delta_t_start_step:])
                x_uni_exp_tem = np.insert(x_uni_exp_tem, 0, 0.0)
                data_exp_tem = np.insert(data_exp_tem, 0, zero_row, axis=0)
                if flag == 'x_50' or flag == 'y_weibull':
                    if opt.sheet_name != 'Q_x_int':
                        ## Input is qx
                        data_exp_tem_raw = self.impl.post.re_calc_distribution(x_uni_exp, qx=data_exp_tem, flag=flag)[0]
                    else:
                        ## Input is Qx
                        data_exp_tem_raw = self.impl.post.re_calc_distribution(x_uni_exp, Qx=data_exp_tem, flag=flag)[0]
                    data_exp_tem = data_exp_tem_raw
                        
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
            
        return x_uni_exp, data_exp