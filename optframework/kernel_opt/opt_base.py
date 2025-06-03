# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import os
import runpy
from pathlib import Path
import warnings
import pandas as pd
import ray
# from ..pbe.dpbe_base import DPBESolver
from .opt_core import OptCore
from .opt_core_multi import OptCoreMulti
from optframework.utils.func.func_read_exp import write_read_exp
from optframework.utils.func.bind_methods import bind_methods_from_module
## For plots
# import matplotlib.pyplot as plt
# from ..utils.plotter import plotter as pt        

class OptBase():
    """
    A class to manage the optimization process for finding the kernel of PBE.
    
    This class is responsible for instantiating either the `opt_algo` or `opt_algo_multi` classes 
    based on the provided configuration. It facilitates passing parameters, executing optimization, 
    generating synthetic data, and visualizing results.
    
    Note
    ----
    This class uses the `bind_methods_from_module` function to dynamically bind methods from external
    modules. Some methods in this class are not explicitly defined here, but instead are imported from
    other files. To fully understand or modify those methods, please refer to the corresponding external
    files, such as `optframework.kernel_opt.opt_base_ray`, from which methods are bound to this class.
    
    Methods
    -------
    __init__(config_path=None, data_path=None)
        Initializes the class with configuration and data paths.
    """
    
    def __init__(self, config_path=None, data_path=None, multi_flag=None):
        """
        Initializes the OptBase class with configuration and data paths.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file. If not provided, a default configuration path is used.
        data_path : str, optional
            Path to the directory where data files will be stored. If not provided, a default path is used.
        
        Raises
        ------
        Exception
            If the requirements file for ray is not found.
        """
        # Get the current script directory and the requirements file path
        # self.pth = os.path.dirname( __file__ )
        self.work_dir = Path(os.getcwd()).resolve()
        # Load the configuration file
        config = self.check_config_path(config_path)
        self.core_params = config['algo_params']
        self.pop_params = config['pop_params']
        if multi_flag is None:
            self.multi_flag = config['multi_flag']
        else:
            self.multi_flag = multi_flag
        self.single_case = config['single_case']
        self.print_highlighted(f'Current operating mode: single_case = {self.single_case}, multi_flag = {self.multi_flag}.',
                               title="INFO", color="cyan")
        
        self.opt_params = config['opt_params']
        self.dim = self.core_params.get('dim', None)
        # Set the data path, use default if not provided
        if data_path is None:
            self.print_highlighted('Data path is not found or is None, default path will be used.',
                                   title="INFO", color="cyan")
            self.data_path = os.path.join(self.work_dir, "data")
        else:
            self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        # Initialize the optimization core algorithm
        self.init_opt_core()
        # Initialize t_vec for file generation
        self.idt_vec = [np.where(self.core.t_all == t_time)[0][0] for t_time in self.core.t_vec]
        self.check_core_params()
        
    def check_core_params(self):
        """
        Check the validity of optimization core parameters.
        """
        ### verbose, delta_flag, method, noise_type, t_vec, t_init...
        pass
        
        
    def check_config_path(self, config_path):
        """
        Checks if the configuration file exists and loads it.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file. If None, a default path is used.
        
        Returns
        -------
        config : dict
            The loaded configuration dictionary.
        
        Raises
        ------
        Exception
            If the configuration file is not found at the specified path.
        """
        # Check if the configuration file exists and load it
        if config_path is None:
            # Use the default configuration file path if none is provided
            # config_path = os.path.join(self.pth, "..","..","config","opt_config.py")
            # config_name = "opt_config"
            config_path = os.path.join(self.work_dir,"config","opt_config.py")
        if not os.path.exists(config_path):
            # Raise an exception if the config file is not found
            raise Exception(f"Warning: Config file not found at: {config_path}.")
        else:
            # Load the configuration from the specified file
            conf = runpy.run_path(config_path)
            config = conf['config']
            self.print_highlighted(f"The Optimization and dPBE are using config file at : {config_path}",
                                   title="INFO", color="cyan")
            return config
        
    def init_opt_core(self):
        """
        Initializes the optimization core based on whether 1D data is used as auxiliary for 2D-PBE kernels.

        The `multi_flag` indicates whether to use 1D data as auxiliary input to help calculate 2D kernels.
        If `multi_flag` is True, the optimization process uses both 1D and 2D data. If False, only 2D data 
        is used for the kernel calculation.

        """
        # Initialize the optimization core based on dimensionality and multi_flag
        if self.dim == 1 and self.multi_flag:
            # If the dimension is 1, the multi algorithm is not applicable
            self.print_highlighted("The multi algorithm does not support 1-D pop!",
                                   title="WARNING", color="yellow")
            self.multi_flag = False
        # Initialize optimization core with or without 1D data based on multi_flag
        if not self.multi_flag:
            self.core = OptCore()
        else:
            self.core = OptCoreMulti()  
        # Initialize attributes for the optimization core
        self.core.init_attr(self.core_params)
        if self.dim == 2 and not self.multi_flag and self.core.calc_init_N:
            raise ValueError("2d PSD can only use exp data to calculate initial conditions if multi_flag is enabled!")
        # Initialize PBE with population parameters and data path
        self.core.init_pbe(self.pop_params, self.data_path)
        
    def generate_data(self, pop_params=None, add_info=""):
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
        if pop_params is None:
            pop_params = self.pop_params
        # Modify the file name if noise is enabled
        if self.core.add_noise:
            # Modify the file name to include noise type and strength
            filename = f"Sim_{self.core.noise_type}_{self.core.noise_strength}"+add_info+".xlsx"
        else:
            filename = "Sim"+add_info+".xlsx"

        # Construct the full path for the output data file
        exp_data_path = os.path.join(self.data_path, filename)
        
        # Check if multi_flag is False, indicating no auxiliary 1D data
        if not self.multi_flag:
            # Calculate the population data based on the given parameters
            self.core.calc_pop(self.core.p, pop_params, self.core.t_all)
            # Only proceed if the calculation status is valid
            if self.core.p.calc_status:
                for i in range(0, self.core.sample_num):
                    # Traverse the file path if multiple samples are generated
                    if self.core.sample_num != 1:
                        exp_data_path=self.core.traverse_path(i, exp_data_path)
                    # print(self.core.exp_data_path)
                    ## Write new data to file
                    self.write_new_data(self.core.p, exp_data_path)
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
            self.core.calc_all_pop(pop_params, self.core.t_all)
            if self.core.p.calc_status and self.core.p_NM.calc_status and self.core.p_M.calc_status:
                for i in range(0, self.core.sample_num):
                    if self.core.sample_num != 1:
                        # Traverse the file paths for multiple samples
                        exp_data_paths = self.core.traverse_path(i, exp_data_paths)
                    # Write data for each dimension to separate files
                    self.write_new_data(self.core.p, exp_data_paths[0])
                    self.write_new_data(self.core.p_NM, exp_data_paths[1])
                    self.write_new_data(self.core.p_M, exp_data_paths[2])
            else:
                return
    def write_new_data(self, pop, exp_data_path):
        """
        Saves the calculation results in the format of experimental data.

        This method saves the population distribution data into an Excel file in the format used for
        experimental data. It supports both smoothed and non-smoothed distributions, and can apply noise 
        to the data if required.

        Parameters
        ----------
        pop : :class:`pop.population`
            The population instance for which data is being generated.
        exp_data_path : str
            The file path where the experimental data will be saved.

        Returns
        -------
        None
        """
        # Return if the population data is invalid (calculation status is False)
        if not pop.calc_status:
            return
        # Get particle size and volume in the dPBE grid
        x_uni = pop.calc_x_uni()
        v_uni = pop.calc_v_uni()
        # Format the simulation times for the output file
        formatted_times = write_read_exp.convert_seconds_to_time(self.core.t_all)
        # Initialize the sumN_uni array to store particle count distributions
        sumN_uni = np.zeros((len(x_uni)-1, len(self.core.t_all)))
        
        for idt in self.idt_vec[1:]:
            if self.core.smoothing:
                # Get the volume distribution at the current time step
                sumvol_uni = pop.return_distribution(t=idt, flag='sum_uni')[0]
                # Skip index=0, as particles with volume 0 theoretically do not exist
                kde = self.core.KDE_fit(x_uni[1:], sumvol_uni[1:])
                # Smooth the distribution using KDE and insert a zero for the first entry
                q3 = self.core.KDE_score(kde, x_uni[1:])
                q3 = np.insert(q3, 0, 0.0)
                # Calculate and normalize Q3 values
                Q3 = self.core.calc_Q3(x_uni, q3)
                Q3 = Q3 / Q3.max()
                # Calculate the final smoothed particle volume distribution
                sumvol_uni = self.core.calc_sum_uni(Q3, sumvol_uni.sum())
                # Store the particle count distribution
                sumN_uni[:, idt] = sumvol_uni[1:] / v_uni[1:]
            else:
                # Use the unsmoothed distribution for this time step
                sumN_uni[:, idt] = pop.return_num_distribution(t=idt, flag='sumN_uni')[0][1:]
        # For initialization data, do not apply smoothing
        for idt in self.core.idt_init:
            sumN_uni[:, idt] = pop.return_num_distribution(t=idt, flag='sumN_uni')[0][1:]
        # Apply noise to the data if noise is enabled
        if self.core.add_noise:
            sumN_uni = self.core.function_noise(sumN_uni)
            
        # Create a DataFrame for the distribution data and set the index name
        df = pd.DataFrame(data=sumN_uni, index=x_uni[1:], columns=formatted_times)
        df.index.name = 'Circular Equivalent Diameter'
        # Save the DataFrame as an Excel file at the specified path
        df.to_excel(exp_data_path)
        return        
    def find_opt_kernels(self, method='kernels', data_names=None, known_params=None):
        """
        Finds optimal kernels for the PBE model by minimizing the difference between 
        simulation results and experimental data.

        This method optimizes kernel parameters for the PBE (Population Balance Equation) model. 
        It supports two optimization methods: 'kernels' and 'delta'. The 'kernels' method optimizes 
        the kernel for each dataset and computes an average kernel, while 'delta' uses averaged 
        delta values before optimization.

        Parameters
        ----------
        method : str, optional
            The optimization method to use. Options are:
                - 'kernels': Optimizes kernel parameters for each data set and averages the results.
                - 'delta': Averages the delta values before optimization, leading to a single kernel.
        data_names : str or list of str, optional
            The name(s) of the experimental data file(s). If multiple datasets are provided, 
            the optimization will be performed for each dataset.
        known_params : list, optional
            Known parameters to be used during optimization. This should match the length of `data_names`.

        Returns
        -------
        dict or list of dict
            A dictionary (or a list of dictionaries for multiple datasets) containing optimized 
            kernels and their respective optimization results.
        """
        # Warn if the component or population parameters have not been set
        if self.core.set_comp_para_flag is False:
            warnings.warn('Component parameters have not been set')
        if self.core.set_init_pop_para_flag is False:
            warnings.warn('Initial PBE parameters have not been set')
        # Warn if data_names are not provided    
        if data_names == None:
            raise ValueError("Please specify the name of the experiment data without labels!")

        self.print_highlighted(f"Now the flag of resume tuning is: {self.core.resume_unfinished}", 
                               title="INFO", color="cyan")
        
        # Helper function to construct full file paths for the data files
        def join_paths(names):
            if isinstance(names, list):
                return [os.path.join(self.data_path, name) for name in names]
            return os.path.join(self.data_path, names)
        
        exp_data_paths = []
        if isinstance(known_params, dict) and not known_params:
            known_params = None
        # Handle multi-flag (whether auxiliary 1D data is used for 2D-PBE)
        if self.multi_flag:
            # We are dealing with multiple datasets
            if not self.single_case:
                # Ensure known_params is of the same length as data_names, even if empty
                if known_params is None:
                    known_params = [None] * len(data_names)
                # Generate file paths for multiple datasets
                for data_names_ex in data_names:
                    exp_data_paths.append(join_paths(data_names_ex))
            else:
                # Single dataset optimization
                exp_data_paths = join_paths(data_names)
        else:
            if not self.single_case:
                if known_params is None:
                    known_params = [None] * len(data_names)
                        
            exp_data_paths = join_paths(data_names)
        # Initialize ray for parallel computation
        # log_to_driver = True if self.core.verbose != 0 else False
        # ray.init(log_to_driver=log_to_driver)
        # ray.init(address=os.environ["ip_head"], log_to_driver=log_to_driver)
        if method == 'kernels':
            # Currently, this method is not implemented
            self.print_highlighted("not coded yet", title="ERROR", color="red")
        elif method == 'delta':
            # Perform multi-job optimization if enabled
            if self.core.multi_jobs:
                result_dict = self.multi_optimierer_ray(self.opt_params,exp_data_paths=exp_data_paths, 
                                                               known_params=known_params)
            else:
                # Perform sequential optimization for multiple datasets
                result_dict = []
                if not self.single_case and not self.core.exp_data:
                    for exp_data_paths_tem, known_params_tem in zip(exp_data_paths, known_params):
                        result_dict_tem = self.optimierer_ray(self.opt_params,exp_data_paths=exp_data_paths_tem,
                                                                    known_params=known_params_tem)
                        result_dict.append(result_dict_tem)
                else:
                    # Perform optimization for a single dataset
                    result_dict = self.optimierer_ray(self.opt_params,exp_data_paths=exp_data_paths,
                                                           known_params=known_params)
        # Print the current actors (for debugging purposes) and shut down ray   
        # self.print_current_actors()
        # ray.shutdown()
        return result_dict
            
    def calc_PSD_delta(self, params, exp_data_path):
        """
        Directly calculates the delta value using the input parameters.
ulated delta by comparing the 
        particle size distribution (PSD) from the input parameters with the experimental data.
        It can also be used to compute the theoretical minimum delta 
        This method is used to validate the accuracy of the calcvalue for synthetically 
        generated data.

        Parameters
        ----------
        params : dict
            The parameters used to calculate the particle size distribution (PSD) and delta value.
        exp_data_path : str or list of str
            The path(s) to the experimental or synthetic PSD data file(s). For multi-dimensional 
            data, this is a list containing paths for both 1D and 2D data.

        Returns
        -------
        tuple
            A tuple containing:
                - delta : float
                    The calculated difference between the input parameters and the experimental PSD.
                - exp_data_path_ori : str
                    The original experimental data path.
        """
        # Initialize particle distribution if necessary
        if self.core.calc_init_N:
            self.core.set_init_N(exp_data_path, init_flag='mean')
        if isinstance(exp_data_path, list):
            # Handle multi-dimensional data (1D + 2D)
            exp_data_path_ori = exp_data_path[0]
            x_uni_exp = []
            data_exp = []
            for exp_data_path_tem in exp_data_path:
                if self.core.exp_data:
                    x_uni_exp_tem, data_exp_tem = self.core.get_all_exp_data(exp_data_path_tem)
                else:
                    x_uni_exp_tem, data_exp_tem = self.core.get_all_synth_data(exp_data_path_tem)
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
        else:
            exp_data_path_ori = exp_data_path
            if self.core.exp_data:
                x_uni_exp, data_exp = self.core.get_all_exp_data(exp_data_path)
            else:
                x_uni_exp, data_exp = self.core.get_all_synth_data(exp_data_path)
        # Calculate the delta value based on the difference between simulated and experimental data
        delta = self.core.calc_delta(params, x_uni_exp, data_exp)
        return delta, exp_data_path_ori
    
    def print_highlighted(self, message, title=None, color="yellow", separator=True, timestamp=True, width=80):
        """
        Print a highlighted message with optional color, timestamp, and separator.
    
        Parameters:
            message (str): The message to print.
            title (str, optional): Title for the message (e.g., "WARNING", "INFO").
            color (str, optional): Color of the message ("red", "green", "yellow", "blue", "cyan", etc.).
            separator (bool, optional): Whether to print a separator line before the message.
            timestamp (bool, optional): Whether to include a timestamp.
            width (int, optional): The width of the separator line.
    
        Colors supported:
            - "red", "green", "yellow", "blue", "magenta", "cyan", "white"
        """
    
        # ANSI color codes
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m"
        }
        
        color_code = colors.get(color.lower(), colors["yellow"])
        
        # Build the output string
        output = ""
        
        if separator:
            output += "=" * width + "\n"  # Print a separator line
        
        if timestamp:
            time_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
            output += f"{time_str} "
        
        if title:
            output += f"[{title.upper()}] "
    
        output += f"{color_code}{message}{colors['reset']}"  # Apply color formatting
    
        print(output, file=sys.stdout)
# Bind methods from another module into this class    
bind_methods_from_module(OptBase, 'optframework.kernel_opt.opt_base_ray')        
