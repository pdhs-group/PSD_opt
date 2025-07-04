# -*- coding: utf-8 -*-
"""
Calculate the difference between the PSD of the simulation results and the experimental data.
"""
import numpy as np
from ray import tune
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from optframework.dpbe.dpbe_base import DPBESolver
from optframework.utils.func.bind_methods import bind_methods_from_module , unbind_methods_from_class

class OptCore():
    """
    Class definition for calculations within the optimization process.

    This class is responsible for handling experimental data processing, calculating 
    the error (delta) between the experimental particle size distribution (PSD) and the 
    PSD generated by the population balance equation (PBE). It accepts trial parameters 
    from the optimizer, processes and computes the results, and passes the calculated delta 
    back to the optimizer to be used as the optimization objective.

    Note
    ----
    This class uses the `bind_methods_from_module` function to dynamically bind methods from external
    modules. Some methods in this class are not explicitly defined here, but instead are imported from
    other files. To fully understand or modify those methods, please refer to the corresponding external
    files.
    """
    def __init__(self):
        """
        Initialize the OptCore class with default values for key attributes.
        
        """
        self.calc_init_N = False
        self.set_init_pop_para_flag = False
        self.set_comp_para_flag = False
        self.init_N_NM = None
        self.init_N_M = None
        self.init_N_2D = None
        self.init_N = None
        self.mean_delta = True
        
 
    def init_attr(self, core_params):
        """
        Initialize the attributes of the class based on the provided core parameters.
        
        This method sets up the time vectors (`t_init` and `t_vec`), initializes the number 
        of time steps, and validates the `delta_t_start_step`. It combines the initial and 
        main time vectors into a single array and calculates the corresponding indices for 
        initialization.
        
        Parameters
        ----------
        core_params : dict
            Dictionary containing the core parameters for initializing the optimization process.
        
        Raises
        ------
        Exception
            If `delta_t_start_step` is out of bounds of the time vector (`t_vec`).
        """
        # Initialize class attributes from core_params dictionary
        for key, value in core_params.items():
            setattr(self, key, value)
        # Convert time vectors to float and store the number of time steps
        self.t_init = self.t_init.astype(float)
        self.t_vec = self.t_vec.astype(float)
        self.num_t_init = len(self.t_init)
        self.num_t_steps = len(self.t_vec)
        # Validate the delta_t_start_step value
        if self.delta_t_start_step >= self.num_t_steps:
            raise Exception("The value of delta_t_start_step must be within the indices range of {self.t_vec}!")
        # Combine the initialization and main time vectors into a single array
        self.t_all = np.concatenate((self.t_init, self.t_vec))
        self.t_all = np.unique(self.t_all)
        # Get indices corresponding to the initial time vector in the combined time array
        self.idt_init = [np.where(self.t_all == t_time)[0][0] for t_time in self.t_init]
        
    def init_pbe(self, pop_params, data_path):
        """
        Initialize the PBE solver.
        
        This method creates and initializes a `DPBESolver` for solving the population balance 
        equations. If the dimension (`dim`) is 2, it also creates a 1D PBE solver to assist 
        with the initialization of the 2D system.
        
        Parameters
        ----------
        pop_params : dict
            The parameters used for initializing the PBE solver.
        data_path : str
            The path to the data directory for loading component parameters.
        """
        # Initialize the PBE solver
        self.p = DPBESolver(dim=self.dim, disc='geo', t_vec=self.t_vec, load_attr=False)
        # If the dimension is 2, also create a 1D population for initialization
        if self.dim == 2:
            self.create_1d_pop(self.t_vec, disc='geo')
        # Set the initial population parameters and component parameters
        self.set_init_pop_para(pop_params)
        self.set_comp_para(data_path)
        self.p.reset_params()
        if self.dim == 2:
            self.p_NM.reset_params()
            self.p_M.reset_params()
        
    def calc_delta(self, params_in, x_uni_exp, data_exp):
        """
        This method calculates the PSD difference (delta). It first converts the input `corr_agg` 
        (agglomeration rate) into the equivalent `CORR_BETA` and `alpha_prim`.Then the method runs 
        the population balance equation (PBE) with the computed parameters and compares the generated
        PSD with the experimental PSD to calculate the delta value.

        Parameters
        ----------
        params_in : dict
            The population parameters.
        x_uni_exp : array-like
            The unique particle diameters in experimental data.
        data_exp : array-like
            The experimental PSD data that will be compared with the PBE-generated PSD.

        Returns
        -------
        float
            The calculated delta value representing the difference between the experimental PSD 
            and the PBE-generated PSD. If the PBE calculation fails, it returns a large positive 
            value (e.g., 10) to indicate failure.
        """
        params = self.check_corr_agg(params_in)
        
        # Run the PBE calculations using the provided parameters
        self.calc_pop(self.p, params, self.t_vec, init_N=self.init_N)
        
        # If the PBE calculation is successful, calculate the delta
        if self.p.calc_status:
            return self.calc_delta_tem(x_uni_exp, data_exp, self.p)
        else:
            # Return a large delta value if the PBE calculation fails
            return 10

    def calc_delta_tem(self, x_uni_exp, data_exp, pop):
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
        pop : :class:`pop.DPBESolver`
            An instance of the PBE solver, which generates the simulated PSD.
 
        Returns
        -------
        float
            The average difference (delta) between the experimental PSD and the simulated PSD, 
            normalized by the number of particle sizes in the experimental data (`x_uni_exp`).
        """
        # If smoothing is enabled, initialize a list to store KDE objects
        if self.smoothing:
            kde_list = []
            
        # Get the unique particle size in PBE
        x_uni = pop.calc_x_uni()
        
        # Loop through time steps to collect the simulation results and convert to PSD
        for idt in range(self.delta_t_start_step, self.num_t_steps):
            if self.smoothing:
                sum_uni = pop.return_distribution(t=idt, flag='sum_uni', q_type=self.dist_type)[0]
                # Volume of particles with index=0 is 0; in theory, such particles do not exist
                kde = self.KDE_fit(x_uni[1:], sum_uni[1:])
                # The qx distribution measured by the Lumisizer is typically matched using the 
                # average values of two measurement nodes, so the corresponding conversion has also been performed here.
                # x_uni_m = (x_uni[:-1]+x_uni[1:]) / 2
                # kde = self.KDE_fit(x_uni_m, sum_uni[1:])
                kde_list.append(kde)
                
        delta_sum = 0 
        # Single sample case
        if self.sample_num == 1:
            qx_mod = np.zeros((len(x_uni_exp), self.num_t_steps-self.delta_t_start_step))
            for idt in range(self.num_t_steps-self.delta_t_start_step):
                if self.smoothing:
                    qx_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp[1:])
                    qx_mod[1:, idt] = qx_mod_tem
                else:
                    qx_mod[:, idt] = pop.return_distribution(t=idt+self.delta_t_start_step, flag='qx', q_type=self.dist_type)[0]
                Qx = self.calc_Qx(x_uni_exp, qx_mod[:, idt]) 
                qx_mod[:, idt] = qx_mod[:, idt] / Qx.max() 
            # Calculate the delta for each cost function type, if is defined.
            for flag, cost_func_type in self.delta_flag:
                data_mod = pop.re_calc_distribution(x_uni_exp, qx=qx_mod, flag=flag)[0]
                delta = self.cost_fun(data_exp, data_mod, cost_func_type, flag)
                delta_sum += delta 
            
            return delta
        
        # Multiple sample case
        else:
            for i in range (0, self.sample_num):
                qx_mod = np.zeros((len(x_uni_exp[i]), self.num_t_steps-self.delta_t_start_step))
                for idt in range(self.num_t_steps-self.delta_t_start_step):
                    if self.smoothing:
                        qx_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp[i][1:])
                        qx_mod[1:, idt] = qx_mod_tem
                    else:
                        qx_mod[:, idt] = pop.return_distribution(t=idt+self.delta_t_start_step, flag='qx')[0]
                    Qx = self.calc_Qx(x_uni_exp[i], qx_mod[:, idt]) 
                    qx_mod[:, idt] = qx_mod[:, idt] / Qx.max()
                # Calculate delta for each cost function type, if is defined.    
                for flag, cost_func_type in self.delta_flag:
                    data_mod = pop.re_calc_distribution(x_uni_exp[i], qx=qx_mod, flag=flag)[0]
                    delta = self.cost_fun(data_exp[i], data_mod, cost_func_type, flag)
                    delta_sum += delta 
                
            delta_sum /= self.sample_num
            return delta_sum
        
    def check_corr_agg(self, params_in):
        """
        Process the `corr_agg` parameter by converting it to  equivalent `CORR_BETA` and `alpha_prim`.
        Since `corr_agg` is not a parameter in the PBE calculations, it is an artificially 
        defined parameter that accounts for the characteristics of Frank's collision model. 
        
        Parameters
        ----------
        params_in : dict
            A dictionary of input parameters that may include `corr_agg`.
        
        Returns
        -------
        dict
            The updated parameters dictionary with `CORR_BETA` and `alpha_prim`, and without `corr_agg`.
        """
        params = params_in.copy()
        
        # If corr_agg is in the parameters, calculate CORR_BETA and alpha_prim if needed
        if "corr_agg" in params:
            corr_agg = params["corr_agg"]
            if "CORR_BETA" not in params:
                CORR_BETA = self.return_syth_beta(corr_agg)
                params["CORR_BETA"] = CORR_BETA
            else:
                CORR_BETA = params["CORR_BETA"]
                print("Detected that CORR_BETA was entered as a known parameter")
            if "alpha_prim" not in params:
                alpha_prim = corr_agg / CORR_BETA
                params["alpha_prim"] = alpha_prim
            else:
                print("Detected that alpha_prim was entered as a known parameter")
            
            # Remove corr_agg from the parameters
            del params["corr_agg"]
        return params
    
    def return_syth_beta(self,corr_agg):
        """
        Calculate and return a synthetic beta value based on `corr_agg`.

        Parameters
        ----------
        corr_agg : array-like
            The correction factors for agglomeration rate.

        Returns
        -------
        float
            The synthetic beta value, which is a scaling factor derived from `corr_agg`.
        """
        max_val = max(corr_agg)
        power = np.log10(max_val)
        power = np.ceil(power)
        return 10**power
    
    def array_dict_transform(self, array_dict_in):
        """
        Transform the dictionary to handle `corr_agg` arrays based on the dimensionality.
        
        This method processes a dictionary containing `corr_agg` values and adjusts them based 
        on the dimensionality of the PBE problem. For 1D problems, it keeps a single value, and 
        for 2D problems, it creates an array with three elements. After transforming the dictionary, 
        it deletes the original keys.
        
        Parameters
        ----------
        array_dict : dict
            A dictionary containing the `corr_agg` values that need to be transformed.
        
        Returns
        -------
        dict
            The transformed dictionary with `corr_agg` as an array.
        """
        array_dict = array_dict_in.copy()
        if self.p.dim == 1:
            array_dict['corr_agg'] = np.array([array_dict['corr_agg_0']])
            del array_dict["corr_agg_0"]
        elif self.p.dim == 2:
            array_dict['corr_agg'] = np.array([array_dict[f'corr_agg_{i}'] for i in range(3)])
            for i in range(3):
                del array_dict[f'corr_agg_{i}']
        return array_dict
        
    def cost_fun(self, data_exp, data_mod, cost_func_type, flag):
        """
        Calculate the cost (error) between experimental and PBE-generated data
        using various cost functions such as MSE, RMSE, MAE, and 
        Kullback-Leibler (KL) divergence.

        Parameters
        ----------
        data_exp : array-like
            The experimental data.
        data_mod : array-like
            The model-generated data from the PBE simulation.
        cost_func_type : str
            The type of cost function to use. Options include:
                - 'MSE': Mean Squared Error
                - 'RMSE': Root Mean Squared Error
                - 'MAE': Mean Absolute Error
                - 'KL': Kullback-Leibler divergence (only for 'qx' or 'Qx').
        flag : str
            A flag indicating whether to use 'qx' or 'Qx' for KL divergence.

        Returns
        -------
        float
            The calculated cost between the experimental and simulated data.
        
        Raises
        ------
        Exception
            If an unsupported cost function type is provided.
        """
        if cost_func_type == 'MSE':
            return mean_squared_error(data_mod, data_exp)
        elif cost_func_type == 'RMSE':
            mse = mean_squared_error(data_mod, data_exp)
            return np.sqrt(mse)
        elif cost_func_type == 'MAE':
            return mean_absolute_error(data_mod, data_exp)
        elif (flag == 'qx' or flag == 'Qx') and cost_func_type == 'KL':
            # Replace very small values with a minimum threshold to avoid division by zero in KL divergence
            data_mod = np.where(data_mod <= 10e-20, 10e-20, data_mod)
            data_exp = np.where(data_exp <= 10e-20, 10e-20, data_exp)
            return entropy(data_mod, data_exp).mean()
        else:
            raise Exception("Current cost function type is not supported")
            
    def print_notice(self, message):
        """
        Print a formatted notice message with a border and padding.
        
        Parameters
        ----------
        message : str
            The message to be displayed in the notice.
        """
        # Add extra empty lines before and after the message
        empty_lines = "\n" * 5
        
        # Define the border and padding
        border = "*" * 80
        padding = "\n" + " " * 10  # 左右留白
        
        # Construct the formatted notice
        notice = f"{empty_lines}{border}{padding}{message}{padding}{border}{empty_lines}"
        
        # Print the formatted notice
        print(notice)

# Bind methods from other modules into this class
bind_methods_from_module(OptCore, 'optframework.kernel_opt.opt_algo_bo')
bind_methods_from_module(OptCore, 'optframework.kernel_opt.opt_data')
bind_methods_from_module(OptCore, 'optframework.kernel_opt.opt_pbe')
bind_methods_from_module(OptCore, 'optframework.dpbe.dpbe_post')
methods_to_remove = ['calc_v_uni','calc_x_uni', 'return_distribution',
                     'return_N_t','calc_mom_t']
unbind_methods_from_class(OptCore, methods_to_remove)


