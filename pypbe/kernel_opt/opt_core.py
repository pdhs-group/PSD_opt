# -*- coding: utf-8 -*-
"""
Calculate the difference between the PSD of the simulation results and the experimental data.
Minimize the difference by optimization algorithm to obtain the kernel of PBE.
"""
import os ,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pypbe.pbe.dpbe_base import bind_methods_from_module, unbind_methods_from_class

class OptCore():
    """
    Class definition for calculations within optimization process class.

    Attributes
    ----------
    n_iter : `int`, optional
        Number of iterations of the optimization process. Default is 100.
    delta_flag : `str`, optional
        Which data from the PSD is used for the calculation. Default is 'q3'. Options include:
        
        - 'q3': Number density distribution
        - 'Q3': Cumulative distribution
        - 'x_10': Particle size corresponding to 10% cumulative distribution
        - 'x_50': Particle size corresponding to 50% cumulative distribution
        - 'x_90': Particle size corresponding to 90% cumulative distribution
    cost_func_type : `str`, optional
        Method for calculating the PSD difference. Default is 'MSE'. Options include:
        
        - 'MSE': Mean Squared Error
        - 'RMSE': Root Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'KL': Kullback–Leibler divergence (Only q3 and Q3 are compatible with KL)
    calc_init_N : `bool`, optional
        Whether to use experimental data to calculate initial conditions. If False, the initial conditions for PBE need to be defined manually. Default is False.
    """
    def __init__(self):
        self.n_iter = 100
        ## delta_flag = 1: use q3
        ## delta_flag = 2: use Q3
        ## delta_flag = 3: use x_10
        ## delta_flag = 4: use x_50
        ## delta_flag = 5: use x_90
        self.delta_flag = 'q3'    
        ## 'MSE': Mean Squared Error
        ## 'RMSE': Root Mean Squared Error
        ## 'MAE': Mean Absolute Error
        ## 'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
        self.cost_func_type = 'MSE'

        self.calc_init_N = False
        self.set_init_pop_para_flag = False
        self.set_comp_para_flag = False
        self.num_bundles = 4
        # self.cpu_per_bundles = 20   
    def calc_delta(self, params_in, x_uni_exp, data_exp):
        """
        Calculate the difference (delta) of PSD.
        
        - The function is exactly the same as :meth:`~.calc_delta`, but receives corr_agg (agglomeration rate). Then calls `~.return_syth_beta` to compute the equivalent CORR_BETA and alpha_prim.
        
        Parameters
        ----------
        corr_agg : `array`
            Agglomeration rate in PBE.
        scale : `int`, optional. Default 1.
            The actual return result is delta*scale. delta is absolute and always positive. 
            Setting scale to 1 is suitable for optimizers that look for minimum values, 
            and -1 is suitable for those that look for maximum values.
        sample_num : `int`, optional. Default 1.
            Set how many sets of experimental data are used simultaneously for optimization.
        exp_data_path : `str`
            path for experimental data.
        """
        params = self.check_corr_agg(params_in)

        self.calc_pop(self.p, params, self.t_vec)
        if self.p.calc_status:
            return self.calc_delta_tem(x_uni_exp, data_exp, self.p)
        else:
            return 10

    def calc_delta_tem(self, x_uni_exp, data_exp, pop):
        """
        Loop through all the experimental data and calculate the average diffences.
        
        Parameters
        ----------
        sample_num : `int`, optional. Default 1.
            Set how many sets of experimental data are used simultaneously for optimization.
        exp_data_path : `str`
            path for experimental data.
        pop : :class:`pop.DPBESolver`
            Instance of the PBE.
            
        Returns   
        -------
        (delta_sum * scale) / x_uni_num : `float`
            Average value of PSD's difference for corresponding to all particle sizes. 
        """
        if self.smoothing:
            kde_list = []
        x_uni = pop.calc_x_uni()
        for idt in range(self.delta_t_start_step, self.num_t_steps):
            if self.smoothing:
                sumvol_uni = pop.return_distribution(t=idt, flag='sumvol_uni')[0]
                ## The volume of particles with index=0 is 0. 
                ## In theory, such particles do not exist.
                kde = self.KDE_fit(x_uni[1:], sumvol_uni[1:])
                kde_list.append(kde)
                
        delta_sum = 0    
        if self.sample_num == 1:
            q3_mod = np.zeros((len(x_uni_exp), self.num_t_steps-self.delta_t_start_step))
            for idt in range(self.num_t_steps-self.delta_t_start_step):
                if self.smoothing:
                    q3_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp[1:])
                    q3_mod[1:, idt] = q3_mod_tem
                else:
                    q3_mod[:, idt] = pop.return_distribution(t=idt+self.delta_t_start_step, flag='q3')[0]
                Q3 = self.calc_Q3(x_uni_exp, q3_mod[:, idt]) 
                q3_mod[:, idt] = q3_mod[:, idt] / Q3.max() 
            q3_mod = np.insert(q3_mod, 0, 0.0, axis=0)
            for flag, cost_func_type in self.delta_flag:
                data_mod = self.re_calc_distribution(x_uni_exp, q3=q3_mod, flag=flag)[0]
                # Calculate the error between experimental data and simulation results
                delta = self.cost_fun(data_exp, data_mod, cost_func_type, flag)
                delta_sum += delta 
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(x_uni_exp)
            return delta / x_uni_num
        else:
            for i in range (0, self.sample_num):
                q3_mod = np.zeros((len(x_uni_exp[i]), self.num_t_steps-self.delta_t_start_step))
                for idt in range(self.num_t_steps-self.delta_t_start_step):
                    if self.smoothing:
                        q3_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp[i][1:])
                        q3_mod[1:, idt] = q3_mod_tem
                    else:
                        q3_mod[:, idt] = pop.return_distribution(t=idt+self.delta_t_start_step, flag='q3')[0]
                    Q3 = self.calc_Q3(x_uni_exp[i], q3_mod[:, idt]) 
                    q3_mod[:, idt] = q3_mod[:, idt] / Q3.max()
                for flag, cost_func_type in self.delta_flag:
                    data_mod = self.re_calc_distribution(x_uni_exp[i], q3=q3_mod, flag=flag)[0]
                    # Calculate the error between experimental data and simulation results
                    delta = self.cost_fun(data_exp[i], data_mod, cost_func_type, flag)
                    delta_sum += delta 
            # Restore the original name of the file to prepare for the next step of training
            delta_sum /= self.sample_num
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(x_uni_exp[i])  
            return delta_sum / x_uni_num
        
    def check_corr_agg(self, params_in):
        params = params_in.copy()
        if "corr_agg" in params:
            corr_agg = params["corr_agg"]
            CORR_BETA = self.return_syth_beta(corr_agg)
            alpha_prim = corr_agg / CORR_BETA
            
            params["CORR_BETA"] = CORR_BETA
            params["alpha_prim"] = alpha_prim
            
            del params["corr_agg"]
        return params
    
    def return_syth_beta(self,corr_agg):
        """
        Calculate and return a synthetic beta value.

        This method calculates the maximum value of `corr_agg`, takes its logarithm to base 10, 
        rounds it up to the nearest integer, and returns 10 raised to the power of this integer.
        
        Parameters
        ----------
        corr_agg : `array`
            The correction factors for aggregation.

        Returns
        -------
        float
            The synthetic beta value
        """
        max_val = max(corr_agg)
        power = np.log10(max_val)
        power = np.ceil(power)
        return 10**power

    def cost_fun(self, data_exp, data_mod, cost_func_type, flag):
        """
        Calculate the difference(cost) between experimental and model data.
        
        This method supports multiple cost function types including Mean Squared Error (MSE), 
        Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Kullback-Leibler (KL) divergence. 
        
        Parameters
        ----------
        data_exp : `array`
            The experimental data.
        data_mod : `array`
            The data generated by the simulation.
        
        Returns
        -------
        float
            The calculated cost based on the specified cost function type.
        """
        if cost_func_type == 'MSE':
            return mean_squared_error(data_mod, data_exp)
        elif cost_func_type == 'RMSE':
            mse = mean_squared_error(data_mod, data_exp)
            return np.sqrt(mse)
        elif cost_func_type == 'MAE':
            return mean_absolute_error(data_mod, data_exp)
        elif (flag == 'q3' or flag == 'Q3') and cost_func_type == 'KL':
            data_mod = np.where(data_mod <= 10e-20, 10e-20, data_mod)
            data_exp = np.where(data_exp <= 10e-20, 10e-20, data_exp)
            return entropy(data_mod, data_exp).mean()
        else:
            raise Exception("Current cost function type is not supported")
            
    def print_notice(self, message):
        # 上下空很多行
        empty_lines = "\n" * 5
        
        # 定义符号和边框
        border = "*" * 80
        padding = "\n" + " " * 10  # 左右留白
        
        # 构建完整的提示消息
        notice = f"{empty_lines}{border}{padding}{message}{padding}{border}{empty_lines}"
        
        # 打印消息
        print(notice)

bind_methods_from_module(OptCore, 'pypbe.kernel_opt.opt_algo_ray')
bind_methods_from_module(OptCore, 'pypbe.kernel_opt.opt_algo_bo')
bind_methods_from_module(OptCore, 'pypbe.kernel_opt.opt_data')
bind_methods_from_module(OptCore, 'pypbe.kernel_opt.opt_pbe')
bind_methods_from_module(OptCore, 'pypbe.pbe.dpbe_post')
methods_to_remove = ['calc_v_uni','calc_x_uni', 'return_distribution', 'return_num_distribution',
                     'return_N_t','calc_mom_t']
unbind_methods_from_class(OptCore, methods_to_remove)


