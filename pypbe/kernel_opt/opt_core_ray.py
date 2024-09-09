# -*- coding: utf-8 -*-
"""
Calculate the difference between the PSD of the simulation results and the experimental data.
Minimize the difference by optimization algorithm to obtain the kernel of PBE.
"""
import os
import time
from ray import tune
from .opt_core import OptCore

class OptCoreRay(OptCore, tune.Trainable):
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
    def __init__(self, *args, **kwargs):
        tune.Trainable.__init__(self, *args, **kwargs)
    def setup(self, config, core_params, pop_params, data_path, x_uni_exp, data_exp, known_params):
        self.init_attr(core_params)
        self.init_pbe(pop_params, data_path)
        self.known_params = known_params
        self.x_uni_exp = x_uni_exp
        self.data_exp = data_exp
        self.reuse_num=0
    
    def step(self):
        start_time = time.time()
        # Special handling for corr_agg based on dimension
        if 'corr_agg_0' in self.config:
            transformed_params = self.array_dict_transform(self.config)
        else:
            transformed_params = self.config
        
        if self.known_params is not None:
            for key, value in self.known_params.items():
                if key in transformed_params:
                    print(f"Warning: Known parameter '{key}' are set for optimization.")
                transformed_params[key] = value
        loss = self.calc_delta(transformed_params, self.x_uni_exp, self.data_exp)
        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time < 2:
            time.sleep(2 - execution_time)
        return {"loss": loss, "reuse_num": self.reuse_num}
    def save_checkpoint(self, checkpoint_dir):
        return None
    def load_checkpoint(self, checkpoint_path):
        pass
    def reset_config(self, new_config):
        # self.config = new_config
        self.reuse_num += 1
        return True
        


