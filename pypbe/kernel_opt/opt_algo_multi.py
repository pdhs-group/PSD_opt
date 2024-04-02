# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:05:42 2023

@author: px2030
"""

from .opt_algo import opt_algo        

class opt_algo_multi(opt_algo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_2d = 1
        
        
    def calc_delta(self, corr_beta=None, alpha_prim=None, scale=1, sample_num=1, exp_data_path=None):  
        self.calc_all_pop(corr_beta, alpha_prim, self.t_vec)
        
        delta = self.calc_delta_tem(sample_num, exp_data_path[0], scale, self.p)
        delta_NM = self.calc_delta_tem(sample_num, exp_data_path[1], scale, self.p_NM)
        delta_M = self.calc_delta_tem(sample_num, exp_data_path[2], scale, self.p_M)
        # increase the weight of the 2D case
        delta_sum = delta * self.weight_2d + delta_NM + delta_M
            
        return delta_sum
    
    def calc_delta_agg(self, params, scale=1, sample_num=1, exp_data_path=None): 
        if "corr_agg" in params:
            corr_agg = params["corr_agg"]
            CORR_BETA = self.return_syth_beta(corr_agg)
            alpha_prim = corr_agg / CORR_BETA
            
            params["CORR_BETA"] = CORR_BETA
            params["alpha_prim"] = alpha_prim
            
            del params["corr_agg"]
        
        self.calc_all_pop(params, self.t_vec)
        if self.p.calc_status == 0 and self.p_NM.calc_status == 0 and self.p_M.calc_status == 0:
            delta = self.calc_delta_tem(sample_num, exp_data_path[0], scale, self.p)
            delta_NM = self.calc_delta_tem(sample_num, exp_data_path[1], scale, self.p_NM)
            delta_M = self.calc_delta_tem(sample_num, exp_data_path[2], scale, self.p_M)
            # increase the weight of the 2D case
            delta_sum = delta * self.weight_2d + delta_NM + delta_M
                
            return delta_sum
        else:
            return scale
        
    def calc_all_pop(self, params=None, t_vec=None):
        self.calc_pop(self.p_NM, params, t_vec)
        self.calc_pop(self.p_M, params, t_vec)
        self.calc_pop(self.p, params, t_vec)       
        
