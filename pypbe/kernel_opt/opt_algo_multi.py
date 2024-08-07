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
        
        
    def calc_delta(self, corr_beta=None, alpha_prim=None, exp_data_path=None):  
        self.calc_all_pop(corr_beta, alpha_prim, self.t_vec)

        if self.p.calc_status:
            delta = self.calc_delta_tem(exp_data_path[0], self.p)
        else:
            delta = 10
        if self.p_NM.calc_status:
            delta_NM = self.calc_delta_tem(exp_data_path[1], self.p_NM)
        else:
            delta_NM = 10
        if self.p_M.calc_status:
            delta_M = self.calc_delta_tem(exp_data_path[2], self.p_M)
        else:
            delta_M = 10
        # increase the weight of the 2D case
        delta_sum = delta * self.weight_2d + delta_NM + delta_M
            
        return delta_sum
    
    def calc_delta_agg(self, params_in,x_uni_exp, data_exp): 
        params = params_in.copy()
        if "corr_agg" in params:
            corr_agg = params["corr_agg"]
            CORR_BETA = self.return_syth_beta(corr_agg)
            alpha_prim = corr_agg / CORR_BETA
            
            params["CORR_BETA"] = CORR_BETA
            params["alpha_prim"] = alpha_prim
            
            del params["corr_agg"]
        
        self.calc_all_pop(params, self.t_vec)
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
            # increase the weight of the 2D case
        delta_sum = delta * self.weight_2d + delta_NM + delta_M
        return delta_sum        

        
    def calc_all_pop(self, params=None, t_vec=None):
        self.calc_pop(self.p_NM, params, t_vec)
        self.calc_pop(self.p_M, params, t_vec)
        self.calc_pop(self.p, params, t_vec)           
        
