# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:05:42 2023

@author: px2030
"""

from opt_algo import opt_algo        

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
    
    def calc_delta_agg(self, corr_agg=None, scale=1, sample_num=1, exp_data_path=None): 
        corr_beta = self.return_syth_beta(corr_agg)
        alpha_prim = corr_agg / corr_beta
        
        self.calc_all_pop(corr_beta, alpha_prim, self.t_vec)
        
        delta = self.calc_delta_tem(sample_num, exp_data_path[0], scale, self.p)
        delta_NM = self.calc_delta_tem(sample_num, exp_data_path[1], scale, self.p_NM)
        delta_M = self.calc_delta_tem(sample_num, exp_data_path[2], scale, self.p_M)
        # increase the weight of the 2D case
        delta_sum = delta * self.weight_2d + delta_NM + delta_M
            
        return delta_sum
        
    def calc_all_pop(self, corr_beta, alpha_prim, t_vec=None):
        self.calc_pop(self.p_NM, corr_beta, alpha_prim[0], t_vec)
        self.calc_pop(self.p_M, corr_beta, alpha_prim[2], t_vec)
        self.calc_pop(self.p, corr_beta, alpha_prim, t_vec)       
        
