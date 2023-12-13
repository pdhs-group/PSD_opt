# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:05:42 2023

@author: px2030
"""

import os
from pop import population
from kernel_opt import kernel_opt        

class multi_opt(kernel_opt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_1d_pop(disc='geo')
        self.weight_2d = 1
        
    def cal_delta(self, corr_beta=None, alpha_prim=None, scale=1, Q3_exp=None, 
                        x_50_exp=None, sample_num=1, exp_data_path=None):       
        self.cal_all_pop(corr_beta, alpha_prim)
        
        delta = self.cal_delta_tem(sample_num, exp_data_path[0], scale, self.p)
        delta_NM = self.cal_delta_tem(sample_num, exp_data_path[1], scale, self.p_NM)
        delta_M = self.cal_delta_tem(sample_num, exp_data_path[2], scale, self.p_M)
        # increase the weight of the 2D case
        delta_sum = delta * self.weight_2d + delta_NM + delta_M
            
        return delta_sum
        
    def cal_all_pop(self, corr_beta, alpha_prim):
        self.cal_pop(self.p_NM, corr_beta, alpha_prim[0])
        self.cal_pop(self.p_M, corr_beta, alpha_prim[2])
        self.cal_pop(self.p, corr_beta, alpha_prim)       
    
    def create_1d_pop(self, disc='geo'):
        
        self.p_NM = population(dim=1,disc=disc)
        self.p_M = population(dim=1,disc=disc)
        self.set_comp_para()
        
    def set_comp_para(self, R_NM=None, R_M=None):
        if R_NM!=None and R_M!=None:
            self.p.R01 = R_NM
            self.p.R03 = R_M
        # parameter for particle component 1 - NM
        self.p_NM.R01 = self.p.R01
        self.p_NM.DIST1 = os.path.join(self.p.pth,"data\\PSD_data\\")+'PSD_x50_1.0E-6_r01_2.9E-7.npy'
        
        # parameter for particle component 2 - M
        self.p_M.R01 = self.p.R03
        self.p_M.DIST1 = os.path.join(self.p.pth,"data\\PSD_data\\")+'PSD_x50_1.0E-6_r01_2.9E-7.npy'