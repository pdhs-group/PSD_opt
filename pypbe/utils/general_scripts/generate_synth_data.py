# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
import numpy as np
import multiprocessing
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../../.."))
from generate_psd import full_psd
import pypbe.kernel_opt.opt_find as opt
import config.opt_config as conf

def calc_function(R01_0, R03_0, dist_path_NM, dist_path_M, var_pop_params):
    #%%  Input for Opt 
    find = opt.opt_find()
     
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    multi_flag = conf.config['multi_flag']
    opt_params = conf.config['opt_params']
    
    find.init_opt_algo(multi_flag, algo_params, opt_params)
    
    find.algo.set_init_pop_para(pop_params)
    
    # find.algo.set_comp_para(R_NM=conf.config['R_NM'], R_M=conf.config['R_M'])
    
    find.algo.weight_2d = conf.config['weight_2d']
    find.algo.calc_init_N = False
    find.algo.set_comp_para(R01_0, R03_0, dist_path_NM, dist_path_M,R01_0_scl=1e-1,R03_0_scl=1e-1)
    
    # find.algo.calc_all_pop(var_pop_params, find.algo.t_vec)
    # calc_status = find.algo.p.calc_status
    # calc_NM_status = find.algo.p_NM.calc_status
    # calc_M_status = find.algo.p_M.calc_status
    b = var_pop_params['CORR_BETA']
    a = var_pop_params['alpha_prim']
    v = var_pop_params['pl_v']
    p1 = var_pop_params['pl_P1']
    p2 = var_pop_params['pl_P2']
    p3 = var_pop_params['pl_P3']
    p4 = var_pop_params['pl_P4']
    p5 = var_pop_params['pl_P5']
    p6 = var_pop_params['pl_P6']
    add_info = f"_para_{b}_{a[0]}_{a[1]}_{a[2]}_{v}_{p1}_{p2}_{p3}_{p4}_{p5}_{p6}"
    # Generate synthetic Data
    find.generate_data(var_pop_params, find.algo.sample_num, add_info=add_info)
    
if __name__ == '__main__':
    generate_new_psd = True
    
    if generate_new_psd:
        ## Input for generating psd-data
        x50 = 2   # /mm
        resigma = 0.15
        minscale = 0.5
        maxscale = 2
        dist_path_1 = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        dist_path_5 = full_psd(x50*5, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        dist_path_10 = full_psd(x50*10, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
    else:
        pth = os.path.dirname( __file__ )
        dist_path_1 = os.path.join(pth, "..", "..", "data", "PSD_data", conf.config['dist_scale_1'])
        dist_path_5 = os.path.join(pth, "..", "..", "data", "PSD_data", conf.config['dist_scale_5'])
        dist_path_10 = os.path.join(pth, "..","..", "data", "PSD_data", conf.config['dist_scale_10'])

    ## define the range of corr_beta
    var_corr_beta = np.array([1e-2, 1e0, 1e2])
    # var_corr_beta = np.array([1e-2])
    ## define the range of alpha_prim 27x3
    values = np.array([0, 0.5, 1])
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    ## The case of all zero α is meaningless, that means no Agglomeration occurs
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]

    ## define the range of v(breakage function)
    var_v = np.array([0.1,1,2])
    # var_v = np.array([0.01])
    ## define the range of P1, P2 for power law breakage rate
    var_P1 = np.array([1e-4,1e-2])
    var_P2 = np.array([0.1,0.5])
    var_P3 = np.array([1e-4,1e-2])
    var_P4 = np.array([0.1,0.5])
    var_P5 = np.array([1e-4,1e-2])
    var_P6 = np.array([0.1,1])
    # var_P1 = np.array([1])
    # var_P2 = np.array([0.0])
    
    ## define the range of particle size scale and minimal size
    dist_path = [dist_path_1] # [dist_path_1, dist_path_10]
    size_scale = np.array([1, 10])
    R01_0 = 'r0_001'
    R03_0 = 'r0_001'

    func_list = []
    for i, dist in enumerate(dist_path):
        ## Reinitialization of pop equations using psd data  
        dist_path_NM = dist_path[0]
        dist_path_M = dist
        scale = size_scale[i]
        for j,corr_beta in enumerate(var_corr_beta):
            for k,alpha_prim in enumerate(var_alpha_prim):
                for l,v in enumerate(var_v):
                    for m1,P1 in enumerate(var_P1):
                        for m2,P2 in enumerate(var_P2):
                            for m3,P3 in enumerate(var_P3):
                                for m4,P4 in enumerate(var_P4):
                                    for m5,P5 in enumerate(var_P5):
                                        for m6,P6 in enumerate(var_P6):
                                            ## Set parameters for PBE
                                            conf_params = {
                                                'pop_params':{
                                                    'CORR_BETA' : corr_beta,
                                                    'alpha_prim' : alpha_prim,
                                                    'pl_v' : v,
                                                    'pl_P1' : P1,
                                                    'pl_P2' : P2,
                                                    'pl_P3' : P3,
                                                    'pl_P4' : P4,
                                                    'pl_P5' : P5,
                                                    'pl_P6' : P6,
                                                    }
                                                }
                                            var_pop_params = conf_params['pop_params']
                                            func_list.append((R01_0, R03_0, dist_path_NM, dist_path_M, var_pop_params))
    pool = multiprocessing.Pool(processes=12)
    pool.starmap(calc_function, func_list)                        
    pool.close()
    pool.join()                        
                   