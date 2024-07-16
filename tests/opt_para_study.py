# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import sys, os
import numpy as np
import multiprocessing
import logging
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from pypbe.kernel_opt import opt_find as opt
from config import opt_config as conf

logging.basicConfig(filename='parallel.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def optimization_process(algo_params,pop_params,multi_flag,opt_params,ori_params,file_name, data_path):
    #%%  Input for Opt 
    find = opt.opt_find()

    ## Update the parameter for PBE
    pop_params.update(ori_params)

    find.init_opt_algo(multi_flag, algo_params, opt_params, data_path)
    
    find.algo.set_init_pop_para(pop_params)
    
    if find.algo.p.process_type == 'breakage':
        USE_PSD = False
        dist_path_NM = None
        dist_path_M = None
    else:
        USE_PSD = True
        dist_path_NM = os.path.join(data_path, "PSD_data", conf.config['dist_scale_1'])
        dist_path_M = os.path.join(data_path, "PSD_data", conf.config['dist_scale_1'])
        
    R_NM = conf.config['R_NM']
    R_M=conf.config['R_M']
    R01_0_scl=conf.config['R01_0_scl']
    R03_0_scl=conf.config['R03_0_scl']
    R01_0 = 'r0_001'
    R03_0 = 'r0_001'
    find.algo.set_comp_para(USE_PSD, R01_0, R03_0, R_NM=R_NM, R_M=R_M,R01_0_scl=R01_0_scl,R03_0_scl=R03_0_scl,
                            dist_path_NM=dist_path_NM, dist_path_M=dist_path_M)
    
    find.algo.weight_2d = conf.config['weight_2d']

    result_dict = \
        find.find_opt_kernels(sample_num=find.algo.sample_num, method='delta', data_name=file_name)

    return result_dict, ori_params

if __name__ == '__main__':
    #%%  Input for Opt
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    multi_flag = conf.config['multi_flag']
    opt_params = conf.config['opt_params']
    weight_2d = conf.config['weight_2d']
    
    noise_type = algo_params['noise_type']
    noise_strength = algo_params['noise_strength']
    delta_flag = algo_params['delta_flag']
    method = algo_params['method']
    n_iter = algo_params['n_iter']
    
    #%% Prepare test data set
    ## define the range of corr_beta
    var_corr_beta = np.array([1e-3,1e-2,1e-1])
    # var_corr_beta = np.array([1e-2])
    ## define the range of alpha_prim 27x3
    values = np.array([0.5,1.0])
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    ## The case of all zero α is meaningless, that means no Agglomeration occurs
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]
    ## For cases where R01 and R03 have the same size, the elements of alpha_prim mirror symmetry 
    ## are equivalent and can be removed to simplify the calculation.
    unique_alpha_prim = []
    for comp in var_alpha_prim:
        comp_reversed = comp[::-1]  
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_alpha_prim):
            unique_alpha_prim.append(comp)
            
    var_alpha_prim = np.array(unique_alpha_prim)

    ## define the range of v(breakage function)
    var_v = np.array([0.7,1,2])
    # var_v = np.array([0.01])    ## define the range of P1, P2 for power law breakage rate
    var_P1 = np.array([1e-3,1e-2,1e-1])
    var_P2 = np.array([0.5,1.0,2.0])
    var_P3 = np.array([1e-3,1e-2,1e-1])
    var_P4 = np.array([0.5,1.0,2.0])


    ## define the range of particle size scale and minimal size
    pth = '/pfs/work7/workspace/scratch/px2030-MC_train'
    data_path = os.path.join(pth,"mix", "data")
    # data_path = r"C:\Users\px2030\Code\PSD_opt\pypbe\data"
    # dist_path_1 = os.path.join(data_path, "PSD_data", conf.config['dist_scale_1'])
    # dist_path = [dist_path_1] # [dist_path_1, dist_path_10]
    # size_scale = np.array([1, 10])
    # R01_0 = 'r0_001'
    # R03_0 = 'r0_001'
    results = []
    for j,corr_beta in enumerate(var_corr_beta):
        for k,alpha_prim in enumerate(var_alpha_prim):
            for l,v in enumerate(var_v):
                for m1,P1 in enumerate(var_P1):
                    for m2,P2 in enumerate(var_P2):
                        for m3,P3 in enumerate(var_P3):
                            for m4,P4 in enumerate(var_P4):
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
                                                }
                                            }
                                        file_name = f"Sim_{noise_type}_{noise_strength}_para_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_{v}_{P1}_{P2}_{P3}_{P4}.xlsx"
                                        file_path = os.path.join(data_path, file_name)
                                        file_path = file_path.replace(".xlsx", "_0.xlsx")
                                        if not os.path.exists(file_path):
                                            continue
                                        var_pop_params = conf_params['pop_params']
                                        result = optimization_process(algo_params,pop_params,multi_flag,opt_params,
                                                      var_pop_params,file_name, data_path)
                                        results.append(result)

    ## save the results in npz
    if multi_flag:
        result_name = f'multi_{delta_flag}_{method}_wight_{weight_2d}_iter_{n_iter}'
    else:
        result_name =  f'{delta_flag}_{method}_wight_{weight_2d}_iter_{n_iter}'
        
    np.savez(f'{result_name}.npz', 
          results=results, 
          )
    
    
    
