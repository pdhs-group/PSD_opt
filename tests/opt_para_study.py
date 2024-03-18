# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import sys, os
import numpy as np
import time
import multiprocessing
import logging
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from pypbe.kernel_opt import opt_find as opt
from config import opt_config as conf

logging.basicConfig(filename='parallel.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def optimization_process(algo_params,pop_params,multi_flag,opt_params,ori_params,data_name):
    #%%  Input for Opt 
    find = opt.opt_find()

    ## Update the parameter for PBE
    pop_params.update(ori_params)

    find.init_opt_algo(multi_flag, algo_params, opt_params)
    
    find.algo.set_init_pop_para(pop_params)
    
    find.algo.set_comp_para(R_NM=conf.config['R_NM'], R_M=conf.config['R_M'])
    
    find.algo.weight_2d = conf.config['weight_2d']

    delta_opt, opt_values = \
        find.find_opt_kernels(sample_num=find.algo.sample_num, method='delta', data_name=data_name)

    return delta_opt, opt_values, ori_params

if __name__ == '__main__':
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    multi_flag = conf.config['multi_flag']
    opt_params = conf.config['opt_params']
    weight_2d = conf.config['weight_2d']
    
    noise_type = algo_params['noise_type']
    noise_strength = algo_params['noise_strength']
    delta_flag = algo_params['delta_flag']
    method = algo_params['method']
    cost_func_type = algo_params['cost_func_type']
    #%% Prepare test data set
    ## define the range of corr_beta
    # var_corr_beta = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    var_corr_beta = np.array([1e-2])
    ## define the range of alpha_prim 27x3
    values = [0, 0.5, 1]
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))
    ## remove element [0, 0, 0]
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]
    
    ## For cases where R01 and R03 have the same size, the elements of alpha_prim mirror symmetry 
    ## are equivalent and can be removed to simplify the calculation.
    unique_alpha_prim = []
    for comp in var_alpha_prim:
        comp_reversed = comp[::-1]  
        if not any(np.array_equal(comp, x) or np.array_equal(comp_reversed, x) for x in unique_alpha_prim):
            unique_alpha_prim.append(comp)
            
    var_alpha_prim = np.array(unique_alpha_prim)
    
    # var_v = np.array([0.01, 1, 2])
    var_v = np.array([0.01])
    ## define the range of P1, P2 for power law breakage rate
    # var_P1 = np.array([1e-6, 1e-4, 1e-2, 1])
    # var_P2 = np.array([0.0, 0.5, 1, 2])
    var_P1 = np.array([1])
    var_P2 = np.array([0.0])
    
    pool = multiprocessing.Pool(processes=12)
    tasks = []
    for j,corr_beta in enumerate(var_corr_beta):
        for k,alpha_prim in enumerate(var_alpha_prim):
            for l,v in enumerate(var_v):
                for m,P1 in enumerate(var_P1):
                    for n,P2 in enumerate(var_P2):
                        ## Set parameters for PBE
                        conf_params = {
                            'pop_params':{
                                'CORR_BETA' : corr_beta,
                                'alpha_prim' : alpha_prim,
                                'pl_v' : v,
                                'pl_P1' : P1,
                                'pl_P2' : P2,
                                }
                            }
                        data_name = f"Sim_{noise_type}_{noise_strength}_para_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_{v}_{P1}_{P2}.xlsx"
                        var_pop_params = conf_params['pop_params']
                        tasks.append((algo_params,pop_params,multi_flag,opt_params,
                                      var_pop_params,data_name))
    
    results = pool.starmap(optimization_process, tasks)

    ## save the results in npz
    if multi_flag:
        result_name = f'multi_{delta_flag}_{method}_{cost_func_type}_wight_{weight_2d}'
    else:
        result_name =  f'{delta_flag}_{method}_{cost_func_type}_wight_{weight_2d}'
        
    np.savez(f'{result_name}.npz', 
          results=results, 
          )
    
    
    
