# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import numpy as np
import time
import opt_find as opt
import multiprocessing
import opt_config as conf

def optimization_process(var_pop_params):
    #%%  Input for Opt 
    find = opt.opt_find()
     
    algo_params = conf.config['algo_params']
    pop_params = conf.config['pop_params']
    multi_flag = conf.config['multi_flag']
    opt_params = conf.config['opt_params']
    ## Update the parameter for PBE
    pop_params.update(var_pop_params)
    
    find.init_opt_algo(multi_flag, algo_params, opt_params)
    
    find.algo.set_init_pop_para(pop_params)
    
    find.algo.set_comp_para(R_NM=conf.config['R_NM'], R_M=conf.config['R_M'])
    
    find.algo.weight_2d = conf.config['weight_2d']

    be = var_pop_params['CORR_BETA']
    al = var_pop_params['alpha_prim']
    pv = var_pop_params['pl_v']
    p1 = var_pop_params['pl_P1']
    p2 = var_pop_params['pl_P2']
    noise_type = algo_params['noise_type']
    noise_strength = algo_params['noise_strength']
    data_name = f"Sim_{noise_type}_{noise_strength}_para_{be}_{al[0]}_{al[1]}_{al[2]}_{pv}_{p1}_{p2}.xlsx"
    
    results = \
        find.find_opt_kernels(sample_num=find.algo.sample_num, method='delta', data_name=data_name)

    return results

if __name__ == '__main__':
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
    
    pool = multiprocessing.Pool(processes=1)
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
                        var_pop_params = conf_params['pop_params']
                        tasks.append((var_pop_params))
    
    results = pool.map(optimization_process, tasks)
    
    data_size = np.array([len(var_corr_beta), len(var_alpha_prim)],len(var_v),len(var_P1),len(var_P2))
    # The asterisk (*) is used in a function call to indicate an "unpacking" operation, 
    # which means that it expands the elements of 'data_size' into individual arguments
    
    
    # for result in results:
    #     i, j, (corr_beta_opt_res, alpha_prim_opt_res, para_diff_res, delta_opt_res, \
    #            corr_agg_res, corr_agg_opt_res, corr_agg_diff_res) = result
    #     corr_beta_opt[i,j] = corr_beta_opt_res
    #     alpha_prim_opt[i,j,:] = alpha_prim_opt_res
    #     para_diff[i,j] = para_diff_res
    #     delta_opt[i,j] = delta_opt_res
        
    #     corr_agg[i,j,:] = corr_agg_res
    #     corr_agg_opt[i,j,:] = corr_agg_opt_res
    #     corr_agg_diff[i,j,:] = corr_agg_diff_res
            
    # ## save the results in npz
    # if multi_flag:
    #     result_name = f'multi_{delta_flag}_{method}_{cost_func_type}_wight_{weight_2d}'
    # else:
    #     result_name =  f'{delta_flag}_{method}_{cost_func_type}_wight_{weight_2d}'
        
    # np.savez(f'{result_name}.npz', 
    #      corr_beta_opt=corr_beta_opt, 
    #      alpha_prim_opt=alpha_prim_opt, 
    #      para_diff=para_diff, 
    #      delta_opt=delta_opt, 
    #      corr_agg = corr_agg,
    #      corr_agg_opt = corr_agg_opt,
    #      corr_agg_diff = corr_agg_diff,
    #      )
    
    
    
