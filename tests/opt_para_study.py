# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import sys, os
from pathlib import Path
import time
import ray
import numpy as np
from optframework.kernel_opt.opt_base import OptBase

if __name__ == '__main__':
    base_path = Path(os.getcwd()).resolve()
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    # test_group = os.environ.get('TEST_GROUP')
    data_path = r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_data\data"
    result_dir = os.path.join(base_path, "opt_results")
    # result_dir = os.path.join(os.environ.get('STORAGE_PATH'), f"opt_results_{test_group}")
    os.makedirs(result_dir, exist_ok=True)
    #%%  Instantiate OptBase.
    opt = OptBase(data_path=data_path)
    
    multi_flag = opt.multi_flag
    
    weight_2d = opt.core.weight_2d
    add_noise = opt.core.add_noise
    noise_type = opt.core.noise_type
    noise_strength = opt.core.noise_strength
    delta_flag = opt.core.delta_flag
    method = opt.core.method
    n_iter = opt.core.n_iter
    
    # n_iter_list = [50, 100, 200, 400, 800, 1600]
    n_iter_list = [10, 20]
    prev_iter = 0
    
    #%% Prepare paths of test data set
    ## define the range of corr_beta
    var_corr_beta = np.array([1.0])
    ## define the range of alpha_prim 27x3
    values = np.array([1e-3])
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
    var_v = np.array([1.0,1.5])
    # var_v = np.array([0.01])    ## define the range of P1, P2 for power law breakage rate
    var_P1 = np.array([1e-4])
    var_P2 = np.array([0.5])
    var_P3 = np.array([1e-4])
    var_P4 = np.array([0.5])

    ## define the range of particle size scale and minimal size
    
    # dist_path_1 = os.path.join(data_path, "PSD_data", conf.config['dist_scale_1'])
    # dist_path = [dist_path_1] # [dist_path_1, dist_path_10]
    # size_scale = np.array([1, 10])
    # R01_0 = 'r0_001'
    # R03_0 = 'r0_001'
    data_names_list = []
    known_params_list = []
    start_time = time.time()
    for j,corr_beta in enumerate(var_corr_beta):
        for k,alpha_prim in enumerate(var_alpha_prim):
            for l,v in enumerate(var_v):
                for m1,P1 in enumerate(var_P1):
                    for m2,P2 in enumerate(var_P2):
                        for m3,P3 in enumerate(var_P3):
                            for m4,P4 in enumerate(var_P4):
                                # Set known parameters for PBE
                                known_params = {
                                    # 'CORR_BETA' : corr_beta,
                                    # 'alpha_prim' : alpha_prim,
                                    # 'pl_v' : v,
                                    # 'pl_P1' : P1,
                                    # 'pl_P2' : P2,
                                    # 'pl_P3' : P3,
                                    # 'pl_P4' : P4,
                                    # 'G': 1,
                                    }
                                # known_params = None
                                if add_noise:
                                    prefix = f"Sim_{noise_type}_{noise_strength}_para"
                                else:
                                    prefix = "Sim_para"
                                data_name = f"{prefix}_{corr_beta}_{alpha_prim[0]}_{alpha_prim[1]}_{alpha_prim[2]}_{v}_{P1}_{P2}_{P3}_{P4}.xlsx"
                                data_path_tem = os.path.join(opt.data_path, data_name)
                                if opt.core.sample_num != 1:
                                    data_path_tem = data_path_tem.replace(".xlsx", "_0.xlsx")
                                if not os.path.exists(data_path_tem):
                                    continue
                                data_names_list.append(data_name)
                                known_params_list.append(known_params)
    if multi_flag:
        data_names_list_tem= []
        for data_name in data_names_list:
            data_name_ex = [
                data_name,
                data_name.replace(".xlsx", "_NM.xlsx"),
                data_name.replace(".xlsx", "_M.xlsx")
            ]
            data_names_list_tem.append(data_name_ex)
        data_names_list = data_names_list_tem
    opt.core.result_dir = result_dir
    ray.init(log_to_driver=True)
    for n_iter in n_iter_list:
        if n_iter <= prev_iter:
            continue
        inc = n_iter - prev_iter
        opt.core.n_iter = int(n_iter)
        opt.core.n_iter_prev = int(prev_iter)
        opt.core.resume_unfinished = prev_iter > 0
        
        if getattr(opt.core, 'resume_unfinished', False):
            prev_path = os.path.join(result_dir, f"{opt.core.n_iter_prev}.sqlite")
            if os.path.exists(prev_path):
                print(f"Loaded previous opt_params for warm start: {prev_path}")
            else:
                print(f"Warning: Previous result not found: {prev_path}") 
        result = opt.find_opt_kernels(method='delta', data_names=data_names_list, known_params=known_params_list)   

        ## save the results in npz
        if multi_flag:
            result_name = f'multi_{delta_flag}_{method}_wight_{weight_2d}_iter_{n_iter}'
        else:
            result_name =  f'{delta_flag}_{method}_wight_{weight_2d}_iter_{n_iter}'
        
        file_path = os.path.join(result_dir, f'{result_name}.npz')
        # if os.path.exists(file_path):
        #     timestamp = time.strftime("%Y%m%d_%H%M%S")
        #     file_path = os.path.join(result_dir, f'{result_name}_{timestamp}.npz')
    
        np.savez(file_path,
              results=result,
              )
        
        prev_iter = n_iter
    ray.shutdown()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time = {elapsed_time} s")
    
