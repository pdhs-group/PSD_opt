# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
from pathlib import Path
import numpy as np
import runpy
import multiprocessing
from optframework.kernel_opt.opt_base import OptBase
from optframework.utils.general_scripts.generate_psd import full_psd

def calc_function(conf_params, data_path, config_path):
    #%%  Input for Opt 
    find = OptBase(data_path=data_path, config_path=config_path)
    if not isinstance(conf_params, dict):
        raise TypeError("conf_params should be a dictionary.")
    b = conf_params['CORR_BETA']
    a = conf_params['alpha_prim']
    v = conf_params['pl_v']
    p1 = conf_params['pl_P1']
    p2 = conf_params['pl_P2']
    p3 = conf_params['pl_P3']
    p4 = conf_params['pl_P4']

    add_info = f"_para_{b}_{a[0]}_{a[1]}_{a[2]}_{v}_{p1}_{p2}_{p3}_{p4}"
    # Generate synthetic Data
    find.generate_data(conf_params, add_info=add_info)
    
if __name__ == '__main__':
    base_path = Path(os.getcwd()).resolve()
    generate_new_psd = True
    # pth = '/pfs/work7/workspace/scratch/px2030-MC_train'
    data_path = os.path.join(base_path,"mix", "data")
    config_path = r"C:\Users\px2030\Code\PSD_opt\tests\config\opt_config.py"
    conf = runpy.run_path(config_path)
    
    if generate_new_psd:
        ## Input for generating psd-data
        x50 = 20   # /um
        resigma = 0.2
        minscale = 0.01
        maxscale = 100
        output_dir = os.path.join(data_path, "PSD_data")
        dist_path_1 = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False, output_dir=output_dir)
        dist_path_5 = full_psd(x50*5, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False, output_dir=output_dir)
        dist_path_10 = full_psd(x50*10, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False, output_dir=output_dir)
    else:
        pth = os.path.dirname( __file__ )
        dist_path_1 = os.path.join(data_path, "PSD_data", conf.config['dist_scale_1'])
        dist_path_5 = os.path.join(data_path, "PSD_data", conf.config['dist_scale_5'])
        dist_path_10 = os.path.join(data_path, "PSD_data", conf.config['dist_scale_10'])

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
    var_v = np.array([1.5])
    # var_v = np.array([0.01])    ## define the range of P1, P2 for power law breakage rate
    var_P1 = np.array([1e-2])
    var_P2 = np.array([0.5])
    var_P3 = np.array([1e-2])
    var_P4 = np.array([2.0])
    
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
                                    ## Set parameters for PBE
                                    conf_params = {
                                        'CORR_BETA' : corr_beta,
                                        'alpha_prim' : alpha_prim,
                                        'pl_v' : v,
                                        'pl_P1' : P1,
                                        'pl_P2' : P2,
                                        'pl_P3' : P3,
                                        'pl_P4' : P4,
                                        }
                                    calc_function(conf_params, data_path, config_path)
                                    # func_list.append(conf_params)
    # pool = multiprocessing.Pool(processes=16)
    # pool.starmap(calc_function, func_list)                        
    # pool.close()
    # pool.join()        
    # with multiprocessing.Pool(processes=2) as pool:
    #     pool.starmap(calc_function, [(conf, data_path, config_path) for conf in func_list])             
                   