# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import opt_find as opt
import numpy as np
from generate_psd import full_psd
import opt_config as conf

if __name__ == '__main__':
    ## Input for Opt
    dim = 2
    t_vec = np.concatenate(([0, 0.1, 0.3, 0.6, 0.9], np.arange(1, 602, 60, dtype=float)))
    delta_flag = 1
    add_noise = True
    smoothing = True
    noise_type='Mul'
    noise_strength = 0.1
    sample_num = 5
    multi_flag = True
    
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
        dist_path_1 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_1'])
        dist_path_5 = os.path.join(pth, "..", "data", "PSD_data", conf.config['dist_scale_5'])
        dist_path_10 = os.path.join(pth, "..","data", "PSD_data", conf.config['dist_scale_10'])
    
    ## define the range of corr_beta
    var_corr_beta = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
    
    ## define the range of alpha_prim 27x3
    values = np.array([0, 0.5, 1])
    a1, a2, a3 = np.meshgrid(values, values, values, indexing='ij')
    var_alpha_prim = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))

    ## define the range of particle size scale and minimal size
    dist_path = [dist_path_1, dist_path_10]
    size_scale = np.array([1, 10])
    R01_0 = 'r0_005'
    R03_0 = 'r0_005'
    ## Instantiate Opt
    find = opt.opt_find()
    find.init_opt_algo(dim, t_vec, add_noise, noise_type, noise_strength, smoothing)

    # for i, dist in enumerate(dist_path):
    #     ## Reinitialization of pop equations using psd data  
    #     dist_path_NM = dist_path[0]
    #     dist_path_M = dist
    #     scale = size_scale[i]
    #     find.algo.set_comp_para(R01_0, R03_0, dist_path_NM, dist_path_M)
        
    #     for corr_beta in var_corr_beta:
    #         for alpha_prim in var_alpha_prim:
    #             ## Set α and β_corr
    #             find.algo.corr_beta = corr_beta
    #             find.algo.alpha_prim = alpha_prim
    #             add_info = f"_para_{find.algo.corr_beta}_{find.algo.alpha_prim[0]}_{find.algo.alpha_prim[1]}_{find.algo.alpha_prim[2]}_{scale}"
    #             # Generate synthetic Data
    #             find.generate_data(sample_num=sample_num, add_info=add_info)
                   