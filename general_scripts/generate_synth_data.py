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
    dim = conf.config['dim']
    t_init = conf.config['t_init']
    t_vec = conf.config['t_vec']
    delta_flag = conf.config['delta_flag']
    add_noise = conf.config['add_noise']
    smoothing = conf.config['smoothing']
    noise_type=conf.config['noise_type']
    noise_strength = conf.config['noise_strength']
    sample_num = conf.config['sample_num']
    
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
    ## The case of all zero α is meaningless, that means no Agglomeration occurs
    var_alpha_prim = var_alpha_prim[~np.all(var_alpha_prim == 0, axis=1)]

    ## define the range of particle size scale and minimal size
    dist_path = [dist_path_1, dist_path_10]
    size_scale = np.array([1, 10])
    R01_0 = 'r0_001'
    R03_0 = 'r0_001'
    ## Instantiate Opt
    find = opt.opt_find()
    find.init_opt_algo(dim, t_init, t_vec, add_noise, noise_type, noise_strength, smoothing)

    for i, dist in enumerate(dist_path):
        ## Reinitialization of pop equations using psd data  
        dist_path_NM = dist_path[0]
        dist_path_M = dist
        scale = size_scale[i]
        find.algo.set_comp_para(R01_0, R03_0, dist_path_NM, dist_path_M)
        
        for corr_beta in var_corr_beta:
            for alpha_prim in var_alpha_prim:
                ## Set α and β_corr
                find.algo.corr_beta = corr_beta
                find.algo.alpha_prim = alpha_prim
                add_info = f"_para_{find.algo.corr_beta}_{find.algo.alpha_prim[0]}_{find.algo.alpha_prim[1]}_{find.algo.alpha_prim[2]}_{scale}"
                # Generate synthetic Data
                find.generate_data(sample_num=sample_num, add_info=add_info)
                   