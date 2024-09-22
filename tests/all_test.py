# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import sys
import os
import ray
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from pypbe.kernel_opt.opt_base import OptBase
from pypbe.utils.general_scripts.generate_psd import full_psd
import config.opt_config as conf

if __name__ == '__main__':
    generate_new_psd = True
    generate_synth_data = True
    run_opt = True
    run_calc_delta = False
    
    pop_params = conf.config['pop_params']
    b = pop_params['CORR_BETA']
    a = pop_params['alpha_prim']
    v = pop_params['pl_v']
    p1 = pop_params['pl_P1']
    p2 = pop_params['pl_P2']
    p3 = pop_params['pl_P3']
    p4 = pop_params['pl_P4']
    
    if generate_new_psd:
        x50 = 20   # /um
        resigma = 0.2
        minscale = 0.01
        maxscale = 100
        dist_path_1 = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        dist_path_5 = full_psd(x50*5, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        dist_path_10 = full_psd(x50*10, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
        print(f'New psd data has been created and saved in path {dist_path_1}')

    opt = OptBase()
    noise_type = opt.core.noise_type
    noise_strength = opt.core.noise_strength
    add_info = f"_para_{b}_{a[0]}_{a[1]}_{a[2]}_{v}_{p1}_{p2}_{p3}_{p4}"
    data_name = f"Sim_{noise_type}_{noise_strength}" + add_info + ".xlsx"
    
    if generate_synth_data:
        opt.generate_data(pop_params, add_info=add_info)
        print(f'The dPBE simulation has been completed. The new synthetic data {data_name} has been saved in path {opt.data_path}.')
        
    exp_data_path = os.path.join(opt.data_path, data_name)
    exp_data_paths = [
        exp_data_path,
        exp_data_path.replace(".xlsx", "_NM.xlsx"),
        exp_data_path.replace(".xlsx", "_M.xlsx")
    ]

    if run_opt:
        result_dict = opt.find_opt_kernels(method='delta', data_names=exp_data_paths)
        print('The optimization process has finished running.')
        print('Please check the output of ray to confirm whether the calculation was successful.')
        
    if run_calc_delta:
        if opt.core.calc_init_N:
            opt.core.set_init_N(exp_data_paths, 'mean')
            
        if isinstance(exp_data_paths, list):
            x_uni_exp = []
            data_exp = []
            for exp_data_path_tem in exp_data_paths:
                if opt.core.exp_data:
                    x_uni_exp_tem, data_exp_tem = opt.core.get_all_exp_data(exp_data_path_tem)
                else:
                    x_uni_exp_tem, data_exp_tem = opt.core.get_all_synth_data(exp_data_path_tem)
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
        else:
            if opt.core.exp_data:
                x_uni_exp, data_exp = opt.core.get_all_exp_data(exp_data_paths)
            else:
                x_uni_exp, data_exp = opt.core.get_all_synth_data(exp_data_paths)  
                
        delta = opt.core.calc_delta(pop_params, x_uni_exp, data_exp)