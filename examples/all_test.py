# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
# import sys
import os
import runpy
from pathlib import Path
import ray
from optframework.kernel_opt.opt_base import OptBase
from optframework.utils.general_scripts.generate_psd import full_psd
from optframework.utils.func.change_config import replace_key_value

if __name__ == '__main__':
    generate_synth_data = False
    run_opt = False
    run_calc_delta = False
    
    ## Get config data
    pth = Path(os.path.dirname(__file__)).resolve()
    conf_pth = os.path.join(pth, "config", "All_Test_config.py")
    conf = runpy.run_path(conf_pth)
    
    pop_params = conf['config']['pop_params']
    b = pop_params['CORR_BETA']
    a = pop_params['alpha_prim']
    v = pop_params['pl_v']
    p1 = pop_params['pl_P1']
    p2 = pop_params['pl_P2']
    p3 = pop_params['pl_P3']
    p4 = pop_params['pl_P4']
    
    ## Generate a new psd data for initail distribution of particle
    x50 = 20   # /um
    resigma = 0.2
    minscale = 0.01
    maxscale = 100
    dist_path = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=False)
    print(f'New psd data has been created and saved in path {dist_path}')
    dist_name = os.path.basename(dist_path)
    # Use the psd data as initial condition
    replace_key_value(conf_pth, 'PSD_R01', dist_name)
    replace_key_value(conf_pth, 'PSD_R03', dist_name)
    
    opt = OptBase(config_path=conf_pth)    
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
        ray.shutdown()
        
    if run_calc_delta:
        if opt.core.calc_init_N:
            opt.core.opt_pbe.set_init_N(exp_data_paths, 'mean')
            
        if isinstance(exp_data_paths, list):
            x_uni_exp = []
            data_exp = []
            for exp_data_path_tem in exp_data_paths:
                if opt.core.exp_data:
                    x_uni_exp_tem, data_exp_tem = opt.core.opt_data.get_all_exp_data(exp_data_path_tem)
                else:
                    x_uni_exp_tem, data_exp_tem = opt.core.opt_data.get_all_synth_data(exp_data_path_tem)
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
        else:
            if opt.core.exp_data:
                x_uni_exp, data_exp = opt.core.opt_data.get_all_exp_data(exp_data_paths)
            else:
                x_uni_exp, data_exp = opt.core.opt_data.get_all_synth_data(exp_data_paths)  
                
        delta = opt.core.calc_delta(pop_params, x_uni_exp, data_exp)