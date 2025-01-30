# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:02:59 2025

@author: px2030
"""
import os
from optframework.kernel_opt.opt_base import OptBase

if __name__ == '__main__':
    ## The following two variables specify the **full path** of the config file 
    ## and the **folder path** of the experimental data. 
    ## If set to `None`, the default paths will be used, 
    ## which are `/config/opt_config.py` and `/data` in the directory of the current script.
    config_path = None
    data_path = None
    
    opt = OptBase(config_path=config_path, data_path=data_path)
    
    data_path_list = []
    ## Define the name of the experimental data, they should be located in the same folder
    data_names = ['exp1.xlsx', 'exp1.xlsx', 'exp1.xlsx']
    ## Define PBE parameters for each experimental data in the form of a dictionary. 
    ## Its essence is to assign values ​​to the attributes of the same name in the PBE instance, 
    ## so not only G, but other parameters can also be adjusted. 
    ## These parameters will override the assignments of config **and optimization process**.
    known_params_list = [{'G': 1.0}, {'G': 2.0}, {'G': 3.0}]
    
    for data_name in data_names:
        ## Combine the file name and path. 
        ## Of course, you can also define the complete file path directly.
        data_path_tem = os.path.join(opt.data_path, data_name)
        data_path_list.append(data_path_tem)
    
    result = opt.find_opt_kernels(method='delta', data_names=data_path_list, known_params=known_params_list)
    