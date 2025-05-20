# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:15 2024

@author: px2030
"""
import sys
import os
import numpy as np
import multiprocessing
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from optframework.kernel_opt.opt_base import OptBase

def transform_parameters(X):
    """
    Transform parameters from sampled values to actual values, considering log scale.
    """
    X_transformed = np.zeros_like(X)
    for i, name in enumerate(problem['names']):
        bounds = problem['bounds'][i]
        log_scale = log_scale_params[name]
        if log_scale:
            # Log scale: power of 10
            X_transformed[:, i] = 10 ** X[:, i]
        else:
            # Linear scale
            X_transformed[:, i] = X[:, i]
    return X_transformed

def transform_parameters_to_dict(X):
    """
    Convert parameter array to a list of parameter dictionaries.
    """
    params_list = []
    X_transformed = transform_parameters(X)
    for params in X_transformed:
        params_dict = {name: value for name, value in zip(problem['names'], params)}
        params_list.append(params_dict)
    return params_list

def evaluate_model(params):
    """
    Run the PBE model for a given set of parameters and return Moment M.
    """
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    base_path = r"C:\Users\px2030\Code\PSD_opt\tests"
    config_path = os.path.join(base_path, "config", "opt_Batch_config.py")
    data_dir = "lognormal_curvefit"  ## "int1d", "lognormal_curvefit", "lognormal_zscore"
    data_path = os.path.join(base_path, "data", data_dir)
    opt = OptBase(config_path=config_path, data_path=data_path)
    data_names_list = [
        "Batch_600_Q0_post.xlsx",
        "Batch_900_Q0_post.xlsx",
        "Batch_1200_Q0_post.xlsx",
        "Batch_1500_Q0_post.xlsx",
        "Batch_1800_Q0_post.xlsx",
    ]
    
    G_flag_list = [
        "Median_Integral", 
        "Median_LocalStirrer", 
        "Mean_Integral", 
        "Mean_LocalStirrer"
    ]
    
    # for G_flag in G_flag_list:
    #     if G_flag == "Median_Integral":
    #         G_datas = [32.0404, 39.1135, 41.4924, 44.7977, 45.6443]
    #     elif G_flag == "Median_LocalStirrer":
    #         G_datas = [104.014, 258.081, 450.862, 623.357, 647.442]
    #     elif G_flag == "Mean_Integral":
    #         # G_datas = [87.2642, 132.668, 143.68, 183.396, 185.225]
    #         G_datas = [87.2642, 132.668]
    #     elif G_flag == "Mean_LocalStirrer":
    #         G_datas = [297.136, 594.268, 890.721, 1167.74, 1284.46]
    
    opt = OptBase(data_path=data_path, multi_flag=False)
    params_trans = params.copy()
    params_trans = opt.core.array_dict_transform(params_trans)
    params_checked = opt.core.check_corr_agg(params_trans)
    opt.core.calc_pop(opt.core.p, params_checked, opt.core.t_vec)
    ## if PBE failed to converge
    if not opt.core.p.calc_status:
        return np.nan
    moment = opt.core.p.calc_mom_t()
    ## Rate of change of particle number
    M00 = (moment[0,0,-1] - moment[0,0,0]) / moment[0,0,0]
    ## How evenly the materials are distributed in a particle, 
    ## scaled by the combined volume of the two materials
    M11 = moment[1,1,-1] / (moment[1,0,0] * moment[0,1,0])
    if moment_flag == "m00":
        return M00
    elif moment_flag == "m11":
        return M11
    elif moment_flag == "m_wight":
        return M00 + 1000*M11
    
def save_to_csv(Si, param_names, N, moment_flag):
    # 步骤 1：保存一阶和总效应敏感性指数
    df_Si = pd.DataFrame({
        'Parameter': param_names,
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf'],
    })
    
    # 保存为 CSV 文件
    df_Si.fillna(0, inplace=True)
    df_Si.to_csv(f'sensitivity_indices_{moment_flag}_{N}.csv', index=False, float_format='%.6f')
    
    # 步骤 2：保存二阶敏感性指数
    S2 = Si['S2']
    S2_conf = Si['S2_conf']
    
    param_pair = []
    S2_values = []
    S2_conf_values = []
    
    num_params = len(param_names)

    for i in range(num_params):
        for j in range(i+1, num_params):
            param_pair.append(f"{param_names[i]} & {param_names[j]}")
            S2_values.append(S2[i, j])
            S2_conf_values.append(S2_conf[i, j])
    
    df_S2 = pd.DataFrame({
        'Parameter Pair': param_pair,
        'S2': S2_values,
        'S2_conf': S2_conf_values,
    })

    # 保存为 CSV 文件
    df_S2.fillna(0, inplace=True)
    df_S2.to_csv(f'second_order_sensitivity_indices_{moment_flag}_{N}.csv', index=False, float_format='%.6f')

if __name__ == '__main__':
    # Define parameter names and ranges
    param_names = ['corr_agg_0', 'pl_v', 'pl_P1', 'pl_P2']
    
    # Define parameter bounds
    problem = {
        'num_vars': 4,
        'names': param_names,
        'bounds': [
            [-8.0, -1.0],    # corr_agg_0
            [0.1, 2.0],     # pl_v
            [-8.0, 2.0],   # pl_P1
            [0.1, 5.0],     # pl_P2
        ]
    }
    
    # Define which parameters are on a logarithmic scale
    ## Note: For parameters that use logarithmic values, sampling is uniform on a logarithmic scale, 
    ## so the result actually reflects the effect of a linear change in the exponential 
    ## of the corresponding parameter. This is consistent with the logic of the optimizer 
    ## (as it should be!).
    log_scale_params = {
        'corr_agg_0': True,
        'pl_v': False,
        'pl_P1': True,
        'pl_P2': False,
    }
    
    # Set the number of sampling points
    N = 4  # Adjust this number based on available computational resources
    
    # Generate sampling points
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    # Transform parameters to get a list of parameter dictionaries
    params_list = transform_parameters_to_dict(param_values)
    
    ## only for test
    mu = evaluate_model(params_list[0])
    
    # pool = multiprocessing.Pool(processes=8)
    # try:
    #     results = pool.map(evaluate_model, params_list)
    # except KeyboardInterrupt:
    #     print("Caught KeyboardInterrupt, terminating workers")
    #     pool.terminate()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     pool.terminate()
    # finally:          
    #     pool.close()
    #     pool.join()                        
    #     results_arr = np.array(results)  

    # # Convert the results to an array format
    # Y = np.array(results)
    # nan_indices = np.isnan(Y)
    # if np.any(nan_indices):
    #     # 将有效（非NaN）的样本用于训练K近邻模型
    #     valid_Y = Y[~nan_indices]
    #     valid_params = param_values[~nan_indices]
        
    #     # 将包含NaN的样本分离出来作为测试集
    #     invalid_params = param_values[nan_indices]
        
    #     # 使用K近邻模型来对缺失值进行插值
    #     # 使用邻近的5个样本来插值，您可以根据需要调整n_neighbors的值
    #     knn = KNeighborsRegressor(n_neighbors=5)
    #     knn.fit(valid_params, valid_Y)
        
    #     # 使用K近邻模型对缺失值进行预测
    #     Y[nan_indices] = knn.predict(invalid_params)
        
    #     # 将无效参数保存到CSV文件中
    #     np.savetxt('invalid_params.csv', invalid_params, delimiter=',')

    # ## Normalization
    # min_max_scaler = MinMaxScaler()
    # normalized_Y = min_max_scaler.fit_transform(Y)

    # # Calculate Sobol' indices
    # Si = sobol.analyze(problem, normalized_Y, calc_second_order=True, print_to_console=True)  
    
    # # save the results
    # save_to_csv(Si, param_names, N, moment_flag)
             