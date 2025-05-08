# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import sys, os
from pathlib import Path
import time
import numpy as np
from optframework.kernel_opt.opt_base import OptBase
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt

def calc_delta_test(known_params_list, exp_data_paths, opt_params=None, visual=False):
    if opt_params is None:
        pop_params =  {
            'pl_v': 1.5,
            'pl_P1': 1e-4,
            'pl_P2': 1,
            'corr_agg': np.array([1e-3])
        }
    else:
        pop_params = opt_params.copy()

    exp_data_paths = [os.path.join(opt.data_path, name) for name in exp_data_paths]        
    opt.core.init_attr(opt.core_params)
    opt.core.init_pbe(opt.pop_params, opt.data_path) 

    x_uni_exp = []
    data_exp = []
    for exp_data_path_tem in exp_data_paths:
        if opt.core.exp_data:
            x_uni_exp_tem, data_exp_tem = opt.core.get_all_exp_data(exp_data_path_tem)
        else:
            x_uni_exp_tem, data_exp_tem = opt.core.get_all_synth_data(exp_data_path_tem)
        x_uni_exp.append(x_uni_exp_tem)
        data_exp.append(data_exp_tem)

    losses = []
    for i in range(len(known_params_list)):
        known_i = known_params_list[i]
        for key, value in known_i.items():
            pop_params[key] = value
        x_i = x_uni_exp[i]
        data_i = data_exp[i]
        loss_i = opt.core.calc_delta(pop_params, x_i, data_i)
        if visual:
            visualize_opt_distribution(x_uni_exp=x_i, data_exp=data_i)
        losses.append(loss_i)

    return losses

def cross_validation(data_names_list, known_params_list, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    N = len(data_names_list)
    for i in range(N):
        print(f"Running cross-validation iteration {i+1}/{N}")

        train_data = [data_names_list[j] for j in range(N) if j != i]
        train_known = [known_params_list[j] for j in range(N) if j != i]
        test_data = [data_names_list[i]]
        # test_known = [known_params_list[i]]

        start_time = time.time()
        result = opt.find_opt_kernels(method='delta', data_names=train_data, known_params=train_known)    
        end_time = time.time()

        opt_params = result['opt_params']
        opt_score = result['opt_score']
        elapsed_time = end_time - start_time

        losses_all = calc_delta_test(known_params_list, data_names_list, opt_params)

        result_dict = {
            'cv_index': i,
            'train_data': train_data,
            'test_data': test_data,
            'opt_params': opt_params,
            'opt_score_train': opt_score,
            'losses_all': losses_all,
            'elapsed_time': elapsed_time
        }

        result_name =  f'cv_iter_{n_iter}_{i}_{data_dir}_{G_flag}.npz'  
        file_path = os.path.join(result_dir, result_name)
        np.savez(file_path, **result_dict)
        print(f"Saved result to {file_path}\n")

def load_cross_validation_results(result_dir, n_iter, data_dir):
    """
    Load and assemble results from saved cross-validation files.

    Parameters
    ----------
    result_dir : str
        Directory where cross-validation result .npz files are saved.
    n_iter : int
        Iteration number used in saved filenames.
    data_dir : str
        Data directory name used in saved filenames.

    Returns
    -------
    opt_params_list : list of dict
        List of optimized parameter dictionaries for each CV iteration.
    opt_scores : list of float
        List of training set optimization scores for each CV iteration.
    losses_all_list : list of list[float]
        List of losses on all data for each CV iteration.
    """
    opt_params_list = []
    opt_scores = []
    losses_all_list = []

    files = sorted([f for f in os.listdir(result_dir) if f.startswith(f"cv_iter_{n_iter}_") and f.endswith(f"_{data_dir}_{G_flag}.npz")])

    for file in files:
        file_path = os.path.join(result_dir, file)
        data = np.load(file_path, allow_pickle=True)
        opt_params_list.append(data['opt_params'].item())  # .item() because it's a dict saved in npz
        opt_scores.append(data['opt_score_train'].item())
        losses_all_list.append(data['losses_all'])

    return opt_params_list, opt_scores, losses_all_list

def visualize_opt_distribution(t_frame=-1, x_uni_exp=None, data_exp=None):
    x_uni, q0, Q0, sum_uni = opt.core.p.return_distribution(t=t_frame, flag='x_uni, qx, Qx,sum_uni', q_type='q0')
    if opt.core.smoothing:
        kde = opt.core.KDE_fit(x_uni[1:], sum_uni[1:])
        q0 = opt.core.KDE_score(kde, x_uni_exp[1:])
        q0 = np.insert(q0, 0, 0.0)
        Q0 = opt.core.calc_Qx(x_uni_exp, q0)
        q0 = q0 / Q0.max()
        x_uni = x_uni_exp
    fig, ax = plt.subplots()
    ax, fig = pt.plot_data(x_uni, q0, fig=fig, ax=ax,
                           xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q0$ / $-$',
                           lbl='opt',clr='b',mrk='o')
    ax, fig = pt.plot_data(x_uni_exp, data_exp[:, t_frame], fig=fig, ax=ax,
                           xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q0$ / $-$',
                           lbl='exp',clr='r',mrk='^')
    
    ax.grid('minor')
    ax.set_xscale('log')
    plt.tight_layout()  
    plt.show()
    return ax, fig        

if __name__ == '__main__':
    base_path = Path(os.getcwd()).resolve()
    config_path = os.path.join(base_path, "config", "opt_Batch_config.py")
    data_dir = "L1_int1d"
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data")
    data_path = os.path.join(base_path, "data", data_dir)
    opt = OptBase(config_path=config_path, data_path=data_path)
    n_iter = opt.core.n_iter
    data_names_list = [
        "Batch_600_Q0_post.xlsx",
        "Batch_900_Q0_post.xlsx",
        "Batch_1200_Q0_post.xlsx",
        "Batch_1500_Q0_post.xlsx",
        "Batch_1800_Q0_post.xlsx",
    ]
    
    G_flag = "Median_Integral"
    if G_flag == "Median_Integral":
        G_datas = [32.0404, 39.1135, 41.4924, 44.7977, 45.6443]
    elif G_flag == "Median_LocalStirrer":
        G_datas = [104.014, 258.081, 450.862, 623.357, 647.442]
    elif G_flag == "Mean_Integral":
        G_datas = [87.2642, 132.668, 143.68, 183.396, 185.225]
    elif G_flag == "Mean_LocalStirrer":
        G_datas = [297.136, 594.268, 890.721, 1167.74, 1284.46]
    else:
        raise ValueError(f"Unknown G_flag: {G_flag}")
        
    known_params_list = [{'G': G_val} for G_val in G_datas]

    result_dir = os.path.join(base_path, "cv_results")
    cross_validation(data_names_list, known_params_list, result_dir)
    
    opt_params_list, opt_scores, losses_all_list = load_cross_validation_results(result_dir, n_iter, data_dir)
    calc_delta_test(known_params_list, data_names_list, opt_params_list[0], visual=True)