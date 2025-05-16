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

def calc_delta_test(known_params_list, exp_data_paths, init_core=True, opt_params=None, visual=False):
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
    if init_core:        
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
            ## The comparison is between the test data and the simulation results for each data set.
            visualize_opt_distribution(x_uni_exp=x_i, data_exp=data_i)
        losses.append(loss_i)

    return losses

def cross_validation(data_names_list, known_params_list, result_dir, G_flag):
    os.makedirs(result_dir, exist_ok=True)
    N = len(data_names_list)
    for i in range(N):
        print(f"Running cross-validation iteration {i+1}/{N}")
        opt.core.data_name_index = i
        train_data = [data_names_list[j] for j in range(N) if j != i]
        train_known = [known_params_list[j] for j in range(N) if j != i]
        test_data = [data_names_list[i]]
        opt.core.data_name_tune = test_data[0]
        # test_known = [known_params_list[i]]

        start_time = time.time()
        result = opt.find_opt_kernels(method='delta', data_names=train_data, known_params=train_known)    
        end_time = time.time()

        opt_params = result['opt_params']
        opt_score = result['opt_score']
        elapsed_time = end_time - start_time

        losses_all = calc_delta_test(known_params_list, data_names_list, False, opt_params)

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

# Modified load function to include G_flag as parameter
def load_cross_validation_results(result_dir, n_iter, data_dir, G_flag):
    """
    Load and assemble cross-validation results for a specific (n_iter, data_dir, G_flag).
    Returns lists of optimized parameters, training scores, and all losses.
    """
    opt_params_list = []
    opt_scores = []
    losses_all_list = []

    # Build filename patterns
    prefix = f"cv_iter_{n_iter}_"
    suffix = f"_{data_dir}_{G_flag}.npz"
    files = sorted([
        f for f in os.listdir(result_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    ])

    # Load each file
    for fname in files:
        path = os.path.join(result_dir, fname)
        data = np.load(path, allow_pickle=True)
        opt_params_list.append(data['opt_params'].item())
        opt_scores.append(data['opt_score_train'].item())
        losses_all_list.append(data['losses_all'])

    return opt_params_list, opt_scores, losses_all_list


def load_all_cv_results(result_dir, n_iter_list, data_dir, G_flag_list):
    """
    Load results across all combinations of n_iter and G_flag.
    Returns a nested dict: results[G_flag][n_iter] = {...}
    """
    results = {}
    for G_flag in G_flag_list:
        results[G_flag] = {}
        for n_iter in n_iter_list:
            opt_params, scores, losses = load_cross_validation_results(
                result_dir, n_iter, data_dir, G_flag
            )
            results[G_flag][n_iter] = {
                'opt_params_list': opt_params,
                'opt_scores': scores,
                'losses_all_list': losses
            }
    return results


def analyze_and_plot_cv_results(results, n_iter_list, G_flag_list, result_dir):
    """
    Analyze and visualize cross-validation statistics and save each figure as JPG.
    Parameters:
      results       : Nested dict from load_all_cv_results
      n_iter_list   : List of iteration counts
      G_flag_list   : List of G_flag strings
      result_dir    : Directory path to save plots
    """
    # Ensure save directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Determine parameter names from first G_flag and n_iter
    first_flag = G_flag_list[0]
    first_iter = n_iter_list[0]
    sample_params = results[first_flag][first_iter]['opt_params_list'][0]
    param_names = list(sample_params.keys())

    # 1. Plot each parameter's mean ± std
    for param in param_names:
        if param == "G":
            continue
        plt.figure()
        for G_flag in G_flag_list:
            means = []
            stds = []
            for n_iter in n_iter_list:
                vals = [p[param] for p in results[G_flag][n_iter]['opt_params_list']]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            plt.errorbar(
                n_iter_list,
                means,
                yerr=stds,
                marker='o',
                capsize=5,
                capthick=1.5,
                elinewidth=1.2,
                label=G_flag
            )
        plt.xlabel('n_iter')
        plt.ylabel(param)
        plt.title(f"Parameter '{param}' Mean ± Std vs n_iter")
        plt.legend()
        plt.grid(True)
        # Save figure
        fname = f"param_{param}_mean_std.jpg"
        plt.savefig(os.path.join(result_dir, fname), format='jpg')
        plt.close()

    # --- Additional normalized mean ± std plot for parameters ---
    for param in param_names:
        if param == "G":
            continue
        plt.figure()
        for G_flag in G_flag_list:
            raw_means = []
            raw_stds = []
            for n_iter in n_iter_list:
                vals = [p[param] for p in results[G_flag][n_iter]['opt_params_list']]
                raw_means.append(np.mean(vals))
                raw_stds.append(np.std(vals))
            # Normalize means by first entry
            base = raw_means[0] if raw_means[0] != 0 else 1.0
            norm_means = [m / base for m in raw_means]
            # Scale stds by their corresponding raw mean (pre-normalization)
            scale_stds = [s / m if m != 0 else 0.0 for s, m in zip(raw_stds, raw_means)]
            plt.errorbar(
                n_iter_list,
                norm_means,
                yerr=scale_stds,
                marker='o',
                capsize=5,
                capthick=1.5,
                elinewidth=1.2,
                label=G_flag
            )
        plt.xlabel('n_iter')
        plt.ylabel(f"Normalized {param}")
        plt.title(f"Parameter '{param}' Normalized Mean ± Std vs n_iter")
        plt.legend()
        plt.grid(True)
        fname = f"param_{param}_normalized_mean_std.jpg"
        plt.savefig(os.path.join(result_dir, fname), format='jpg')
        plt.close()

    # 2. Plot training scores' mean ± std
    plt.figure()
    for G_flag in G_flag_list:
        means = []
        stds = []
        for n_iter in n_iter_list:
            scores = results[G_flag][n_iter]['opt_scores']
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        plt.errorbar(
            n_iter_list,
            means,
            yerr=stds,
            marker='o',
            capsize=5,
            capthick=1.5,
            elinewidth=1.2,
            label=G_flag
        )
    plt.xlabel('n_iter')
    plt.ylabel('Training Score')
    plt.title('Training Score Mean ± Std vs n_iter')
    plt.legend()
    plt.grid(True)
    # Save figure
    plt.savefig(os.path.join(result_dir, 'training_score_mean_std.jpg'), format='jpg')
    plt.close()

    # 3. Plot test data loss' mean ± std
    plt.figure()
    for G_flag in G_flag_list:
        means = []
        stds = []
        for n_iter in n_iter_list:
            losses_all = results[G_flag][n_iter]['losses_all_list']
            test_losses = [losses_all[i][i] for i in range(len(losses_all))]
            means.append(np.mean(test_losses))
            stds.append(np.std(test_losses))
        plt.errorbar(
            n_iter_list,
            means,
            yerr=stds,
            marker='o',
            capsize=5,
            capthick=1.5,
            elinewidth=1.2,
            label=G_flag
        )
    plt.xlabel('n_iter')
    plt.ylabel('Test Data Loss')
    plt.title('Test Data Loss Mean ± Std vs n_iter')
    plt.legend()
    plt.grid(True)
    # Save figure
    plt.savefig(os.path.join(result_dir, 'test_data_loss_mean_std.jpg'), format='jpg')
    plt.close()
    
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
    ax, fig = pt.plot_data(x_uni[:30], q0[:30], fig=fig, ax=ax,
                           xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q0$ / $-$',
                           lbl='opt',clr='b',mrk='o')
    ax, fig = pt.plot_data(x_uni_exp[:30], data_exp[:30, t_frame], fig=fig, ax=ax,
                           xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q0$ / $-$',
                           lbl='exp',clr='r',mrk='^')
    
    ax.grid('minor')
    # ax.set_xscale('log')
    plt.tight_layout()  
    plt.show()
    return ax, fig        

if __name__ == '__main__':
    base_path = Path(os.getcwd()).resolve()
    config_path = os.path.join(base_path, "config", "opt_Batch_config.py")
    data_dir = "int1d"  ## "int1d", "lognormal_curvefit", "lognormal_zscore"
    # tmp_path = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmp_path, "data", data_dir)
    data_path = os.path.join(base_path, "data", data_dir)
    opt = OptBase(config_path=config_path, data_path=data_path)
    data_names_list = [
        "Batch_600_Q0_post.xlsx",
        "Batch_900_Q0_post.xlsx",
        # "Batch_1200_Q0_post.xlsx",
        # "Batch_1500_Q0_post.xlsx",
        # "Batch_1800_Q0_post.xlsx",
    ]
    
    G_flag_list = [
        # "Median_Integral", 
        # "Median_LocalStirrer", 
        "Mean_Integral", 
        # "Mean_LocalStirrer"
    ]
    n_iter = opt.core.n_iter
    # n_iter_list = [100, 200, 400, 800, 1200, 2400, 4800, 9600]
    n_iter_list = [10]
    prev = 0
    result_dir = os.path.join(base_path, "cv_results")
    for n_iter in n_iter_list:
        inc = n_iter - prev
        opt.core.n_iter = int(inc)
        # flag for optimierer_ray
        resume_flag = (prev > 0)
        opt.core.resume_unfinished = resume_flag
        prev = n_iter
        for G_flag in G_flag_list:
            if G_flag == "Median_Integral":
                G_datas = [32.0404, 39.1135, 41.4924, 44.7977, 45.6443]
            elif G_flag == "Median_LocalStirrer":
                G_datas = [104.014, 258.081, 450.862, 623.357, 647.442]
            elif G_flag == "Mean_Integral":
                # G_datas = [87.2642, 132.668, 143.68, 183.396, 185.225]
                G_datas = [87.2642, 132.668]
            elif G_flag == "Mean_LocalStirrer":
                G_datas = [297.136, 594.268, 890.721, 1167.74, 1284.46]
            else:
                raise ValueError(f"Unknown G_flag: {G_flag}")
                
            known_params_list = [{'G': G_val} for G_val in G_datas]
        
            cross_validation(data_names_list, known_params_list, result_dir, G_flag)
    
    # Load everything
    # result_dir = os.path.join(r"C:\Users\px2030\Code\Ergebnisse\Batch_opt\opt_results", "group6+10")
    # results = load_all_cv_results(result_dir, n_iter_list, data_dir, G_flag_list)
    
    # Analyze & visualize
    # analyze_and_plot_cv_results(results, n_iter_list, G_flag_list, result_dir)
    
    ## calculate PBE 
    # G_flag = "Mean_LocalStirrer"
    # if G_flag == "Median_Integral":
    #     G_datas = [32.0404, 39.1135, 41.4924, 44.7977, 45.6443]
    # elif G_flag == "Median_LocalStirrer":
    #     G_datas = [104.014, 258.081, 450.862, 623.357, 647.442]
    # elif G_flag == "Mean_Integral":
    #     G_datas = [87.2642, 132.668, 143.68, 183.396, 185.225]
    # elif G_flag == "Mean_LocalStirrer":
    #     G_datas = [297.136, 594.268, 890.721, 1167.74, 1284.46]
    # else:
    #     raise ValueError(f"Unknown G_flag: {G_flag}")
    # known_params_list = [{'G': G_val} for G_val in G_datas]
    # opt_params = results[G_flag][800]['opt_params_list'][0]
    # calc_delta_test(known_params_list, data_names_list, opt_params, visual=True)