# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:57:59 2025

@author: Haoran Ji 
e-mail: haoran.ji@kit.edu
"""
import os
import numpy as np
from optframework.kernel_opt.opt_base import OptBase
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt
from optframework.utils.general_scripts.convert_exp_data_Batch import interpolate_Qx
import pandas as pd

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
        evaluated_params = list(data['all_params'])
        evaluated_rewards = list(data['all_score'])
        
    return opt_params_list, opt_scores, losses_all_list, evaluated_params, evaluated_rewards

def load_all_cv_results(result_dir, n_iter_list, data_dir, G_flag_list):
    """
    Load results across all combinations of n_iter and G_flag.
    Returns a nested dict: results[G_flag][n_iter] = {...}
    """
    results = {}
    for G_flag in G_flag_list:
        results[G_flag] = {}
        for n_iter in n_iter_list:
            opt_params, scores, losses, evaluated_params, evaluated_rewards = load_cross_validation_results(
                result_dir, n_iter, data_dir, G_flag
            )
            results[G_flag][n_iter] = {
                'opt_params_list': opt_params,
                'opt_scores': scores,
                'losses_all_list': losses,
                'evaluated_params': evaluated_params,
                'evaluated_rewards': evaluated_rewards
            }
    return results

def add_opt_params_mean(results):
    for G_flag in results:
        for n_iter in results[G_flag]:
            opt_list = results[G_flag][n_iter]["opt_params_list"]
            keys = opt_list[0].keys()
            means = {key: np.mean([d[key] for d in opt_list]) for key in keys}
            results[G_flag][n_iter]["opt_params_mean"] = means
    return results

def plot_summary_two_charts(results, n_iter_list, G_flag_list, result_dir):
    """
    Plot summary charts for a single G_flag from cross-validation results.

    Parameters:
        results      : Nested dict from load_all_cv_results
        n_iter_list  : List of iteration counts
        G_flag_list  : List of G_flag strings (only one expected)
        result_dir   : Directory path to save plots
    """
    os.makedirs(result_dir, exist_ok=True)
    G_flag = G_flag_list[0]

    # Chart 1: Training/Test mean and std over n_iter (twin y-axes)
    train_means, train_stds = [], []
    test_means, test_stds = [], []

    for n_iter in n_iter_list:
        train_scores = results[G_flag][n_iter]['opt_scores']
        test_losses = [results[G_flag][n_iter]['losses_all_list'][i][i] for i in range(len(train_scores))]
        train_means.append(np.mean(train_scores))
        train_stds.append(np.std(train_scores))
        test_means.append(np.mean(test_losses))
        test_stds.append(np.std(test_losses))

    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax1 = fig.subplots()

    ax2 = ax1.twinx()
    ax1.plot(n_iter_list, train_means, 'o-', markersize=8, linewidth=2, label='Train Mean', color='tab:blue')
    ax1.plot(n_iter_list, test_means, 's-', markersize=8, linewidth=2, label='Test Mean', color='tab:green')
    ax2.plot(n_iter_list, train_stds, 'o--', markersize=8, linewidth=2, label='Train Std', color='tab:blue')
    ax2.plot(n_iter_list, test_stds, 's--', markersize=8, linewidth=2, label='Test Std', color='tab:green')

    ax1.set_xlabel("n_iter")
    ax1.set_ylabel("Mean Score", color='black')
    ax2.set_ylabel("Std of Score", color='black')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', ncol=2)

    plt.title("Train/Test Mean & Std over Iterations")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "train_test_mean_std_combined.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 2: Normalized parameter std
    param_names = list(results[G_flag][n_iter_list[0]]['opt_params_list'][0].keys())
    param_names = [p for p in param_names if p != 'G']  # exclude G

    plt.figure(figsize=(10, 6), dpi=300)
    for i, param in enumerate(param_names):
        raw_means = []
        raw_stds = []
        for n_iter in n_iter_list:
            vals = [p[param] for p in results[G_flag][n_iter]['opt_params_list']]
            raw_means.append(np.mean(vals))
            raw_stds.append(np.std(vals))
        norm_stds = [s / m if m != 0 else 0.0 for s, m in zip(raw_stds, raw_means)]
        marker = marker_styles[i % len(marker_styles)]
        plt.plot(n_iter_list, norm_stds, marker=marker, markersize=8, linewidth=2, label=param)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("n_iter")
    plt.ylabel("Normalized Std of Parameters")
    plt.title("Normalized Parameter Std vs n_iter")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "params_normalized_std_combined.jpg"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # save the mean and stds as excel file
    rows = []
    for idx, n_iter in enumerate(n_iter_list):
        row = {
            "n_iter": n_iter,
            "train_mean": train_means[idx],
            "train_std": train_stds[idx],
            "test_mean": test_means[idx],
            "test_std": test_stds[idx],
        }
        for param in param_names:
            vals = [p[param] for p in results[G_flag][n_iter]['opt_params_list']]
            param_mean = np.mean(vals)
            param_std = np.std(vals)
            norm_std = param_std / param_mean if param_mean != 0 else 0.0
            row[f"{param}_mean"] = param_mean
            row[f"{param}_norm_std"] = norm_std
        rows.append(row)
    
    df = pd.DataFrame(rows)
    excel_path = os.path.join(result_dir, "summary_data.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"[Saved] Summary data table to: {excel_path}")

def plot_x_mean(results, G_flag):
    if G_flag == "Median_Integral":
        n = 3.3700
        G_datas = [39.1135, 41.4924, 44.7977, 45.6443]
    elif G_flag == "Median_LocalStirrer":
        n = 0.6417
        G_datas = [258.081, 450.862, 623.357, 647.442]
    elif G_flag == "Mean_Integral":
        n = 1.6477
        G_datas = [132.668, 143.68, 183.396, 185.225]
    elif G_flag == "Mean_LocalStirrer":
        n = 0.8154
        G_datas = [594.268, 890.721, 1167.74, 1284.46]
    else:
        raise ValueError(f"Unknown G_flag: {G_flag}")
    
    known_params_list = [{'G': G_val} for G_val in G_datas]
    n_iter_list = [200, 800, 6400]
    time_points = [0, 5, 10, 45]
    linestyles = {200: '-', 800: '--', 6400: ':'}
    sample_names = ["n=900", "n=1200", "n=1500", "n=1800"]
    
    excel_path = os.path.join(result_dir, "x_mean.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    for i in range(len(known_params_list)):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(time_points, x_exp_mean[i], 'x', color='black', label="Exp")
        data_dict = {'Time': time_points, 'Exp': x_exp_mean[i]}

        for n_iter in n_iter_list:
            opt_params = results[G_flag][n_iter]['opt_params_list'][i]
            _, x_mod_mean = calc_delta_test(
                [known_params_list[i]],
                [data_names_list[i]],
                init_core=True,  
                opt_params=opt_params
            )
            x_mod_mean = np.array(x_mod_mean)
            if isinstance(x_mod_mean[0], list):
                x_mod_mean = x_mod_mean[0]
            ax.plot(time_points, x_mod_mean, linestyle=linestyles[n_iter], label=f"Sim ({n_iter})")
            data_dict[f"Sim_{n_iter}"] = x_mod_mean

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Mean Diameter (m)")
        ax.set_yscale('log')
        ax.set_title(f"Experimental vs Simulated Mean Volume - {G_flag} - Sample {i+1}")
        ax.grid(True, which='both')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        df = pd.DataFrame(data_dict)
        df.to_excel(writer, index=False, sheet_name=sample_names[i])
    writer.close()
    print(f"[Saved] x_mean comparison to: {excel_path}")

def calc_delta_test(known_params_list, exp_data_paths, init_core=True, opt_params=None,
                    ax=None, fig=None, index=''):
    if opt_params is None:
        pop_params =  {
            'pl_v': 1.5,
            'pl_P1': 1e-4,
            'pl_P2': 1,
            'corr_agg': np.array([1e-3])
        }
    else:
        if 'corr_agg_0' in opt_params:
            opt_params = opt.core.array_dict_transform(opt_params)
        pop_params = opt_params.copy()
        
    exp_data_paths = [os.path.join(opt.data_path, name) for name in exp_data_paths]
    if init_core:        
        opt.core.init_attr(opt.core_params)
        opt.core.init_pbe(opt.pop_params, opt.data_path) 

    x_uni_exp = []
    data_exp = []
    for exp_data_path_tem in exp_data_paths:
        x_uni_exp_tem, data_exp_tem = opt.core.p.get_all_data(exp_data_path_tem)
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
        losses.append(loss_i)
        
        mom = opt.core.p.calc_mom_t()
        x_mod_mean = (mom[1,0,:] / mom[0,0,:] *6 /np.pi)**(1/3)

    return losses, x_mod_mean

def plot_x_50(results, G_flag):
    if G_flag == "Median_Integral":
        n = 3.3700
        G_datas = [39.1135, 41.4924, 44.7977, 45.6443]
    elif G_flag == "Median_LocalStirrer":
        n = 0.6417
        G_datas = [258.081, 450.862, 623.357, 647.442]
    elif G_flag == "Mean_Integral":
        n = 1.6477
        G_datas = [132.668, 143.68, 183.396, 185.225]
    elif G_flag == "Mean_LocalStirrer":
        n = 0.8154
        G_datas = [594.268, 890.721, 1167.74, 1284.46]
    else:
        raise ValueError(f"Unknown G_flag: {G_flag}")

    known_params_list = [{'G': G_val} for G_val in G_datas]
    n_iter_list = [200, 800, 6400]
    time_points = [0, 5, 10, 45]
    linestyles = {200: '-', 800: '--', 6400: ':'}
    sample_names = ["n=900", "n=1200", "n=1500", "n=1800"]
    
    excel_path = os.path.join(result_dir, "x_50.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    for i in range(len(known_params_list)):
        fig, ax = plt.subplots(figsize=(8, 6))
        opt_params_dict = {n_iter: results[G_flag][n_iter]['opt_params_list'][i] for n_iter in n_iter_list}
        
        exp_x50, _ = calc_x50([known_params_list[i]], [data_names_list[i]], init_core=True, opt_params=opt_params_dict[200])
        ax.plot(time_points, exp_x50, 'x', color='black', label='Exp')
        data_dict = {'Time': time_points, 'Exp': exp_x50}

        for n_iter in n_iter_list:
            _, mod_x50 = calc_x50([known_params_list[i]], [data_names_list[i]], init_core=True, opt_params=opt_params_dict[n_iter])
            ax.plot(time_points, mod_x50, linestyle=linestyles[n_iter], label=f"Sim ({n_iter})")
            data_dict[f"Sim_{n_iter}"] = mod_x50
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Median Diameter (m)")
        ax.set_yscale('log')
        ax.set_title(f"Exp vs Sim Median Volume - {G_flag} - Sample {i+1}")
        ax.grid(True, which='both')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        df = pd.DataFrame(data_dict)
        df.to_excel(writer, index=False, sheet_name=sample_names[i])
    writer.close()
    print(f"[Saved] x_50 comparison to: {excel_path}")
    
def calc_x50(known_params_list, exp_data_paths, init_core=True, opt_params=None,
                    ax=None, fig=None, index=''):
    if opt_params is None:
        pop_params =  {
            'pl_v': 1.5,
            'pl_P1': 1e-4,
            'pl_P2': 1,
            'corr_agg': np.array([1e-3])
        }
    else:
        if 'corr_agg_0' in opt_params:
            opt_params = opt.core.array_dict_transform(opt_params)
        pop_params = opt_params.copy()
        
    exp_data_paths = [os.path.join(opt.data_path, name) for name in exp_data_paths]
    if init_core:        
        opt.core.init_attr(opt.core_params)
        opt.core.init_pbe(opt.pop_params, opt.data_path) 
        opt.core.sheet_name = 'Q_x_int'
        opt.core.delta_flag = [('x_50','MSE')]

    x_uni_exp = []
    data_exp = []
    for exp_data_path_tem in exp_data_paths:
        x_uni_exp_tem, data_exp_tem = opt.core.p.get_all_data(exp_data_path_tem)
        x_uni_exp.append(x_uni_exp_tem)
        data_exp.append(data_exp_tem)
    
    x_50_exp = np.array(data_exp).reshape(-1)
    for i in range(len(known_params_list)):
        known_i = known_params_list[i]
        for key, value in known_i.items():
            pop_params[key] = value
        x_i = x_uni_exp[i]
        data_i = data_exp[i]
        _ = opt.core.calc_delta(pop_params, x_i, data_i)
        x_50_mod = np.zeros(len(opt.core.p.t_vec))
        for t_frame, t in enumerate(opt.core.p.t_vec):
            x_50_mod[t_frame] = opt.core.p.return_distribution(t=t_frame, flag='x_50', q_type='q0')[0]

    return x_50_exp, x_50_mod

def re_calc_lognormal_results():
    N = len(data_names_list)
    for n_iter in n_iter_list:
        for G_flag in G_flag_list:
            if G_flag == "Median_Integral":
                n = 2.6428 if data_dir == "lognormal_curvefit" else 3.1896
                G_datas = [32.0404, 39.1135, 41.4924, 44.7977, 45.6443]
            elif G_flag == "Median_LocalStirrer":
                n = 0.4723 if data_dir == "lognormal_curvefit" else 0.5560
                G_datas = [104.014, 258.081, 450.862, 623.357, 647.442]
            elif G_flag == "Mean_Integral":
                n = 1.1746 if data_dir == "lognormal_curvefit" else 1.3457
                G_datas = [87.2642, 132.668, 143.68, 183.396, 185.225]
            elif G_flag == "Mean_LocalStirrer":
                n = 0.5917 if data_dir == "lognormal_curvefit" else 0.7020
                G_datas = [297.136, 594.268, 890.721, 1167.74, 1284.46]
                # G_datas = [297.136, 594.268, 890.721, 1167.74]
            else:
                raise ValueError(f"Unknown G_flag: {G_flag}")
            known_params_list = [{'G': G_val} for G_val in G_datas]
            
            # Build filename patterns
            prefix = f"cv_iter_{n_iter}_"
            suffix = f"_{data_dir}_{G_flag}.npz"
            files = sorted([
                f for f in os.listdir(result_dir)
                if f.startswith(prefix) and f.endswith(suffix)
            ])
            # Load each file
            for i, fname in enumerate(files):
                path = os.path.join(result_dir, fname)
                with np.load(path, allow_pickle=True) as data:
                    data_dict = dict(data)
                opt_params = data_dict['opt_params'].item()
                losses_all_re_calc, _ = calc_delta_test(known_params_list, data_names_list, False, opt_params)
                train_losses_re_calc = [losses_all_re_calc[j] for j in range(N) if j != i]
                opt_scores_re_calc = sum(train_losses_re_calc) / len(train_losses_re_calc)
                data_dict['losses_all'] = losses_all_re_calc
                data_dict['opt_score_train'] = opt_scores_re_calc
                np.savez(path, **data_dict)
                

def visualize_opt_distribution(t_frame=-1, x_uni_exp=None, data_exp=None, 
                               ax=None, fig=None, index=0):
    x_uni, q0, Q0, sum_uni = opt.core.p.return_distribution(t=t_frame, flag='x_uni, qx, Qx,sum_uni', q_type='q0')
    if opt.core.smoothing:
        kde = opt.core.KDE_fit(x_uni[1:], sum_uni[1:])
        q0 = opt.core.KDE_score(kde, x_uni_exp[1:])
        q0 = np.insert(q0, 0, 0.0)
        Q0 = opt.core.calc_Qx(x_uni_exp, q0)
        q0 = q0 / Q0.max()
        x_uni = x_uni_exp
    fig, ax = plt.subplots()
    ax, fig = pt.plot_data(x_uni, Q0, fig=fig, ax=ax,
                           xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q0$ / $-$',
                           lbl='opt',clr='b',mrk='o')
    ax, fig = pt.plot_data(x_uni_exp, data_exp[:, t_frame], fig=fig, ax=ax,
                           xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q0$ / $-$',
                           lbl='exp',clr='r',mrk='^')
    
    # color_list = ['b', 'gold', 'g', 'r', 'violet']
    # ax, fig = pt.plot_data(x_uni[:30], Q0[:30], fig=fig, ax=ax, plt_type='scatter', 
    #                        xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
    #                        ylbl='number distribution of agglomerates $q0$ / $-$',
    #                        lbl=f'opt_{index}',clr=color_list[i],mrk='o')
    # ax, fig = pt.plot_data(x_uni_exp[:30], data_exp[:30, t_frame], fig=fig, ax=ax,
    #                        xlbl=r'Agglomeration size $x_\mathrm{A}$ / $-$',
    #                        ylbl='number distribution of agglomerates $q0$ / $-$',
    #                        lbl=f'exp_{index}',clr=color_list[i],mrk='^')
    
    ax.grid('minor')
    # ax.set_xscale('log')
    plt.tight_layout()  
    plt.show()
    return ax, fig  
    
def main():
    results = load_all_cv_results(result_dir, n_iter_list, data_dir, G_flag_list)
    add_opt_params_mean(results)
    
    plot_summary_two_charts(results, n_iter_list, G_flag_list, result_dir)
    plot_x_mean(results, G_flag_list[0])
    plot_x_50(results, G_flag_list[0])
    
    return results
    
if __name__ == '__main__':
    base_path = r"C:\Users\px2030\Code\PSD_opt\tests"
    config_path = os.path.join(base_path, "config", "opt_Batch_config.py")
    data_dir = "int1d" ## "int1d", "lognormal_curvefit"
    data_path = os.path.join(base_path, "data", data_dir)
    opt = OptBase(config_path=config_path, data_path=data_path)
    data_names_list = [
        # "Batch_600_Q0_post.xlsx",
        "Batch_900_Q0_post.xlsx",
        "Batch_1200_Q0_post.xlsx",
        "Batch_1500_Q0_post.xlsx",
        "Batch_1800_Q0_post.xlsx",
    ]
    n_iter_list = [200, 400, 800, 1600, 2400, 4000, 6400]

    marker_styles = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    result_dir = os.path.join(r"C:\Users\px2030\Code\Ergebnisse\Batch_opt\opt_results", "cv_results_group41")
    
    if data_dir == "lognormal_curvefit":
        x_exp_mean = np.array([
            # [0.08523817, 0.08326844, 0.08579775, 0.07277097],
               [0.08523817, 0.07365923, 0.07167854, 0.07170555],
               [0.08523817, 0.07324782, 0.07380572, 0.06823556],
               [0.08523817, 0.07450765, 0.07019528, 0.06936281],
               [0.08523817, 0.07652142, 0.0740287 , 0.06236208]]) * 1e-6
        G_flag_list = ["Mean_Integral"]
        
    elif data_dir == "int1d":
        x_exp_mean = np.array([
            # [0.11826945, 0.12083028, 0.11184293, 0.21994005],
               [0.11826945, 0.12277321, 0.09504571, 0.08820781],
               [0.11826945, 0.09741461, 0.09254452, 0.28119003],
               [0.11826945, 0.0992418 , 0.09677272, 0.09395102],
               [0.11826945, 0.09177996, 0.08894885, 0.08688184]
               ]) * 1e-6
        G_flag_list = ["Median_LocalStirrer"]

        G_flag_list = [
            # "Median_Integral", 
            # "Median_LocalStirrer", 
            # "Mean_Integral", 
            "Mean_LocalStirrer"
        ]
    # re_calc_lognormal_results()
    
    results = main()