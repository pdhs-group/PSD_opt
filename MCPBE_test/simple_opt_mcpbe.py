# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:00 2024

@author: px2030
"""
import os
from pathlib import Path
import ray
from optframework import OptBase
import numpy as np
import time
import matplotlib.pyplot as plt

def normal_test():
    start_time = time.time()
    result_dict = \
        opt.find_opt_kernels(method='delta', data_names=exp_data_paths, known_params=known_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")
    
    file_path = os.path.join(result_dir, f'pure_CB_result_{n_iter}.npz')
    np.savez(file_path, results=result_dict)
    
    return result_dict
    
def return_pop_distribution():
    # Berechne x_uni
    x_uni = opt.core.p.post.calc_x_uni()
    
    # Erstelle eine Liste der Verteilungen für die ersten 12 Zeitschritte
    Q3 = [np.array(opt.core.p.post.return_distribution(t=i, flag='Qx')).reshape(len(x_uni)) for i in range(len(opt.core.t_vec))]
    
    # Zusammenfügen zu einer (52, 12) Matrix
    Q3_matrix = np.column_stack(Q3)
    
    return x_uni, Q3_matrix


def calc_delta_test(var_delta=False, pop_params=None, plot=False):
    """
    Run a single delta-evaluation with a given parameter set and
    optionally plot:

      - experimental vs. model PSD CDF at each time step
      - experimental vs. model mean volume x_50(t)
    """
    if pop_params is None:
        pop_params = opt.pop_params

    # Re-initialize core and adapter
    opt.core.init_attr(opt.core_params)
    opt.core.init_pbe(opt.pop_params, opt.data_path)
    if opt.core.calc_init_N:
        opt.core.opt_pbe.calc_init_from_data(exp_data_paths, 'mean')

    # Load experimental data (currently only single dataset supported)
    if isinstance(exp_data_paths, list):
        x_uni_exp = []
        data_exp = []
        for exp_data_path_tem in exp_data_paths:
            x_uni_exp_tem, data_exp_tem = opt.core.p.get_all_data(exp_data_path_tem)
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
    else:
        x_uni_exp, data_exp = opt.core.p.get_all_data(exp_data_paths)

    if var_delta:
        # time-resolved delta (not used for plotting at the moment)
        delta_arr = np.zeros(len(opt.core.t_vec))
        for start_step in range(1, len(opt.core.t_vec)):
            opt.core.delta_t_start_step = start_step
            delta_arr[start_step] = opt.core.calc_delta(pop_params, x_uni_exp, data_exp)
        return delta_arr
    else:
        # Single scalar delta using full time horizon
        delta = opt.core.calc_delta(pop_params, x_uni_exp, data_exp)

        # ------------------------------------------------------------------
        # Optional plotting
        # ------------------------------------------------------------------
        if plot:
            p = opt.core.p
            t_vec = np.asarray(opt.core.t_vec, dtype=float)

            # Model data Q(x, t), shape (Nx, Nt)
            data_mod = getattr(p, "data_mod", None)
            if data_mod is None:
                raise RuntimeError("Adapter has no data_mod; make sure solve() was called.")

            # x-grid (we assume single dataset case)
            if isinstance(x_uni_exp, list):
                x_plot = np.asarray(x_uni_exp[0], dtype=float)
                data_exp_plot = np.asarray(data_exp[0], dtype=float)
            else:
                x_plot = np.asarray(x_uni_exp, dtype=float)
                data_exp_plot = np.asarray(data_exp, dtype=float)

            # --- 3.1 PSD CDF comparison at each time step ---
            Nx, Nt = data_exp_plot.shape
            assert data_mod.shape == data_exp_plot.shape, (
                f"Shape mismatch: data_exp {data_exp_plot.shape}, "
                f"data_mod {data_mod.shape}"
            )

            n_cols = min(4, Nt)
            n_rows = int(np.ceil(Nt / n_cols))
            fig_psd, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(4 * n_cols, 3 * n_rows),
                                         squeeze=False)

            for j in range(Nt):
                ax = axes[j // n_cols][j % n_cols]
                ax.plot(x_plot, data_exp_plot[:, j], label="exp")
                ax.plot(x_plot, data_mod[:, j], "--", label="model")
                ax.set_xscale("log")
                ax.set_xlabel("diameter x")
                ax.set_ylabel("CDF Q(x)")
                ax.set_title(f"t = {t_vec[j]:.3g}")
            axes[0][0].legend()
            fig_psd.tight_layout()

            # --- 3.2 x_50(t) comparison ---
            x_50_exp = getattr(p, "x_50_exp", None)
            x_50_mod = getattr(p, "x_50_mod", None)
            data_flag = getattr(opt.core, "data_flag", "Q0")
            if data_flag.startswith("Q0"):
                y_label = "mean volume x_50 of Q0"
            else:  # Q3 / Q3_X_50
                y_label = "mean volume x_50 of Q3"
            if (x_50_exp is not None) or (x_50_mod is not None):
                fig_vm, ax_vm = plt.subplots(figsize=(5, 4))
                if x_50_exp is not None:
                    ax_vm.plot(t_vec, x_50_exp, "o-", label="x_50_exp")
                if x_50_mod is not None:
                    ax_vm.plot(t_vec, x_50_mod, "s--", label="x_50_mod")
                ax_vm.set_xlabel("time")
                ax_vm.set_ylabel(y_label)
                ax_vm.set_title("Mean particle volume vs. time")
                ax_vm.legend()
                fig_vm.tight_layout()

            plt.show()
            print("Model ratio :", x_50_mod[-1]/x_50_mod[0])
        return x_uni_exp, data_exp, delta


if __name__ == '__main__':
    ## Instantiate OptBase.
    ## The OptBase class determines how the experimental 
    ## data is used, while algo determines the optimization process.
    base_path = Path(os.getcwd()).resolve()
    config_path = os.path.join(base_path, "config", "opt_config_mcpbe.py")
    data_name = "CB_pur_N2000.h5"
    
    result_dir = os.path.join(base_path, "opt_results")
    data_path = os.path.join(base_path, "data_mcpbe_CB")
    # tmpdir = os.environ.get('TMP_PATH')
    # data_path = os.path.join(tmpdir, "data_mcpbe_CB")
    # test_group = os.environ.get('TEST_GROUP')
    # result_dir = os.path.join(os.environ.get('STORAGE_PATH'), f"opt_results_{test_group}")
    os.makedirs(result_dir, exist_ok=True)
    
    opt = OptBase(config_path=config_path, data_path=data_path)
    exp_data_paths = os.path.join(data_path, data_name)
     
    known_params = {
        # 'CORR_BETA' : 1.0,
        # 'alpha_prim' : [1e-3,1e-3,0.1],
        # 'pl_v' : v,
        # 'pl_P1' : P1,
        # 'pl_P2' : P2,
        # 'pl_P3' : P3,
        # 'pl_P4' : P4,
        }
    
    # n_iter_list = [10,20,30,40,50,60]
    # # n_iter_list = [10]
    # prev_iter = 0
    # opt.core.result_dir = result_dir
    
    # ray.init(log_to_driver=True)
    # # Run optimization
    # for n_iter in n_iter_list:
    #     if n_iter <= prev_iter:
    #         continue
    #     inc = n_iter - prev_iter
    #     # opt.core.n_iter = int(n_iter)
    #     opt.core.n_iter = int(inc)
    #     opt.core.n_iter_prev = int(prev_iter)
    #     opt.core.resume_unfinished = prev_iter > 0
    #     if getattr(opt.core, 'resume_unfinished', False):
    #         prev_path = os.path.join(result_dir, f"{opt.core.n_iter_prev}.sqlite")
    #         if os.path.exists(prev_path):
    #             print(f"Loaded previous opt_params for warm start: {prev_path}")
    #         else:
    #             print(f"Warning: Previous result not found: {prev_path}") 
    #     result_dict = normal_test()
    #     prev_iter = n_iter
    # ray.shutdown()
    
    result_to_analyse = os.path.join(base_path, "opt_results_N2000_Q0", "pure_CB_result_8000.npz")
    with np.load(result_to_analyse, allow_pickle=True) as data:
        results = data['results'].item()
    pop_params = results['opt_params']
    pop_params['lmc_gamma'] = 5
    pop_params['lmc_int_bre'] = 0.5
    pop_params['lmc_energy_exp'] = 3
    pop_params['lmc_lambda_E'] = 1e-8
    pop_params['lmc_NO_FRAG'] = 4
    pop_params['CORR_BETA'] = 1e1
    start_time = time.time()
    x_uni_test, Q3_test , delta = calc_delta_test(var_delta=False, pop_params=pop_params, plot=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The execution of optimierer takes：{elapsed_time} seconds")