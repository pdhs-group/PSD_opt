# -*- coding: utf-8 -*-
"""
Comprehensive Test Script for OptFramework

This script provides a comprehensive test suite for all major functionalities 
of the OptFramework library, including particle size distribution (PSD) generation,
synthetic data creation, optimization, and solver validation.

The test script performs the following operations sequentially:

1. **PSD Data Generation**: Creates initial particle size distribution data that
   serves as the computational initial condition for subsequent simulations.

2. **Synthetic Data Generation**: Generates synthetic data for optimizer testing.
   This process internally invokes the DPBESolver to perform numerical solving
   and creates artificial datasets that mimic experimental data.

3. **Optimization Testing**: The optimizer reads the synthetic data and performs
   optimization to test the functionality of the optimization algorithms.

4. **Delta Calculation**: Executes `run_calc_delta` which performs a single 
   cost function calculation within the optimization process. This is a common
   debugging approach for optimizers to verify the objective function computation.

5. **Solver Validation**: Calls three different solvers from the library:
   - DPBESolver (Discrete Population Balance Equation Solver)
   - PBMSolver (Population Balance Model Solver)  
   - MCPBESolver (Monte Carlo Population Balance Equation Solver)
   
   All three solvers simulate the same input parameters and their results are
   compared for consistency validation.

Expected Output:
    If the program runs successfully, it will generate comparison plots showing
    the 0th, 1st, and 2nd order moments. The results from all three solvers
    should be essentially identical.

Usage:
    Run this script directly to execute the full test suite:
    
    ```python
    python all_test.py
    ```
    
    Individual test components can be controlled by modifying the boolean flags:
    - `generate_synth_data`: Controls synthetic data generation
    - `run_opt`: Controls optimization testing
    - `run_calc_delta`: Controls delta calculation testing  
    - `run_validation`: Controls solver validation testing

Dependencies:
    - numpy: Numerical computations
    - ray: Parallel processing for optimization
    - pathlib: Path handling
    - optframework: Main optimization framework modules

Note:
    This script serves as both a functionality test and a demonstration of
    the OptFramework library capabilities. It can be used for regression
    testing, performance validation, and as a reference implementation.

Created on Thu Jan  4 14:53:00 2024

@author: Haoran Ji
"""
import os
import runpy
from pathlib import Path
import numpy as np
import ray
from optframework import OptBase
from optframework import PBEValidation
from optframework.utils.general_scripts.generate_psd import full_psd
from optframework.utils.func.change_config import replace_key_value

if __name__ == '__main__':
    generate_synth_data = True
    run_opt = True
    run_calc_delta = True
    run_validation = True
    
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
    add_info = (
    f"_para_{b}"
    f"_{a[0]:.1e}_{a[1]:.1e}_{a[2]:.1e}"
    f"_{v}_{p1:.1e}_{p2}_{p3:.1e}_{p4}"
    )
    data_name = f"Sim_{noise_type}_{noise_strength}" + add_info + ".xlsx"
    
    if generate_synth_data:
        opt.core.p.generate_data(opt.data_path, opt.multi_flag, pop_params, add_info=add_info)
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
                x_uni_exp_tem, data_exp_tem = opt.core.p.get_all_data(exp_data_path_tem)
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
        else:
            x_uni_exp, data_exp = opt.core.p.get_all_data(exp_data_paths)
                
        delta = opt.core.calc_delta(pop_params, x_uni_exp, data_exp)
    
    if run_validation:
        dim = 1
        grid = 'geo'
        NS = 20
        t = np.arange(0, 1, 0.1, dtype=float)
        v = PBEValidation(1, 'geo', 20, 2, 'sum', 'mix', t=t, c=1e-2,
                          beta0=1e-16, use_psd=True, dist_path=dist_path)
        v.P1 = 1e12
        v.init_all()
        v.calculate_case(calc_mc=True, calc_pbm=True)
        v.init_plot(size = 'half', extra = True, mrksize=6)
        v.plot_all_moments(REL=True)
        v.show_plot()