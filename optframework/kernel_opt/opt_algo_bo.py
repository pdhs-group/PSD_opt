# -*- coding: utf-8 -*-
"""
serial BayesianOptimization, only for test and comparison
"""
from bayes_opt import BayesianOptimization

def optimierer_bo(self, opt_params, hyperparameter=None, exp_data_paths=None,known_params=None):
    if self.calc_init_N:
        self.set_init_N(exp_data_paths, init_flag='mean')
    if isinstance(exp_data_paths, list):
        ## When set to multi, the exp_data_paths entered here is a list 
        ## containing one 2d data name and two 1d data names.
        x_uni_exp = []
        data_exp = []
        for exp_data_paths_tem in exp_data_paths:
            if self.exp_data:
                x_uni_exp_tem, data_exp_tem = self.get_all_exp_data(exp_data_paths_tem)
            else:
                x_uni_exp_tem, data_exp_tem = self.get_all_synth_data(exp_data_paths_tem)
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
    else:
        ## When not set to multi or optimization of 1d-data, the exp_data_paths 
        ## contain the name of that data.
        if self.exp_data:
            x_uni_exp, data_exp = self.get_all_exp_data(exp_data_paths)
        else:
            x_uni_exp, data_exp = self.get_all_synth_data(exp_data_paths)
            
    pbounds = {}
    transform = {}
    # Prepare bounds and transformation based on parameters definition
    for param, info in opt_params.items():
        bounds = info['bounds']
        log_scale = info.get('log_scale', False)
        pbounds[param] = bounds
        if log_scale:
            transform[param] = lambda x: 10**x
        else:
            transform[param] = lambda x: x

    def objective(scale, **kwargs):
        transformed_params = {}
        for param, func in transform.items():
            transformed_params[param] = func(kwargs[param])
        # Special handling for corr_agg based on dimension
        if 'corr_agg_0' in transformed_params:
            transformed_params = self.array_dict_transform(transformed_params)
        if known_params is not None:
            for key, value in known_params.items():
                if key in transformed_params:
                    print(f"Warning: Known parameter '{key}' are set for optimization.")
                transformed_params[key] = value
        return self.calc_delta_agg(transformed_params, x_uni_exp, data_exp)*scale
        
    scale = -1  ## BayesianOptimization find the maximum
    bayesian_objective = lambda **kwargs: objective(scale, **kwargs)
    opt = BayesianOptimization(f=bayesian_objective, pbounds=pbounds, random_state=1, allow_duplicate_points=True)
    opt.maximize(init_points=5, n_iter=self.n_iter)
    opt_values = {param: transform[param](opt.max['params'][param]) for param in opt_params}
    delta_opt = -opt.max['target']
    return delta_opt, opt_values