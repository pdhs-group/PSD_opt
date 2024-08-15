# -*- coding: utf-8 -*-
"""
Optimization algorithm based on ray-Tune
"""
import uuid
import os
import numpy as np
# import math
import ray
from ray import tune, train
# from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# from ray.util.placement_group import placement_group, placement_group_table
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hebo import HEBOSearch
from optuna.samplers import GPSampler,CmaEsSampler,TPESampler,NSGAIIISampler,QMCSampler

def optimierer_agg_bundles(self, opt_params, init_points=4, hyperparameter=None, exp_data_paths=None):
    """
    Optimize the corr_agg based on :meth:~.calc_delta_agg. 
    Results are saved in corr_agg_opt.
    
    Parameters
    ----------
    method : str
        Which algorithm to use for optimization.
    init_points : int, optional. Default 4.
        Number of steps for random exploration in BayesianOptimization.
    sample_num : int, optional. Default 1.
        Set how many sets of experimental data are used simultaneously for optimization.
    exp_data_path : str
        path for experimental data.
        
    Returns   
    -------
    delta_opt : float
        Optimized value of the objective.
    """
    self.RT_space = {}
    
    for param, info in opt_params.items():
        bounds = info['bounds']
        log_scale = info.get('log_scale', False)
        
        if log_scale:
            self.RT_space[param] = tune.loguniform(10**bounds[0], 10**bounds[1])
        else:
            self.RT_space[param] = tune.uniform(bounds[0], bounds[1])

    total_cpus = ray.cluster_resources().get("CPU", 1)
    
    ## Add all data (groups for multi_flag=True case) to the task queue
    task_queue = ray.util.queue.Queue()
    queue_length = 0
    for path in exp_data_paths:
        task_queue.put(path)
        queue_length += 1

    self.check_num_bundles(queue_length, total_cpus)
    num_bundles_max = self.num_bundles
    results = []
    # 获取最优结果
    while not task_queue.empty():
        run_tune_task = self.create_remote_worker()
        active_tasks = []
        for i in range(num_bundles_max):
            if not task_queue.empty():
                exp_data_path = task_queue.get()
                queue_length -= 1
                task = run_tune_task.remote(exp_data_path)
                active_tasks.append(task)

        # 等待所有当前活跃的任务完成
        completed_tasks = ray.get(active_tasks)
        if queue_length != 0:
            self.check_num_bundles(queue_length, total_cpus)
        for task_result, path in completed_tasks:
            best_result = task_result.get_best_result(metric="loss", mode="min")
            results.append({
                "opt_params": best_result.config,
                "opt_score": best_result.metrics["loss"],
                "file_path": path
            })
   
    return results

def check_num_bundles(self, queue_length, total_cpus):
    if queue_length < self.num_bundles:
        self.num_bundles = queue_length
    self.cpus_per_bundle = int(max(1, total_cpus//self.num_bundles))
    
def create_remote_worker(self):
    @ray.remote(num_cpus=self.cpus_per_bundle)
    def run_tune_task(exp_data_path):
        if self.calc_init_N:
            self.set_init_N(exp_data_path, init_flag='mean')
        algo = self.create_algo()
        if isinstance(exp_data_path, list):
            x_uni_exp = []
            data_exp = []
            for exp_data_path_tem in exp_data_path:
                if self.exp_data:
                    x_uni_exp_tem, data_exp_tem = self.get_all_exp_data(exp_data_path_tem)
                else:
                    x_uni_exp_tem, data_exp_tem = self.get_all_synth_data(exp_data_path_tem)
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
            data_name = os.path.basename(exp_data_path[0])
        else:
            if self.exp_data:
                x_uni_exp, data_exp = self.get_all_exp_data(exp_data_path)
            else:
                x_uni_exp, data_exp = self.get_all_synth_data(exp_data_path)
            data_name = os.path.basename(exp_data_path)
            
        if data_name.startswith("Sim_"):
            data_name = data_name[len("Sim_"):]
        if data_name.endswith(".xlsx"):
            data_name = data_name[:-len(".xlsx")]
            
        def objective_func(config):
            # message = f"The value is: {data_exp[0][0][10,10]}"
            # self.print_notice(message)
            return self.objective(config, x_uni_exp, data_exp)
        objective_func.__name__ = "objective_func" + uuid.uuid4().hex[:8]
        tuner = tune.Tuner(
            tune.with_resources(
                objective_func,
                {"cpu": self.cpus_per_trail}
            ),
            param_space=self.RT_space,
            tune_config=tune.TuneConfig(
                num_samples=self.n_iter,
                search_alg=algo,
                max_concurrent_trials=self.cpus_per_bundle
            ),
            run_config=train.RunConfig(
                storage_path=self.tune_storage_path,
                name = data_name, 
                verbose = 0,
                log_to_file=True,
            ),
        )
        return tuner.fit(), exp_data_path
    return run_tune_task

def optimierer_agg(self, opt_params, init_points=4, hyperparameter=None, exp_data_path=None):
    """
    Optimize the corr_agg based on :meth:`~.calc_delta_agg`. 
    Results are saved in corr_agg_opt.
    
    Parameters
    ----------
    method : `str`
        Which algorithm to use for optimization.
    init_points : `int`, optional. Default 4.
        Number of steps for random exploration in BayesianOptimization.
    sample_num : `int`, optional. Default 1.
        Set how many sets of experimental data are used simultaneously for optimization.
    exp_data_path : `str`
        path for experimental data.
        
    Returns   
    -------
    delta_opt : `float`
        Optimized value of the objective.
    """
    if self.calc_init_N:
        self.set_init_N(exp_data_path, init_flag='mean')
    if isinstance(exp_data_path, list):
        ## When set to multi, the exp_data_path entered here is a list 
        ## containing one 2d data name and two 1d data names.
        x_uni_exp = []
        data_exp = []
        for exp_data_path_tem in exp_data_path:
            if self.exp_data:
                x_uni_exp_tem, data_exp_tem = self.get_all_exp_data(exp_data_path_tem)
            else:
                x_uni_exp_tem, data_exp_tem = self.get_all_synth_data(exp_data_path_tem)
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
    else:
        ## When not set to multi or optimization of 1d-data, the exp_data_path 
        ## contain the name of that data.
        if self.exp_data:
            x_uni_exp, data_exp = self.get_all_exp_data(exp_data_path)
        else:
            x_uni_exp, data_exp = self.get_all_synth_data(exp_data_path)
        
    RT_space = {}
    
    for param, info in opt_params.items():
        bounds = info['bounds']
        log_scale = info.get('log_scale', False)
        
        if log_scale:
            RT_space[param] = tune.loguniform(10**bounds[0], 10**bounds[1])
        else:
            RT_space[param] = tune.uniform(bounds[0], bounds[1])
   
    algo = self.create_algo()
        
    if isinstance(exp_data_path, list):
        data_name = os.path.basename(exp_data_path[0])
    else:
        data_name = os.path.basename(exp_data_path)
        
    if data_name.startswith("Sim_"):
        data_name = data_name[len("Sim_"):]
    if data_name.endswith(".xlsx"):
        data_name = data_name[:-len(".xlsx")]
    # 运行Ray Tune进行超参数搜索
    trainable_with_resources  = tune.with_resources(lambda config: self.objective(config, x_uni_exp, data_exp), 
                                                    resources=tune.PlacementGroupFactory([
        {"CPU": self.cpus_per_trail}
    ]))
    
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=RT_space,
        tune_config=tune.TuneConfig(
            num_samples=self.n_iter,
            # scheduler=scheduler,
            search_alg=algo
        ),
        run_config=train.RunConfig(
        storage_path =self.tune_storage_path,
        name = data_name, 
        verbose = 1,
        # log_to_file=("stdout.log", "stderr.log")
        )
    )
    
    results = tuner.fit()
    # 获取最优结果
    opt_result = results.get_best_result(metric="loss", mode="min")
    opt_params = opt_result.config
    opt_score = opt_result.metrics["loss"]
    
    result_dict = {
        "opt_score": opt_score,
        "opt_params": opt_params,
        "file_path": exp_data_path
    }
    
    return result_dict

# Objective function considering the log scale transformation if necessary
def objective(self, config, x_uni_exp, data_exp):
    # Special handling for corr_agg based on dimension
    if 'corr_agg_0' in config:
        transformed_params = self.array_dict_transform(config)
    else:
        transformed_params = config
    # checkpoint_config=train.CheckpointConfig(
    #     checkpoint_at_end=True,  # 仅在任务结束时保存检查点
    #     checkpoint_frequency=0   # 禁用中间检查点保存
    # ),  
    checkpoint_config = None
    
    loss = self.calc_delta_agg(transformed_params, x_uni_exp, data_exp)
    train.report({"loss": loss}, checkpoint=checkpoint_config)
    
def array_dict_transform(self, array_dict):
    # Special handling for array in dictionary like corr_agg based on dimension
        if self.p.dim == 1:
            array_dict['corr_agg'] = np.array([array_dict['corr_agg_0']])
            del array_dict["corr_agg_0"]
        elif self.p.dim == 2:
            array_dict['corr_agg'] = np.array([array_dict[f'corr_agg_{i}'] for i in range(3)])
            for i in range(3):
                del array_dict[f'corr_agg_{i}']
        return array_dict
    
def create_algo(self):
    if self.method == 'HEBO': 
        return HEBOSearch(metric="loss", mode="min")
    elif self.method == 'GP': 
        return OptunaSearch(metric="loss", mode="min", sampler=GPSampler())
    elif self.method == 'TPE': 
        return OptunaSearch(metric="loss", mode="min", sampler=TPESampler())
    elif self.method == 'Cmaes':    
        return OptunaSearch(metric="loss", mode="min", sampler=CmaEsSampler())
    elif self.method == 'NSGA':    
        return OptunaSearch(metric="loss", mode="min", sampler=NSGAIIISampler())
    elif self.method == 'QMC':    
        return OptunaSearch(metric="loss", mode="min", sampler=QMCSampler())