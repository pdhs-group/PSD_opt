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
import ray.util.multiprocessing as mp
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hebo import HEBOSearch
from optuna.samplers import GPSampler,CmaEsSampler,TPESampler,NSGAIIISampler,QMCSampler
from ray.tune.search import ConcurrencyLimiter
from .opt_core_ray import OptCoreRay
from .opt_core_multi_ray import OptCoreMultiRay

from ray._private import state

def print_current_actors(self):
    actors = state.actors()
    print(f"Number of actors: {len(actors)}")
    for actor_id, actor_info in actors.items():
        print(f"Actor ID: {actor_id}, State: {actor_info['State']}")


def multi_optimierer_ray(self, opt_params, exp_data_paths=None, known_params=None):
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
    exp_data_paths : str
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

    ## Add all data (groups for multi_flag=True case) to the job queue
    job_queue = [(paths, params) for paths, params in zip(exp_data_paths, known_params)]
    queue_length = len(job_queue)

    self.check_num_jobs(queue_length)
    num_jobs_max = self.core.num_jobs
    results = []
    def worker(job_queue_slice):
        job_results = []
        for paths, params in job_queue_slice:
            result = self.optimierer_ray(exp_data_paths=paths, known_params=params)
            job_results.append(result)
        return job_results
        
    job_slices = [job_queue[i::num_jobs_max] for i in range(num_jobs_max)]

    with mp.Pool(num_jobs_max) as pool:
        results_batches = pool.map(worker, job_slices)

    # Flatten the list of results
    for batch in results_batches:
        results.extend(batch)
   
    return results

def check_num_jobs(self, queue_length):
    if queue_length < self.core.num_jobs:
        self.core.num_jobs = queue_length

def optimierer_ray(self, opt_params=None, exp_data_paths=None,known_params=None):
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
    exp_data_paths : `str`
        path for experimental data.
        
    Returns   
    -------
    delta_opt : `float`
        Optimized value of the objective.
    """

    if self.core.calc_init_N:
        self.core.set_init_N(exp_data_paths, init_flag='mean')
    if isinstance(exp_data_paths, list):
        ## When set to multi, the exp_data_paths entered here is a list 
        ## containing one 2d data name and two 1d data names.
        x_uni_exp = []
        data_exp = []
        for exp_data_paths_tem in exp_data_paths:
            if self.core.exp_data:
                x_uni_exp_tem, data_exp_tem = self.core.get_all_exp_data(exp_data_paths_tem)
            else:
                x_uni_exp_tem, data_exp_tem = self.core.get_all_synth_data(exp_data_paths_tem)
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
        data_name = os.path.basename(exp_data_paths[0])
    else:
        ## When not set to multi or optimization of 1d-data, the exp_data_paths 
        ## contain the name of that data.
        if self.core.exp_data:
            x_uni_exp, data_exp = self.core.get_all_exp_data(exp_data_paths)
        else:
            x_uni_exp, data_exp = self.core.get_all_synth_data(exp_data_paths)
        data_name = os.path.basename(exp_data_paths)
        
    if opt_params is not None:    
        self.RT_space = {}
        
        for param, info in opt_params.items():
            bounds = info['bounds']
            log_scale = info.get('log_scale', False)
            
            if log_scale:
                self.RT_space[param] = tune.loguniform(10**bounds[0], 10**bounds[1])
            else:
                self.RT_space[param] = tune.uniform(bounds[0], bounds[1])
   
    algo = self.create_algo()
        
    if data_name.startswith("Sim_"):
        data_name = data_name[len("Sim_"):]
    if data_name.endswith(".xlsx"):
        data_name = data_name[:-len(".xlsx")]

    def trial_dirname_creator(trial):
        return f"trial_{trial.trial_id}"
    # objective_func.__name__ = "objective_func" + uuid.uuid4().hex[:8]
    if self.dim == 1:
        trainable = tune.with_parameters(OptCoreRay, core_params=self.core_params, pop_params=self.pop_params,
                                         data_path=self.data_path,
                                         x_uni_exp=x_uni_exp, data_exp=data_exp, known_params=known_params)
    else:
        trainable = tune.with_parameters(OptCoreMultiRay, core_params=self.core_params, pop_params=self.pop_params,
                                         data_path=self.data_path,
                                         x_uni_exp=x_uni_exp, data_exp=data_exp, known_params=known_params)    
    trainable_with_resources  = tune.with_resources(trainable, 
                                                  resources=tune.PlacementGroupFactory([{"CPU": self.core.cpus_per_trail}]),
    )
    # trainable_with_resources  = tune.with_resources(trainable,                             
    #     {"cpu": self.core.cpus_per_trail}, 
    # )
    
    tuner = tune.Tuner(
        trainable_with_resources,
        # trainable,
        param_space=self.RT_space,
        tune_config=tune.TuneConfig(
            num_samples=self.core.n_iter,
            # scheduler=scheduler,
            search_alg=algo,
            reuse_actors=True,
            trial_dirname_creator=trial_dirname_creator,
        ),
        run_config=train.RunConfig(
        storage_path =self.core.tune_storage_path,
        name = data_name, 
        verbose = 1,
        stop={"training_iteration": 1},
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
        "file_path": exp_data_paths
    }

    return result_dict
    
def create_algo(self, batch=False):
    if self.core.method == 'HEBO': 
        search_alg = HEBOSearch(metric="loss", mode="min")
    elif self.core.method == 'GP': 
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=GPSampler())
    elif self.core.method == 'TPE': 
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=TPESampler())
    elif self.core.method == 'Cmaes':    
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=CmaEsSampler())
    elif self.core.method == 'NSGA':    
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=NSGAIIISampler())
    elif self.core.method == 'QMC':    
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=QMCSampler())
    if self.core.max_concurrent is None:
        return search_alg
    else:
        return ConcurrencyLimiter(search_alg, max_concurrent=self.core.max_concurrent, batch=batch)