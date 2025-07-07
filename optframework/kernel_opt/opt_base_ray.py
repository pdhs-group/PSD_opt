# -*- coding: utf-8 -*-
"""
Optimization algorithm based on ray-Tune
"""
# import uuid
import os
import sqlite3
from filelock import FileLock
import json
# import numpy as np
# import math
# import ray
from ray import tune
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
    """
    Print the current number of active Ray actors.

    Returns
    -------
    None
    """
    actors = state.actors()
    print(f"Number of actors: {len(actors)}")
    for actor_id, actor_info in actors.items():
        print(f"Actor ID: {actor_id}, State: {actor_info['State']}")


def multi_optimierer_ray(self, opt_params_space, exp_data_paths=None, known_params=None):
    """
    Optimize PBE parameters using multiple Ray Tuners, managed via Ray multiprocessing.

    This method enables multiple instances of Ray Tune to run concurrently, each tuning 
    with different experimental data sets or parameters. It splits the job queue based on 
    the number of available jobs and distributes the tasks across multiple processes 
    managed by Ray's multiprocessing pool.

    Parameters
    ----------
    opt_params_space : dict
        A dictionary of optimization parameters, where each key corresponds to a parameter name 
        and its value contains information about the bounds and scaling (logarithmic or linear).
    exp_data_paths : list of str, optional
        Paths to the experimental data for each tuning instance.
    known_params : list of dict, optional
        Known parameters to be passed to the optimization process.
    
    Returns
    -------
    list
        A list of dictionaries, where each dictionary contains the optimized parameters and 
        the corresponding objective score (delta).
    """
    # --- build the Ray Tune search space once ---
    self.RT_space = {}
    for name, info in opt_params_space.items():
        if "fixed" in info:
            # just hand back the literal value
            self.RT_space[name] = info["fixed"]
        else:
            lo, hi = info["bounds"]
            if info.get("log_scale", False):
                self.RT_space[name] = tune.loguniform(10**lo, 10**hi)
            else:
                self.RT_space[name] = tune.uniform(lo, hi)

    # Build the job queue
    job_queue = list(zip(exp_data_paths or [], known_params or []))
    queue_length = len(job_queue)

    # Ensure num_jobs does not exceed queue length
    self.check_num_jobs(queue_length)
    num_workers = self.core.num_jobs

    # The worker will pull its slice of the queue, look up warm-start params,
    # and pass them into optimierer_ray
    def worker(job_queue_slice):
        job_results = []
        for paths, params in job_queue_slice:
            # call optimierer_ray with warm start
            result = self.optimierer_ray(
                exp_data_paths=paths,
                known_params=params,
            )
            job_results.append(result)
        return job_results

    # split into roughly equal slices
    job_slices = [job_queue[i::num_workers] for i in range(num_workers)]

    # dispatch with multiprocessing
    results = []
    with mp.Pool(num_workers) as pool:
        batches = pool.map(worker, job_slices)
    for batch in batches:
        results.extend(batch)

    return results

def check_num_jobs(self, queue_length):
    """
    Adjust the number of concurrent Tuners to the available job queue length.

    Parameters
    ----------
    queue_length : int
        The length of the job queue, representing the number of data sets available for tuning.

    Returns
    -------
    None
    """
    if queue_length < self.core.num_jobs:
        self.core.num_jobs = queue_length

def _save_warm_params(db_path, data_name, all_params, all_score):
    assert len(all_params) == len(all_score)
    lock_path = db_path + ".lock"

    with FileLock(lock_path):  # Ensure concurrent safety
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create a table（if not exist）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS warm_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_name TEXT,
            param_json TEXT,
            score REAL
        )
        """)

        # delete the old data_name (if exist)
        cursor.execute("DELETE FROM warm_params WHERE data_name = ?", (data_name,))

        # insert new values
        records = [(data_name, json.dumps(p), s) for p, s in zip(all_params, all_score)]
        cursor.executemany("INSERT INTO warm_params (data_name, param_json, score) VALUES (?, ?, ?)", records)

        conn.commit()
        conn.close()

        
def _load_warm_params(db_path, data_name):
    if not os.path.exists(db_path):
        return [], []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT param_json, score FROM warm_params WHERE data_name = ?", (data_name,))
    rows = cursor.fetchall()
    conn.close()

    all_params = [json.loads(p) for p, _ in rows]
    all_score = [s for _, s in rows]
    return all_params, all_score


def optimierer_ray(self, opt_params_space=None, exp_data_paths=None,known_params=None, evaluated_params=None):
    """
    Optimize PBE parameters using Ray's Tune module, based on the delta calculated by the method `calc_delta_agg`.

    This method utilizes Ray Tune for hyperparameter optimization. It sets up the search 
    space and runs the optimization process, either for single or multi-dimensional 
    PBEs. The results are saved and returned in a dictionary 
    that contains the optimized parameters and the corresponding objective score.

    Parameters
    ----------
    opt_params_space : dict, optional
        A dictionary of optimization parameters, where each key corresponds to a parameter name 
        and its value contains information about the bounds and scaling (logarithmic or linear).
    exp_data_paths : list of str, optional
        Paths to the experimental or synthetic data to be used for optimization. For multi-case, 
        this should be a list containing the paths for 1D and 2D data.
    known_params : dict, optional
        Known parameters to be passed to the optimization process.

    Returns
    -------
    dict
        A dictionary containing:
            - "opt_score": The optimized objective score (delta value).
            - "opt_params_space": The optimized parameters from the search space.
            - "file_path": The path(s) to the experimental data used for optimization.
    """
    
    # evaluated_params = getattr(self.core, 'evaluated_params', None)
    # evaluated_rewards = getattr(self.core, 'evaluated_rewards', None)
    evaluated_params = None
    evaluated_rewards = None
    
    # Prepare experimental data (either for 1D or 2D)
    if isinstance(exp_data_paths, list):
        # When set to multi, the exp_data_paths entered here is a list containing one 2d data name and two 1d data names.
        x_uni_exp = []
        data_exp = []
        for exp_data_paths_tem in exp_data_paths:
            if self.core.exp_data:
                x_uni_exp_tem, data_exp_tem = self.core.get_all_exp_data(exp_data_paths_tem)
            else:
                x_uni_exp_tem, data_exp_tem = self.core.get_all_synth_data(exp_data_paths_tem)
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
        data_name = getattr(self.core, 'data_name_tune', os.path.basename(exp_data_paths[0]))
    else:
        # When not set to multi or optimization of 1d-data, the exp_data_paths contain the name of that data.
        if self.core.exp_data:
            x_uni_exp, data_exp = self.core.get_all_exp_data(exp_data_paths)
        else:
            x_uni_exp, data_exp = self.core.get_all_synth_data(exp_data_paths)
        data_name = os.path.basename(exp_data_paths)
        
    # Reuse the previous parameters as warm-up for new optimization
    resume_unfinished = getattr(self.core, 'resume_unfinished', False)
    result_dir = getattr(self.core, 'result_dir', self.core.tune_storage_path)
    if resume_unfinished:
        n_prev = getattr(self.core, 'n_iter_prev', 0)
        warm_params_path = os.path.join(result_dir, f"{n_prev}.sqlite")
        evaluated_params, evaluated_rewards = _load_warm_params(warm_params_path, data_name)
        evaluated_rewards = None
            
    # Set up the Ray Tune search space    
    self.RT_space = {}
    for name, info in opt_params_space.items():
        if "fixed" in info:
            # just hand back the literal value
            self.RT_space[name] = info["fixed"]
        else:
            lo, hi = info["bounds"]
            if info.get("log_scale", False):
                self.RT_space[name] = tune.loguniform(10**lo, 10**hi)
            else:
                self.RT_space[name] = tune.uniform(lo, hi)
    # Create the search algorithm
    algo = self.create_algo(evaluated_params=evaluated_params, evaluated_rewards=evaluated_rewards)
    # Clean up the data name for output storage 
    if data_name.startswith("Sim_"):
        data_name = data_name[len("Sim_"):]
    if data_name.endswith(".xlsx"):
        data_name = data_name[:-len(".xlsx")]
    # Define the directory name creator for each trial
    def trial_dirname_creator(trial):
        return f"trial_{trial.trial_id}"
    # Set up the trainable function based on the multi_flag
    if not self.multi_flag:
        trainable = tune.with_parameters(OptCoreRay, core_params=self.core_params, pop_params=self.pop_params,
                                         data_path=self.data_path, exp_data_paths=exp_data_paths,
                                         x_uni_exp=x_uni_exp, data_exp=data_exp, known_params=known_params, 
                                         exp_case=self.core.exp_data)
    else:
        trainable = tune.with_parameters(OptCoreMultiRay, core_params=self.core_params, pop_params=self.pop_params,
                                         data_path=self.data_path, exp_data_paths=exp_data_paths,
                                         x_uni_exp=x_uni_exp, data_exp=data_exp, known_params=known_params,
                                         exp_case=self.core.exp_data)    
    # Define the resources used for each trial using PlacementGroupFactory
    trainable_with_resources  = tune.with_resources(trainable, 
                                                  resources=tune.PlacementGroupFactory([{"CPU": self.core.cpus_per_trail}]),
    )
    # trainable_with_resources  = tune.with_resources(trainable,                             
    #     {"cpu": self.core.cpus_per_trail}, 
    # )
    
    # checkpoint_path_save = os.path.join(self.core.tune_storage_path, f"{data_name}_checkpoint_{n_save}.pkl")
    # if resume_unfinished:
    #     # checkpoint_path_re = os.path.join(self.core.tune_storage_path, f"{data_name}_checkpoint_{n_prev}.pkl")
    #     # Resume tuning
    #     # algo.restore_from_dir(
    #     #     os.path.join(self.core.tune_storage_path, data_name))
    #     # algo.restore(checkpoint_path_re)
    #     # Set up a new Ray Tune Tuner
    #     tuner = tune.Tuner(
    #         trainable_with_resources,
    #         param_space=self.RT_space,
    #         tune_config=tune.TuneConfig(
    #             num_samples=self.core.n_iter,
    #             search_alg=algo,
    #             reuse_actors=True,
    #             trial_dirname_creator=trial_dirname_creator,
    #         ),
    #         run_config=tune.RunConfig(
    #         storage_path =self.core.tune_storage_path,
    #         name = data_name,
    #         verbose = self.core.verbose, # verbose=0: no trial info, 1: basic info, 2: detailed info
    #         stop={"training_iteration": 1},
    #         )
    #     )
    # else:
    # Set up a new Ray Tune Tuner
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=self.RT_space,
        tune_config=tune.TuneConfig(
            num_samples=self.core.n_iter,
            search_alg=algo,
            reuse_actors=True,
            trial_dirname_creator=trial_dirname_creator,
        ),
        run_config=tune.RunConfig(
        storage_path =self.core.tune_storage_path,
        name = data_name,
        verbose = self.core.verbose, # verbose=0: no trial info, 1: basic info, 2: detailed info
        stop={"training_iteration": 1},
        )
    )
    # Run the optimization process
    results = tuner.fit()
    # algo.save(checkpoint_path_save)
    
    all_params = []
    all_score = []
    for trial in results:
        config = trial.config
        score = trial.metrics.get("loss", None)
        if score is not None:
            all_params.append(config)
            all_score.append(score)

    warm_params_path = os.path.join(result_dir, f"{self.core.n_iter}.sqlite")
    _save_warm_params(warm_params_path, data_name, all_params, all_score)
    # Get the best result from the optimization
    opt_result = results.get_best_result(metric="loss", mode="min")
    opt_params = opt_result.config
    opt_exp_data_paths = opt_result.metrics["exp_paths"]
    opt_score = opt_result.metrics["loss"]
    result_dict = {
        "opt_score": opt_score,
        "opt_params": opt_params,
        "file_path": opt_exp_data_paths,
    }

    return result_dict
    
def create_algo(self, batch=False, evaluated_params=None, evaluated_rewards=None):
    """
    Create and return the search algorithm to be used for hyperparameter optimization.

    This method creates a search algorithm based on the `method` attribute of the core object. 
    It supports a variety of search algorithms from the Optuna library, including Bayesian 
    optimization (`GP`), tree-structured Parzen estimators (`TPE`), covariance matrix adaptation 
    evolution strategy (`Cmaes`), NSGA-II (`NSGA`), and quasi-Monte Carlo sampling (`QMC`). 
    Optionally, it can also limit the number of concurrent trials.
    
    The number of concurrent trials controls the parallelism of the optimization process. In theory, 
    increasing the number of concurrent trials speeds up the calculation, but it may reduce the 
    convergence rate due to less frequent information sharing between trials. Empirically, a range of 
    4-12 concurrent trials tends to work well. 

    The `batch` parameter controls whether a new batch of trials is submitted only after the current 
    batch finishes all trials. Note that for some algorithms, the batch setting is fixed; for example, 
    the HEBO algorithm always uses batching.

    Parameters
    ----------
    batch : bool, optional
        Whether to use batch mode for the concurrency limiter. Default is False.

    Returns
    -------
    search_alg : object
        The search algorithm instance to be used for optimization.

    """
    # if self.core.method == 'HEBO': 
    #     search_alg = HEBOSearch(metric="loss", mode="min", random_state_seed=self.core.random_seed)
    if self.core.method == 'GP': 
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=GPSampler(seed=self.core.random_seed), 
                                  points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
    elif self.core.method == 'TPE': 
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=TPESampler(seed=self.core.random_seed), 
                                  points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
    elif self.core.method == 'Cmaes':    
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=CmaEsSampler(seed=self.core.random_seed), 
                                  points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
    elif self.core.method == 'NSGA':    
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=NSGAIIISampler(seed=self.core.random_seed), 
                                  points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
    elif self.core.method == 'QMC':    
        search_alg = OptunaSearch(metric="loss", mode="min", sampler=QMCSampler(scramble=True, seed=self.core.random_seed), 
                                  points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
    else:
        raise ValueError(f"Unsupported sampler detected: {self.core.method}")
    # If no concurrency limit is set, return the search algorithm directly    
    if self.core.max_concurrent is None:
        return search_alg
    else:
        # Limit the number of concurrent trials using ConcurrencyLimiter
        return ConcurrencyLimiter(search_alg, max_concurrent=self.core.max_concurrent, batch=batch)