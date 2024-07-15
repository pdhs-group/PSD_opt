# -*- coding: utf-8 -*-
"""
Using Baysian-Optimization to find the Hyper-Parameter of ANN
"""
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
import time
from pypbe.bond_break.bond_break_ANN_F import ANN_bond_break
import pickle
## package for Baysian-Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
## package for Ray Tune
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
## available Optuna samplers https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
from optuna.samplers import GPSampler, TPESampler

BO_space  = [
    Integer(10, 100, name='epochs'),
    Integer(16, 128, name='batch_size'),
    Real(1e-6, 1e-2, "log-uniform", name='learning_rate'),
    Categorical(['sgd', 'adam', 'rmsprop'], name='optimizer_type'),
    Real(1e-6, 1e-2, "log-uniform", name='l1_factor'),
    Real(0.1, 0.5, name='dropout_rate'),
    Integer(1, 5, name='num_layers'),
    Integer(10, 512, name='init_neurons')
]

@use_named_args(BO_space)
def BO_train_model(epochs, batch_size, learning_rate, optimizer_type, l1_factor,
                dropout_rate, num_layers, init_neurons):
    
    ann = ANN_bond_break(m_dim,m_NS,m_S,m_global_seed)
    ann.save_model = False
    ann.save_validate_results = False
    ann.print_status = False
    
    ann.batch_size = batch_size
    ann.learning_rate = learning_rate
    ann.optimizer_type = optimizer_type
    ann.l1_factor = l1_factor
    ann.dropout_rate = dropout_rate
    ann.num_layers = num_layers
    ann.init_neurons = init_neurons
    
    # ann.processing_train_data()
    ann.split_data_set(n_splits=m_n_splits)
    results = ann.cross_validation(epochs, 1)
    
    return results[0,0] * m_mse_weight + results[0,2] * m_frag_num_weight

def run_BO(result_path, n_steps):
    res_gp = gp_minimize(BO_train_model, BO_space, n_calls=n_steps, random_state=0)
    best_score = res_gp.fun
    best_parameters = res_gp.x
    
    result_dict = {
        "best_score": best_score,
        "best_parameters": {
            "epochs": best_parameters[0],
            "batch_size": best_parameters[1],
            "learning_rate": best_parameters[2],
            "optimizer_type": best_parameters[3],
            "l1_factor": best_parameters[4],
            "dropout_rate": best_parameters[5],
            "num_layers": best_parameters[6],
            "init_neurons": best_parameters[7]
        }
    }
    
    with open(result_path, "wb") as f:
        pickle.dump(result_dict, f)
    return result_dict
        
RT_space = {
    'epochs': tune.randint(10, 100),
    'batch_size': tune.randint(16, 128),
    'learning_rate': tune.loguniform(1e-6, 1e-2),
    'optimizer_type': tune.choice(['sgd', 'adam', 'rmsprop']),
    'l1_factor': tune.loguniform(1e-6, 1e-2),
    'dropout_rate': tune.uniform(0.1, 0.5),
    'num_layers': tune.randint(1, 5),
    'init_neurons': tune.randint(10, 512)
}

def RT_train_model(config):
    ann = ANN_bond_break(m_dim, m_NS, m_S, m_global_seed)
    ann.save_model = False
    ann.save_validate_results = False
    ann.print_status = False

    ann.batch_size = config['batch_size']
    ann.learning_rate = config['learning_rate']
    ann.optimizer_type = config['optimizer_type']
    ann.l1_factor = config['l1_factor']
    ann.dropout_rate = config['dropout_rate']
    ann.num_layers = config['num_layers']
    ann.init_neurons = config['init_neurons']

    # ann.processing_train_data()
    ann.split_data_set(n_splits=m_n_splits)
    results = ann.cross_validation(config['epochs'], 1)
    loss=results[0,0] * m_mse_weight + results[0,2] * m_frag_num_weight
    train.report({"loss": loss})
    
def run_ray_tune(result_path, n_steps):
    # 初始化Ray
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))}},
             # num_cpus=12
             )
    
    # 使用ASHAScheduler进行早停
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=1000,
    #     grace_period=1,
    #     reduction_factor=2)
    
    # 使用Bayesian Optimization作为搜索算法
    algo = OptunaSearch(metric="loss", mode="min", sampler=GPSampler())
    
    # 运行Ray Tune进行超参数搜索
    tuner = tune.Tuner(
        RT_train_model,
        param_space=RT_space,
        tune_config=tune.TuneConfig(
            num_samples=n_steps,
            # scheduler=scheduler,
            search_alg=algo
        ),
    )
    
    results = tuner.fit()
    # 获取最优结果
    best_result = results.get_best_result(metric="loss", mode="min")
    best_config = best_result.config
    best_score = best_result.metrics["loss"]
    
    result_dict = {
        "best_score": best_score,
        "best_parameters": best_config
    }
    
    with open(result_path, "wb") as f:
        pickle.dump(result_dict, f)

    # 关闭Ray
    ray.shutdown()
    return result_dict
    
def read_best_params_and_train_model(result_path):
    with open(result_path, "rb") as f:
        load_result = pickle.load(f)
        
    ann = ANN_bond_break(m_dim,m_NS,m_S,m_global_seed)
    ann.save_model = True
    ann.save_validate_results = True
    ann.print_status = True
    
    epochs = load_result["best_parameters"]["epochs"]
    ann.batch_size = load_result["best_parameters"]["batch_size"]
    ann.learning_rate = load_result["best_parameters"]["learning_rate"]
    ann.optimizer_type = load_result["best_parameters"]["optimizer_type"]
    ann.l1_factor = load_result["best_parameters"]["l1_factor"]
    ann.dropout_rate = load_result["best_parameters"]["dropout_rate"]
    ann.num_layers = load_result["best_parameters"]["num_layers"]
    ann.init_neurons = load_result["best_parameters"]["init_neurons"]
    
    # ann.processing_train_data()
    ann.split_data_set(n_splits=m_n_splits)
    results = ann.cross_validation(epochs, 1)
    return results
    
if __name__ == '__main__':
    ## The value of random seed (int value) itself is not important.
    ## But fixing random seeds can ensure the consistency and comparability of the results.
    ## The reverse improves the robustness of the model (set m_global_seed=0)
    m_global_seed = 0
    
    m_dim = 1
    m_NS = 50
    m_S = 1.3
    m_n_splits = 5
    m_mse_weight = 1e4
    m_frag_num_weight = 1e-2
    
    result_path = "best_results.pkl"
    n_steps = 400
    
    start_time = time.time()
    # best_result = run_BO(result_path, n_steps)
    best_result = run_ray_tune(result_path, n_steps)
    end_time = time.time()
    opt_time = end_time - start_time
    
    # test_result = read_best_params_and_train_model(result_path)
    
    