# -*- coding: utf-8 -*-
"""
Using Baysian-Optimization to find the Hyper-Parameter of ANN
"""
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
import time
from pypbe.bond_break.bond_break_ANN_F import ANN_bond_break
import pickle
## package for Ray Tune
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
## available Optuna samplers https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
from optuna.samplers import GPSampler, TPESampler, NSGAIIISampler, QMCSampler

RT_space = {
    'epochs': tune.randint(10, 200),
    'batch_size': tune.randint(16, 128),
    'learning_rate': tune.loguniform(1e-6, 1e-2),
    'optimizer_type': tune.choice(['sgd', 'adam', 'rmsprop']),
    'l1_factor': tune.loguniform(1e-6, 1e-2),
    'dropout_rate': tune.uniform(0.1, 0.5),
    'num_layers': tune.randint(1, 5),
    'init_neurons': tune.randint(10, 512)
}

def RT_train_model(config, ann):
    epochs = config['epochs']
    ann.batch_size = config['batch_size']
    ann.learning_rate = config['learning_rate']
    ann.optimizer_type = config['optimizer_type']
    ann.l1_factor = config['l1_factor']
    ann.dropout_rate = config['dropout_rate']
    ann.num_layers = config['num_layers']
    ann.init_neurons = config['init_neurons']
    # epochs = 1000
    # ann.batch_size = 64
    # ann.learning_rate = 1e-4
    # ann.optimizer_type = 'rmsprop'
    # ann.l1_factor = 1e-4
    # ann.dropout_rate = 0.2
    # ann.num_layers = 2
    # ann.init_neurons = 64
    
    # ann.processing_train_data()
    model_path = train.get_context().get_trial_dir()
    results = ann.cross_validation(epochs, 1, model_path)
    
    ## single objective with weighted sum
    loss = results[0,0] * m_mse_weight + results[0,2] * m_frag_num_weight
    train.report({"loss": loss})
    
    ## single objective with penalty points
    # loss = results[0,0] if results[0,2] <= 10 else results[0,0] + 1e6
    # train.report({"loss": loss})

    ## multi objective(for TPE, NSGA, QMC)
    # loss1 = results[0,0]
    # loss2 = results[0,2]
    # train.report({"loss1": loss1, "loss2": loss2})
    # print(f"The mse is {results[0,0]}, the frag_error is {results[0,2]}")
    
def run_ray_tune(result_path, n_steps):
    ## Initialize ANN training Instance 
    ann = ANN_bond_break(m_dim, m_NS, m_S, m_global_seed)
    ann.save_model = False
    ann.save_validate_results = False
    ann.print_status = False
    ## For unicluster
    # tmpdir = os.environ.get('TMPDIR')
    # ann.path_scaler = os.path.join(tmpdir,'Inputs_scaler.pkl')
    # ann.path_all_data = os.path.join(tmpdir,'output_data_vol.pkl')
    ## Read training data and split them
    ann.split_data_set(n_splits=m_n_splits)
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
    # algo = OptunaSearch(metric=["loss1", "loss2"], mode=["min", "min"], sampler=TPESampler())
    # 运行Ray Tune进行超参数搜索
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(RT_train_model,ann=ann),
            {"cpu": 2}
        ),
        param_space=RT_space,
        tune_config=tune.TuneConfig(
            num_samples=n_steps,
            # scheduler=scheduler,
            search_alg=algo,
            reuse_actors=True,
        ),
        run_config=train.RunConfig(
        verbose = 0,
        storage_path =r"C:\Users\px2030\Code\Ray_Tune"  
        )
    )
    
    results = tuner.fit()
    # 获取最优结果
    best_result = results.get_best_result(metric="loss", mode="min")
    # best_result = results.get_best_result(metric="loss1", mode="min")
    best_config = best_result.config
    best_score = best_result.metrics["loss"]
    # best_score = best_result.metrics["loss1"]
    best_model_path = best_result.path
    
    result_dict = {
        "best_score": best_score,
        "best_parameters": best_config,
        "best_model_path": best_model_path
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
    # epochs = 152
    # ann.batch_size = 86
    # ann.learning_rate = 0.000409987
    # ann.optimizer_type = 'rmsprop'
    # ann.l1_factor = 0.00381228
    # ann.dropout_rate = 0.463052
    # ann.num_layers = 1
    # ann.init_neurons = 41
    
    # ann.processing_train_data()
    ann.split_data_set(n_splits=m_n_splits)
    results = ann.cross_validation(epochs, 1, model_path=None, validate_only=True)
    if m_dim == 1:
        ann.plot_1d_F(epochs, 20, vol_dis=False)
    elif m_dim == 2:
        ann.plot_2d_F(epochs, 20, vol_dis=True)
    return ann, results
    
if __name__ == '__main__':
    ## The value of random seed (int value) itself is not important.
    ## But fixing random seeds can ensure the consistency and comparability of the results.
    ## The reverse improves the robustness of the model (set m_global_seed=0)
    m_global_seed = 42
    
    m_dim = 1
    m_NS = 50
    m_S = 1.3
    m_n_splits = 5
    m_mse_weight = 1e4
    m_frag_num_weight = 1e-2
    
    result_path = "best_results.pkl"
    n_steps = 100
    
    start_time = time.time()
    best_result = run_ray_tune(result_path, n_steps)
    end_time = time.time()
    opt_time = end_time - start_time
    
    # ann, test_result = read_best_params_and_train_model(result_path)
    # loss = test_result[0,0] * m_mse_weight + test_result[0,2] * m_frag_num_weight
    