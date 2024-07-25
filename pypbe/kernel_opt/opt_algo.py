# -*- coding: utf-8 -*-
"""
Calculate the difference between the PSD of the simulation results and the experimental data.
Minimize the difference by optimization algorithm to obtain the kernel of PBE.
"""
import os
import numpy as np
# import math
import ray
from ray import tune, train
# from ray.util.placement_group import placement_group
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hebo import HEBOSearch
from optuna.samplers import GPSampler,CmaEsSampler,TPESampler,NSGAIIISampler,QMCSampler
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import interp1d
## import internal package
from ..dpbe import population
from ..utils.func.func_read_exp import write_read_exp

class opt_algo():
    """
    Class definition for calculations within optimization process class.

    Attributes
    ----------
    n_iter : `int`, optional
        Number of iterations of the optimization process. Default is 100.
    delta_flag : `str`, optional
        Which data from the PSD is used for the calculation. Default is 'q3'. Options include:
        
        - 'q3': Number density distribution
        - 'Q3': Cumulative distribution
        - 'x_10': Particle size corresponding to 10% cumulative distribution
        - 'x_50': Particle size corresponding to 50% cumulative distribution
        - 'x_90': Particle size corresponding to 90% cumulative distribution
    cost_func_type : `str`, optional
        Method for calculating the PSD difference. Default is 'MSE'. Options include:
        
        - 'MSE': Mean Squared Error
        - 'RMSE': Root Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'KL': Kullback–Leibler divergence (Only q3 and Q3 are compatible with KL)
    calc_init_N : `bool`, optional
        Whether to use experimental data to calculate initial conditions. If False, the initial conditions for PBE need to be defined manually. Default is False.
    """
    def __init__(self):
        self.n_iter = 100
        ## delta_flag = 1: use q3
        ## delta_flag = 2: use Q3
        ## delta_flag = 3: use x_10
        ## delta_flag = 4: use x_50
        ## delta_flag = 5: use x_90
        self.delta_flag = 'q3'    
        ## 'MSE': Mean Squared Error
        ## 'RMSE': Root Mean Squared Error
        ## 'MAE': Mean Absolute Error
        ## 'KL': Kullback–Leibler divergence(Only q3 and Q3 are compatible with KL) 
        self.cost_func_type = 'MSE'

        self.calc_init_N = False
        self.set_init_pop_para_flag = False
        self.set_comp_para_flag = False
        self.num_bundles = 4
        # self.cpu_per_bundles = 20
    #%%  Optimierer    
    def calc_delta_agg(self, params_in, x_uni_exp, data_exp):
        """
        Calculate the difference (delta) of PSD.
        
        - The function is exactly the same as :meth:`~.calc_delta`, but receives corr_agg (agglomeration rate). Then calls `~.return_syth_beta` to compute the equivalent CORR_BETA and alpha_prim.
        
        Parameters
        ----------
        corr_agg : `array`
            Agglomeration rate in PBE.
        scale : `int`, optional. Default 1.
            The actual return result is delta*scale. delta is absolute and always positive. 
            Setting scale to 1 is suitable for optimizers that look for minimum values, 
            and -1 is suitable for those that look for maximum values.
        sample_num : `int`, optional. Default 1.
            Set how many sets of experimental data are used simultaneously for optimization.
        exp_data_path : `str`
            path for experimental data.
        """
        params = params_in.copy()
        if "corr_agg" in params:
            corr_agg = params["corr_agg"]
            CORR_BETA = self.return_syth_beta(corr_agg)
            alpha_prim = corr_agg / CORR_BETA
            
            params["CORR_BETA"] = CORR_BETA
            params["alpha_prim"] = alpha_prim
            
            del params["corr_agg"]

        self.calc_pop(self.p, params, self.t_vec)
        if self.p.calc_status:
            return self.calc_delta_tem(x_uni_exp, data_exp, self.p)
        else:
            return 10

    def calc_delta_tem(self, x_uni_exp, data_exp, pop):
        """
        Loop through all the experimental data and calculate the average diffences.
        
        Parameters
        ----------
        sample_num : `int`, optional. Default 1.
            Set how many sets of experimental data are used simultaneously for optimization.
        exp_data_path : `str`
            path for experimental data.
        pop : :class:`pop.population`
            Instance of the PBE.
            
        Returns   
        -------
        (delta_sum * scale) / x_uni_num : `float`
            Average value of PSD's difference for corresponding to all particle sizes. 
        """
        if self.smoothing:
            kde_list = []
        x_uni = self.calc_x_uni(pop)
        for idt in range(self.delta_t_start_step, self.num_t_steps):
            if self.smoothing:
                sumvol_uni = pop.return_distribution(t=idt, flag='sumvol_uni')[0]
                ## The volume of particles with index=0 is 0. 
                ## In theory, such particles do not exist.
                kde = self.KDE_fit(x_uni[1:], sumvol_uni[1:])
                kde_list.append(kde)
                
        delta_sum = 0    
        if self.sample_num == 1:
            q3_mod = np.zeros((len(x_uni_exp), self.num_t_steps-self.delta_t_start_step))
            for idt in range(self.num_t_steps-self.delta_t_start_step):
                if self.smoothing:
                    q3_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp[1:])
                    q3_mod[1:, idt] = q3_mod_tem
                else:
                    q3_mod[:, idt] = pop.return_distribution(t=idt+self.delta_t_start_step, flag='q3')[0]
                Q3 = self.calc_Q3(x_uni_exp, q3_mod[:, idt]) 
                q3_mod[:, idt] = q3_mod[:, idt] / Q3.max() 
            q3_mod = np.insert(q3_mod, 0, 0.0, axis=0)
            for flag, cost_func_type in self.delta_flag:
                data_mod = self.re_calc_distribution(x_uni_exp, q3=q3_mod, flag=flag)[0]
                # Calculate the error between experimental data and simulation results
                delta = self.cost_fun(data_exp, data_mod, cost_func_type, flag)
                delta_sum += delta 
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(x_uni_exp)
            return delta / x_uni_num
        else:
            for i in range (0, self.sample_num):
                q3_mod = np.zeros((len(x_uni_exp[i]), self.num_t_steps-self.delta_t_start_step))
                for idt in range(self.num_t_steps-self.delta_t_start_step):
                    if self.smoothing:
                        q3_mod_tem = self.KDE_score(kde_list[idt], x_uni_exp[i][1:])
                        q3_mod[1:, idt] = q3_mod_tem
                    else:
                        q3_mod[:, idt] = pop.return_distribution(t=idt+self.delta_t_start_step, flag='q3')[0]
                    Q3 = self.calc_Q3(x_uni_exp[i], q3_mod[:, idt]) 
                    q3_mod[:, idt] = q3_mod[:, idt] / Q3.max()
                for flag, cost_func_type in self.delta_flag:
                    data_mod = self.re_calc_distribution(x_uni_exp[i], q3=q3_mod, flag=flag)[0]
                    # Calculate the error between experimental data and simulation results
                    delta = self.cost_fun(data_exp[i], data_mod, cost_func_type, flag)
                    delta_sum += delta 
            # Restore the original name of the file to prepare for the next step of training
            delta_sum /= self.sample_num
            # Because the number of x_uni is different in different pop equations, 
            # the average value needs to be used instead of the sum.
            x_uni_num = len(x_uni_exp[i])  
            return delta_sum / x_uni_num
        
    def get_all_exp_data(self, exp_data_path):
        if self.sample_num == 1:
            x_uni_exp, sumN_uni_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:]) 
            x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
            sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
            vol_uni = np.tile((1/6)*np.pi*x_uni_exp**3, (self.num_t_steps-self.delta_t_start_step, 1)).T
            sumvol_uni_exp = sumN_uni_exp * vol_uni
            sumvol_uni_exp = np.insert(sumvol_uni_exp, 0, 0.0, axis=0)
            x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
            for flag, _ in self.delta_flag:
                data_exp = self.re_calc_distribution(x_uni_exp, sum_uni=sumvol_uni_exp, flag=flag)[0]

        else:
            x_uni_exp = []
            data_exp = []
            for i in range (0, self.sample_num):
                exp_data_path = self.traverse_path(i, exp_data_path)
                x_uni_exp_tem, sumN_uni_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:])
                x_uni_exp_tem = np.insert(x_uni_exp_tem, 0, 0.0)
                sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
                vol_uni = np.tile((1/6)*np.pi*x_uni_exp_tem**3, (self.num_t_steps-self.delta_t_start_step, 1)).T
                sumvol_uni_exp = sumN_uni_exp * vol_uni
                for flag, _ in self.delta_flag:
                    data_exp_tem = self.re_calc_distribution(x_uni_exp_tem, sum_uni=sumvol_uni_exp, flag=flag)[0] 
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
        return x_uni_exp, data_exp
    
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
        RT_space = {}
        
        for param, info in opt_params.items():
            bounds = info['bounds']
            log_scale = info.get('log_scale', False)
            
            if log_scale:
                RT_space[param] = tune.loguniform(10**bounds[0], 10**bounds[1])
            else:
                RT_space[param] = tune.uniform(bounds[0], bounds[1])

        total_cpus = ray.cluster_resources().get("CPU", 1)
        num_bundles = self.num_bundles
        if len(exp_data_paths) < num_bundles:
            num_bundles = len(exp_data_paths)
        cpus_per_bundle = int(max(1, total_cpus//num_bundles))
        
        @ray.remote(num_cpus=cpus_per_bundle)
        def run_tune_task(exp_data_path):
            algo = self.create_algo()
            if isinstance(exp_data_path, list):
                x_uni_exp = []
                data_exp = []
                for exp_data_path_tem in exp_data_path:
                    x_uni_exp_tem, data_exp_tem = self.get_all_exp_data(exp_data_path_tem)
                    x_uni_exp.append(x_uni_exp_tem)
                    data_exp.append(data_exp_tem)
                data_name = os.path.basename(exp_data_path[0])
            else:
                x_uni_exp, data_exp = self.get_all_exp_data(exp_data_path)
                data_name = os.path.basename(exp_data_path)
                
            if data_name.startswith("Sim_"):
                data_name = data_name[len("Sim_"):]
            if data_name.endswith(".xlsx"):
                data_name = data_name[:-len(".xlsx")]

            tuner = tune.Tuner(
                tune.with_resources(
                    lambda config: self.objective(config, x_uni_exp, data_exp), 
                    {"cpu": 2}
                ),
                param_space=RT_space,
                tune_config=tune.TuneConfig(
                    num_samples=self.n_iter,
                    search_alg=algo,
                    max_concurrent_trials=cpus_per_bundle
                ),
                run_config=train.RunConfig(
                    storage_path=self.tune_storage_path,
                    name = data_name, 
                    verbose = 2,
                    log_to_file=True,
                ),
            )
            return tuner.fit(), exp_data_path
        
        ## Add all data (groups for multi_flag=True case) to the task queue
        task_queue = ray.util.queue.Queue()
        for path in exp_data_paths:
            task_queue.put(path)
            
        results = []
        # 获取最优结果
        while not task_queue.empty():
            active_tasks = []
            for i in range(num_bundles):
                if not task_queue.empty():
                    exp_data_path = task_queue.get()
                    task = run_tune_task.remote(exp_data_path)
                    active_tasks.append(task)

            # 等待所有当前活跃的任务完成
            completed_tasks = ray.get(active_tasks)
            for task_result, path in completed_tasks:
                best_result = task_result.get_best_result(metric="loss", mode="min")
                results.append({
                    "opt_params": best_result.config,
                    "opt_score": best_result.metrics["loss"],
                    "file_path": path
                })
                
        return results
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
        if isinstance(exp_data_path, list):
            x_uni_exp = []
            data_exp = []
            for exp_data_path_tem in exp_data_path:
                x_uni_exp_tem, data_exp_tem = self.get_all_exp_data(exp_data_path_tem)
                x_uni_exp.append(x_uni_exp_tem)
                data_exp.append(data_exp_tem)
        else:
            x_uni_exp, data_exp = self.get_all_exp_data(exp_data_path)
            
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
            {"CPU": 1}
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
            "opt_parameters": opt_params,
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
        
    def create_algo(self):
        if self.method == 'HEBO': 
            return HEBOSearch(metric="loss", mode="min")
        elif self.method == 'GP': 
            return OptunaSearch(metric="loss", mode="min", sampler=GPSampler())
        elif self.method == 'TPS': 
            return OptunaSearch(metric="loss", mode="min", sampler=TPESampler())
        elif self.method == 'Cmaes':    
            return OptunaSearch(metric="loss", mode="min", sampler=CmaEsSampler())
        elif self.method == 'NSGA':    
            return OptunaSearch(metric="loss", mode="min", sampler=NSGAIIISampler())
        elif self.method == 'QMC':    
            return OptunaSearch(metric="loss", mode="min", sampler=QMCSampler())

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
                    
    def return_syth_beta(self,corr_agg):
        """
        Calculate and return a synthetic beta value.

        This method calculates the maximum value of `corr_agg`, takes its logarithm to base 10, 
        rounds it up to the nearest integer, and returns 10 raised to the power of this integer.
        
        Parameters
        ----------
        corr_agg : `array`
            The correction factors for aggregation.
    
        Returns
        -------
        float
            The synthetic beta value
        """
        max_val = max(corr_agg)
        power = np.log10(max_val)
        power = np.ceil(power)
        return 10**power
    
    def cost_fun(self, data_exp, data_mod, cost_func_type, flag):
        """
        Calculate the difference(cost) between experimental and model data.
        
        This method supports multiple cost function types including Mean Squared Error (MSE), 
        Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Kullback-Leibler (KL) divergence. 
        
        Parameters
        ----------
        data_exp : `array`
            The experimental data.
        data_mod : `array`
            The data generated by the simulation.
        
        Returns
        -------
        float
            The calculated cost based on the specified cost function type.
        """
        if cost_func_type == 'MSE':
            return mean_squared_error(data_mod, data_exp)
        elif cost_func_type == 'RMSE':
            mse = mean_squared_error(data_mod, data_exp)
            return np.sqrt(mse)
        elif cost_func_type == 'MAE':
            return mean_absolute_error(data_mod, data_exp)
        elif (flag == 'q3' or flag == 'Q3') and cost_func_type == 'KL':
            data_mod = np.where(data_mod <= 10e-20, 10e-20, data_mod)
            data_exp = np.where(data_exp <= 10e-20, 10e-20, data_exp)
            return entropy(data_mod, data_exp).mean()
        else:
            raise Exception("Current cost function type is not supported")
    #%% Data Process  
    ## Read the experimental data and re-interpolate the particle distribution 
    ## of the experimental data according to the simulation results.
    def read_exp(self, exp_data_path, t_vec):  
        """
        Reads experimental data from a specified path and processes it.
    
        Parameters
        ----------
        exp_data_path : `str`
            Path to the experimental data.
    
        Returns
        -------
        `tuple of array`
            - `x_uni_exp`: An array of unique particle sizes.
            - `sumN_uni_exp`: An array of sum of number concentrations for the unique particle sizes.
        """
        exp_data = write_read_exp(exp_data_path, read=True)
        df = exp_data.get_exp_data(t_vec)
        x_uni_exp = df.index.to_numpy()
        sumN_uni_exp = df.to_numpy()
        return x_uni_exp, sumN_uni_exp
    
    def function_noise(self, ori_data):
        """
        Adds noise to the original data based on the specified noise type.
        
        The method supports four types of noise: Gaussian ('Gaus'), Uniform ('Uni'), 
        Poisson ('Po'), and Multiplicative ('Mul'). The type of noise and its strength 
        are determined by the `noise_type` and `noise_strength` attributes, respectively.
        
        Parameters
        ----------
        ori_data : `array`
            The original data to which noise will be added.
        
        Returns
        -------
        `array`
            The noised data.
        
        Notes
        -----
        - Gaussian noise is added with mean 0 and standard deviation `noise_strength`.
        - Uniform noise is distributed over [-`noise_strength`/2, `noise_strength`/2).
        - Poisson noise uses `noise_strength` as lambda (expected value of interval).
        - Multiplicative noise is Gaussian with mean 1 and standard deviation `noise_strength`,
          and it is multiplied by the original data instead of being added.
        """
        rows, cols = ori_data.shape
        noise = np.zeros((rows, cols))
        if self.noise_type == 'Gaus':
            # The first parameter 0 represents the mean value of the noise, 
            # the second parameter is the standard deviation of the noise,
            for i in range(cols):
                noise[:, i] = np.random.normal(0, self.noise_strength, rows)              
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Uni':
            # Noises are uniformly distributed over the half-open interval [low, high)
            for i in range(cols):
                noise[:, i] = np.random.uniform(low=-self.noise_strength/2, high=self.noise_strength/2, size=rows)
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Po':
            for i in range(cols):
                noise[:, i] = np.random.poisson(self.noise_strength, rows)
            noised_data = ori_data + noise
            
        elif self.noise_type == 'Mul':
            for i in range(cols):
                noise[:, i] = np.random.normal(1, self.noise_strength, rows)
            noised_data = ori_data * noise
        # Cliping the data out of range  
        noised_data = np.clip(noised_data, 0, np.inf)
        return noised_data

    ## Kernel density estimation
    ## data_ori must be a quantity rather than a relative value!
    def KDE_fit(self, x_uni_ori, data_ori, bandwidth='scott', kernel_func='epanechnikov'):
        """
        Fit a Kernel Density Estimation (KDE) to the original data.
        
        This method applies KDE to the original data using specified bandwidth and kernel function. 
        
        Parameters
        ----------
        x_uni_ori : `array`
            The unique values of the data variable, must be a one-dimensional array.
        data_ori : `array`
            The original data corresponding to `x_uni_ori`.
        bandwidth : `str`, optional
            The bandwidth to use for the kernel density estimation. Defaults to 'scott'.
        kernel_func : `str`, optional
            The kernel function to use for KDE. Defaults to 'epanechnikov'.
        
        Returns
        -------
        sklearn.neighbors.kde.KernelDensity
            The fitted KDE model.
        """    
        # KernelDensity requires input to be a column vector
        # So x_uni_re must be reshaped
        x_uni_ori_re = x_uni_ori.reshape(-1, 1)
        # Avoid divide-by-zero warnings when calculating KDE
        data_ori_adjested = np.where(data_ori <= 0, 1e-20, data_ori)      
        kde = KernelDensity(kernel=kernel_func, bandwidth=bandwidth)
        kde.fit(x_uni_ori_re, sample_weight=data_ori_adjested)  
        return kde
    
    def KDE_score(self, kde, x_uni_new):
        """
        Evaluate the KDE model on new data points and normalize the results.
        
        Parameters
        ----------
        kde : sklearn.neighbors.kde.KernelDensity
            The fitted KDE model from :meth:`~.KDE_fit`.
        x_uni_new :`array`
            New unique data points where the KDE model is evaluated.
        
        Returns
        -------
        `array`
            The smoothed and normalized data based on the KDE model.
        """
        x_uni_new_re = x_uni_new.reshape(-1, 1) 
        data_smoothing = np.exp(kde.score_samples(x_uni_new_re))
        
        # Flatten a column vector into a one-dimensional array
        data_smoothing = data_smoothing.ravel()
        ## normalize the data for value range after smoothing
        # data_smoothing = data_smoothing/np.trapz(data_smoothing,x_uni_new)
        Q3 = self.calc_Q3(x_uni_new, data_smoothing)
        data_smoothing = data_smoothing / Q3[-1]
        return data_smoothing
    
    def traverse_path(self, label, path_ori):
        """
        Update the file path or list of paths based on the label.
        
        Parameters
        ----------
        label : `int`
            The label to update in the file path(s).
        path_ori : `str` or `list` of `str`
            The original file path or list of file paths to be updated.
        
        Returns
        -------
        `str` or `list` of `str`
            The updated file path.
        """
        def update_path(path, label):
            if label == 0:
                return path.replace(".xlsx", f"_{label}.xlsx")
            else:
                return path.replace(f"_{label-1}.xlsx", f"_{label}.xlsx")
    
        if isinstance(path_ori, list):
            return [update_path(path, label) for path in path_ori]
        else:
            return update_path(path_ori, label)        
    #%% PBE    
    def create_1d_pop(self, disc='geo'):
        """
        Instantiate one-dimensional populations for both non-magnetic (NM) and magnetic (M) particles.
        """
        self.p_NM = population(dim=1,disc=disc)
        self.p_M = population(dim=1,disc=disc)
            
    def calc_pop(self, pop, params=None, t_vec=None):
        """
        Configure and calculate the PBE.
        """
        self.set_pop_para(pop, params)
        
        if not self.calc_init_N:
            pop.full_init(calc_alpha=False)
        else:
            pop.calc_F_M()
            pop.calc_B_R()
            pop.calc_int_B_F()
        
        if t_vec is None: pop.solve_PBE(t_vec=self.t_vec)      
        else: pop.solve_PBE(t_vec=t_vec) 
        
    def set_init_pop_para(self,pop_params):
        
        self.set_pop_para(self.p, pop_params)
        
        if hasattr(self, 'p_NM'):
            self.set_pop_para(self.p_NM, pop_params)
        if hasattr(self, 'p_M'):
            self.set_pop_para(self.p_M, pop_params)
            ## P3 and P4 correspond to the breakage rate parameters of magnetic particles
            if 'BREAKRVAL' in pop_params and pop_params['BREAKRVAL'] == 4:
                self.p_M.pl_P1 = pop_params['pl_P3']
                self.p_M.pl_P2 = pop_params['pl_P4']
        
        self.set_init_pop_para_flag = True

    def set_pop_para(self, pop, params_in):
        params = params_in.copy()
        if params is None:
            return
        self.set_pop_attributes(pop, params)
        ## Because alpha_prim can be an arry, it needs to be handled separatedly 
        if self.dim == 1:
            if 'corr_agg' in params:
                params['CORR_BETA'] = self.return_syth_beta(params['corr_agg'])
                params['alpha_prim'] = params['corr_agg'] / params['CORR_BETA']
                del params["corr_agg"]
            if 'alpha_prim' in params:
                if params['alpha_prim'].ndim != 0:
                    pop.alpha_prim = params['alpha_prim'][0]
                else:
                    pop.alpha_prim = params['alpha_prim']
        elif self.dim == 2:
            if 'corr_agg' in params:
                params['CORR_BETA'] = self.return_syth_beta(params['corr_agg'])
                params['alpha_prim'] = params['corr_agg'] / params['CORR_BETA']
                del params["corr_agg"]
            if 'alpha_prim' in params:
                alpha_prim_value = params['alpha_prim']
                if pop is self.p:
                    alpha_prim_temp = np.zeros(4)
                    alpha_prim_temp[0] = alpha_prim_value[0]
                    alpha_prim_temp[1] = alpha_prim_temp[2] = alpha_prim_value[1]
                    alpha_prim_temp[3] = alpha_prim_value[2]
                    pop.alpha_prim = alpha_prim_temp
                elif pop is self.p_NM:
                    pop.alpha_prim = alpha_prim_value[0]
                elif pop is self.p_M:
                    pop.alpha_prim = alpha_prim_value[2]
            if 'pl_P3' and 'pl_P4' in params:
                if pop is self.p_M:
                    pop.pl_P1 = params['pl_P3']
                    pop.pl_P2 = params['pl_P4']
        if 'CORR_BETA' in params:
            pop.CORR_BETA = params['CORR_BETA']

    def set_pop_attributes(self, pop, params):
        for key, value in params.items():
            if key != 'alpha_prim':
                setattr(pop, key, value)
        
    def set_comp_para(self, USE_PSD, R01_0='r0_005', R03_0='r0_005', dist_path_NM=None, dist_path_M=None,
                      R_NM=2.9e-7, R_M=2.9e-7,R01_0_scl=1,R03_0_scl=1):
        """
        Set component parameters for non-magnetic and magnetic particle.
        
        Configures the particle size distribution (PSD) parameters from provided paths or sets
        default values.
        
        Parameters
        ----------
        R01_0 : `str`, optional
            Key for accessing the initial radius of NM particles from the PSD dictionary. Defaults to 'r0_005'.
        R03_0 : `str`, optional
            Key for accessing the initial radius of M particles from the PSD dictionary. Defaults to 'r0_005'.
        dist_path_NM : `str`, optional
            Path to the file containing the PSD dictionary for NM particles. If None, default radii are used.
        dist_path_M : `str`, optional
            Path to the file containing the PSD dictionary for M particles. If None, default radii are used.
        R_NM : `float`, optional
            Default radius for NM particles if `dist_path_NM` is not provided. Defaults to 2.9e-7.
        R_M : `float`, optional
            Default radius for M particles if `dist_path_M` is not provided. Defaults to 2.9e-7.
        """
        self.p.USE_PSD = USE_PSD
        if self.p.USE_PSD:
            if dist_path_NM is None or dist_path_M is None:
                raise Exception("Please give the full path to all PSD data!")
            psd_dict_NM = np.load(dist_path_NM,allow_pickle=True).item()
            psd_dict_M = np.load(dist_path_M,allow_pickle=True).item()
            self.p.DIST1 = dist_path_NM
            self.p.DIST3 = dist_path_M
            self.p.R01 = psd_dict_NM[R01_0] * R01_0_scl
            self.p.R03 = psd_dict_M[R03_0] * R03_0_scl
        else:
            self.p.R01 = R_NM * R01_0_scl
            self.p.R03 = R_M * R03_0_scl
        if self.dim > 1:
            ## Set particle parameter for 1D PBE
            self.p_NM.USE_PSD = self.p_M.USE_PSD = self.p.USE_PSD
            # parameter for particle component 1 - NM
            self.p_NM.R01 = self.p.R01
            self.p_NM.DIST1 = self.p.DIST1
            
            # parameter for particle component 2 - M
            self.p_M.R01 = self.p.R03
            self.p_M.DIST1 = self.p.DIST3
        self.set_comp_para_flag = True
        
    def calc_all_R(self):
        """
        Calculate the radius for particles in all PBEs.
        """
        self.p.calc_R()
        self.p_NM.calc_R()
        self.p_M.calc_R()
    
    ## only for 1D-pop, 
    def set_init_N(self, exp_data_paths, init_flag):
        """
        Initialize the number concentration N for 1D populations based on experimental data.
        
        Parameters
        ----------
        sample_num : `int`
            The number of sets of experimental data used for initialization.
        exp_data_paths : `list of str`
            Paths to the experimental data for initialization.
        init_flag : `str`
            The method to use for initialization: 'int' for interpolation or 'mean' for averaging
            the initial sets.
        """
        self.calc_all_R()
        self.set_init_N_1D(self.p_NM, exp_data_paths[1], init_flag)
        self.set_init_N_1D(self.p_M, exp_data_paths[2], init_flag)
        self.p.N = np.zeros((self.p.NS, self.p.NS, len(self.p.t_vec)))
        self.p.N[1:, 1, 0] = self.p_NM.N[1:, 0]
        self.p.N[1, 1:, 0] = self.p_M.N[1:, 0]
    
    def set_init_N_1D(self, pop, exp_data_path, init_flag):
        """
        Initialize the number concentration N for a single 1D population using experimental data.
        
        It processes the experimental data to align with the population's discrete size grid,
        using either interpolation or averaging based on the `init_flag`. Supports processing
        multiple samples of experimental data for averaging purposes.
        
        Parameters
        ----------
        pop : :class:`pop.population`
            The population instance (either NM or M) to initialize.
        sample_num : `int`
            Number of experimental data sets used for initialization.
        exp_data_path : `str`
            Path to the experimental data file.
        init_flag : `str`
            Initialization method: 'int' for interpolation, 'mean' for averaging.
        """
        x_uni = self.calc_x_uni(pop)
        if self.sample_num == 1:
            x_uni_exp, sumN_uni_init_sets = self.read_exp(exp_data_path, self.t_init[1:])
        else:
            exp_data_path=self.traverse_path(0, exp_data_path)
            x_uni_exp, sumN_uni_tem = self.read_exp(exp_data_path, self.t_init[1:])
            sumN_uni_all_samples = np.zeros((len(x_uni_exp), len(self.t_init[1:]), self.sample_num))
            sumN_uni_all_samples[:, :, 0] = sumN_uni_tem
            for i in range(1, self.sample_num):
                exp_data_path=self.traverse_path(i, exp_data_path)
                _, sumN_uni_tem = self.read_exp(exp_data_path, self.t_init[1:])
                sumN_uni_all_samples[:, :, i] = sumN_uni_tem
            sumN_uni_init_sets = sumN_uni_all_samples.mean(axis=2)
            
        sumN_uni_init = np.zeros(len(x_uni))
            
        if init_flag == 'int':
            for idx in range(len(x_uni_exp)):
                interp_time = interp1d(self.t_init[1:], sumN_uni_init_sets[idx, :], kind='linear', fill_value="extrapolate")
                sumN_uni_init[idx] = interp_time(0.0)

        elif init_flag == 'mean':
            sumN_uni_init = sumN_uni_init_sets.mean(axis=1)
                
        ## Remap q3 corresponding to the x value of the experimental data to x of the PBE
        # kde = self.KDE_fit(x_uni_exp, q3_init)
        # sumV_uni = self.KDE_score(kde, x_uni)
        # q3_init = sumV_uni / sumV_uni.sum()
        inter_grid = interp1d(x_uni_exp, sumN_uni_init, kind='linear', fill_value="extrapolate")
        sumN_uni_init = inter_grid(x_uni)
                
        pop.N = np.zeros((pop.NS, len(pop.t_vec)))
        ## Because sumN_uni_init[0] = 0
        pop.N[:, 0]= sumN_uni_init
        thr = 1e-5
        pop.N[pop.N < (thr * pop.N[1:, 0].max())]=0   
        pop.N[:, 0] *= pop.V_unit
        
    def calc_v_uni(self, pop):
        """
        Calculate unique volume values for a given population.
        """
        return np.setdiff1d(pop.V, [-1])*1e18
    
    def calc_x_uni(self, pop):
        """
        Calculate unique particle diameters from volume values for a given population.
        """
        v_uni = self.calc_v_uni(pop)
        # Because the length unit in the experimental data is millimeters 
        # and in the simulation it is meters, so it needs to be converted 
        # before use.
        # x_uni = np.zeros(len(v_uni))
        x_uni = (6*v_uni/np.pi)**(1/3)
        return x_uni
        
    def re_calc_distribution(self, x_uni, q3=None, sum_uni=None, flag='all'):
        """
        Recalculate distribution metrics for a given population and distribution data.
        
        Can operate on either q3 or sum_uni distribution data to calculate Q3, q3, and particle
        diameters corresponding to specific percentiles (x_10, x_50, x_90).
        
        Parameters
        ----------
        x_uni : `array`
            Unique particle diameters.
        q3 : `array`, optional
            q3 distribution data.
        sum_uni : `array`, optional
            sum_uni distribution data.
        flag : `str`, optional
            Specifies which metrics to return. Defaults to 'all', can be a comma-separated list
            of 'q3', 'Q3', 'x_10', 'x_50', 'x_90'.
        
        Returns
        -------
        `tuple`
            Selected distribution metrics based on the `flag`. Can include q3, Q3, x_10, x_50,
            and x_90 values.
        """
        if q3 is not None:
            q3_new = q3
            Q3_new = np.apply_along_axis(lambda q3_slice: 
                                     self.calc_Q3(x_uni, q3=q3_slice), 0, q3)

        else:
            Q3_new = np.apply_along_axis(lambda sum_uni_slice: 
                                     self.calc_Q3(x_uni, sum_uni=sum_uni_slice), 0, sum_uni)
            q3_new = np.apply_along_axis(lambda Q3_slice: 
                                          self.calc_q3(Q3_slice, x_uni), 0, Q3_new)
        
        dim = q3_new.shape[1]
        x_10_new = np.zeros(dim)
        x_50_new = np.zeros(dim)
        x_90_new = np.zeros(dim)
        for idx in range(dim):
            x_10_new[idx] = np.interp(0.1, Q3_new[:, idx], x_uni)
            x_50_new[idx] = np.interp(0.5, Q3_new[:, idx], x_uni)
            x_90_new[idx] = np.interp(0.9, Q3_new[:, idx], x_uni)
        outputs = {
        'q3': q3_new,
        'Q3': Q3_new,
        'x_10': x_10_new,
        'x_50': x_50_new,
        'x_90': x_90_new,
        }
        
        if flag == 'all':
            return outputs.values()
        else:
            flags = flag.split(',')
            return tuple(outputs[f.strip()] for f in flags if f.strip() in outputs)
        
    def calc_Q3(self, x_uni, q3=None, sum_uni=None):
        """
        Calculate the cumulative distribution Q3 from q3 or sum_uni distribution data.
        """
        Q3 = np.zeros_like(q3) if q3 is not None else np.zeros_like(sum_uni)
        if q3 is None:
            Q3 = np.cumsum(sum_uni)/sum_uni.sum()
        else:
            for i in range(1, len(Q3)):
                    # Q3[i] = np.trapz(q3[:i+1], x_uni[:i+1])
                    ## Euler backward
                    Q3[i] = Q3[i-1] + q3[i] * (x_uni[i] - x_uni[i-1])
        return Q3
    def calc_sum_uni(self, Q3, sum_total):
        """
        Calculate the sum_uni distribution from the Q3 cumulative distribution and total sum.
        """
        sum_uni = np.zeros_like(Q3)
        # sum_uni[0] = sum_total * Q3[0]
        for i in range(1, len(Q3)):
            sum_uni[i] = sum_total * max((Q3[i] -Q3[i-1] ), 0)
        return sum_uni
    def calc_q3(self, Q3, x_uni):
        """
        Calculate the q3 distribution from the Q3 cumulative distribution.
        """
        q3 = np.zeros_like(Q3)
        q3[1:] = np.diff(Q3) / np.diff(x_uni)
        return q3
    
# @ray.remote
# class TuneWorker:
#     def __init__(self, sample_num, pg, RT_space, algo, tune_storage_path, n_iter, opt_algo_instance):
#         self.sample_num = sample_num
#         self.pg = pg
#         self.RT_space = RT_space
#         self.algo = algo
#         self.tune_storage_path = tune_storage_path
#         self.n_iter = n_iter
#         self.opt_algo_instance = opt_algo_instance

#     def run_tune_task(self, exp_data_path):
#         def objective(config, exp_data_path):
#             # Special handling for corr_agg based on dimension
#             if 'corr_agg_0' in config:
#                 transformed_params = self.opt_algo_instance.array_dict_transform(config)
            
#             checkpoint_config = None
#             loss = self.opt_algo_instance.calc_delta_agg(transformed_params, sample_num=self.sample_num, exp_data_path=exp_data_path)
#             train.report({"loss": loss}, checkpoint=checkpoint_config)
        
#         if isinstance(exp_data_path, list):
#             data_name = os.path.basename(exp_data_path[0])
#         else:
#             data_name = os.path.basename(exp_data_path)
            
#         if data_name.startswith("Sim_"):
#             data_name = data_name[len("Sim_"):]
#         if data_name.endswith(".xlsx"):
#             data_name = data_name[:-len(".xlsx")]

#         tuner = tune.Tuner(
#             tune.with_resources(
#                 lambda config: objective(config, exp_data_path), 
#                 resources = self.pg
#             ),
#             param_space=self.RT_space,
#             tune_config=tune.TuneConfig(
#                 num_samples=self.n_iter,
#                 search_alg=self.algo,
#             ),
#             run_config=train.RunConfig(
#                 storage_path=self.tune_storage_path,
#                 name = data_name
#             ),
#         )
#         return tuner.fit(), exp_data_path