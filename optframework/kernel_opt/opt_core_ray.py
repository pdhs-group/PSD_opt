# -*- coding: utf-8 -*-
"""
Calculate the difference between the PSD of the simulation results and the experimental data.
Minimize the difference by optimization algorithm to obtain the kernel of PBE.
"""
import os
import time
from ray import tune
from .opt_core import OptCore

class OptCoreRay(OptCore, tune.Trainable):
    """
    An extension of the OptCore class that integrates with Ray Tune for optimization.

    This class inherits from both OptCore and tune.Trainable, making it a Ray Tune actor 
    for managing optimization iterations. It allows for the running of the PBE calculation 
    with Ray Tune's iterative optimization framework, and implements the necessary methods 
    to handle Ray Tune's checkpointing, configuration resetting, and optimization step control.

    Attributes
    ----------
    reuse_num : int
        A counter tracking the number of times the Actor has been reused.
    actor_wait : bool
        A flag indicating whether the actor should wait between iterations if the execution 
        time is too short.
    """
    def __init__(self, *args, **kwargs):
        # Initialize tune.Trainable to prepare the class as a Ray Tune actor
        tune.Trainable.__init__(self, *args, **kwargs)
        
    def setup(self, config, core_params, pop_params, data_path, 
              exp_data_paths, x_uni_exp, data_exp, known_params, exp_case):
        """
        Set up the environment for the optimization task.
        
        This method initializes the core attributes and PBE solver for optimization. It also 
        stores the experimental data and known parameters for use during the optimization process.
        
        Parameters
        ----------
        config : dict
            The configuration dictionary used by Ray Tune to pass in optimization parameters.
        core_params : dict
            The core parameters used for setting up the OptCore.
        pop_params : dict
            The parameters for the PBE.
        data_path : str
            The path to the experimental data.
        x_uni_exp : array-like
            The unique particle diameters in experimental data.
        data_exp : array-like
            The experimental PSD data.
        known_params : dict
            A dictionary of known parameters that will be used to override any conflicting 
            optimization parameters.
        """
        # Initialize OptCore attributes and PBE setup
        self.init_attr(core_params)
        self.init_pbe(pop_params, data_path)
        
        # Initialize the number concentration N if required
        if self.calc_init_N:
            self.set_init_N(exp_data_paths, init_flag='mean')
        else:
            self.init_N_NM = None
            self.init_N_M = None
            self.init_N_2D = None
            self.init_N = None
            
        # Store experimental data and known parameters
        self.known_params = known_params
        self.x_uni_exp = x_uni_exp
        self.data_exp = data_exp
        self.exp_data_paths = exp_data_paths
        self.exp_case = exp_case
        self.reuse_num=0
        self.actor_wait=False
    
    def step(self):
        """
        Perform one step of optimization.

        This method is called by Ray Tune to execute a single iteration of the optimization. 
        It calculates the loss (delta) between the experimental and simulated data based on 
        the current configuration parameters.

        Returns
        -------
        dict
            A dictionary containing the loss (delta) and the reuse count for the current Actor.
        """
        start_time = time.time()
        # Transform the input parameters if they include corr_agg for dimensional handling
        if 'corr_agg_0' in self.config:
            transformed_params = self.array_dict_transform(self.config)
        else:
            transformed_params = self.config
            
        if not self.exp_case:
            # Apply known parameters to override any conflicting optimization parameters
            if self.known_params is not None:
                for key, value in self.known_params.items():
                    if key in transformed_params:
                        print(f"Warning: Known parameter '{key}' are set for optimization.")
                    transformed_params[key] = value
                    
            # print(f"The paramters actually entered calc_delta are {transformed_params}")
            # Calculate the loss (delta) using the transformed parameters
            loss = self.calc_delta(transformed_params, self.x_uni_exp, self.data_exp)
        else:
            losses = []
            for i in range(len(self.known_params)):
                known_i = self.known_params[i]
                for key, value in known_i.items():
                    transformed_params[key] = value
                x_i = self.x_uni_exp[i]
                data_i = self.data_exp[i]
                loss_i = self.calc_delta(transformed_params, x_i, data_i)
                losses.append(loss_i)
            loss = sum(losses) / len(losses)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # If execution time is too short, introduce a delay to simulate actor waiting
        # This is done because Ray Tune introduces a delay when managing and communicating between actors.
        # If the iteration is too fast, it may cause unknown errors where actors are repeatedly created
        # and discarded, leading to a large number of ineffective operations and impacting system performance.
        if execution_time < 2 and self.actor_wait:
            time.sleep(2 - execution_time)
        return {"loss": loss, "reuse_num": self.reuse_num, "exp_paths": self.exp_data_paths}
    def save_checkpoint(self, checkpoint_dir):
        """
        Save the checkpoint. This method is required by Ray Tune but is not used in this implementation.
        """
        pass
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint. This method is required by Ray Tune but is not used in this implementation.
        """
        pass
    def reset_config(self, new_config):
        """
        Reset the Actor for reuse in subsequent iterations.

        This method is called when Ray Tune needs to reset the Actor between iterations.
        The reuse counter is incremented each time this method is called.

        Returns
        -------
        bool
            Always returns True, indicating successful configuration reset.
        """
        self.reuse_num += 1
        return True
        


