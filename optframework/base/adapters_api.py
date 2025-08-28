# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:38:37 2025

@author: px2030
"""

from typing import Any, Protocol, runtime_checkable, Dict, Type
import numpy as np
from .dpbe_adapter import DPBEAdapter

@runtime_checkable
class SolverProtocol(Protocol):
    """
    Protocol defining the essential interface for solver adapters.
    
    Specifies the required methods that all solver adapters must implement
    to ensure consistent interaction with optimization frameworks.
    """
    
    def set_comp_para(self, data_path: str) -> None: 
        """
        Set component parameters that cannot be easily passed through config files.
        
        This method is called once after solver instantiation and initial parameter
        passing. It handles parameters that require runtime information, such as
        PSD file paths with dynamic base paths. Other one-time initialization
        operations can also be placed here.
        
        Parameters
        ----------
        data_path : str
            Base path for data files, typically obtained dynamically at runtime
            
        Notes
        -----
        - Called exactly once after solver instantiation
        - Can include other one-time initialization operations
        - If no operations needed, implement as `return None`
        """
        ...
        
    def reset_params(self) -> None: 
        """
        Reset and update solver parameters that depend on other attributes.
        
        This method is called multiple times: first after set_comp_para(), then
        after each update to solver attributes. It ensures that dependent
        parameters are properly synchronized with their controlling attributes.
        
        Notes
        -----
        - Called first after set_comp_para()
        - Called after each attribute update
        - Should update dynamically dependent parameters
        - If no updates needed, implement as `return None`
        """
        ...
        
    def calc_matrix(self, init_N) -> None: 
        """
        Calculate parameter matrices required by the solver.
        
        This method is called once before each solving session to compute
        necessary matrices based on current solver state and initial conditions.
        
        Parameters
        ----------
        init_N : array-like or None
            Initial number concentrations or other initial conditions.
            
            - If None: Solver should obtain initial conditions internally 
              (e.g., from files or default values)
            - If provided: Use these runtime-calculated initial conditions
        
        Notes
        -----
        - Called once before each solve session
        - If no calculations needed, implement as `return None`
        """
        ...
        
    def solve(self, t_vec) -> None: 
        """
        Execute the solver to compute results over specified time vector.
        
        This method is called once or multiple times per optimizer iteration,
        depending on Ray Actor configuration. After this method returns, the
        solver must have complete calculation results available.
        
        Parameters
        ----------
        t_vec : array-like
            Time vector specifying output points for the simulation
            
        Notes
        -----
        - Called 1+ times per optimizer iteration
        - Results should be accessible after method completion
        """
        ...
        
    def get_all_data(self, exp_data_path) -> tuple[np.ndarray, np.ndarray]: 
        """
        Read and return all experimental data for comparison.

        This method is called once before optimization begins. It should return
        experimental particle size/volume data and corresponding PSD data
        (qx/Qx etc.) for 1D or 2D systems. Handles both single file paths
        and lists of file paths for multi-algorithm or repeated measurements.
        
        Parameters
        ----------
        exp_data_path : str or list of str
            Path to experimental data file(s). Can be single path or list
            depending on multi-algorithm usage and repeated measurements
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First array: particle diameter/volume data
            Second array: corresponding PSD data (qx/Qx etc.)
            
        Notes
        -----
        - Called once before optimization starts
        - Should handle both single files and file lists
        - Returned data must be compatible with calc_delta_pop()
        """
        ...
        
    def calc_delta_pop(self, x_uni_exp, data_exp) -> float: 
        """
        Calculate population difference metric between simulation and experiment.
        
        This method computes the discrepancy between solver results and
        experimental data for one optimizer iteration. The specific calculation
        method should be customized based on the application requirements.
        
        Parameters
        ----------
        x_uni_exp : array-like
            Experimental particle diameter/volume data
        data_exp : array-like
            Experimental PSD data (qx/Qx etc.)
            
        Returns
        -------
        float
            Overall average difference metric representing simulation-experiment
            discrepancy. Lower values indicate better agreement.
            
        Notes
        -----
        - Called once per optimizer iteration
        - Return value used as optimization objective
        - Must be compatible with data from get_all_data()
        """
        ...
        
    def close(self) -> None: 
        """
        Close solver and release memory resources.
        
        This method is called after the solver has been used for a certain
        number of iterations to clean up resources and free memory.
        
        Notes
        -----
        - Called after predetermined number of solver uses
        - Should release all allocated memory and resources
        - No return value required
        """
        ...

def validate_solver(obj: Any) -> None:
    """
    Validate that an object implements the SolverProtocol interface.
    
    Checks if the object has all required methods and they are callable.
    
    Parameters
    ----------
    obj : Any
        Object to validate against SolverProtocol
        
    Raises
    ------
    TypeError
        If object lacks any required callable method
    """
    essential_methods = [
        "set_comp_para", 
        "reset_params", 
        "calc_matrix", 
        "solve", 
        "get_all_data", 
        "calc_delta_pop", 
        "close"
    ]
    for method_name in essential_methods:
        if not hasattr(obj, method_name) or not callable(getattr(obj, method_name)):
            raise TypeError(f"Adapter lacks callable '{method_name}()'")

# Registry of available solver adapters
REGISTRY: Dict[str, Type] = {
    "dpbe": DPBEAdapter,
    # "other": OtherAdapter,
}

def make_solver(name: str, **params) -> SolverProtocol:
    """
    Factory function to create solver adapter instances.
    
    Creates and validates solver adapters based on the specified type.
    
    Parameters
    ----------
    name : str
        Name of the solver type (must be in REGISTRY)
    \*\*params
        Keyword arguments passed to the adapter constructor
        
    Returns
    -------
    SolverProtocol
        Validated solver adapter instance
        
    Raises
    ------
    ValueError
        If solver name is not found in REGISTRY
    TypeError
        If created adapter doesn't implement SolverProtocol
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown solver '{name}'. Available: {list(REGISTRY)}")
    adapter = REGISTRY[name](**params)
    validate_solver(adapter)
    return adapter