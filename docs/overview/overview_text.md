# Overview

## 1. Model

The **{doc}`Model <model>`** introduces the implemented Population Balance Equation (PBE) solvers, including:

- **DPBESolver** (grid-based, deterministic discretization)  
- **PBMSolver** (moment-based methods: QMOM, GQMOM, CQMOM)  
- **MCPBESolver** (Monte Carlo stochastic simulation)  
- **ExtruderPBESolver** (multi-region extension of DPBE)  

It also explains the physical basis of PBE, the algorithms behind each solver, and provides a performance and stability comparison across methods.


## 2. Class Structure

The **{doc}`Class Structure <class_structure>`** documentation describes:

- The design principles of all solver classes  
- Their modular organization (core, postprocessing, visualization)  
- Key attributes and methods essential for extending or customizing the solvers  

This section helps users understand the internal logic of the library and how the main solvers are constructed.


## 3. Config Data

The **{doc}`Config Data <config_data>`** section explains:

- The role of configuration files in setting solver parameters  
- Why configs are used (to ensure reproducibility and flexible parameter control)  
- How to override defaults using `__init__`, config files, or manual attribute changes  
- Priority rules among these methods  

This provides a unified way to manage simulation parameters consistently across different solvers.

## 4. Optimization Framework

The **{doc}Optimization Framework <optimization_framework>** documentation introduces:

- The structure of the optimization modules (`kernel_opt`, `kernel_opt_extruder`)  
- How Ray Tune is used to enable scalable, parallel optimization  
- Methods to integrate solver runs into optimization loops  
- Guidelines on defining objective functions and interpreting optimization results  

This section shows how to couple the PBE solvers with large-scale parameter studies and optimization tasks.


```{toctree}
:maxdepth: 1

```
