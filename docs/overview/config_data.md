# Config Data

**Config data** provides initialization parameters to solvers and optimizers.  
It is written as a **Python dictionary**, not JSON, in order to:  

- Allow direct use of `numpy` arrays  
- Enable preprocessing and scripting flexibility  

During class instantiation, the config data is automatically loaded.  
If no config path is provided, the program attempts to read a default file from:

```
config/*_config.py
```

in the current working directory.


## 1. General Solvers

For all standard solvers (`DPBESolver`, `PBMSolver`, `MCPBESolver`),  
the config data is a **flat dictionary** (single level).  

- Each key corresponds to an attribute of the base solver class.  
- On initialization, all keys are directly mapped to class attributes.  

This ensures simple and transparent parameter control.


## 2. ExtruderPBESolver

Because `ExtruderPBESolver` involves **multiple computational regions**,  
its config structure is slightly more complex.  

- The top-level dictionary contains two keys:
  - **`geom_params`** → parameters for the extruder geometry  
  - **`pbe_params`** → PBE-related parameters for each region  

- Inside **`pbe_params`**:
  - **`global`** → shared parameters across all regional DPBE solvers (e.g., grid definition)  
  - **`local_0`, `local_1`, ...** → region-specific parameters, where `local_i` defines the parameters for the *i-th region*  

This design allows each region to have its own DPBE configuration while sharing common structural definitions.


## 3. Optimization Framework

For the optimization framework (`OptBase`), the config data has a **multi-layered structure**.  

- **`multi_flag` / `single_case`**  
  - Control whether `OptBase` uses `OptCore` or `OptCoreMulti`.  

- **`algo_params`**  
  - Defines optimization algorithm parameters (population size, iteration limits, etc.).  

- **`pop_params`**  
  - Provides control parameters for the embedded DPBE solver.  

- **`opt_params`**  
  - Specifies kernel parameters to be optimized, along with their search ranges.  

The nested dictionaries (`algo_params`, `pop_params`, `opt_params`) are passed directly to attributes of `OptCore` or `OptCoreMulti`, depending on the optimization mode.

```{toctree}
:maxdepth: 1

```
