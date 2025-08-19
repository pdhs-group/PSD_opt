(quickstart)= 
# Quick Start

The fastest way to get started is by adapting the example scripts provided in the `optframework/examples` folder.  
Each script demonstrates the basic usage of one or more core classes, and you can quickly modify them to fit your own needs.

All classes in this library are built following a consistent design philosophy, so their usage is very similar.  
Here we use `DPBESolver` (see `optframework/examples/simple_dpbe.py`) as an example.

### 1. Importing the Class

```python
from optframework.dpbe import DPBESolver
````

### 2. Instantiating the Solver

Each instance represents an independent PBE solver with its own set of parameters.
During initialization, only the PBE dimension `dim` must be specified manually.
All other parameters have default values but can be customized in three ways:

1. **Pass parameters directly to `__init__()`**
2. **Use a config file (recommended)**

   * A config file is simply a Python script that defines a dictionary, where each key corresponds to a class attribute.
   * The config is loaded at the end of `__init__()`.
   * If no config path is provided, the solver attempts to load `config/PBM_config.py` from the current working directory.
3. **Modify attributes manually after initialization** (not recommended)

   * Since initialization may compute intermediate variables based on attributes, manual changes can desynchronize values unless recalculations are also done.

**Priority:** `manual modification (3) > config file (2) > init arguments (1)`

Example:

```python
p = DPBESolver(dim=dim, NS=10, S=1.2) 
# In PBM_config.py, "S": 1.3
p.S = 1.4
```

In this case:

* `S = 1.2` (from init args) is first applied.
* It is overridden by the config file value `S = 1.3`.
* Finally, it is manually updated to `S = 1.4`.

### 3. Running the Solver

By default, the solver instance contains only parameters and basic initialization methods.
The actual computational methods are grouped in namespaces:

* `p.core` → methods for matrix construction and PBE solving
* `p.post` → post-processing methods
* `p.visualization` → visualization methods

All intermediate results are still stored in the base solver instance `p`, making them easy to access.

Example workflow:

```python
p.core.full_init(calc_alpha=False)
p.core.solve_PBE()
N = p.N
```

Now, `N` contains the computed particle number density distribution on the discretized grid.

---

### 4. Using Other Classes

Other solvers (e.g., `ExtruderPBESolver`, `PBMSolver`, `MCPBESolver`) follow the same usage pattern.
Please refer to the corresponding example scripts in the `examples` folder for details.
