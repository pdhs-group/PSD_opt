# Class Structure

## Overview

In this project, each solver is implemented as a **dedicated class**.  
All parameters related to material properties, particle kinetics, and simulation control are defined as **class attributes**.  
Meanwhile, operations such as grid construction, data processing, matrix computations, and solving the PBE are organized into **class methods**.

The overall class design follows two main principles:

1. **Base class with method wrappers**  
   - Each solver is structured around its own **base class**, which stores solver-specific attributes, intermediate variables, matrices, and essential methods.  
   - Additional functionality is grouped into **method classes**, which are automatically instantiated and linked to the base class as attributes.  
   - From the user’s perspective, only the base class needs to be instantiated and used directly.  

2. **Shared base class for common functionality**  
   - Since many solvers share common parameters and methods, all solver base classes now **inherit from a global `BaseSolver` class**.  
   - `BaseSolver` provides reusable methods such as:  
     - `_init_base_parameters` → defines common parameters (e.g., kernel parameters)  
     - `_load_attributes` → loads attributes from config files and assigns them to the solver instance  
   - This inheritance ensures consistency across solvers and avoids redundant code.  

3. **Parameter passing via config data**  
   - Parameters are mainly provided through **config data**, a Python file containing a dictionary where each key corresponds to a class attribute.  
   - Using Python rather than JSON enables direct definition of `numpy` arrays and allows preprocessing steps within the config.  

---

## Solver Class Structures

### 1. DPBESolver

- **Method classes:**
  - `DPBECore` → accessible as `*.core`  
    - Contains core DPBE-related matrix computations and the final solver.  
  - `DPBEPost` → accessible as `*.post`  
    - Provides post-processing methods for simulation results.  
  - `DPBEVisual` → accessible as `*.visualization`  
    - Provides visualization methods for particle size distribution (PSD).  

---

### 2. ExtruderPBESolver

- **Structure:**  
  - This solver only has a **base class**.  
  - Internally, it instantiates a `DPBESolver` and delegates most computations to it.  
  - As a result, it shares many methods with DPBE.  
- **Unique features:**  
  - Construction of **multi-region computational matrices**.  
  - Solution of **multi-region PBEs** with inter-region convection terms.  

---

### 3. PBMSolver

- **Method classes:**
  - `PBMCore` → accessible as `*.core`  
    - Responsible for generating initial moments and solving the moment equations.  
  - `PBMPost` → accessible as `*.post`  
    - Provides methods for reconstructing moments from QMOM-calculated nodes and weights, as well as post-processing routines.  
  - `PBMQuickTest` → accessible as `*.quick_test`  
    - Contains diagnostic tools and test routines for QMOM algorithms, primarily used for debugging.  

---

### 4. MCPBESolver

- **Structure:**  
  - Currently implemented as a **single monolithic class**.  
  - Unlike other solvers, its structure has not yet been refactored into base and method classes, as development is ongoing.  

---

## Optimization Framework Classes

### 5. OptBase

The optimization framework follows a slightly different design philosophy compared to solvers, but the outer interface is still based on a **base class** (`OptBase`) that users instantiate directly.

- **Linked components:**
  - `OptBaseRay` → accessible as `*.base_ray`  
    - Provides methods to initialize and manage the **Ray Tune** framework.  
  - `OptCore` and `OptCoreMulti` → accessible as `*.core`  
    - Define the optimization **cost functions**.  
    - `OptCoreMulti` extends `OptCore` to enable **1D+2D data optimization** (see related publication).  

- **Subcomponents of `OptCore`:**
  - `OptData` → accessible as `*.core.opt_data`  
    - Contains methods for data handling and preprocessing.  
  - `OptPBE` → accessible as `*.core.opt_pbe`  
    - Provides methods for running PBE-based computations (currently implemented using `DPBESolver`).  

- **Actor classes for Ray Tune:**
  - `OptCoreRay` and `OptCoreMultiRay`  
    - Inherit from `OptCore` and `OptCoreMulti` respectively.  
    - Used internally when running **Ray Tune in Actor mode**.  
    - Define the behavior of each optimization iteration.  
    - Not intended for direct external use.  

```{toctree}
:maxdepth: 1

```
