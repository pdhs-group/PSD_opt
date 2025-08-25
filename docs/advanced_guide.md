# Advanced Guide: Adding a Custom Kernel Model

This guide explains how to add a **new kernel model** for existing particle kinetics (**agglomeration** or **breakage**).  
The changes involve three parts:

1. **Class layer:** define required variables as class attributes.  
2. **JIT functions:** implement the model’s math and extend function interfaces.  
3. **Runtime config:** provide the new parameters via config data.

> We use **`DPBESolver`** as the working example. (Integrations for `PBMSolver` and `MCPBESolver` are in progress and structurally more involved.)

## 1. Class Layer: Define Parameters & Defaults

You can define defaults in two places:

- **Global to all solvers:** in `base_solver.py` → `_init_base_parameters`  
  - Pros: every solver that inherits `BaseSolver` gets the attribute.  
  - Good for parameters shared by multiple solvers (e.g., generic kernel switches, exponents).

- **Specific to DPBESolver:** in `optframework/dpbe.py` → `__init__`  
  - Pros: keeps solver-specific parameters local to DPBE only.

If needed, also update any **matrix/build** routines to precompute constants derived from your new parameters.

**Example (pseudo-code):**
```python
# base_solver.py
class BaseSolver()
    def _init_base_parameters(self):
        # existing common params ...
        self.kernel_name = getattr(self, "kernel_name", "my_new_kernel")
        self.kappa = getattr(self, "kappa", 1.0)        # float default
        self.alpha = getattr(self, "alpha", 0)          # int default
        # arrays should be C-contiguous later when passed into JIT

# optframework/dpbe.py
class DPBESolver(BaseSolver):
    def __init__(self, **kwargs):
        # existing specific params ...
        self.kernel_name = getattr(self, "kernel_name", "my_new_kernel")
        self.kappa = getattr(self, "kappa", 1.0)        # float default
        self.alpha = getattr(self, "alpha", 0)          # int default
        # arrays should be C-contiguous later when passed into JIT
````

## 2. JIT Layer: Add the Mathematical Model & Wire Parameters

Navigate to:

* **Agglomeration kernels:** `optframework/utils/func/jit_kernel_agg.py`
* **Breakage kernels:** `optframework/utils/func/jit_kernel_break.py`

There are three families of functions to consider:

### A. Agglomeration rate

* Add your model to **`calc_beta(...)`** in `jit_kernel_agg.py`.
* Expose your new parameters as **additional inputs** to `calc_beta` and to the wrapper JIT functions (`calc_F_M` family) that call it.
* Also update the wrapper that collects parameters from the solver instance and forwards them to the JIT call.

**Example (simplified):**

```python
# jit_kernel_agg.py
@njit
def calc_beta(COLEVAL, CORR_BETA, G, R, idx1, idx2, kappa, alpha):
    ...
    if COLEVAL == 0:
        # existing model A
        ...
    elif COLEVAL == 1:
        # existing model B
        ...
    elif COLEVAL == 10:  # your new model id
        # your formula using kappa, alpha, etc.
        return kappa * (v_i**0.5 + v_j**0.5) + alpha
    else:
        ...

@njit
def calc_F_M_2D(..., kappa, alpha):
    ...
    # pass through to calc_beta and use parameters consistently
    beta = calc_beta(COLEVAL, CORR_BETA, G, R, (a, b), (i, j), kappa, alpha)
    ...

def calc_F_M_2D(solver):
    return calc_F_M_2D_jit(
        # existing parameters or matrix...
        float(kappa),
        int(alpha),
        )
```

**Wrapper parameter formatting rules:**

* **Integers:** `int(...)`
* **Floats:** `float(...)`
* **Arrays:** `np.ascontiguousarray(...)` before passing into JIT



### B. Breakage rate

* For breakage **rate** models, add to:

  * `calc_break_rate_1d`, `calc_break_rate_2d`, and/or `calc_break_rate_2d_flat` (Monte Carlo usage) in `jit_kernel_break.py`.
* As with agglomeration, extend **function signatures** and **propagate parameters** through the JIT wrappers (`calc_B_R` family).

### C. Breakage fragment distribution

This part is **more involved**:

* The model goes into **`breakage_func_1d` / `breakage_func_2d`**.
* New parameters must be threaded through multiple places in `jit_kernel_break.py`.
  The fastest way: **search for** `v,q,BREAKFVAL` and **extend the signature** everywhere to `v, q, BREAKFVAL, <your_params>`.
* Finally, add matching formatted inputs in **`calc_int_B_F_2D_GL`** , ensuring ints/floats/arrays are cast/contiguous as above.


## 3. Runtime: Add Parameters in Config

Add your new parameters and switches to the config file used to instantiate the solver.
Because configs are **Python dictionaries**, you can put scalars, lists, or `numpy` arrays directly.

**Example `dpbe_config.py`:**

```python
config = {
    ...
    # choose your kernel variant
    "kernel_name": "my_new_kernel",
    "model_flag": 10,

    # your new parameters
    "kappa": 0.95,          # float
    "alpha": 2,             # int
    "shape_arr": np.array([1.0, 0.5], dtype=np.float64),  # array
}
```

> The loader (`_load_attributes`) in `BaseSolver` maps these keys to instance attributes automatically.
> If you also set defaults in `_init_base_parameters`, the config values will **override** those defaults.


## 4. Validation


Run **`optframework/examples/simple_dpbe.py`** with your new model.


