# Optimization Framework

## 1. Motivation

In population balance modeling, the **kernel parameters** (parameters within agglomeration and breakage models) determine whether the PBE accurately reflects the physical environment and material properties.  

- Many kernel parameters are **not standard physical quantities**, and some lack direct physical meaning.  
- Obtaining precise kernel parameters through experiments or theoretical derivations is often extremely difficult.  

A common approach is therefore to use **experimental PSD (Particle Size Distribution) data** and determine the kernel parameters via **inverse modeling / optimization**.  

In this framework, the target data (`data_exp`) is usually the PSD measured over time in experiments, while the model results (`data_mod`) are obtained from solving the **dPBE**.  

## 2. Basic Workflow

1. **Provide experimental data (`data_exp`)**  
   - Typically PSD at multiple time points.  
   - Can be density distributions $q$ or cumulative distributions $Q$.  
   - The coordinate system (particle volume grid) must be consistent.  

2. **Initialize optimizer**  
   - Random kernel parameters (within predefined ranges) are fed into the dPBE solver.  
   - The solver produces simulated PSD (`data_mod`).  
   - ⚠️ Recommendation:  
     - Either construct the dPBE grid identical to the experimental grid, or  
     - Interpolate experimental data onto the solver grid in preprocessing.  
   - Otherwise, set `smoothing = True` in config data. This applies **KDE (Kernel Density Estimation)** to map `data_mod` onto the experimental grid.  
     - Note: KDE may underestimate peaks when the grid is very dense, so direct alignment is preferred.  

3. **Compute cost function**  
   - Defined as the “error” between `data_mod` and `data_exp`.  
   - Configurable in `config data`. Example:  
     ```python
     'delta_flag': [('qx', 'MSE')]
     ```
     means: convert both datasets to density distribution $q_x$, then compute Mean Squared Error (MSE).  

4. **Update kernel parameters**  
   - After one iteration, the error \$delta$ is recorded.  
   - The optimizer updates its predictive model and proposes a new candidate parameter set.  

5. **Repeat iterations**  
   - Steps 3–4 are repeated until the maximum number of iterations is reached.  
   - The optimizer selects the parameter set with the smallest error as the optimal kernel parameters.  

## 3. Ray Tune Framework

This project uses **Ray Tune** to accelerate optimization with distributed parallelization.  

- Each dPBE simulation runs in parallel threads.  
- Results are exchanged asynchronously to guide subsequent iterations.  
- ⚖️ Trade-off: synchronization is not real-time.  
  - Serial execution with 100 iterations > 2-thread parallel 100 iterations > 4-thread parallel 100 iterations.  
  - Recommended concurrency: **2–6 threads** (controlled by `max_concurrent`).  

### Actors in Ray Tune

Ray Tune introduces the concept of **Actors**, which wrap optimization experiments as persistent class instances.  

- In this framework, these are `OptCoreRay` and `OptCoreRayMulti`.  
- Each actor encapsulates its own solver instance (e.g., dPBE), which can be reused across iterations to reduce initialization overhead.  

#### Caveats with Actors

1. **Resource leakage**  
   - Long-term repeated runs of dPBE may cause cumulative resource leaks.  
   - To prevent this, actors periodically destroy and restart their dPBE instances.  
   - Configurable in `opt_params['max_reuse']` (number of runs before restart).  

2. **Too fast iterations**  
   - If a single iteration is faster than 1 second, inter-thread communication may lag, causing bottlenecks.  
   - A wait mechanism is added: if runtime < `wait_time`, the process pauses until `wait_time` is reached.  
   - Configurable in `opt_params['wait_time']`.  

## 4. Usage Recommendations

1. **Iteration count**  
   - Depends on data quality, number of kernel parameters, parameter ranges, and degree of parallelization.  
   - Hard to predict in advance → recommended: set as high as possible within resource limits, then monitor convergence.  

2. **Resume from checkpoint**  
   - Optimization can be resumed from saved states, enabling long or interrupted runs.  

3. **Parameter search ranges**  
   - Choosing good search ranges is nontrivial.  
   - Suggested workflow:  
     - Start with a small number of iterations for testing.  
     - Guided algorithms quickly narrow parameter space.  
     - If a parameter consistently converges to the boundary, expand or shift the search range accordingly.  

```{toctree}
:maxdepth: 1

```
