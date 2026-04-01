# WMCPBE Validation Scripts Introduction

This document introduces the usage of the three validation and analysis scripts in this folder:

- `advance_validation`
- `reconstruction_monitor`
- `WMCPBE_sensitivity_analysis`

The goal is to explain them in a consistent order:

1. what the script is used for
2. what inputs it expects
3. how those inputs should be configured
4. what outputs it produces


## 1. Shared Design Logic

These scripts are designed around the same general workflow:

1. define a physical case
2. define one reference dPBE configuration
3. define one or more WMCPBE configurations
4. build a common initial condition
5. run the target solver workflow
6. compare against a reference quantity
7. save all numerical results to Excel for later post-processing

Common supporting modules:

- `validation.py`
  - provides the common configuration objects
  - provides the standard `ValidationRunner`
- `pbe_validation_advance.py`
  - provides the 2D Dirichlet-type initial condition workflow
- `plotter_new.py`
  - provides the unified plotting style

In all workflows, the most important configuration objects are:

- `CaseConfig`
- `DPBEVariantConfig`
- `WMCPBEVariantConfig`
- optionally `DirichletInitialCondition`


## 2. Basic Input Objects

### 2.1 `CaseConfig`

`CaseConfig` defines the physical problem itself.

Typical fields:

- `dim`
  - particle-space dimension
  - `1` for 1D
  - `2` for 2D
- `kernel`
  - for example `"const"` or `"sum"`
- `process`
  - `"agglomeration"`
  - `"breakage"`
  - `"mix"`
- `t_vec`
  - time vector used for simulation and output
- `x`
  - reference particle size / scale parameter used by the framework
- `beta0`
  - agglomeration kernel parameter
- `p1`, `p2`
  - breakage-related parameters
- `use_psd`
  - whether dPBE uses PSD-file-based initialization

This object answers:

"What physical PBE problem are we solving?"


### 2.2 `DPBEVariantConfig`

This defines the reference dPBE discretization.

Typical fields:

- `name`
- `grid`
- `ns`
- `s`

In the new workflows, dPBE is mainly used for one of two roles:

- building the canonical initial state
- acting as a numerical reference in advanced validation


### 2.3 `WMCPBEVariantConfig`

This defines one WMCPBE calculation setup.

Typical fields:

- `name`
- `repeats`
- `base_seed`
- `maxiter`
- `enabled`
- `attrs`

The most important WMCPBE parameters are passed through `attrs`.

Typical examples inside `attrs`:

- `a0`
- `V_eff_init`
- `recon_enable`
- `recon_method`
- `recon_bins`
- `recon_N_max`
- `recon_RS_target`
- `break_dW_min`
- `break_dW_max`
- `agg_dW_min`
- `agg_dW_max`

This object answers:

"How exactly should this WMCPBE variant be simulated?"


### 2.4 `DirichletInitialCondition`

This is used by the 2D advanced workflows.

Fields:

- `alpha_x`
- `alpha_y`
- `alpha_rest`
- `x_min_scale`
- `x_max_scale`
- `y_min_scale`
- `y_max_scale`
- `total_number`
- `volume_concentration`

This object defines a truncated 2D Dirichlet-type initial distribution on the dPBE grid.

Important rule:

- if both `total_number` and `volume_concentration` are given, the workflow uses `volume_concentration`


## 3. `advance_validation`

### 3.1 Purpose

`advance_validation.py` is the main professional post-processing workflow for:

- analytical solution
- dPBE
- one or more WMCPBE variants

It is focused on 2D cases and advanced comparison metrics.


### 3.2 Main Components

The workflow is implemented mainly by:

- `PBEValidationAdvanced`
- `Dirichlet2DValidationRunner`

The entry example is:

- `advance_validation.py`


### 3.3 Input Order

To use `advance_validation`, configure inputs in this order.

#### Step 1: Define the physical case

Create a `CaseConfig`.

Typical example:

```python
case = CaseConfig(
    dim=2,
    kernel="const",
    process="breakage",
    t_vec=np.arange(0.0, 10.0 + 1e-12, 1.0),
    x=2e-1,
    beta0=1e-3,
    p1=3e-2,
    p2=1.0,
    use_psd=False,
)
```


#### Step 2: Define one reference dPBE variant

Only one dPBE variant should be enabled in the advanced workflow.

Typical example:

```python
dpbe_variants = [
    DPBEVariantConfig(name="dPBE", grid="geo", ns=15, s=2),
]
```


#### Step 3: Define one or more WMCPBE variants

Typical example:

```python
wmcpbe_variants = [
    WMCPBEVariantConfig(
        name="WMCPBE (fine)",
        repeats=4,
        attrs={
            "a0": 100000,
            "V_eff_init": 1000,
            "recon_N_max": 4000,
            "recon_bins": 30,
            "recon_method": "4PMC",
        },
    ),
    WMCPBEVariantConfig(
        name="WMCPBE (ref)",
        repeats=5,
        attrs={
            "a0": 100000,
            "V_eff_init": 1000,
            "recon_N_max": 4000,
            "recon_bins": 30,
            "recon_method": "4PMC",
            "break_dW_max": 10,
            "agg_dW_max": 5,
        },
    ),
]
```


#### Step 4: Build the full `ValidationConfig`

Typical example:

```python
config = ValidationConfig(
    case=case,
    dpbe_variants=dpbe_variants,
    wmcpbe_variants=wmcpbe_variants,
    qmom_variants=[],
    reference_dpbe_name="dPBE",
)
```


#### Step 5: Define the initial distribution

Typical example:

```python
init_dist = DirichletInitialCondition(
    alpha_x=2.5,
    alpha_y=3.0,
    alpha_rest=4.0,
    x_min_scale=2.0,
    x_max_scale=0.5,
    y_min_scale=2.0,
    y_max_scale=0.5,
    total_number=5e5,
    volume_concentration=None,
)
```


#### Step 6: Create the workflow object and run

```python
advanced = PBEValidationAdvanced(config=config, init_dist=init_dist)
result = advanced.run()
```


### 3.4 Available Outputs

After `run()`, the usual workflow is:

```python
advanced.print_moment_error_summary(result)
advanced.plot_selected_moments(result, relative=True)
advanced.plot_psd_snapshot(result, t_index=-1, two_d=False, marginal=True, total=True)
advanced.plot_error_time_pareto(result)
advanced.plot_moment_variances(result)
advanced.show()
```

Meaning of each method:

- `print_moment_error_summary`
  - prints summary errors of selected moments
- `plot_selected_moments`
  - plots `M00`, `M01`, `M11`, `M02` versus time
- `plot_psd_snapshot`
  - plots PSD snapshots
  - supports:
    - `two_d`
    - `marginal`
    - `total`
- `plot_error_time_pareto`
  - plots aggregated error vs CPU time
- `plot_moment_variances`
  - plots normalized variances for WMCPBE


### 3.5 Excel Exports

Each print/plot method writes one Excel workbook before plotting/printing.

Default export location:

- `scripts/pbe_validation/new/exports`

Typical contents:

- `metadata` sheet
- data sheets for curves, surfaces, labels, and error values

This is intended for later processing in Origin or similar tools.


## 4. `reconstruction_monitor`

### 4.1 Purpose

`reconstruction_monitor.py` is used to study reconstruction-induced errors inside the special solver:

- `wmcpbe_recon_debug`

It monitors the error introduced by each reconstruction event itself, rather than the full solver error over time.

The monitored quantities are:

- relative error of `M00`
- relative error of `M01`
- relative error of `M11`
- relative error of `M02`
- L1 error of the reconstructed `N(x, y)`


### 4.2 Internal Logic

This workflow relies on extra monitoring fields added inside:

- `mcpbe/src/wmcpbe_recon_debug/reconstruction_mixin.py`

Each time reconstruction happens, the solver stores:

- reconstruction time
- event index
- particle count before / after reconstruction
- pre-reconstruction moments
- post-reconstruction moments
- relative errors
- `N(x, y)` L1 error on a common 2D histogram grid


### 4.3 Input Order

The input structure is intentionally close to `advance_validation`.

#### Step 1: Define `CaseConfig`

Use a 2D case.

#### Step 2: Define one enabled dPBE reference

This is used to build the canonical initial state.

#### Step 3: Define one or more WMCPBE variants

These should contain reconstruction-related settings such as:

- `recon_method`
- `recon_bins`
- `recon_N_max`
- `recon_monitor_psd_bins`

#### Step 4: Define the initial distribution

Use `DirichletInitialCondition`, same as in `advance_validation`.

#### Step 5: Create the monitor and run

```python
monitor = ReconstructionMonitor(config=config, init_dist=init_dist)
result = monitor.run()
```


### 4.4 Available Outputs

Typical usage:

```python
monitor.print_summary(result)
monitor.plot_moment_errors(result)
monitor.plot_psd_l1_error(result)
monitor.show()
```

Meaning:

- `print_summary`
  - prints the max and final reconstruction error levels
- `plot_moment_errors`
  - plots the reconstruction-induced moment error history
- `plot_psd_l1_error`
  - plots the L1 error history of `N(x, y)`


### 4.5 Excel Exports

Default export location:

- `scripts/pbe_validation/new/exports_reconstruction_monitor`

Important note:

The summary export also contains a `raw_events` sheet, which stores every reconstruction event from every run.

This is useful when:

- you want to rebuild statistics manually
- you want to compare early and late reconstruction stages
- you want to post-process event-level data in Origin


## 5. `WMCPBE_sensitivity_analysis`

### 5.1 Purpose

`WMCPBE_sensitivity_analysis.py` performs variance-based sensitivity analysis on selected WMCPBE parameters.

The current default metric is:

- aggregated moment error

The script is designed so that more error metrics can be added later through a metric registry.


### 5.2 Main Design

The workflow uses:

- one physical `CaseConfig`
- one reference dPBE variant
- one enabled WMCPBE template variant
- a list of sensitivity parameters
- Saltelli/Sobol sampling from `SALib`

For each sampled parameter set:

1. a new WMCPBE config is built from the template
2. `ValidationRunner` runs the solver workflow
3. the chosen error metric is computed
4. the scalar response is stored

Then Sobol indices are computed from all model evaluations.


### 5.3 Input Order

#### Step 1: Define `CaseConfig`

This is the physical case to be studied.

#### Step 2: Define one reference dPBE variant

This is required because `ValidationRunner` still builds canonical initial states from dPBE.

#### Step 3: Define exactly one enabled WMCPBE template variant

This is important.

For sensitivity analysis, the script expects one and only one enabled WMCPBE configuration.

That template provides the baseline solver settings.

Typical example:

```python
wmcpbe_variants = [
    WMCPBEVariantConfig(
        name="WMCPBE template",
        repeats=4,
        attrs={
            "a0": 60000,
            "V_eff_init": 1000,
            "recon_enable": True,
            "recon_method": "4PMC",
            "break_dW_min": 10.0,
            "break_dW_max": 10.0,
            "agg_dW_min": 10.0,
            "agg_dW_max": 10.0,
            "recon_bins": 24,
            "recon_N_max": 3500,
            "recon_RS_target": 1200,
        },
    ),
]
```


#### Step 4: Define the sensitivity parameters

Each parameter is described by `SensitivityParameter`.

Fields:

- `name`
- `bounds`
- `kind`
- `targets`

Examples:

```python
parameters = [
    SensitivityParameter(
        name="delta_w",
        bounds=(2.0, 30.0),
        kind="float",
        targets=("break_dW_min", "break_dW_max", "agg_dW_min", "agg_dW_max"),
    ),
    SensitivityParameter(
        name="recon_bins",
        bounds=(10.0, 40.0),
        kind="int",
        targets=("recon_bins",),
    ),
    SensitivityParameter(
        name="recon_N_max",
        bounds=(1500.0, 6000.0),
        kind="int",
        targets=("recon_N_max",),
    ),
]
```

Interpretation:

- `delta_w` controls several solver attributes together
- `recon_bins` and `recon_N_max` map to one solver attribute each


#### Step 5: Create the analyzer

Typical example:

```python
analyzer = WMCPBESensitivityAnalyzer(
    config=config,
    parameters=parameters,
    metric_name="aggregated_moment_error",
    sample_size=16,
    calc_second_order=False,
)
```

Important fields:

- `metric_name`
  - currently default and recommended: `"aggregated_moment_error"`
- `sample_size`
  - Sobol/Saltelli base sample size
- `calc_second_order`
  - whether to compute second-order Sobol indices


#### Step 6: Run

```python
result = analyzer.run()
```


### 5.4 Current Metric Interface

The current built-in metric is:

- `aggregated_moment_error`

This metric is computed against the analytical solution.

Current implementation:

- in 1D: uses `M00`, `M10`, `M20`
- in 2D: uses `M00`, `M01`, `M11`, `M02`

The structure is intentionally written so that more metrics can be added later, for example:

- max relative moment error
- final-time moment error
- PSD L1 error
- reconstruction-based error


### 5.5 Outputs

The script currently prints:

- first-order Sobol indices
- total-order Sobol indices

It saves one Excel file containing:

- `metadata`
- `sample_records`
- `first_order`
- `total_order`
- `second_order` if enabled

Default export location:

- `scripts/pbe_validation/new/exports_wmcpbe_sensitivity`


## 6. Recommended Usage Strategy

If you are not sure which script to use, this order is usually the most practical:

### Step A: Start from `advance_validation`

Use it when you want to answer:

- Is this WMCPBE setup accurate?
- How does it compare with dPBE and the analytical solution?
- How do moments, PSD, and variances evolve?


### Step B: Use `reconstruction_monitor`

Use it when you want to answer:

- How much error is introduced by reconstruction itself?
- Which reconstruction settings are responsible for the largest local distortion?
- Is the global solver error mainly coming from reconstruction?


### Step C: Use `WMCPBE_sensitivity_analysis`

Use it when you want to answer:

- Which WMCPBE parameter matters the most?
- Is the solution more sensitive to `ΔW`, `recon_bins`, `recon_N_max`, or `a0`?
- Which parameter should be tuned first?


## 7. Practical Notes

### 7.1 Repeats and cost

For WMCPBE-based workflows:

- more `repeats` improve estimator stability
- but sensitivity analysis can become expensive very quickly

A common workflow is:

1. use small `repeats` and small sample size for debugging
2. increase them after the pipeline is verified


### 7.2 Export-first workflow

All new workflows are designed with export-first logic:

- each plot/print writes its numerical content to Excel
- plotting is only one layer on top of the stored data

This is intentional and recommended if you later use:

- Origin
- Excel
- MATLAB
- custom statistical scripts


### 7.3 Reference choice

Be careful about what is used as the reference:

- in advanced moment comparison:
  - analytical solution is the reference
- in PSD snapshot comparison:
  - `WMCPBE (ref)` can be the PSD reference
- in reconstruction monitoring:
  - the pre-reconstruction state is the reference
- in sensitivity analysis:
  - the scalar metric is still defined against the analytical solution


## 8. Summary

These three scripts answer three different levels of questions:

- `advance_validation`
  - full-solver accuracy and comparison
- `reconstruction_monitor`
  - reconstruction-event-induced error
- `WMCPBE_sensitivity_analysis`
  - parameter importance ranking

Together they form a practical workflow:

1. validate accuracy
2. isolate reconstruction effects
3. quantify parameter sensitivity

This is the recommended order for systematic WMCPBE studies.
