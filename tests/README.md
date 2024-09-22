### Executable Scripts

This directory contains several executable scripts. Among them, you can use `all_test.py` to run the most important modules of the project and check the integrity of their functionality.

#### Structure of `all_test.py`:

At the beginning, you need to define four boolean values: `generate_new_psd`, `generate_synth_data`, `run_opt`, and `run_calc_delta`, which determine which modules will be tested.  
**Note**: Some modules require the results of previous modules to run properly. For example, you need at least one complete PSD file before the optimization module can start running.

1. **`generate_new_psd`**:  
   This section generates PSD data that follows either a normal or log-normal distribution. This data is used to initialize the dPBE equation (but not for optimization!). The parameters for this section are directly defined within the script.

2. **`generate_synth_data`**:  
   This section computes the dPBE and extracts the PSD data, saving it into an Excel file. The numbers in the file represent particle number concentration (scaled by `V_unit`). The parameters used for the dPBE calculation and other settings like sample size are defined in `config/opt_config`.

3. **`run_opt`**:  
   This section uses the file specified by `data_name` for optimization. The optimization parameters are defined in `config/opt_config`.  
   **Note**: If Ray fails during execution, you might need to manually run `ray.shutdown()` before restarting the optimization.

4. **`run_calc_delta`**:  
   This section serves as an example of testing an individual function. It performs a complete iteration of the optimization process by reading data from the file specified by `data_name`. Using the dPBE parameters from `config/opt_config`, it calculates a PSD and then computes the difference between this PSD and the data read from the file. This process allows you to track variable changes and debug the calculations in detail.

### `simple_dpbe`

This file serves as an example of using the `DPBESolver` module. By default, its parameters are read from `config/PBE_config.py`. Otherwise, the user needs to provide the path to the config file or manually set the parameters via the instance's attributes.

After completing the PBE simulation, this script performs simple post-processing on the results, including:

- Calculating the change in total system volume from the initial conditions to the end of the simulation to check if mass conservation is upheld.
- Printing the PSD at the time step defined by `t_frame`.
- Printing the PSD over time to observe its evolution.