# PSD_opt
### Section 1: Project Modules

This project mainly consists of three fundamental modules:

1. **dPBE (discrete Population Balance Equation) Solver - `pbe`**: The current dPBE solver is capable of calculating particle aggregation and breakage processes, supporting both single and dual-component systems.
2. **Ray Tune-based Optimization Framework - `kernel_opt`**: This module takes one or more PSD (Particle Size Distribution) data sets as input and uses optimization algorithms to search for kernels in the dPBE model. The `kernel_opt` module requires the use of the `pbe` module mentioned above.  
   _Note_: **What are kernels?** Kernels are parameters in the dPBE that describe the dynamics of aggregation and breakage processes. They are usually related to material properties and experimental conditions.
3. **Monte Carlo-based Breakage Simulation - `bond_break`**: This original model is designed for specific particles, but with the help of ANN (Artificial Neural Network) acceleration, it can be extended to a particle system. This module is still under development and is currently independent of the other two modules.

For a detailed explanation of each module, refer to the respective sections.

See the [documentation](docs/_build/html/index.html) for more information! (You have to open it locally in your browser currently, because private repo)

### Section 2: Installation

To download the project, use the following command:

```bash
git clone https://github.com/pdhs-group/PSD_opt.git PSD_opt
```

This will download the entire project into a folder named `PSD_opt`. Then navigate into this folder:

```bash
cd PSD_opt
```

To install the required external environment, it is recommended to create a Python virtual environment. This makes it easier to manage package versions. First, ensure that Python 3 is installed, and then run the following command:

```bash
python -m venv PSD_opt_env
```

This will create a folder named `PSD_opt_env` in the current directory, which will contain the virtual environment. After that, activate the virtual environment:

- **On Windows**:
  
  ```bash
  PSD_opt_env\Scripts\activate
  ```

- **On Linux**:

  ```bash
  source PSD_opt_env/bin/activate
  ```

Next, install all external packages within the current virtual environment:

```bash
pip install -r requirements_ray.txt
```

_Remember_: All subsequent usage of this project must be done within the virtual environment.

To exit the virtual environment, simply run:

```bash
deactivate
```

### Section 3: Project File Structure

- **config**: Contains parameter settings for running both the `pbe` and `PSD_opt` modules. When running `pbe` independently, it reads from `PBE_config`. When running `kernel_opt`, it only reads from `opt_config.py`, and the contents of `PBE_config` are ignored.
- **docs**: Contains documentation for all classes, variables, and functions.
- **pypbe**: Includes the core code of the modules and some scripts for extended use.
- **tests**: Contains examples of direct usage for each module.