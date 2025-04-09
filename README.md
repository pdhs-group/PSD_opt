# PSD_opt
### Section 1: Project Modules

This project mainly consists of three fundamental modules:

1. **dPBE (discrete Population Balance Equation) Solver - `dpbe.DPBESolver`**: The current dPBE solver is capable of calculating particle aggregation and breakage processes, supporting both single and dual-component systems.
   - **Uni-grid**: Corresponds to the general discretization method, suitable for standard verification but may contain bugs.  
   - **Geo-grid**: Uses the Cell Average Technique (CAT). The primary development is based on geo-grid, and it is the recommended option for use.
  
2. **mcPBE (Monte Carlo Population Balance Equation) Solver - `mcpbe.MCPBESolver`**: The mcPBE solver uses the Monte Carlo random method to solve the PBE. The kernels used are the same as those in the dPBE solver. It supports both single and dual-component systems.

3. **PBM (Population Balance Moments) Solver - `pbm.PBMSolver`**: The PBM solver uses the Quadrature Method of Moments (QMOM) to solve the PBE. It includes various QMOM algorithms such as HyQMOM, GQMOM, and CQMOM. It supports both single and dual-component systems.

4. **Ray Tune-based Optimization Framework - `kernel_opt`**: This module takes one or more PSD (Particle Size Distribution) data sets as input and uses optimization algorithms to search for kernels in the dPBE model. The `kernel_opt` module requires the use of the `pbe` module mentioned above.  
   _Note_: **What are kernels?** Kernels are parameters in the dPBE that describe the dynamics of aggregation and breakage processes. They are usually related to material properties and experimental conditions.
5. **dPBE-Extruder Solver (beta) - `pbe.ExtruderPBESolver`**: Extends the calculation of a single-region dPBE to multiple regions, such as an extruder. Essentially, it solves multiple sets of dPBEs with identical dimensions and allows interactions between them through particle mass flow. This module is still in the testing phase.

For a detailed explanation of each module, refer to the respective sections.

### Section 2: Installation

### **Important Notes**
- **Supported Python version: 3.9 - 3.12**
- The project has not yet been tested on Linux systems, so some issues might arise.
- If you choose the first installation method (manual installation), you need to manually add the `PSD_opt` folder to Python's search path for it to work correctly. However, this step is not required if you use the second method (pre-packaged wheel file), which is recommended.

#### **Method 1: Manual Installation**
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
pip install -r requirements.txt
```

_Remember_: All subsequent usage of this project must be done within the virtual environment.

To exit the virtual environment, simply run:

```bash
deactivate
```

#### **Method 2: Using the Pre-packaged Wheel File (Recommended)**

The project has been packaged into a Python wheel file, which is available in the **Releases** section of the repository. Follow these steps:

1. Download the wheel file, for example: `optframework-0.3.0-py3-none-any.whl`.
2. Install the package using the following command:

   ```bash
   pip install .\optframework-0.3.0-py3-none-any.whl
   ```

This method installs the project as a Python library into the environment, making it easier to use.

Additionally, the wheel file comes with test scripts that can be executed directly to verify that the installation was successful.

### Section 3: Basic Usage

If you downloaded the entire project manually, you can find some test scripts and introductory guidance in the `tests` folder.

If you installed the library using the wheel file, please extract the `tests` archive located alongside the wheel file to obtain the same set of scripts.

These scripts provide examples and a starting point for understanding and testing the project's functionality.

### Section 4: Project File Structure

- **docs**: Contains documentation for all classes, variables, and functions.
- **optframework**: Includes the core code of the modules and some scripts for extended use.
- **tests**: Contains examples of direct usage for each module. These examples should be able to run directly.
  - **config**: This project requires a large number of parameters for simulation and optimization processes. To handle this complexity, all parameter settings are defined in `config.py` scripts containing Python dictionaries. Using Python scripts instead of JSON files allows direct definition of numpy arrays and offers simpler and more intuitive type control.  

    1. These config scripts are typically automatically read during the instantiation of classes. The keys in the dictionary correspond to class attribute names, and the values are the content of those attributes.  
    **Note**: If a key name is entered incorrectly, it will still be read as a class attribute without raising an error! (A validation mechanism is needed.)  

    2. For the functions used in the examples below, if a config file path is not explicitly specified, the method will attempt to locate the parameter file in the `config` folder within the current directory. The expected file names for each example are listed in the descriptions below.  

  - **data**: Similar to the `config` folder, if the save path for the initial condition PSD or the PSD dataset used for optimization is not explicitly specified, they will be saved in this folder. The same applies to reading these files.
  - **all_test.py**: This script tests the main functionality of the project's modules. At the beginning of the main function, there are four keywords corresponding to the tests of each module. You can set them to `False` to skip the respective module tests. The testing sequence for each module in the script is as follows:
    1. The script first generates a PSD normal distribution for particle volumes, which will serve as the initial condition for all subsequent dPBE-related calculations.
    2. It generates a time-dependent PSD dataset that can be used for optimization. This step essentially calculates one or more dPBEs.
    3. The dataset generated in step 2 is used for optimization. By default, the optimization only performs 20 iterations, so it is normal for there to be significant differences between the optimized kernel parameters and the original kernel parameters.
    4. Tests a single iteration from the optimization process. This step does not launch the optimization framework but calculates the optimization objective once (directly using the original kernel parameters, so the result should be small). This process is typically used for debugging.
   
    The script reads parameters from `config/opt_config_all_test.py` for computation.  
    **Note**: If you modify the dPBE parameters in `opt_config_all_test.py`, you must re-run step 2 to regenerate the PSD dataset before running steps 3 or 4.
  - **simple_dpbe.py**: Independently tests the functionality of the dPBE module. It reads parameters from `config/PBE_config.py` and generates a PSD animation as well as a PSD plot for the final time step.
  - **simple_extruder.py**: Tests the `dPBE-Extruder` extension module (beta). By default, it reads parameters for all regions from `config/Extruder0_config.py`. To assign different parameters for each region, refer to the annotations within the script.
  - **simple_opt.py**: Independently tests the optimization framework module. It reads parameters from `config/opt_config.py`.
  - **simple_mcpbe.py**: Tests the `MCPBESolver`.
  - **simple_pbm**: Tests the `PBMSolver`.