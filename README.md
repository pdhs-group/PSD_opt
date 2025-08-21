# PSD_opt
### Section 1: Project Modules

This project mainly consists of three fundamental modules:

1. **dPBE (discrete Population Balance Equation) Solver - `dpbe.DPBESolver`**: The current dPBE solver is capable of calculating particle aggregation and breakage processes, supporting both single and dual-component systems.
   - **Uni-grid**: Corresponds to the general discretization method, suitable for standard verification but may contain bugs.  
   - **Geo-grid**: Uses the Cell Average Technique (CAT). The primary development is based on geo-grid, and it is the recommended option for use.
  
2. **mcPBE (Monte Carlo Population Balance Equation) Solver - `mcpbe.MCPBESolver`**: The mcPBE solver uses the Monte Carlo random method to solve the PBE. The kernels used are the same as those in the dPBE solver. It supports both single and dual-component systems.

3. **PBM (Population Balance Moments) Solver - `pbm.PBMSolver`**: The PBM solver uses the Quadrature Method of Moments (QMOM) to solve the PBE. It includes various QMOM algorithms such as HyQMOM, GQMOM, and CQMOM. It supports both single and dual-component systems.

4. **Ray Tune-based Optimization Framework - `kernel_opt`**: This module takes one or more PSD (Particle Size Distribution) data sets as input and uses optimization algorithms to search for kernel parameters in the dPBE model. The `kernel_opt` module requires the use of the `dPBE` module mentioned above.  
   _Note_: **What are kernel and kernel parameters?** Kernel models refer to the models used in the PBE to describe particle dynamics, such as agglomeration and breakage processes. They are typically mathematical formulas based on internal variables and include multiple parameters. These parameters are called **kernel parameters**.  Kernel parameters essentially represent influencing factors such as material properties, experimental conditions, and other environmental effects.
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
After installation, you can run the script `optframework/examples/all_test.py`, which tests the basic functionality of all core classes.  If the program finishes without errors, it indicates that the essential modules have been correctly installed.

The project provides example scripts directly within the `optframework/examples` folder.  

They serve as a starting point for understanding and testing the functionality of `DPBESolver`, `ExtruderPBESolver`, `PBMSolver`, `MCPBESolver`, and related modules.


### Section 4: Project File Structure

- **docs**: Contains scripts for building documentation.
- **optframework**: The core project library, including all computational kernels and some utility scripts.
  - **dpbe**: Includes the class `DPBESolver`, a discrete PBE solver. Also contains `ExtruderPBESolver`, an extension for multi-region PBE.
  - **pbm**: Contains the class `PBMSolver`, a PBE solver based on the QMOM method.
  - **mcpbe**: Contains the class `MCPBESolver`, a PBE solver based on the Monte Carlo method.
  - **kernel_opt**: The class of kernel parameter optimizer `OptBase`. Built on Ray Tune, it supports highly parallel large-scale computations. Calls `DPBESolver` for PSD evaluations.
  - **kernel_opt_extruder**: Similar to `kernel_opt` but extended to multi-region applications with `ExtruderPBESolver`.
  - **validation**: Provides validation classes `PBEValidation` for `DPBESolver`, `PBMSolver`, and `MCPBESolver`, comparing them with analytical solutions where available.
  - **utils**: Contains other low-level functions, preprocessing and postprocessing scripts, such as pre-compiled JIT-accelerated scripts for large matrix operations in particle dynamics.
  - **examples**: Contains basic usage and postprocessing examples for `DPBESolver`, `ExtruderPBESolver`, `PBMSolver`, `MCPBESolver`, `OptBase`, and `PBEValidation`.

> **Note**: Except for `ExtruderPBESolver`, all solvers are limited to a single closed PBE computational domain.

📖 For detailed documentation of each class and script, please refer to: [http://pdhs-group.com/PSD_opt/](http://pdhs-group.com/PSD_opt/)
