# pypbe

### Folder Contents

- **bond_break**: Contains the original module class for `bond_break` and its ANN training class (still under development).
- **data**: Stores the default path for the PSD data used by `kernel_opt` for optimization. If another path is manually specified, the specified path will be used.
  - **Note**: The `data` folder typically needs a subfolder named `PSD_data`. This subfolder contains the PSD data used to determine the initial conditions for the dPBE. You can generate this data using the script located at `PSD_opt\pypbe\utils\general_scripts\generate_psd.py`.
- **kernel_opt**: Contains classes related to the optimization framework.
- **pbe**: Contains classes for the dPBE solver.
- **utils**: Contains additional classes, functions, and scripts used for calculations or data processing.
