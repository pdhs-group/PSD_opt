# pypbe

### Folder Contents

- **kernel_opt**: Contains classes related to the optimization framework.
- **pbe**: Contains classes for the dPBE and dPBE-Extruder solver.
- **utils**: Contains additional functions and scripts used for calculations or data processing.
  - **func**: Includes general mathematical computation functions, functions for reading and writing PSD datasets, and functions for the detailed calculations of the dPBE partial differential equation (most of which are accelerated using JIT pre-compilation).
  - **func_temp**: Contains scripts primarily used for debugging or as backups, most of which are not actively maintained.
  - **general_scripts**: Mainly consists of scripts for post-processing dPBE or optimization results, as well as scripts for generating initial condition PSDs.
  - **plotter**: Provides functions for quickly generating plots, contributed by Frank Rhein.

