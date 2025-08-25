(attributes_overview)= 
# Attributes

## Common Attributes of the Solver Class

### General Information

The following tables summarize the most important class attributes for performing PBE simulations. These attributes are defined in the `_init_base_parameters` method of the `BaseSolver` class in `base_solver.py`.

> Note: Additional attributes may be defined in specific solver implementations.

### Simulation Parameters
| Attribute | Type | Default | Description |  
|---|---|---|---|
| `dim` | `int` | $-$ | Dimension of the PBE (1=1D, 2=2D, 3=3D). The 3D case has not been fully adapted yet |
| `t_vec` | `array-like` | `None` | Simulation time vector |
| `t_total` | `float` | 601 | Total simulation time [second] |
| `t_write` | `float` | 100 | Output time interval. If `t_vec` is not specified, use `t_total` and `t_write` to construct `t_vec` |
| `work_dir` | `Path` | Current working directory | Baseline path for the simulation |

### Agglomeration Kernel Parameters
| Attribute | Type | Default | Description | 
|---|---|---|---|
| `COLEVAL` | `int` | 1 | Case for calculation of beta (collision frequency). 1: Chin 1998, 2: Tsouris 1995, 3: Constant kernel, 4: Volume-sum kernel |
| `SIZEEVAL` | `int` | 1 | Case for implementation of size dependency. 1: No size dependency, 2: Model from Soos2007 |
| `X_SEL` | `float` | 0.310601 | Size dependency parameter for Selomulya2003 / Soos2006 |
| `Y_SEL` | `float` | 1.06168 | Size dependency parameter for Selomulya2003 / Soos2006 |
| `POTEVAL` | `int` | 1 | Case for interaction potentials in DLVO theory. Case 2 is faster, Case 3 uses pre-defined alphas |
| `alpha_prim` | `array-like` | `np.ones(dim**2)` | Collision efficiency array |
| `CORR_BETA` | `float` | 25.0 | Correction term for collision frequency describing external factors influence |

### Breakage Kernel Parameters
| Attribute | Type | Default | Description | 
|---|---|---|---|
| `BREAKRVAL` | `int` | 3 | Case for breakage rate calculation. 1: Constant, 2: Volume-sum, 3: Jeldres 2018 Power Law, 4: Jeldres 2018 volume fraction |
| `BREAKFVAL` | `int` | 3 | Case for breakage function calculation. 1-2: Leong2023 models, 3-5: Diemer Olson 2002 models |
| `process_type` | `str` | `"breakage"` | Process type: `"agglomeration"`, `"breakage"`, or `"mix"` |
| `pl_v` | `int` | 2 | Number of fragments in product function of power law |
| `pl_q` | `int` | 1 | Parameter describing breakage type in product function of power law |
| `pl_P1` | `float` | 0.01 | 1st parameter in power law for breakage rate (1D/2D) |
| `pl_P2` | `float` | 1 | 2nd parameter in power law for breakage rate (1D/2D) |
| `pl_P3` | `float` | 0.01 | 3rd parameter in power law for breakage rate (2D) |
| `pl_P4` | `float` | 1 | 4th parameter in power law for breakage rate (2D) |
| `B_F_type` | `str` | `'int_func'` | Breakage function calculation method: `'int_func'`, `'MC_bond'`, or `'ANN_MC'` |
| `work_dir_MC_BOND` | `str` | `work_dir/bond_break/int_B_F.npz` | Path to MC bond breakage data file |

### Particle Size Distribution Parameters
| Attribute | Type | Default | Description | 
|---|---|---|---|
| `USE_PSD` | `bool` | `True` | Whether to use .npy file to initialize particle distribution (False = monodisperse primary particles) |
| `DIST1_path` | `str` | `work_dir/data/PSD_data` | Path to PSD data directory for component 1 |
| `DIST2_path` | `str` | `work_dir/data/PSD_data` | Path to PSD data directory for component 2 |
| `DIST3_path` | `str` | `work_dir/data/PSD_data` | Path to PSD data directory for component 3 |

### DLVO Theory Parameters
| Attribute | Type | Default | Description | 
|---|---|---|---|
| `PSI1` | `float` | 0.001 | Surface potential component 1 [V] - NM1 |
| `PSI2` | `float` | 0.001 | Surface potential component 2 [V] - NM2 |
| `PSI3` | `float` | -0.04 | Surface potential component 3 [V] - M |
| `A_NM1NM1` | `float` | 10e-21 | Hamaker constant for NM1-NM1 interaction [J] |
| `A_NM2NM2` | `float` | 10e-21 | Hamaker constant for NM2-NM2 interaction [J] |
| `A_MM` | `float` | 80e-21 | Hamaker constant for M-M interaction [J] |

### Hydrophobic Interaction Parameters
| Attribute | Type | Default | Description | 
|---|---|---|---|
| `C1_NM1NM1` | `float` | 0 | Short-range hydrophobic interaction parameter NM1-NM1 [N/m] |
| `C2_NM1NM1` | `float` | 0 | Long-range hydrophobic interaction parameter NM1-NM1 [N/m] |
| `C1_MNM1` | `float` | 0 | Short-range hydrophobic interaction parameter M-NM1 [N/m] |
| `C2_MNM1` | `float` | 0 | Long-range hydrophobic interaction parameter M-NM1 [N/m] |
| `C1_MM` | `float` | 0 | Short-range hydrophobic interaction parameter M-M [N/m] |
| `C2_MM` | `float` | 0 | Long-range hydrophobic interaction parameter M-M [N/m] |
| `C1_NM2NM2` | `float` | 0 | Short-range hydrophobic interaction parameter NM2-NM2 [N/m] |
| `C2_NM2NM2` | `float` | 0 | Long-range hydrophobic interaction parameter NM2-NM2 [N/m] |
| `C1_MNM2` | `float` | 0 | Short-range hydrophobic interaction parameter M-NM2 [N/m] |
| `C2_MNM2` | `float` | 0 | Long-range hydrophobic interaction parameter M-NM2 [N/m] |
| `C1_NM1NM2` | `float` | 0 | Short-range hydrophobic interaction parameter NM1-NM2 [N/m] |
| `C2_NM1NM2` | `float` | 0 | Long-range hydrophobic interaction parameter NM1-NM2 [N/m] |
| `LAM1` | `float` | 1.2e-9 | Range of short-range hydrophobic interactions [m] |
| `LAM2` | `float` | 10e-9 | Range of long-range hydrophobic interactions [m] |
| `X_CR` | `float` | 2e-9 | Alternative range criterion for hydrophobic interactions [m] |

### Experimental/Process Parameters
| Attribute | Type | Default | Description | 
|---|---|---|---|
| `c_mag_exp` | `float` | 0.01 | Volume concentration of magnetic particles [Vol-%] |
| `Psi_c1_exp` | `float` | 1 | Concentration ratio component 1 (V_NM1/V_M) [-] |
| `Psi_c2_exp` | `float` | 1 | Concentration ratio component 2 (V_NM2/V_M) [-] |
| `G` | `float` | 1 | Shear rate [1/s] |
| `V_unit` | `float` | 1 | Unit volume used to calculate total particle concentration |

### Calculated Parameters
The following parameters are automatically calculated in the `_reset_params` method:

| Attribute | Type | Description | 
|---|---|---|
| `t_num` | `int` | Number of time steps based on `t_vec` |
| `cv_1` | `float` | Volume concentration of NM1 particles [Vol-%] |
| `cv_2` | `float` | Volume concentration of NM2 particles [Vol-%] |
| `V01` | `float` | Total volume concentration of component 1 [unit/unit] - NM1 |
| `V02` | `float` | Total volume concentration of component 2 [unit/unit] - NM2 |
| `V03` | `float` | Total volume concentration of component 3 [unit/unit] - M |
| `N01` | `float` | Total number concentration of primary particles component 1 [1/m³] - NM1 |
| `N02` | `float` | Total number concentration of primary particles component 2 [1/m³] - NM2 |
| `N03` | `float` | Total number concentration of primary particles component 3 [1/m³] - M |
| `DIST1` | `str` | Full path to PSD file for component 1 |
| `DIST2` | `str` | Full path to PSD file for component 2 |
| `DIST3` | `str` | Full path to PSD file for component 3 |

## DPBESolver Specific Attributes

### General Information

The following attributes are specific to the `DPBESolver` class and are defined in the `__init__` method of `dpbe_base.py`. These attributes extend the common `BaseSolver` attributes for discrete Population Balance Equation (dPBE) solving.

### Discretization Parameters
| Attribute | Type | Default | Description |
|---|---|---|---|
| `disc` | `str` | `'geo'` | Discretization scheme: `'geo'` for geometric grid, `'uni'` for uniform grid |
| `NS` | `int` | 12 | Grid parameter [-] |
| `S` | `float` | 2 | Geometric grid ratio (V_e[i] = S*V_e[i-1]). Actual primary particle size is R[1] = ((1+S)/2)**(1/3)*R01 |
| `solve_algo` | `str` | `"ivp"` | Solution algorithm: `"ivp"` uses integrate.solve_ivp, `"radau"` uses RK.radau_ii_a (debug only) |
| `aggl_crit` | `float` | 1000.0 | Upper volume limit for agglomeration (grid-based) |

### Material Parameters
| Attribute | Type | Default | Description |
|---|---|---|---|
| `R01` | `float` | 2.9e-7 | Basic radius component 1 [m] - NM1 |
| `R02` | `float` | 2.9e-7 | Basic radius component 2 [m] - NM2 |
| `R03` | `float` | 2.9e-7 | Basic radius component 3 [m] - M3 (magnetic component) |

### Physical Constants
| Attribute | Type | Default | Description |
|---|---|---|---|
| `KT` | `float` | 4.0434e-21 | Boltzmann constant × Temperature [J] (k*T at 293K) |
| `MU0` | `float` | 4π×10⁻⁷ | Permeability constant vacuum [N/A²] |
| `EPS0` | `float` | 8.854e-12 | Permittivity constant vacuum [F/m] |
| `EPSR` | `float` | 80 | Permittivity material factor [-] |
| `E` | `float` | 1.602e-19 | Electron charge [C] |
| `NA` | `float` | 6.022e23 | Avogadro number [1/mol] |
| `MU_W` | `float` | 1e-3 | Viscosity water [Pa*s] |
| `EPS` | `float` | Calculated | Permittivity (EPSR×EPS0) [F/m] |

### Process Parameters
| Attribute | Type | Default | Description |
|---|---|---|---|
| `I` | `float` | 1.0 | Ionic strength [mol/m³] (converted from mol/L) |

### Computational Options
| Attribute | Type | Default | Description |
|---|---|---|---|
| `JIT_FM` | `bool` | `True` | Whether to precompile FM (Formation Matrix) calculation |
| `JIT_BF` | `bool` | `True` | Whether to precompile BF (Breakage Function) calculation |

### Submodules
The following submodules are automatically instantiated:
- `core`: DPBECore instance for core PBE functionality
- `post`: DPBEPost instance for post-processing
- `visualization`: DPBEVisual instance for visualization

## PBMSolver Specific Attributes

### General Information

The following attributes are specific to the `PBMSolver` class and are defined in the `__init__` method of `pbm_base.py`. These attributes extend the common `BaseSolver` attributes for Population Balance Moment (PBM) method solving.

### Moment Method Parameters
| Attribute | Type | Default | Description |
|---|---|---|---|
| `n_order` | `int` | 5 | Order parameter where n_order×2 is the order of the moments [-] |
| `n_add` | `int` | 10 | Number of additional nodes [-] |
| `GQMOM` | `bool` | `False` | Flag for using Generalized Quadrature Method of Moments |
| `GQMOM_method` | `str` | `"gaussian"` | Method for GQMOM implementation |
| `nu` | `int` | 1 | Exponent for the correction in gaussian-GQMOM |

### Integration Tolerance Parameters
| Attribute | Type | Default | Description |
|---|---|---|---|
| `atol_min` | `float` | 1e-16 | Minimum absolute tolerance |
| `atol_scale` | `float` | 1e-9 | Scaling factor for absolute tolerance |
| `rtol` | `float` | 1e-6 | Relative tolerance |

### Runtime Calculated Attributes
The following attributes are calculated during runtime:
| Attribute | Type | Description |
|---|---|---|
| `moments_norm` | `array` | Normalized moments |
| `moments_norm_factor` | `array` | Normalization factors for moments |
| `atolarray` | `array` | Absolute tolerance array for integration |
| `rtolarray` | `array` | Relative tolerance array for integration |
| `x_max` | `float` | Maximum x-coordinate for normalization |
| `moments` | `array` | Calculated moments |
| `indices` | `array` | 2D moment indices for CHY or C method |

### Submodules
The following submodules are automatically instantiated:
- `post`: PBMPost instance for post-processing
- `quick_test`: PBMQuickTest instance for quick testing
- `core`: PBMCore instance for core PBM functionality

## MCPBESolver Specific Attributes

### General Information

The following attributes are specific to the `MCPBESolver` class and are defined in the `__init__` method of `mcpbe.py`. These attributes extend the common `BaseSolver` attributes for Monte Carlo Population Balance Equation solving.

### Simulation Parameters
| Attribute | Type | Default | Description |
|---|---|---|---|
| `c` | `array` | `np.full(dim,0.1e-2)` | Concentration array of components |
| `x` | `array` | `np.full(dim,1e-6)` | (Mean) equivalent diameter of primary particles for each component [m] |
| `x2` | `array` | `np.full(dim,1e-6)` | (Mean) equivalent diameter of primary particles for each component (bi-modal case) [m] |
| `a0` | `float` | 1000.0 | Total amount of particles in control volume (initially) [-] |
| `CDF_method` | `str` | `"disc"` | Method for fragment distribution: `"disc"` uses discrete points, `"conti"` uses continuous functions |
| `VERBOSE` | `bool` | `False` | Whether to print detailed calculation information |

### Initial Condition Parameters
| Attribute | Type | Default | Description |
|---|---|---|---|
| `PGV` | `array` | `np.full(dim, 'mono')` | Initial particle size distribution type for each component: `'mono'`, `'norm'`, `'weibull'` |
| `SIG` | `array` | `np.full(dim,0.1)` | (Relative) standard deviation of normal distribution (STD = SIG×v) |
| `PGV2` | `array` | `None` | Second mode distribution type for bi-modal distributions (None for mono-modal) |
| `SIG2` | `array` | `None` | Standard deviation for second mode in bi-modal distributions |

### Runtime Variables
The following variables are calculated during initialization:
| Attribute | Type | Description |
|---|---|---|
| `V_flat` | `array` | Volume matrix where each column represents one particle/agglomerate |
| `v2` | `array` | Volume of primary particles (bi-modal case) |
| `n` | `array` | Number concentration for each component |
| `n2` | `array` | Number concentration for bi-modal case |
| `n0` | `float` | Total number of primary particles |
| `Vc` | `float` | Control volume |
| `a` | `array` | Total number of primary particles (integer) |
| `a2` | `array` | Total number of primary particles for bi-modal case (integer) |
| `a_tot` | `int` | Final total number of primary particles in control volume |
| `X` | `array` | Equivalent diameter calculated from total volume |
| `t` | `list` | Time array |
| `betaarray` | `array` | Beta array for collision calculations |
| `frag_num` | `float` | Expected number of fragments from breakage |
| `break_rate` | `array` | Breakage rate array |
| `break_func` | `array` | Breakage function array |

### Process Control Flags
| Attribute | Type | Description |
|---|---|---|
| `process_agg` | `bool` | Whether agglomeration process is active |
| `process_break` | `bool` | Whether breakage process is active |

### Data Storage Arrays
| Attribute | Type | Description |
|---|---|---|
| `V_save` | `list` | List storing volume matrices at different time steps |
| `Vc_save` | `list` | List storing control volumes at different time steps |
| `V0` | `array` | Initial volume matrix |
| `X0` | `array` | Initial equivalent diameter |
| `V0_save` | `list` | List storing initial volume matrices |
| `step` | `int` | Current step counter |
| `MACHINE_TIME` | `float` | Calculation time (set after solving) |

### Method Parameters
Additional parameters for breakage methods:
| Attribute | Type | Description |
|---|---|---|
| `rel_frag` | `array` | Relative fragment sizes (1D case) |
| `rel_frag1` | `array` | Relative fragment sizes for component 1 (2D case) |
| `rel_frag3` | `array` | Relative fragment sizes for component 3 (2D case) |
| `cdf_interp` | `function` | Interpolation function for continuous CDF method |