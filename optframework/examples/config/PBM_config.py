import numpy as np

config = {
    "n_order": 2,                          # Order of the moments [-]
    "n_add": 1,                          # Number of additional nodes [-] 
    "GQMOM": False,
    "GQMOM_method": "gaussian",
    "nu": 1,   

    "t_total": 31,  
    # Total simulation time in seconds.

    "t_write": 2,  
    # Interval in seconds for writing output data (e.g., simulation results).

    "process_type": "mix",  
    # Type of process being simulated.
    # "agglomeration": pure agglomeration
    # "breakage": pure breakage
    # "mix": both agglomeration and breakage

    "V_unit": 1e-0,  
    # Volume unit used for normalization of N (particle number concentration). 
    # Setting a smaller value generally does not affect the relative relationships between N (i.e., the PSD),
    # but helps reduce the stiffness of matrices during calculations, leading to faster solver convergence.

    "USE_PSD": False,  
    # Flag indicating whether a particle size distribution (PSD) should be used. If True, 
    # the solver will use the provided PSD files to initialize N.

    "DIST1_path": None,  
    # File path to the PSD data for NM1 particles. If None, default location(pypbe/data/PSD_data) will be used.

    "DIST3_path": None,  
    # File path to the PSD data for M particles. If None, default location will be used.

    "DIST1_name": "PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy",  
    # Name of the file containing the PSD for NM1 particles.

    "DIST3_name": "PSD_x50_2.0E-5_RelSigmaV_2.0E-1.npy",  
    # Name of the file containing the PSD for M particles.

    "COLEVAL": 4,  
    # Flag that determines which model to use for calculating collision frequency.
    # Can be checked in dpbe_core.py's `calc_F_M`.

    "EFFEVAL": 2,  
    # Flag that determines which model to use for calculating collision efficiency.
    # Can be checked in dpbe_core.py's `calc_F_M`.

    "SIZEEVAL": 1,  
    # Flag that determines whether to account for damping effects due to particle volume growth 
    # during aggregation. This is handled in dpbe_core.py's `calc_F_M`.

    "CORR_BETA": 1e-1,
    # Correction factor for the collision frequency kernel, controlling the rate of aggregation.

    'alpha_prim': np.array([1,1,1,1]),  
    # 'alpha_prim': np.array([1]),
    # Factors for collision efficiency.
    # The length of the alpha_prim array must be the square of the dpbe's dimensionality (dim^2).

    "BREAKRVAL": 2,  
    # Flag that determines which model to use for calculating breakage rate.
    # Can be checked in dpbe_core.py's `calc_B_R`.

    "BREAKFVAL": 2,  
    # Flag that determines which model to use for calculating the fragment distribution function.
    # Can be checked in dpbe_core.py's `calc_int_B_F`.

    "pl_v": 2.0,  
    # Parameter in fragment distribution function.
    
    
    ### To ensure the monotonicity of the breakage rate, this setting has been deprecated, 
    ### and all particle volumes are scaled by the volume of the smallest particle.
    # "V1_mean": 5e-13,
    # Mean volume of NM1 particles (in cubic meters).

    # "V3_mean": 5e-13,
    # Mean volume of M particles (in cubic meters).
    
    "pl_P1": 1e-2,  
    "pl_P2": 1,  
    "pl_P3": 1e-2,  
    "pl_P4": 1,  
    # Parameters for breakage rate kernel.
    "G": 1,

}