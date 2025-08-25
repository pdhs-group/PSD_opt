import numpy as np

config = {
    "NS": 10,  
    # Number of size classes for discretizing particle populations (grid points).

    "S": 4,  
    # Geometric ratio used to define the spacing in the size grid for 'geo' discretization. 

    # "R01": 8.677468940430804e-07,
    # "R03": 8.677468940430804e-07,
    "R01": 8.116913897613351e-07,  
    # Radius of primary NM1 particles (in meters).

    "R03": 8.116913897613351e-07,  
    # Radius of primary M particles (in meters).

    "process_type": "mix",  
    # Type of process being simulated.
    # "agglomeration": pure agglomeration
    # "breakage": pure breakage
    # "mix": both agglomeration and breakage

    "solver": "ivp",  
    # Numerical solver used to integrate the PBE.

    # "V_unit": 1e-7,  
    # Volume unit used for normalization of N (particle number concentration). 
    # Setting a smaller value generally does not affect the relative relationships between N (i.e., the PSD),
    # but helps reduce the stiffness of matrices during calculations, leading to faster solver convergence.

    "USE_PSD": True,  
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

    "COLEVAL": 1,  
    # Flag that determines which model to use for calculating collision frequency.
    # Can be checked in dpbe_core.py's `calc_F_M`.

    "SIZEEVAL": 1,  
    # Flag that determines whether to account for damping effects due to particle volume growth 
    # during aggregation. This is handled in dpbe_core.py's `calc_F_M`.
    
    "aggl_crit": 100,  
    # Critical particle size for agglomeration. Agglomeration will be limited to particles larger than this size.

    "CORR_BETA": 1,
    # Correction factor for the collision frequency kernel, controlling the rate of aggregation.

    'alpha_prim': np.array([1,1,1,1]),  
    # 'alpha_prim': np.array([1]),
    # Factors for collision efficiency.
    # The length of the alpha_prim array must be the square of the dpbe's dimensionality (dim^2).

    "BREAKRVAL": 4,  
    # Flag that determines which model to use for calculating breakage rate.
    # Can be checked in dpbe_core.py's `calc_B_R`.

    "BREAKFVAL": 5,  
    # Flag that determines which model to use for calculating the fragment distribution function.
    # Can be checked in dpbe_core.py's `calc_int_B_F`.

    "pl_v": 2.0,  
    # Parameter in fragment distribution function.
    
    "pl_P1": 1e-2,  
    "pl_P2": 1,  
    "pl_P3": 1e-2,  
    "pl_P4": 1,  
    # Parameters for breakage rate kernel.

}