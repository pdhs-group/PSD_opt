import numpy as np

config = {
    
    "tA": 100,
    # Agglomeration time [s]
    
    "savesteps": 11,
    # Numer of equally spaced, saved timesteps [-]
    
    "a0": 1e2,
    # Total amount of particles in control volume (initially)
    
    "c": np.array([0.5,0.5]), 
    # Volume concentration array of components [m3/m3]
    ## The Volume concentration of components specifies the proportion of the two primary particles 
    ## in the initial total amount a_0. 
    ## It affects/scales also the control volume to calculate the PBE!
    
    "x": np.array([1e-6, 1e-6]),
    # (Mean) equivalent diameter of primary particles for each component
    
    "PGV": np.array(['mono','mono']),
    # PGV defines which initial particle size distribution is assumed for each component
    # 'mono': Monodisperse at x = x[i]
    # 'norm': Normal distribution at x_mean = x[i] with sigma defined in SIG 
    # 'weibull': Weibull Distribution
    
    "VERBOSE": True,

    "process_type": "mix",  
    # Type of process being simulated.
    # "agglomeration": pure agglomeration
    # "breakage": pure breakage
    # "mix": both agglomeration and breakage

    "CDF_method": "disc",  

    "USE_PSD": True,  
    # Flag indicating whether a particle size distribution (PSD) should be used. If True, 
    # the solver will use the provided PSD files to initialize N.
    # If False, N will be initialized in a quasi-monodisperse form based on process_type:
    # - For process_type="agglomeration", the initial state assumes only the smallest particles (primary particles) are present.
    # - For process_type="breakage", the initial state assumes only the largest particles are present.
    # - For process_type="mix", the initial state assumes both the smallest and largest particles are present.
    # Specific configurations can be found in the init_N() function.

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

    "EFFEVAL": 1,  
    # Flag that determines which model to use for calculating collision efficiency.
    # Can be checked in dpbe_core.py's `calc_F_M`.

    "SIZEEVAL": 1,  
    # Flag that determines whether to account for damping effects due to particle volume growth 
    # during aggregation. This is handled in dpbe_core.py's `calc_F_M`.
    
    "CORR_BETA": 1e-2,
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

    "pl_v": 1.0,  
    # Parameter in fragment distribution function.

    "V1_mean": 1e-15,
    # Mean volume of NM1 particles (in cubic meters).

    "V3_mean": 1e-15,  
    # Mean volume of M particles (in cubic meters).
    
    "pl_P1": 1e-2,  
    "pl_P2": 1,  
    "pl_P3": 1e-2,  
    "pl_P4": 1,  
    # Parameters for breakage rate kernel.
    "G": 1,

}