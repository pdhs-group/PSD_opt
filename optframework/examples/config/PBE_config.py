import numpy as np

config = {
    "NS": 10,  
    # Number of size classes for discretizing particle populations (grid points).

    "S": 4,  
    # Geometric ratio used to define the spacing in the size grid for 'geo' discretization. 

    "R01": 1e-6,  
    # Radius of primary NM1 particles (in meters) for uni-grid.
    # In a geometric grid, volumes are calculated as midpoints between volume edges (V_e).
    # Therefore, when using a geo-grid, the specified value here corresponds to the radius
    # of the left edge of the grid, V_e[1]. The actual primary particle size is given by 
    # V_e[1] * (1 + S) / 2, where S is the geometric spacing factor. R[1] = ((1+S)/2)**(1/3)*R01
    
    "R03": 1e-6,  
    # Radius of primary M particles (in meters).

    "t_total": 10*60+1,  
    # Total simulation time in seconds.

    "t_write": 10,  
    # Interval (in time steps) for writing output data (e.g., simulation results).

    "process_type": "mix",  
    # Type of process being simulated.
    # "agglomeration": pure agglomeration
    # "breakage": pure breakage
    # "mix": both agglomeration and breakage

    "solver": "ivp",  
    # Numerical solver used to integrate the PBE.
    'c_mag_exp': 1e-2,
    "V_unit": 1e-12,  
    # Volume unit used for normalization of N (particle number concentration). 
    # Setting a smaller value generally does not affect the relative relationships between N (i.e., the PSD),
    # but helps reduce the stiffness of matrices during calculations, leading to faster solver convergence.

    "USE_PSD": False,  
    # Flag indicating whether a particle size distribution (PSD) should be used. If True, 
    # the solver will use the provided PSD files to initialize N.
    # If False, N will be initialized in a quasi-monodisperse form based on process_type:
    # - For process_type="agglomeration", the initial state assumes only the smallest particles (primary particles) are present.
    # - For process_type="breakage", the initial state assumes only the largest particles are present.
    # - For process_type="mix", the initial state assumes that all particles are concentrated at the center grid index.

    "DIST1_path": None,  
    # File path to the PSD data for NM1 particles. If None, default location(pypbe/data/PSD_data) will be used.

    "DIST3_path": None,  
    # File path to the PSD data for M particles. If None, default location will be used.

    "DIST1_name": "Name_of_your_PSD.npy",  
    # Name of the file containing the PSD for NM1 particles.

    "DIST3_name": "Name_of_your_PSD.npy",  
    # Name of the file containing the PSD for M particles.

    "COLEVAL": 1,
    # Flag that determines which model to use for calculating collision frequency.
    # Can be checked in dpbe_core.py's `calc_F_M`.

    "EFFEVAL": 1,
    # Flag that determines which model to use for calculating collision efficiency.
    # Can be checked in dpbe_core.py's `calc_F_M`.

    "SIZEEVAL": 1,
    # Flag that determines whether to account for damping effects due to particle volume growth 
    # during aggregation. This is handled in dpbe_core.py's `calc_F_M`.
    
    "aggl_crit": 100,  
    # Critical particle size for agglomeration. Agglomeration will be limited to particles larger than this size.

    "CORR_BETA": 1e-3,
    # Correction factor for the collision frequency kernel, controlling the rate of aggregation.

    # 'alpha_prim': np.array([1,1,1,1]),
    'alpha_prim': np.array([1]),
    # Factors for collision efficiency.
    # The length of the alpha_prim array must be the square of the dpbe's dimensionality (dim^2).

    "BREAKRVAL": 4,  
    # Flag that determines which model to use for calculating breakage rate.
    # Can be checked in dpbe_core.py's `calc_B_R`.

    "BREAKFVAL": 5,  
    # Flag that determines which model to use for calculating the fragment distribution function.
    # Can be checked in dpbe_core.py's `calc_int_B_F`.

    "pl_v": 2,  
    # Parameter in fragment distribution function.

    "pl_P1": 1e13,  
    "pl_P2": 1,  
    "pl_P3": 1e13,  
    "pl_P4": 1,  
    # Parameters for breakage rate kernel.
    "G": 1,

}