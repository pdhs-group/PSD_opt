import numpy as np

config = {
    "NS": 8,
    "S": 4,
    "R01": 8.677468940430804e-07,
    "R03": 8.677468940430804e-07,

    "t_total": 3601,
    "t_write": 100,
    "process_type": "mix",
    "solver": "ivp",
    "V_unit": 1e-15,
    "USE_PSD": True,
    "DIST1_path": None,
    "DIST3_path": None,
    "DIST1_name": "PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy",
    "DIST3_name": "PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy",

    "COLEVAL": 1,
    "EFFEVAL": 1,
    "SIZEEVAL": 1,
    "aggl_crit": 100,
    "CORR_BETA": 1e-2,
    'alpha_prim': np.array([1,1,1,1]),

    "BREAKRVAL": 4,
    "BREAKFVAL": 5,
    "pl_v": 0.7,
    "pl_P1": 1e-3,
    "pl_P2": 0.5,
    "pl_P3": 1e-3,
    "pl_P4": 0.5
}