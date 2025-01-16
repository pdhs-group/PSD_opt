import numpy as np
import math
from numba import jit
import optframework.utils.func.jit_pbm_qmom as qmom
import optframework.utils.func.jit_kernel_agg as kernel_agg
import optframework.utils.func.jit_kernel_break as kernel_break

def filter_negative_nodes(xi, wi, threshold=1e-7):
    """
    Filter the nodes (xi) and weights (wi) based on the condition:
    - If xi[i] < 0 and wi[i] < threshold, remove xi[i] and wi[i].
    - If xi[i] < 0 but wi[i] >= threshold, raise an error.

    Parameters:
        xi (array-like): Array of nodes.
        wi (array-like): Array of weights corresponding to the nodes.
        threshold (float): Threshold for weights below which the corresponding nodes are removed.

    Returns:
        tuple: Filtered (xi, wi) arrays.
    """
    xi = np.array(xi)
    wi = np.array(wi)

    if len(xi) != len(wi):
        raise ValueError("xi and wi must have the same length.")

    # Indices where xi < 0
    negative_indices = np.where(xi < 0)[0]

    # Check each negative xi
    for idx in negative_indices:
        if wi[idx] >= threshold:
            raise ValueError(
                f"Node xi[{idx}] = {xi[idx]} is negative, but its weight wi[{idx}] = {wi[idx]} exceeds the threshold {threshold}."
            )

    # Remove elements where xi < 0 and wi < threshold
    valid_indices = [i for i in range(len(xi)) if xi[i] >= 0 or wi[i] >= threshold]

    # Filter xi and wi
    xi_filtered = xi[valid_indices]
    wi_filtered = wi[valid_indices]

    return xi_filtered, wi_filtered

def get_dMdt_1d(t, moments_normal, x_max, GQMOM, GQMOM_method, 
                moments_norm_factor, n_add, nu, 
                COLEVAL, CORR_BETA, G, alpha_prim, EFFEVAL, 
                SIZEEVAL, X_SEL, Y_SEL, 
                V1_mean, pl_P1, pl_P2, BREAKRVAL, 
                v, q, BREAKFVAL, type_flag):
    dMdt = np.zeros(moments_normal.shape)
    dMdt_norm = np.zeros(moments_normal.shape)
    # Check if moments are unrealizable
    if moments_normal[0] <= 0:
        raise ValueError("Wheeler: Moments are NOT realizable (moment[0] <= 0.0).")
    
    m = len(moments_normal)
    n = m // 2  # Number of xi based on available moments
    # moments_normal = np.array([moments[k] / x_max**k for k in range(m)])
    if not GQMOM:
        xi, wi = qmom.calc_qmom_nodes_weights(moments_normal, n, adaptive=False)
    else:
        xi, wi = qmom.calc_gqmom_nodes_weights(moments_normal, n, n_add, GQMOM_method, nu)
        xi, wi = filter_negative_nodes(xi, wi)
        n += n_add
    
    xi *= x_max
    ## 因为PBE中的计算考虑了零点，所以这里对于PBM使用的时候坐标会发生一定偏移
    V = np.zeros(n+1)
    R = np.zeros(n+1)
    V[1:] = xi
    R[1:] = (V[1:]*3/(4*math.pi))**(1/3)

    F_M_tem = kernel_agg.calc_F_M_1D(n+2, COLEVAL, CORR_BETA, G, R, 
                                 alpha_prim, EFFEVAL, SIZEEVAL, X_SEL, Y_SEL)
    F_M = F_M_tem[1:,1:]
    B_R_tem = np.zeros(n+1)
    B_F_intxk = np.zeros((m, n))
    B_R_tem = kernel_break.breakage_rate_1d(B_R_tem, V, V1_mean, pl_P1, pl_P2, G, BREAKRVAL)
    B_R = B_R_tem[1:]

    xs1 = np.array([-9.681602395076260859e-01,
                -8.360311073266357695e-01,
                -6.133714327005903577e-01,
                -3.242534234038089158e-01,
                0.000000000000000000e+00,
                3.242534234038089158e-01,
                6.133714327005903577e-01,
                8.360311073266357695e-01,
                9.681602395076260859e-01,])
    ws1 = np.array([8.127438836157471758e-02,
                1.806481606948571184e-01,
                2.606106964029356599e-01,
                3.123470770400028074e-01,
                3.302393550012596712e-01,
                3.123470770400028074e-01,
                2.606106964029356599e-01,
                1.806481606948571184e-01,
                8.127438836157471758e-02])
    for k in range(m):
        for i in range(n):
            argsk = (V[i+1],v,q,BREAKFVAL,k)
            func = kernel_break.breakage_func_1d_xk
            B_F_intxk[k, i] = kernel_break.gauss_legendre(func, 0.0, V[i+1], xs1, ws1, args=argsk)
    
    for k in range(m):
        dMdt_agg_ij = 0.0
        dMdt_break_i = 0.0
        for i in range(n):
            for j in range(n):
                dMdt_agg_ij += 0.5 * wi[i]*wi[j]*F_M[i, j]*((xi[i]+xi[j])**k-xi[i]**k-xi[j]**k)
            dMdt_break_i += wi[i] * B_R[i] * B_F_intxk[k, i] - wi[i] * xi[i]**k * B_R[i]
        dMdt[k] = dMdt_agg_ij + dMdt_break_i
    dMdt_norm = dMdt / moments_norm_factor
    
    return dMdt_norm


        