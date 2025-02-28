import numpy as np
import math
from numba import jit
import optframework.utils.func.jit_pbm_qmom as qmom
import optframework.utils.func.jit_kernel_agg as kernel_agg
import optframework.utils.func.jit_kernel_break as kernel_break
import optframework.utils.func.jit_pbm_chyqmom as chyqmom
from scipy.optimize import root
from scipy.optimize import least_squares
from numba import jit

def filter_negative_nodes(xi, wi, threshold=1e-7):
    """
    Filter out negative nodes with small weights or raise an error if negative nodes have significant weights.
    
    Parameters:
        xi (array-like): Array of nodes (abscissas).
        wi (array-like): Array of weights corresponding to the nodes.
        threshold (float): Weight threshold below which negative nodes can be removed.

    Returns:
        tuple: Filtered (xi, wi) arrays with negative nodes removed if their weights are below threshold.
    
    Raises:
        ValueError: If a negative node has a weight exceeding the threshold.
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

def hyqmom_newton_correction(xi, wi, moments, method="lm"):
    """
    Correct QMOM abscissas and weights through Newton iteration to satisfy moment equations,
    particularly addressing negative abscissas.
    
    Parameters:
        xi (array-like): Quadrature abscissas (may contain negative values).
        wi (array-like): Quadrature weights.
        moments (array-like): Target moments [M0, M1, M2, ...].
        method (str): Optimization method for least squares solver.
    
    Returns:
        tuple: Corrected (xi, wi) arrays that better satisfy the moment equations.
    """
    SMALL = 1e-8
    # Handle negative abscissas
    negative_idx = xi < 0
    if np.any(negative_idx):
        # Replace negative values with a small positive value
        xi[negative_idx] = SMALL
    else:
        # No correction needed if no negative values
        return xi, wi
    
    xi_free_idx = ~negative_idx  # Indices of abscissas to optimize
    xi_free = xi[xi_free_idx]
    wi_free = wi
    
    # Define residual function for optimization
    def residuals(variables):
        """
        Calculate residuals between current moments and target moments
        """
        xi_new = xi.copy()
        xi_new[xi_free_idx] = variables[:len(xi_free)]
        wi_new = variables[len(xi_free):]
    
        # Focus on first three moments for stability
        return np.array([np.sum(wi_new * (xi_new ** k)) - moments[k] for k in range(3)])
    
    # Initial guess combining free abscissas and weights
    x0 = np.concatenate([xi_free, wi_free])
    
    # Solve using least squares optimization
    sol = least_squares(residuals, x0, method="trf", ftol=1e-3)
    if not sol.success:
        raise ValueError(f"Optimization failed to converge: {sol.message}")

    # Extract optimized values
    xi[xi_free_idx] = sol.x[:len(xi_free)]
    wi = sol.x[len(xi_free):]

    return xi, wi
    
@jit(nopython=True)
def get_dMdt_1d(t, moments, x_max, GQMOM, GQMOM_method, 
                moments_norm_factor, n_add, nu, 
                COLEVAL, CORR_BETA, G, alpha_prim, EFFEVAL, 
                SIZEEVAL, V_unit, X_SEL, Y_SEL, 
                V1_mean, pl_P1, pl_P2, BREAKRVAL, 
                v, q, BREAKFVAL, type_flag):
    """
    Calculate the moment derivatives for 1D population balance equations, handling 
    agglomeration and/or breakage processes.
    
    Parameters:
        t (float): Current time.
        moments (array): Current moments.
        x_max (float): Maximum coordinate value for scaling.
        GQMOM (bool): Flag to use Generalized QMOM instead of standard QMOM.
        GQMOM_method (str): Method for GQMOM calculations ("gaussian").
        moments_norm_factor (array): Normalization factors for moments.
        n_add (int): Number of additional nodes for GQMOM.
        nu (float): Exponent for the correction in gaussian-GQMOM.
        COLEVAL (int): Case for collision kernel calculation.
        CORR_BETA (float): Correction term for collision frequency.
        G (float): Shear rate [1/s].
        alpha_prim (float): Primary particle interaction parameter.
        EFFEVAL (int): Case for collision efficiency.
        SIZEEVAL (int): Case for size dependency.
        V_unit (float): Unit volume used for concentration calculations.
        X_SEL (float): Size dependency parameter.
        Y_SEL (float): Size dependency parameter.
        V1_mean (float): Mean volume for breakage model.
        pl_P1 (float): First parameter in power law for breakage rate.
        pl_P2 (float): Second parameter in power law for breakage rate.
        BREAKRVAL (int): Breakage rate model selector.
        v (float): Number of fragments in product function of power law.
        q (float): Parameter describing the breakage type in product function.
        BREAKFVAL (int): Breakage fragment distribution model selector.
        type_flag (str): Process type: "agglomeration", "breakage", or "mix".
    
    Returns:
        array: (Normalized) moment derivatives (dM/dt).
    
    Raises:
        ValueError: If moments are not realizable.
    """
    dMdt = np.zeros(moments.shape)
    dMdt_norm = np.zeros(moments.shape)
    # Check if moments are realizable
    if moments[0] <= 0:
        raise ValueError("Wheeler: Moments are NOT realizable (moment[0] <= 0.0).")
    
    m = len(moments)
    n = m // 2  # Number of xi based on available moments
    adaptive = False
    use_central=False
    
    # Calculate quadrature nodes and weights
    if not GQMOM:
        xi, wi, n = qmom.calc_qmom_nodes_weights(moments, n, adaptive, use_central)
    else:
        xi, wi, n = qmom.calc_gqmom_nodes_weights(moments, n, n_add, GQMOM_method, nu, adaptive, use_central)
        # xi, wi = filter_negative_nodes(xi, wi)
        n += n_add

    # Check for negative nodes (debugging)
    if np.any(xi<0):
        print(f"t = {t}")
        print(f"xi = {xi}, wi = {wi}")
        
    # Scale nodes back to original domain
    xi *= x_max
    
    # Prepare volume and radius arrays (with zero padding at index 0)
    V = np.zeros(n+1)
    R = np.zeros(n+1)
    V[1:] = xi
    R[1:] = (V[1:]*3/(4*math.pi))**(1/3)  # Convert volume to radius
    
    # Calculate agglomeration terms if needed
    if type_flag == "agglomeration" or type_flag == "mix":
        # Get agglomeration frequency matrix
        F_M_tem = kernel_agg.calc_F_M_1D(n+2, COLEVAL, CORR_BETA, G, R, 
                                     alpha_prim, EFFEVAL, SIZEEVAL, X_SEL, Y_SEL)
        F_M = F_M_tem[1:,1:] / V_unit
        
    # Calculate breakage terms if needed
    if type_flag == "breakage" or type_flag == "mix":
        B_F_intxk = np.zeros((m, n))
        # Calculate breakage rates for each node
        B_R = kernel_break.breakage_rate_1d(V[1:], V1_mean, pl_P1, pl_P2, G, BREAKRVAL)

        # Gauss-Legendre quadrature points and weights for integration
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
                    
        # Calculate breakage daughter distribution integral for each moment and node
        for k in range(m):
            for i in range(n):
                argsk = (V[i+1],v,q,BREAKFVAL,k)
                func = kernel_break.breakage_func_1d_xk
                B_F_intxk[k, i] = kernel_break.gauss_legendre(func, 0.0, V[i+1], xs1, ws1, args=argsk)
    
    # Calculate moment derivatives for each moment order
    for k in range(m):
        dMdt_agg_ij = 0.0
        dMdt_break_i = 0.0
        for i in range(n):
            # Agglomeration contribution: birth - death
            if type_flag == "agglomeration" or type_flag == "mix":
                for j in range(n):
                    dMdt_agg_ij += 0.5 * wi[i]*wi[j]*F_M[i, j]*((xi[i]+xi[j])**k-xi[i]**k-xi[j]**k)
            
            # Breakage contribution: birth - death
            if type_flag == "breakage" or type_flag == "mix":
                dMdt_break_i += wi[i] * B_R[i] * B_F_intxk[k, i] - wi[i] * xi[i]**k * B_R[i]
                
        # Total derivative for moment k
        dMdt[k] = dMdt_agg_ij + dMdt_break_i
        
    # Normalize derivatives
    dMdt_norm = dMdt / moments_norm_factor
    
    return dMdt_norm

@jit(nopython=True)
def get_dMdt_2d(t, moments, n, indices, COLEVAL, CORR_BETA, G, alpha_prim, EFFEVAL, 
                SIZEEVAL, V_unit, X_SEL, Y_SEL, 
                V1_mean, V3_mean, pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, 
                v, q, BREAKFVAL, type_flag):
    """
    Calculate the moment derivatives for 2D population balance equations, handling
    agglomeration and/or breakage processes with bivariate distributions.
    
    Parameters:
        t (float): Current time.
        moments (array): Current moments of the bivariate distribution.
        n (int): Number of quadrature nodes per dimension.
        indices (array): 2D array of moment indices (i,j) corresponding to moments.
        COLEVAL (int): Case for collision kernel calculation.
        CORR_BETA (float): Correction term for collision frequency.
        G (float): Shear rate [1/s].
        alpha_prim (array): Primary particle interaction parameters array.
        EFFEVAL (int): Case for collision efficiency.
        SIZEEVAL (int): Case for size dependency.
        V_unit (float): Unit volume used for concentration calculations.
        X_SEL (float): Size dependency parameter.
        Y_SEL (float): Size dependency parameter.
        V1_mean (float): Mean volume of component 1.
        V3_mean (float): Mean volume of component 3 (magnetic component).
        pl_P1 (float): First parameter in power law for breakage rate.
        pl_P2 (float): Second parameter in power law for breakage rate.
        pl_P3 (float): Third parameter in power law for breakage rate.
        pl_P4 (float): Fourth parameter in power law for breakage rate.
        BREAKRVAL (int): Breakage rate model selector.
        v (float): Number of fragments in product function of power law.
        q (float): Parameter describing the breakage type in product function.
        BREAKFVAL (int): Breakage fragment distribution model selector.
        type_flag (str): Process type: "agglomeration", "breakage", or "mix".
    
    Returns:
        array: Moment derivatives (dM/dt).
    
    Raises:
        ValueError: If moments are not realizable.
    """
    n0 = n
    dMdt = np.zeros(moments.shape)
    # Check if moments are realizable
    if moments[0] <= 0:
        raise ValueError("Wheeler: Moments are NOT realizable (moment[0] <= 0.0).")

    # Calculate conditional quadrature for 2D distribution
    xi, wi, n = qmom.calc_cqmom_2d(moments, n, indices, use_central=True)
    # for Debug
    # moments_cqmom = np.zeros_like(moments)
    # for idx, _ in enumerate(moments_cqmom):
    #     indice = indices[idx]
    #     moments_cqmom[idx] = chyqmom.quadrature_2d(wi, xi, indice)
    # print(np.mean(abs(moments_cqmom-moments)/moments))

    # Print warning if number of nodes was reduced
    if n0 > n:
        print(f"Warning: At t = {t}, the moments are NOT realizable, abscissas reduced to {n}.")
    
    # Initialize volume arrays
    V = np.ones((n, n))
    V1 = xi[0,:]  # First component volumes
    V3 = xi[1,:]  # Second component volumes
    V_flat = np.ones(n*n)
    
    # Initialize radius and composition arrays with padding
    R = np.ones((n+1, n+1))
    X1 = np.ones((n+1, n+1))  # Composition fraction of first component
    X3 = np.ones((n+1, n+1))  # Composition fraction of second component
    
    # Calculate volume, radius, and composition for each node
    for i in range(n):
        for j in range(n):
            V[i,j] = V1[i*n+j] + V3[i*n+j]  # Total volume
            V_flat[i*n+j] = V[i,j]
            X1[i+1,j+1] = V1[i*n+j] / V[i,j]  # Composition fraction
            X3[i+1,j+1] = V3[i*n+j] / V[i,j]
    
    # Debug output for negative volumes
    if np.any(V<0):
        print(f"t = {t}")
        print(f"xi = {xi}, wi = {wi}")
    
    # Convert volume to radius
    R[1:, 1:] = (V*3/(4*math.pi))**(1/3)
    
    # Calculate agglomeration terms if needed
    if type_flag == "agglomeration" or type_flag == "mix":
        # Get agglomeration frequency matrix for 2D
        F_M_tem = kernel_agg.calc_F_M_2D(n+2, COLEVAL, CORR_BETA, G, R, X1, X3, EFFEVAL, 
                                     alpha_prim, SIZEEVAL, X_SEL, Y_SEL)
        F_M = F_M_tem[1:,1:,1:,1:] / V_unit
        
    # Calculate breakage terms if needed
    if type_flag == "breakage" or type_flag == "mix":
        # Initialize integral container for daughter distribution
        B_F_intxk = np.zeros((2*n, n, 2*n, n))
        
        # Calculate breakage rates for each node
        B_R_flat = kernel_break.breakage_rate_2d_flat(V_flat, V1, V3, V1_mean, V3_mean, G,
                                                     pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL)
        B_R = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                # if V1[i] < eta or V3[j] < eta:
                #     continue
                B_R[i,j] = B_R_flat[i*n+j]

        # Gauss-Legendre quadrature points and weights for integration
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
        xs3 = xs1
        ws3 = ws1
        
        # Calculate breakage daughter distribution integral for each moment and node
        for idx, _ in enumerate(dMdt):
            k = indices[idx,0]  # First component moment order
            l = indices[idx,1]  # Second component moment order
            for i in range(n):
                for j in range(n):
                    # if V1[i*n+j] < eta or V3[i*n+j] < eta:
                    #     continue
                    # Calculate bivariate daughter distribution integral
                    argsk = (V1[i*n+j],V3[i*n+j],v,q,BREAKFVAL,k,l)
                    func = kernel_break.breakage_func_2d_x1kx3l
                    B_F_intxk[k, i, l, j] = kernel_break.dblgauss_legendre(
                        func, 0.0, V1[i*n+j], 0.0, V3[i*n+j], 
                        xs1, ws1, xs3, ws3, args=argsk
                    )
                    # argsk = (V1[i*n+j],V3[i*n+j],v,q,BREAKFVAL,1,eta)
                    # func1 = kernel_break.breakage_func_2d_x1k_trunc
                    # func2 = kernel_break.breakage_func_2d_x3k_trunc
                    # norm_fac1 = kernel_break.dblgauss_legendre(func1, eta, V1[i*n+j], eta, V3[i*n+j], xs1, ws1, xs3,ws3,args=argsk)
                    # norm_fac2 = kernel_break.dblgauss_legendre(func2, eta, V1[i*n+j], eta, V3[i*n+j], xs1, ws1, xs3,ws3,args=argsk)
                    # argsk_trunc = (V1[i*n+j],V3[i*n+j],v,q,BREAKFVAL,k,l,eta)
                    # func_norm = kernel_break.breakage_func_2d_trunc
                    # B_F_intxk_trunk = kernel_break.dblgauss_legendre(func_norm, eta, V1[i*n+j], eta, V3[i*n+j], xs1, ws1, xs3,ws3,args=argsk_trunc)
                    # B_F_intxk[k, i, l, j] = B_F_intxk_trunk / ((norm_fac1 + norm_fac2) / V[i,j])

    # Calculate moment derivatives for each tracked moment
    for idx, _ in enumerate(dMdt):
        k = indices[idx,0]  # First component moment order
        l = indices[idx,1]  # Second component moment order
        dMdt_agg_ijab = 0.0
        dMdt_break_ij = 0.0
        for i in range(n):
            for j in range(n):
                # Agglomeration contribution: birth - death
                if type_flag == "agglomeration" or type_flag == "mix":
                    for a in range(n):
                        for b in range(n):
                            dMdt_agg_ijab += 0.5 * wi[i*n+j]*wi[a*n+b]*F_M[i,j,a,b]*(
                                (xi[0,i*n+j]+xi[0,a*n+b])**k*(xi[1,i*n+j]+xi[1,a*n+b])**l
                                -xi[0,i*n+j]**k*xi[1,i*n+j]**l
                                -xi[0,a*n+b]**k*xi[1,a*n+b]**l)
                            
                # Breakage contribution: birth - death
                if type_flag == "breakage" or type_flag == "mix":
                    dMdt_break_ij += (wi[i*n+j] * B_R[i,j] * B_F_intxk[k,i,l,j] - wi[i*n+j]
                                      * xi[0,i*n+j]**k * xi[1,i*n+j]**l * B_R[i,j])
                    
        # Total derivative for moment (k,l)
        dMdt[idx] = dMdt_agg_ijab + dMdt_break_ij
    
    return dMdt

