import numpy as np
import scipy.stats as stats
from numba import jit
from .jit_pbm_chyqmom import compute_central_moments_2d, compute_central_moments_1d

### This Function is from the open source library PyQBMMlib (https://github.com/sbryngelson/PyQBMMlib/tree/master)
### Note: There is actually a problem with the adaptive function. 
### It should dynamically adjust the recursive coefficient and the number of nodes/weights 
### based on various judgments. But judgments were tested here but adjustment was actually not implemented.
@jit(nopython=True)
def calc_qmom_nodes_weights(moments, n, adaptive, use_central):
    """
    Compute nodes (ξ_i) and weights (w_i) using QMOM with the adaptive Wheeler algorithm.

    Parameters:
        moments (list or numpy.ndarray): Array of moments [M0, M1, ..., M2n-1].
        n (int): Number of quadrature nodes.
        adaptive (bool): If True, use adaptive criteria to refine the solution.
        use_central (bool): If True, use central moments.

    Returns:
        tuple: (x, w, n), where `x` are the nodes (ξ_i), `w` are the weights (w_i), and `n` is the number of nodes.
    """
    # Set adaptivity thresholds
    rmax = 1e-8  # weight ratio threshold
    cutoff = 0   # stability cutoff for sigma diagonal

    if n == 1 or (adaptive and moments[0] < rmax):
        # Trivial case: one node only.
        w = moments[0]
        x = moments[1] / moments[0]
        return np.array([x]), np.array([w]), n

    if use_central:
        # Convert raw moments to central moments and get shift bx.
        bx, central_moments = compute_central_moments_1d(moments)
        mom = central_moments
    else:
        mom = moments

    # Compute recurrence coefficients via Wheeler algorithm.
    a, b, n = calc_qmom_recurrence(mom, n, adaptive, cutoff)
    # Obtain quadrature nodes and weights from the Jacobi matrix.
    x, w = recurrence_jacobi_nodes_weights(mom, a, b)

    if use_central:
        x += bx  # shift nodes back to original scale
        w *= moments[0]  # scale weights accordingly

    return x, w, n

@jit(nopython=True)
def calc_qmom_recurrence(moments, n, adaptive, cutoff):
    """
    Calculate recurrence coefficients for QMOM using the Wheeler algorithm.
    
    Parameters:
        moments (list or numpy.ndarray): Array of moments [M0, M1, ..., M2n-1].
        n (int): Number of quadrature nodes.
        adaptive (bool): If True, use adaptive criteria to refine the solution.
        cutoff (float): Minimum diagonal element to consider for stability.

    Returns:
        tuple: (a, b, n), where `a` and `b` are the recurrence coefficients, and `n` is the number of nodes.
    """
    # Initialize modified moments (σ) and recurrence coefficients (a, b)
    a = np.zeros(n)
    b = np.zeros(n)
    sigma = np.zeros((2*n + 1, 2*n + 1))

    # Populate the first row of sigma with moments.
    sigma[1, 1:] = moments

    a[0] = moments[1] / moments[0]
    b[0] = 0

    # Loop to compute higher-order sigma and extract recurrence coefficients.
    for k in range(2, n + 1):
        for l in range(k, 2*n - k + 2):
            sigma[k, l] = sigma[k - 1, l + 1] - a[k - 2]*sigma[k - 1, l] - b[k - 2]*sigma[k - 2, l]
        a[k - 1] = sigma[k, k + 1]/sigma[k, k] - sigma[k - 1, k]/sigma[k - 1, k - 1]
        b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Adaptive adjustment of the number of nodes
    if adaptive:
        # Check and reduce node count if any sigma[k,k] falls below cutoff.
        for k in range(n, 1, -1):
            if sigma[k, k] <= cutoff:
                n = k - 1
                if n == 1:
                    w = moments[0]
                    x = moments[1] / moments[0]
                    return np.array([x]), np.array([w]), n
        # Recompute recurrence coefficients for reduced n.
        a = np.zeros(n)
        b = np.zeros(n)
        sigma = np.zeros((2*n + 1, 2*n + 1))
        sigma[1, 1:] = moments
        a[0] = moments[1] / moments[0]
        b[0] = 0
        for k in range(2, n + 1):
            for l in range(k, 2*n - k + 2):
                sigma[k, l] = sigma[k - 1, l + 1] - a[k - 2]*sigma[k - 1, l] - b[k - 2]*sigma[k - 2, l]
            a[k - 1] = sigma[k, k + 1]/sigma[k, k] - sigma[k - 1, k]/sigma[k - 1, k - 1]
            b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Validate moments realizability.
    if b.min() < 0:
        raise ValueError("Moments in Wheeler_moments are not realizable!")
    return a, b, n

@jit(nopython=True)
def recurrence_jacobi_nodes_weights(moments, a, b):
    """
    Construct Jacobi matrix and solve for eigenvalues and eigenvectors to get nodes and weights.

    Parameters:
        moments (list or numpy.ndarray): Array of moments [M0, M1, ..., M2n-1].
        a (list or numpy.ndarray): Recurrence coefficient a.
        b (list or numpy.ndarray): Recurrence coefficient b.

    Returns:
        tuple: (x, w), where `x` are the nodes (ξ_i) and `w` are the weights (w_i).
    """
    # Compute square roots for off-diagonal entries.
    sqrt_b = np.sqrt(b[1:])
    jacobi = np.diag(a) + np.diag(sqrt_b, -1) + np.diag(sqrt_b, 1)
    # Eigen-decomposition of the Jacobi matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(jacobi)
    idx = eigenvalues.argsort()
    x = eigenvalues[idx].real  # Sorted nodes
    eigenvectors = eigenvectors[:, idx].real
    # First row of eigenvectors gives the squared weights.
    w = moments[0] * eigenvectors[0, :]**2
    return x, w

@jit(nopython=True)
def calc_gqmom_nodes_weights(moments, n, n_add, method, nu, adaptive, cutoff):
    """
    Compute nodes (ξ_i) and weights (w_i) using Generalized QMOM (GQMOM).

    Parameters:
        moments (list or numpy.ndarray): Array of moments [M0, M1, ..., M2n-1].
        n (int): Number of quadrature nodes.
        n_add (int): Number of additional quadrature nodes.
        method (str): Method for GQMOM ("gaussian", "gamma", "lognormal", "beta").
        nu (float): Exponent for the correction.
        adaptive (bool): If True, use adaptive criteria to refine the solution.
        cutoff (float): Minimum diagonal element to consider for stability.

    Returns:
        tuple: (x, w, n), where `x` are the nodes (ξ_i), `w` are the weights (w_i), and `n` is the number of nodes.
    """
    # Compute regular recurrence coefficients.
    a_reg, b_reg, n = calc_qmom_recurrence(moments, n, adaptive, cutoff)
    
    if method == "gaussian":
        a, b = calc_gqmom_recurrence_real(a_reg, b_reg, n_add, nu)
    elif method == "gamma":
        a, b = calc_gqmom_recurrence_realplus(moments, a_reg, b_reg, n_add, ndf_type="gamma")
    elif method == "lognormal":
        a, b = calc_gqmom_recurrence_realplus(moments, a_reg, b_reg, n_add, ndf_type="lognormal")
    elif method == "beta":
        a, b = calc_gqmom_recurrence_beta(a_reg, b_reg, n_add)
    else:
        raise ValueError("The input method for GQMOM is not available. Supported methods are: gaussian, gamma, lognormal, and beta.")
    
    # Compute nodes and weights for the extended recurrence.
    x, w = recurrence_jacobi_nodes_weights(moments, a, b)
    return x, w, n

@jit(nopython=True)
def calc_gqmom_recurrence_real(a_reg, b_reg, n_add, nu):
    """
    Correct recurrence coefficients a and b for (Gaussian) Generalized QMOM.

    Parameters:
        a_reg (list or numpy.ndarray): List of regular recurrence coefficient a.
        b_reg (list or numpy.ndarray): List of regular recurrence coefficient b.
        n_add (int): Number of additional quadrature nodes.
        nu (float): Exponent for the correction.

    Returns:
        tuple: Updated a and b lists.
    """
    n_reg = len(a_reg)
    n_nodes = n_reg + n_add

    if n_add <= 0:
        return a_reg, b_reg  # No extra nodes needed

    an = sum(a_reg) / n_reg  # Average of regular a coefficients
    a = list(a_reg) + [0]*n_add
    b = list(b_reg) + [0]*n_add

    # Extend coefficients by assigning average and scaled b.
    for i in range(n_reg, n_nodes):
        a[i] = an
        b[i] = b[n_reg-1] * (float(i-1)/float(n_reg-1))**nu

    b[n_nodes-1] = b[n_reg-1] * (float(n_nodes-1)/float(n_reg-1))**nu
    return np.array(a), np.array(b)

@jit(nopython=True)
def calc_gqmom_recurrence_beta(a_reg, b_reg, n_add):
    """
    Correct recurrence coefficients a and b for (Beta) Generalized QMOM.

    Parameters:
        a_reg (list or numpy.ndarray): List of regular recurrence coefficient a.
        b_reg (list or numpy.ndarray): List of regular recurrence coefficient b.
        n_add (int): Number of additional quadrature nodes.

    Returns:
        tuple: Updated a and b lists.
    """
    n_reg = len(a_reg)
    n_max_nodes = n_reg + n_add
    if n_add <= 0:
        return a_reg, b_reg

    a = list(a_reg) + [0]*n_add
    b = list(b_reg) + [0]*n_add

    zetas = calc_zetas(a, b, n_reg, n_max_nodes)
    p = np.zeros_like(zetas)
    # Compute intermediate p values
    for i in range(1, 2*n_reg):
        p[i] = zetas[i] / (1 - p[i-1])
    
    alpha_coeff = (1.0 - p[1] - 2*p[2] + p[1]*p[2]) / p[2]
    beta_coeff = (p[1] - p[2] - p[1]*p[2]) / p[2]

    # Extend p and update zetas for extra nodes.
    pJ_2n_1 = (beta_coeff + n_reg) / (2*n_reg + alpha_coeff + beta_coeff)
    pJ_2n = (n_reg - 1) / (2*n_reg - 2 + 1 + alpha_coeff + beta_coeff)
    for i in range(n_reg, n_max_nodes):
        pJ_2i_1 = (beta_coeff + i) / (2*n_reg)
        pJ_2i = i / (2*i + 1 + alpha_coeff + beta_coeff)
        if i != n_reg:
            if p[2*n_reg-1] <= pJ_2n_1 or pJ_2n_1 >= pJ_2i_1:
                p[2*i-1] = p[2*n_reg-1] * pJ_2i_1 / pJ_2n_1
            else:
                p[2*i-1] = (p[2*n_reg-1]*(1-pJ_2i_1)+pJ_2i_1-pJ_2n_1)/(1-pJ_2n_1)
        if p[2*n_reg-2] <= pJ_2n or pJ_2n >= pJ_2i:
            p[2*i] = p[2*n_reg-2] * pJ_2i / pJ_2n
        else:
            p[2*i] = (p[2*n_reg-2]*(1-pJ_2i)+pJ_2i-pJ_2n)/(1-pJ_2n)
        if i != n_reg:
            zetas[2*i-1] = p[2*i-1]*(1.0-p[2*i-3])
        zetas[2*i] = p[2*i]*(1.0-p[2*i-1])
    # Update a and b using the new zetas.
    a[0] = zetas[1]
    for i in range(1, n_max_nodes):
        a[i] = zetas[2*i] + zetas[2*i+1]
    for i in range(1, n_max_nodes):
        b[i] = zetas[2*i-1]*zetas[2*i]
    
    return np.array(a), np.array(b)

@jit(nopython=True)
def calc_gqmom_recurrence_realplus(moments, a_reg, b_reg, n_add, ndf_type):
    """
    Correct recurrence coefficients for (Gamma/Lognormal) Generalized QMOM using R+ moments.
    
    Parameters:
        moments (list or numpy.ndarray): List of input moments [M0, M1, M2, ...].
        a_reg (list or numpy.ndarray): Recurrence coefficient a to be updated.
        b_reg (list or numpy.ndarray): Recurrence coefficient b to be updated.
        n_add (int): Number of additional quadrature nodes.
        ndf_type (str): Type of node distribution ("gamma" or "lognormal").
    
    Returns:
        tuple: Updated a, b.
    """
    n_reg = len(a_reg)
    n_max_nodes = n_reg + n_add
    if n_add <= 0:
        return a_reg, b_reg

    a = list(a_reg) + [0]*n_add
    b = list(b_reg) + [0]*n_add
    zetas = calc_zetas(a, b, n_reg, n_max_nodes)
    
    if ndf_type == "gamma":
        m1_sqr = moments[1] ** 2
        alpha_coeff = m1_sqr / (moments[0]*moments[2] - m1_sqr) - 1.0
        for i in range(n_reg, n_max_nodes):
            if i != n_reg:
                zetas[2*i-1] = (i+alpha_coeff)*zetas[2*n_reg-1] / (n_reg+alpha_coeff)
            zetas[2*i] = i*zetas[2*n_reg-2] / (n_reg-1)
        zetas[2*n_max_nodes-1] = (n_max_nodes+alpha_coeff)*zetas[2*n_reg-1] / (n_reg+alpha_coeff)
        
    elif ndf_type == "lognormal":
        eta = np.sqrt(moments[0]*moments[2] / (moments[1]**2))
        for i in range(n_reg, n_max_nodes):
            if i != n_reg:
                zetas[2*i-1] = eta ** (4*(i - n_reg)) * zetas[2*n_reg-1]
            zetas[2*i] = eta ** (2*(i+1 - n_reg)) * ((eta ** (2*i) - 1.0) / (eta ** (2*n_reg-2) - 1.0)) * zetas[2*n_reg-2]
        zetas[2*n_max_nodes-1] = eta ** (4*(n_max_nodes-n_reg)) * zetas[2*n_reg-1]
    
    a[0] = zetas[1]
    for i in range(1, n_max_nodes):
        a[i] = zetas[2*i] + zetas[2*i+1]
    for i in range(1, n_max_nodes):
        b[i] = zetas[2*i-1] * zetas[2*i]
    
    return np.array(a), np.array(b)

@jit(nopython=True)
def calc_zetas(a, b, n_reg, n_max_nodes):
    """
    Set regular zetas array from constraint: a_i=zeta_2i+zeta_(2i+1),b_i=zeta_(2i-1)*zeta_2i
    Currently, zetas[0] is not being used, which means there is redundancy in the array length. 
    If this needs to be modified, all indices in the calculation process must be adjusted accordingly.
    
    The paper mentions that i should go up to n. 
    However, in the program, the indices of a and b start from zero. 
    Calculating  zetas[2n] requires a[n] to be known, but at this point, 
    we only have regular, a[n-1]. Therefore, i is limited to n-1 here. 
    This results in the generation of zetas[2n-2] and zetas[2n-1]. 
    The corresponding Laguerre polynomials are actually zetas_L[2i-1] for i=n and zetas_L[2i] for i=n-1! 
    This necessitates corresponding adjustments in the subsequent calculations for the extended nodes.
    
    Returns:
        numpy.ndarray: zetas arrays.
    """
    zetas = np.zeros(2 * n_max_nodes)
    if a[0] != 0:
        zetas[1] = a[0]
    else:
        zetas[1] = np.sqrt(b[1])
    for i in range(1, n_reg):
        # Compute zetas pair for each i.
        zetas[2*i] = b[i] / zetas[2*i-1]
        zetas[2*i+1] = a[i] - zetas[2*i]
    return zetas

@jit(nopython=True)
def vander_rybicki(x, q):
    """
    Solve the Vandermonde linear system using the Rybicki algorithm.

    Parameters:
        x (numpy.ndarray): Node vector (1D array).
        q (numpy.ndarray): Right-hand side vector (1D array).

    Returns:
        numpy.ndarray: Solution vector.
    """
    n = len(x)
    w = np.zeros(n)
    c = np.zeros(n)

    if n == 1:
        w[0] = q[0]
        return w

    # Compute the coefficients of the characteristic polynomial
    c[n-1] = -x[0]
    for i in range(1, n):
        xx = -x[i]
        for j in range(n-2, n-i-2, -1):
            c[j] += xx * c[j+1]
        c[n-1] += xx

    # Solve for each component
    for i in range(n):
        xx = x[i]
        t = 1.0
        b = 1.0
        s = q[n-1]

        for k in range(n-1, 0, -1):
            b = c[k] + xx * b
            s += q[k-1] * b
            t = xx * t + b

        w[i] = s / t

    return w

@jit(nopython=True)
def conditional_mom_sys_solve(M_matrix, u, R_diag):
    """
    Solve the conditional moment system using the Rybicki algorithm.

    Parameters:
        M_matrix (numpy.ndarray): Moment matrix.
        u (numpy.ndarray): Node vector.
        R_diag (numpy.ndarray): Diagonal of the weight matrix.

    Returns:
        numpy.ndarray: Solution matrix.
    """
    N1, N2 = M_matrix.shape
    if len(u) != N1 or len(R_diag) != N1:
        raise ValueError("u and R_diag must match the number of rows in M_matrix")

    R1_matrix = np.zeros((N1, N2))
    ## for debug
    M_prime = M_matrix
    # R1_debug = np.zeros((N1, N2))
    # V = np.vander(u, increasing=True)
    # V = np.dot(np.transpose(V), np.diag(R_diag))
    # R1_debug = np.linalg.solve(V, M_matrix)

    for col in range(N2):
        R1_matrix[:, col] = vander_rybicki(u, M_prime[:, col]) / R_diag

    return R1_matrix
    
@jit(nopython=True)
def calc_cqmom_2d(moments, n, indices, use_central=True):
    """
    Compute nodes and weights for 2D Conditional QMOM (CQMOM).

    Parameters:
        moments (list or numpy.ndarray): Array of moments.
        n (int): Number of quadrature nodes.
        indices (list or numpy.ndarray): Moment indices.
        use_central (bool): If True, use central moments.

    Returns:
        tuple: (abscissas, weights, n), where `abscissas` are the nodes, `weights` are the weights, and `n` is the number of nodes.
    """
    m = 2*n
    mom00, bx, by, central_moments = compute_central_moments_2d(moments, indices)
    
    M1 = central_moments[:m] if use_central else moments[:m]
    x1, w1, n = calc_qmom_nodes_weights(M1, n, True, use_central)
    
    M_matrix = np.zeros((n, m-1))
    for i in range(n):
        if use_central:
            M_matrix[i] = central_moments[m+i*(m-1): m+(i+1)*(m-1)]
        else:
            M_matrix[i] = moments[m+i*(m-1): m+(i+1)*(m-1)]
    
    R1_matrix = conditional_mom_sys_solve(M_matrix, x1, w1)
    ones_column = np.ones((R1_matrix.shape[0], 1))
    R1_matrix = np.hstack((ones_column, R1_matrix))
    
    x2 = np.zeros((n, n))
    w2 = np.zeros((n, n))
    for i in range(n):
        x2_tem, w2_tem, n_tem = calc_qmom_nodes_weights(R1_matrix[i], n, True, use_central)
        x2[i, :n_tem], w2[i, :n_tem] = x2_tem, w2_tem
    
    if use_central:
        x1 += bx  # Adjust nodes by x-shift from central moments.
        x2 += by  # Adjust nodes by y-shift.
        w1 *= mom00  # Scale weights with zeroth moment.
        
    abscissas = np.zeros((2, n*n), dtype=np.float64)
    weights = np.zeros(n*n, dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(n):
            abscissas[0, idx] = x1[i]
            abscissas[1, idx] = x2[i, j]
            weights[idx] = w1[i] * w2[i, j]
            idx += 1
    
    return abscissas, weights, n

@jit(nopython=True)
def quadrature_2d(x1, w1, x2, w2, moment_index):
    """
    Perform 2D quadrature to compute the moment.

    Parameters:
        x1 (numpy.ndarray): Nodes for the first dimension.
        w1 (numpy.ndarray): Weights for the first dimension.
        x2 (numpy.ndarray): Nodes for the second dimension.
        w2 (numpy.ndarray): Weights for the second dimension.
        moment_index (list or numpy.ndarray): Moment index.

    Returns:
        float: Computed moment.
    """
    mu = 0.0
    N1, N2 = x2.shape
    # Sum over all 2D nodes weighted by powers specified in moment_index.
    for i in range(N1):
        for j in range(N2):
            mu += w1[i] * w2[i, j] * (x1[i] ** moment_index[0]) * (x2[i,j] ** moment_index[1])
    return mu








