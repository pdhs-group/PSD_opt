# -*- coding: utf-8 -*-
"""
Test script for QMOM method. The GQMOM algorithm comes from the paper
"The Generalized Quadrature Method of Moments"

@author: px2030
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt

def calc_gqmom_nodes_weights(moments, n_add, method="Gaussian", nu=1):
    n = len(moments) // 2
    
    # Check if moments are unrealizable
    if moments[0] <= 0:
        raise ValueError("Wheeler: Moments are NOT realizable (moment[0] <= 0.0).")
        
    # calculate regular recurrence coefficients
    a_reg, b_reg = calc_qmom_recurrence(moments, n)
    
    if method == "gaussian":
        a, b = calc_gqmom_recurrence_real(a_reg, b_reg, n_add, nu)
    elif method == "gamma":
        a, b = calc_gqmom_recurrence_realplus(moments, a_reg, b_reg, n_add, ndf_type="gamma")
    elif method == "lognormal":
        a, b = calc_gqmom_recurrence_realplus(moments, a_reg, b_reg, n_add, ndf_type="lognormal")
    elif method == "beta":
        a, b = calc_gqmom_recurrence_beta(a_reg, b_reg, n_add)
    else:
        raise ValueError("The input method for GQMOM is not available. \
                         \n Supported methods are: gaussian, gamma, lognormal, and beta.")
    
    return recurrence_jacobi_nodes_weights(a, b)

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
    # Number of regular quadrature nodes (length of a)
    n_reg = len(a_reg)

    # Total number of nodes
    n_nodes = n_reg + n_add

    # Check realizability: If no additional nodes are possible, return original Gauss coefficients
    if n_add <= 0:
        return a_reg, b_reg  # Use Gauss if no additional nodes are possible

    # Calculate the mean of the first n_reg a values
    an = sum(a_reg) / n_reg

    # Extend a and b to accommodate additional nodes
    a = list(a_reg) + [0] * n_add
    b = list(b_reg) + [0] * n_add

    # Correct a and b for the additional nodes
    for i in range(n_reg, n_nodes):
        a[i] = an
        b[i] = b[n_reg - 1] * (float(i - 1) / float(n_reg - 1)) ** nu

    # Correct the last b
    b[n_nodes - 1] = b[n_reg - 1] * (float(n_nodes - 1) / float(n_reg - 1)) ** nu

    return a, b

def calc_gqmom_recurrence_beta(a_reg, b_reg, n_add, ndf_type="gamma"):
    # Number of regular quadrature nodes
    n_reg = len(a_reg)
    
    # Total number of nodes
    n_max_nodes = n_reg + n_add
    
    # If no additional quadrature nodes, use Gauss and return
    if n_add <= 0:
        return a_reg, b_reg
    
    # Extend a and b to accommodate additional nodes
    a = list(a_reg) + [0] * n_add
    b = list(b_reg) + [0] * n_add
    
    zetas = calc_zetas(a, b, n_reg, n_max_nodes)
    p = np.zeros_like(zetas)
    for i in range(1, 2*n_reg):
        p[i] = zetas[i] / (1 - p[i-1])
    
    alpha_coeff = (1.0 - p[1] - 2 * p[2] + p[1] * p[2]) / p[2]
    beta_coeff = (p[1] - p[2] - p[1] * p[2]) / p[2]
    
    ## The following loop differs from the paper; see the comments in the `calc_zetas` function for details.
    pJ_2n_1 = (beta_coeff + n_reg) / (2 * n_reg + alpha_coeff + beta_coeff)
    pJ_2n = (n_reg - 1) / (2 * n_reg - 2 + 1 + alpha_coeff + beta_coeff)
    
    for i in range(n_reg, n_max_nodes):
        pJ_2i_1 = (beta_coeff + i) / (2 * n_reg)
        pJ_2i = i / (2 * i + 1 + alpha_coeff + beta_coeff)
        if i != n_reg:
            if p[2*n_reg-1] <= pJ_2n_1 or pJ_2n_1 >= pJ_2i_1:
                p[2*i-1] = p[2*n_reg-1] * pJ_2i_1 / pJ_2n_1
            else:
                p[2*i-1] = (p[2*n_reg-1] * (1 - pJ_2i_1) + pJ_2i_1 - pJ_2n_1) / (1 - pJ_2n_1)
        if p[2*n_reg-2] <= pJ_2n or pJ_2n >= pJ_2i:
            p[2*i] = p[2*n_reg-2] * pJ_2i / pJ_2n
        else:
            p[2*i] = (p[2*n_reg-2] * (1 - pJ_2i) + pJ_2i - pJ_2n) / (1 - pJ_2n)
        if i != n_reg:
            zetas[2*i-1] = p[2*i-1] * (1.0 - p[2*i-3])
        zetas[2*i] = p[2*i] * (1.0 - p[2*i-1])
    # Update alpha (a) values
    a[0] = zetas[1]
    for i in range(1, n_max_nodes):
        a[i] = zetas[2 * i] + zetas[2 * i + 1]
    
    # Update beta (b) values
    for i in range(1, n_max_nodes):
        b[i] = zetas[2 * i - 1] * zetas[2 * i]
    
    return a, b

def calc_gqmom_recurrence_realplus(moments,a_reg, b_reg, n_add, ndf_type="gamma"):
    """
    Correct recurrence coefficients for (Gamma/Lognormal)Generalized QMOM using R+ moments.
    
    Parameters:
        moments (list or numpy.ndarray): List of input moments [M0, M1, M2, ...].
        zetas (list or numpy.ndarray): Array to store zeta values (modified recurrence coefficients).
        a (list or numpy.ndarray): Recurrence coefficient a to be updated.
        b (list or numpy.ndarray): Recurrence coefficient b to be updated.
        n_add (int): Number of additional quadrature nodes.
        ndf_type (str): Type of node distribution ("gamma" or "lognormal").
    
    Returns:
        tuple: Updated a, b.
    """
    # Number of regular quadrature nodes
    n_reg = len(a_reg)
    
    # Total number of nodes
    n_max_nodes = n_reg + n_add
    
    # If no additional quadrature nodes, use Gauss and return
    if n_add <= 0:
        return a_reg, b_reg
    
    # Extend a and b to accommodate additional nodes
    a = list(a_reg) + [0] * n_add
    b = list(b_reg) + [0] * n_add
    
    zetas = calc_zetas(a, b, n_reg, n_max_nodes)
    
    if ndf_type == "gamma":
        # Calculate alpha coefficient
        m1_sqr = moments[1] ** 2
        alpha_coeff = m1_sqr / (moments[0] * moments[2] - m1_sqr) - 1.0
        for i in range(n_reg, n_max_nodes):
            ## The following loop differs from the paper; see the comments in the `calc_zetas` function for details.
            if i != n_reg:
                zetas[2*i-1] = (i + alpha_coeff) * zetas[2*n_reg-1] / (n_reg + alpha_coeff)
            zetas[2*i] = i*zetas[2*n_reg-2] / (n_reg-1)
        zetas[2*n_max_nodes-1] = (n_max_nodes + alpha_coeff) * zetas[2*n_reg-1] / (n_reg + alpha_coeff)
        
    elif ndf_type == "lognormal":
        # Calculate eta
        eta = np.sqrt(moments[0] * moments[2] / (moments[1] ** 2))
        ## The following loop differs from the paper; see the comments in the `calc_zetas` function for details.
        for i in range(n_reg, n_max_nodes):
            if i != n_reg:
                zetas[2 * i - 1] = (
                    eta ** (4 * (i - n_reg))
                    * zetas[2 * n_reg - 1]
                )
            zetas[2 * i] = (
                eta ** (2 * (i + 1 - n_reg))
                * ((eta ** (2 * i) - 1.0) / (eta ** (2 * n_reg - 2) - 1.0))
                * zetas[2 * n_reg - 2]
            )
        zetas[2 * n_max_nodes - 1] = (
            eta ** (4 * (n_max_nodes - n_reg))
            * zetas[2 * n_reg - 1]
        )
    
    # Update alpha (a) values
    a[0] = zetas[1]
    for i in range(1, n_max_nodes):
        a[i] = zetas[2 * i] + zetas[2 * i + 1]
    
    # Update beta (b) values
    for i in range(1, n_max_nodes):
        b[i] = zetas[2 * i - 1] * zetas[2 * i]
    
    return a, b

def recurrence_jacobi_nodes_weights(a, b):
    # Construct Jacobi matrix and solve for eigenvalues and eigenvectors
    sqrt_b = np.sqrt(b[1:])
    jacobi = np.diag(a) + np.diag(sqrt_b, -1) + np.diag(sqrt_b, 1)

    # Compute eigenvalues (nodes) and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(jacobi)
    idx = eigenvalues.argsort()
    x = eigenvalues[idx].real  # Sorted nodes
    eigenvectors = eigenvectors[:, idx].real
    w = moments[0] * eigenvectors[0, :] ** 2  # Compute weights
    return x, w

def calc_qmom_recurrence(moments, n, adaptive=False, cutoff=0):
    # Initialize modified moments (σ) and recurrence coefficients (a, b)
    nu = moments  # Alias for simplicity
    a = np.zeros(n)
    b = np.zeros(n)
    sigma = np.zeros((2 * n + 1, 2 * n + 1))

    # Construct the first row of σ using the moments
    sigma[1, 1:] = nu

    # Compute the first recurrence coefficients
    a[0] = nu[1] / nu[0]
    b[0] = 0

    # Compute recurrence coefficients using the Wheeler algorithm
    for k in range(2, n + 1):
        for l in range(k, 2 * n - k + 2):
            sigma[k, l] = (
                sigma[k - 1, l + 1]
                - a[k - 2] * sigma[k - 1, l]
                - b[k - 2] * sigma[k - 2, l]
            )
        a[k - 1] = sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
        b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Adaptive adjustment of the number of nodes
    if adaptive:
        for k in range(n, 1, -1):
            if sigma[k, k] <= cutoff:  # Check for numerical stability
                n = k - 1
                if n == 1:
                    w = moments[0]
                    x = moments[1] / moments[0]
                    return np.array([x]), np.array([w])

        # Recalculate recurrence coefficients for the reduced node count
        a = np.zeros(n)
        b = np.zeros(n)
        sigma = np.zeros((2 * n + 1, 2 * n + 1))
        sigma[1, 1:] = nu

        a[0] = nu[1] / nu[0]
        b[0] = 0
        for k in range(2, n + 1):
            for l in range(k, 2 * n - k + 2):
                sigma[k, l] = (
                    sigma[k - 1, l + 1]
                    - a[k - 2] * sigma[k - 1, l]
                    - b[k - 2] * sigma[k - 2, l]
                )
            a[k - 1] = (
                sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
            )
            b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Check for realizability of moments
    if b.min() < 0:
        raise ValueError("Moments in Wheeler_moments are not realizable!")
    return a, b

### This Function is from the open source bib PyQBMMlib(https://github.com/sbryngelson/PyQBMMlib/tree/master)
def calc_qmom_nodes_weights(moments, adaptive=False):
    """
    Compute nodes (ξ_i) and weights (w_i) using QMOM with the adaptive Wheeler algorithm.

    Parameters:
        moments (list or numpy.ndarray): Array of moments [M0, M1, ..., M2n-1].
        adaptive (bool): If True, use adaptive criteria to refine the solution.

    Returns:
        tuple: (x, w), where `x` are the nodes (ξ_i) and `w` are the weights (w_i).
    """
    n = len(moments) // 2  # Number of nodes based on available moments

    # Parameters for adaptivity
    rmax = 1e-8  # Ratio threshold for weights
    eabs = 1e-8  # Absolute error threshold for adaptivity
    cutoff = 0   # Minimum diagonal element to consider

    # Check if moments are unrealizable
    if moments[0] <= 0:
        raise ValueError("Wheeler: Moments are NOT realizable (moment[0] <= 0.0).")

    # Special case for a single node or extremely small moments
    if n == 1 or (adaptive and moments[0] < rmax):
        w = moments[0]
        x = moments[1] / moments[0]
        return np.array([x]), np.array([w])
    
    # calculate recurrence coefficients
    a, b = calc_qmom_recurrence(moments, n, adaptive, cutoff)
    x, w = recurrence_jacobi_nodes_weights(a, b)

    # Adaptive criteria: Refine nodes and weights if enabled
    if adaptive:
        for n1 in range(n, 0, -1):
            if n1 == 1:
                return np.array([moments[1] / moments[0]]), np.array([moments[0]])

            # Check the minimum and maximum distance between nodes
            dab = np.min([np.abs(x[i] - x[:i]) for i in range(1, n1)], axis=0)
            mab = np.max([np.abs(x[i] - x[:i]) for i in range(1, n1)], axis=0)

            mindab = np.min(dab)
            maxmab = np.max(mab)

            if np.min(w) / np.max(w) > rmax and mindab / maxmab > eabs:
                return x, w
    else:
        return x, w

def create_ndf(distribution="normal", x_range=(0, 100), points=1000, **kwargs):
    """
    Create a normalized distribution function (NDF).
    PS: Actually they are Probability Density Function!

    Parameters:
        distribution (str): Type of distribution ("normal", "gamma", "lognormal", "beta").
        x_range (tuple): Range of the variable (start, end). Defaults to (0, 100).
        points (int): Number of points in the range. Defaults to 5000.
        kwargs: Additional parameters for the selected distribution.

    Returns:
        tuple: (x, ndf), where x is the variable range and ndf is the distribution function values.
    """
    # Generate the x variable
    x = np.linspace(x_range[0], x_range[1], points)

    # Ensure x_range is valid for gamma and lognormal distributions
    if distribution in ["gamma", "lognormal"] and x_range[0] < 0:
        raise ValueError(f"{distribution.capitalize()} distribution requires x_range[0] >= 0.")

    # Define the distribution based on the input type
    if distribution == "normal":
        mean = kwargs.get("mean", 50)  # Default mean
        std_dev = kwargs.get("std_dev", 10)  # Default standard deviation
        ndf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    elif distribution == "gamma":
        shape = kwargs.get("shape", 2)  # Default shape parameter
        scale = kwargs.get("scale", 1)  # Default scale parameter
        ndf = stats.gamma.pdf(x, shape, scale=scale)

    elif distribution == "lognormal":
        mean = kwargs.get("mean", 1)  # Default mean of log-space
        sigma = kwargs.get("sigma", 0.5)  # Default standard deviation of log-space
        ndf = stats.lognorm.pdf(x, sigma, scale=np.exp(mean))

    elif distribution == "beta":
        a = kwargs.get("a", 2)  # Default alpha parameter
        b = kwargs.get("b", 2)  # Default beta parameter
        if not (0 <= x_range[0] < x_range[1] <= 1):
            raise ValueError("Beta distribution requires x_range in [0, 1].")
        ndf = stats.beta.pdf(x, a, b)

    else:
        raise ValueError("Unsupported distribution type. Choose from 'normal', 'gamma', 'lognormal', 'beta'.")

    return x, ndf

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
    zetas[1] = a[0]
    for i in range(1, n_reg):
        zetas[2*i] = b[i] / zetas[2*i-1]
        # i <= 2n
        # if i == n_reg: continue
        zetas[2*i+1] = a[i] - zetas[2*i]
    return zetas
    
def plot_moments_comparison(moments, moments_QMOM, moments_GQMOM):
    """
    Plot a visual comparison of QMOM and GQMOM moments against original moments.

    Parameters:
        moments (array-like): Original moments (true values).
        moments_QMOM (array-like): Moments calculated using QMOM.
        moments_GQMOM (array-like): Moments calculated using GQMOM.
    """
    fig=plt.figure()
    ori_ax = fig.add_subplot(1,2,1)   
    rel_ax = fig.add_subplot(1,2,2)  
    # Calculate relative errors
    relative_error_QMOM = np.abs((moments_QMOM - moments) / moments)
    relative_error_GQMOM = np.abs((moments_GQMOM - moments) / moments)

    # Define the orders of moments
    orders = np.arange(len(moments))

    # Plot 1: Original values comparison
    ori_ax, fig = pt.plot_data(orders, moments, fig=fig, ax=ori_ax,
                            xlbl='Order of Moment',
                            ylbl='Moment Value',
                            lbl='Original Moments (True)',
                            clr='k',mrk='o')
    ori_ax, fig = pt.plot_data(orders, moments_QMOM, fig=fig, ax=ori_ax,
                            lbl='QMOM Moments',
                            clr='b',mrk='o')
    ori_ax, fig = pt.plot_data(orders, moments_GQMOM, fig=fig, ax=ori_ax,
                            lbl='GQMOM Moments',
                            clr='r',mrk='o')
    
    rel_ax, fig = pt.plot_data(orders, relative_error_QMOM, fig=fig, ax=rel_ax,
                            xlbl='Order of Moment',
                            ylbl='Relative Error',
                            lbl='Relative Error (QMOM)',
                            clr='b',mrk='o')
    rel_ax, fig = pt.plot_data(orders, relative_error_GQMOM, fig=fig, ax=rel_ax,
                            lbl='Relative Error (GQMOM)',
                            clr='r',mrk='o')
    ori_ax.grid('minor')
    ori_ax.set_yscale('log')
    rel_ax.grid('minor')
    plt.title('Comparison of Moments')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def plot_NDF_comparison(x, NDF, NDF_QMOM, NDF_GQMOM):
    """
    Plot a visual comparison of QMOM and GQMOM NDF against original NDF.

    Parameters:
        NDF (array-like): Original NDF (true values).
        NDF_QMOM (array-like): NDF calculated using QMOM.
        NDF_GQMOM (array-like): NDF calculated using GQMOM.
    """
    fig=plt.figure()

    # Plot 1: Original values comparison
    ax, fig = pt.plot_data(x, NDF, fig=fig, ax=None,
                            xlbl='x',
                            ylbl='NDF',
                            lbl='Original(True)',
                            clr='k',mrk='o')
    ax, fig = pt.plot_data(x, NDF_QMOM, fig=fig, ax=ax,
                            lbl='QMOM',
                            clr='b',mrk='o')
    ax, fig = pt.plot_data(x, NDF_GQMOM, fig=fig, ax=ax,
                            lbl='GQMOM',
                            clr='r',mrk='o')
    
    ax.grid('minor')
    plt.title('Comparison of NDF')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def plot_nodes_weights_comparision(x, NDF, nodes, weights, nodes_G, weights_G):
    fig=plt.figure()
    # ax1 = fig.add_subplot(1,3,1)   
    # ax2 = fig.add_subplot(1,3,2)
    # ax3 = fig.add_subplot(1,3,3)
    # Plot 1: Original values comparison
    ax1, fig = pt.plot_data(x, NDF, fig=fig, ax=None,
                            xlbl='x',
                            ylbl='NDF',
                            lbl='Original(True)',
                            clr='k',mrk='o')
    
    ax2 = ax1.twinx()
    ax2, fig = pt.plot_data(nodes, weights, fig=fig, ax=ax2,
                            lbl='QMOM',
                            clr='b',mrk='o')
    ax2, fig = pt.plot_data(nodes_G, weights_G, fig=fig, ax=ax2,
                            lbl='GQMOM',
                            clr='r',mrk='o')
    
    ax1.grid('minor')
    ax2.grid('minor')
    # ax3.grid('minor')
    plt.title('Comparison of nodes and weights')
    plt.tight_layout()
    plt.legend()
    plt.show()
def NDF_approx(x, nodes, weights, width=1e-1):
    """
    Approximate NDF/Dirac delta function using a sum of Gaussian distributions.

    Parameters:
        x (numpy.ndarray): Points where the function is evaluated.
        nodes (numpy.ndarray): Positions of delta peaks.
        weights (numpy.ndarray): Weights of the delta peaks.
        width (float): Standard deviation of the Gaussian kernel.

    Returns:
        numpy.ndarray: Approximated δ function values.
    """
    NDF_ap = np.zeros_like(x)
    for pos, weight in zip(nodes, weights):
        NDF_ap += weight * stats.norm.pdf(x, loc=pos, scale=width)
    norm_factor = np.trapz(NDF_ap, x)
    NDF_ap /= norm_factor
    return NDF_ap

if __name__ == '__main__':
    # Initialize the number density function
    # x, NDF = create_ndf(distribution="normal", x_range=(-100, 100), mean=10, std_dev=30)
    # x, NDF = create_ndf(distribution="gamma", x_range=(0, 50), shape=5, scale=1)
    # x, NDF = create_ndf(distribution="lognormal", x_range=(0, 100), mean=0.1, sigma=1)
    # x, NDF = create_ndf(distribution="beta", x_range=(0, 1), a=2, b=2)
    # Compute the 0th to 9th order moments using numerical integration
    
    # multiple modal NDF
    # x, NDF1 = create_ndf(distribution="normal", x_range=(0, 100), mean=50, std_dev=10)
    # x, NDF2 = create_ndf(distribution="normal", x_range=(0, 100), mean=10, std_dev=5)
    # x, NDF3 = create_ndf(distribution="normal", x_range=(0, 100), mean=80, std_dev=3)
    x, NDF1 = create_ndf(distribution="gamma", x_range=(0, 50), shape=10, scale=1)
    x, NDF2 = create_ndf(distribution="gamma", x_range=(0, 50), shape=5, scale=1)
    x, NDF3 = create_ndf(distribution="gamma", x_range=(0, 50), shape=1, scale=1)
    # x, NDF1 = create_ndf(distribution="beta", x_range=(0, 1), a=2, b=3)
    # x, NDF2 = create_ndf(distribution="beta", x_range=(0, 1), a=1, b=3)
    # x, NDF3 = create_ndf(distribution="beta", x_range=(0, 1), a=2, b=2)
    NDF = NDF1 + NDF2 + NDF3
    
    n = 5
    moments = np.array([np.trapz(NDF * (x ** k), x) for k in range(2*n)])
    
    nodes, weights = calc_qmom_nodes_weights(moments)
    nodes_G, weights_G = calc_gqmom_nodes_weights(moments, 100, method="Gamma")
    
    # NDF_QMOM = NDF_approx(x, nodes, weights)
    # NDF_GQMOM = NDF_approx(x, nodes_G, weights_G)
    
    moments_QMOM = np.zeros_like(moments)
    moments_GQMOM = np.zeros_like(moments)
    for i in range(2*n):
        moments_QMOM[i] = sum(weights * nodes**i)
        moments_GQMOM[i] = sum(weights_G * nodes_G**i)
    pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    # plot_NDF_comparison(x, NDF, NDF_QMOM, NDF_GQMOM)
    plot_nodes_weights_comparision(x, NDF, nodes, weights, nodes_G, weights_G)
    plot_moments_comparison(moments, moments_QMOM, moments_GQMOM)
    
    
