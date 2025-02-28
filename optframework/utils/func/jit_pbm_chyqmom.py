import numpy as np
import math
from numba import jit

@jit(nopython=True)
def sign(q):
    """
    Return the sign of the input number.
    
    Parameters:
        q (float): Input number.
        
    Returns:
        int: 1 if q > 0, 0 if q == 0, and -1 if q < 0.
    """
    if q > 0:
        return 1
    elif q == 0:
        return 0
    else:
        return -1

@jit(nopython=True)
def hyqmom2(moments):
    """
    Invert moments to obtain a two-node quadrature rule.
    
    Parameters:
        moments (array-like): The input moments.
        
    Returns:
        tuple: (x, w) where x are the abscissas and w are the weights.
    """

    n = 2
    w = np.zeros(n)
    x = np.zeros(n)

    # Equal weights for two-node quadrature
    w[0] = moments[0] / 2.0
    w[1] = w[0]

    # Calculate central moments and use them to position the abscissas
    bx, central_moments = compute_central_moments_1d(moments)
    c2 = central_moments[2]  

    # Enforce minimum variance for numerical stability
    if c2 < 10 ** (-12):
        c2 = 10 ** (-12)
        
    # Position nodes symmetrically around mean (±√c2)
    x[0] = bx - math.sqrt(c2)
    x[1] = bx + math.sqrt(c2)

    return x, w


@jit(nopython=True)
def hyqmom3(moments, max_skewness=30, checks=True):
    """
    Invert moments to obtain a three-node quadrature rule.
    
    Parameters:
        moments (array-like): The input moments.
        max_skewness (float): Maximum allowed skewness (default: 30).
        checks (bool): Flag to perform validity checks (default: True).
        
    Returns:
        tuple: (x, w) where x are the abscissas and w are the weights.
    """

    n = 3
    etasmall = 10 ** (-10)
    verysmall = 10 ** (-14)
    realsmall = 10 ** (-14)

    w = np.zeros(n)
    x = np.zeros(n)
    xp = np.zeros(n)
    xps = np.zeros(n)
    rho = np.zeros(n)

    # Edge case: zero total weight
    if moments[0] <= verysmall and checks:
        w[1] = moments[0]
        return x, w

    # Compute central moments
    bx, central_moments = compute_central_moments_1d(moments)
    c2 = central_moments[2]  
    c3 = central_moments[3]  
    c4 = central_moments[4] 

    # Realizability checks and corrections
    if checks:
        if c2 < 0:
            if c2 < -verysmall:
                raise ValueError("c2 negative in three node HYQMOM")
        else:
            # Check Hamburger moment constraint: c2*c4 - c2^3 - c3^2 ≥ 0
            realizable = c2 * c4 - c2 ** 3 - c3 ** 2
            if realizable < 0:
                if c2 >= etasmall:
                    # Calculate normalized skewness (q) and kurtosis (eta)
                    q = c3 / math.sqrt(c2) / c2
                    eta = c4 / c2 / c2
                    if abs(q) > verysmall:
                        # Adjust q to make moments realizable
                        slope = (eta - 3) / q
                        det = 8 + slope ** 2
                        qp = 0.5 * (slope + math.sqrt(det))
                        qm = 0.5 * (slope - math.sqrt(det))
                        if q > 0:
                            q = qp
                        else:
                            q = qm
                    else:
                        q = 0

                    # Recompute eta, c3 and c4 to ensure realizability
                    eta = q ** 2 + 1
                    c3 = q * math.sqrt(c2) * c2
                    c4 = eta * c2 ** 2
                    if realizable < -(10.0 ** (-6)):
                        raise ValueError("c4 small in HYQMOM3")
                else:
                    # For very small variance, make distribution symmetric
                    c3 = 0.0
                    c4 = c2 ** 2.0

    # Scale factor based on standard deviation
    scale = math.sqrt(c2)
    if checks and c2 < etasmall:
        # For near-zero variance cases
        q = 0
        eta = 1
    else:
        # Normalized skewness and kurtosis parameters
        q = c3 / math.sqrt(c2) / c2
        eta = c4 / c2 / c2

    # Limit skewness if too large
    if q ** 2 > max_skewness ** 2:
        slope = (eta - 3) / q
        if q > 0:
            q = max_skewness
        else:
            q = -max_skewness
        eta = 3 + slope * q
        if checks:
            realizable = eta - 1 - q ** 2
            if realizable < 0:
                eta = 1 + q ** 2

    # Calculate standardized abscissas using q and eta
    xps[0] = (q - math.sqrt(4 * eta - 3 * q ** 2)) / 2.0
    xps[1] = 0.0  # Middle node at mean
    xps[2] = (q + math.sqrt(4 * eta - 3 * q ** 2)) / 2.0

    # Calculate corresponding weights
    dem = 1.0 / math.sqrt(4 * eta - 3 * q ** 2)
    prod = -xps[0] * xps[2]
    prod = max(prod, 1 + realsmall)

    rho[0] = -dem / xps[0]
    rho[1] = 1 - 1 / prod
    rho[2] = dem / xps[2]

    # Normalize weights
    srho = np.sum(rho)
    rho = rho / srho
    if min(rho) < 0:
        raise ValueError("Negative weight in HYQMOM")

    # Scale standardized abscissas back to original scale
    scales = np.sum(rho * xps ** 2) / np.sum(rho)
    xp = xps * scale / math.sqrt(scales)

    # Final weights and positions
    w = moments[0] * rho
    x = xp
    x = bx + x  # Shift by mean
    return x, w

@jit(nopython=True)
def chyqmom4(moments, indices, max_skewness=30):
    """
    Invert 2D moments to obtain a four-node quadrature rule via the CHyQMOM method.
    
    Parameters:
        moments (array-like): Input moments.
        indices (array-like): Moment indices (2D) used for central moments.
        max_skewness (float): Maximum allowed skewness (default: 30).
        
    Returns:
        tuple: (x, w) where x is a list with two arrays [x, y] of abscissas and w are the weights.
    """
    n = 4
    w = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    
    # Calculate zeroth moment, means, and central moments
    mom00, bx, by, central_moments = compute_central_moments_2d(moments, indices)
    c20 = central_moments[3]  # Variance in x
    c11 = central_moments[4]  # Covariance
    c02 = central_moments[5]  # Variance in y
    
    # Get x-quadrature using 2-node HyQMOM (1D)
    M1 = np.array([1, 0, c20])
    xp, rho = hyqmom2(M1)
    
    # Calculate conditional means for y|x using correlation
    yf = c11 * xp / c20
    
    # Compute remaining variance for y after accounting for correlation
    mu2avg = c02 - np.sum(rho * yf ** 2)
    mu2avg = max(mu2avg, 0)  # Ensure non-negative
    mu2 = mu2avg
    
    # Get y-quadrature for remaining variance
    M3 = np.array([1, 0, mu2])
    xp3, rh3 = hyqmom2(M3)
    yp21 = xp3[0]
    yp22 = xp3[1]
    rho21 = rh3[0]
    rho22 = rh3[1]

    # Tensorize weights and nodes
    w[0] = rho[0] * rho21
    w[1] = rho[0] * rho22
    w[2] = rho[1] * rho21
    w[3] = rho[1] * rho22
    w = mom00 * w  # Scale by zeroth moment

    # x nodes (2 distinct x-positions)
    x[0] = xp[0]
    x[1] = xp[0]
    x[2] = xp[1]
    x[3] = xp[1]
    x = bx + x  # Shift by mean

    # y nodes (conditional mean plus remaining variance)
    y[0] = yf[0] + yp21
    y[1] = yf[0] + yp22
    y[2] = yf[1] + yp21
    y[3] = yf[1] + yp22
    y = by + y  # Shift by mean

    x = [x, y]
    return x, w


@jit(nopython=True)
def chyqmom9(moments, indices, max_skewness=30, checks=True):
    """
    Invert 2D moments to obtain a nine-node quadrature rule via the CHyQMOM method.
    
    Parameters:
        moments (array-like): Input moments.
        indices (array-like): Moment indices for the 2D moments.
        max_skewness (float): Maximum allowed skewness (default: 30).
        checks (bool): Flag to perform validity checks (default: True).
        
    Returns:
        tuple: (x, w) where x is a list with two arrays [x, y] of abscissas and w are the weights.
    """
    n = 9
    w = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)

    csmall = 10.0 ** (-10)
    verysmall = 10.0 ** (-14)

    # Calculate zeroth moment, means, and central moments
    mom00, bx, by, central_moments = compute_central_moments_2d(moments, indices)
    
    c20 = central_moments[3]  
    c11 = central_moments[4]  
    c02 = central_moments[5]  
    c30 = central_moments[6]  
    c03 = central_moments[7]  
    c40 = central_moments[8]  
    c04 = central_moments[9]  
    
    # Get x-quadrature using 3-node HyQMOM (1D)
    M1 = np.array([1, 0, c20, c30, c40])
    xp, rho = hyqmom3(M1, max_skewness, checks)
    
    # Special case: negligible variance in x
    if checks and c20 < csmall:
        rho[0] = 0.0
        rho[1] = 1.0
        rho[2] = 0.0
        yf = 0 * xp
        
        # Use y moments directly with HyQMOM
        M2 = np.array([1, 0, c02, c03, c04])
        xp2, rho2 = hyqmom3(M2, max_skewness, checks)
        yp21 = xp2[0]
        yp22 = xp2[1]
        yp23 = xp2[2]
        rho21 = rho2[0]
        rho22 = rho2[1]
        rho23 = rho2[2]
    else:
        # Calculate conditional means for y|x using correlation
        yf = c11 * xp / c20
        
        # Compute remaining variance for y after accounting for correlation
        mu2avg = c02 - np.sum(rho * (yf ** 2.0))
        mu2avg = max(mu2avg, 0.0)
        mu2 = mu2avg
        mu3 = 0 * mu2
        mu4 = mu2 ** 2.0
        
        # If sufficient remaining variance, compute conditional higher moments
        if mu2 > csmall:
            # Calculate normalized skewness and kurtosis for y|x
            q = (c03 - np.sum(rho * (yf ** 3.0))) / mu2 ** (3.0 / 2.0)
            eta = (
                c04 - np.sum(rho * (yf ** 4.0)) - 6 * np.sum(rho * (yf ** 2.0)) * mu2
            ) / mu2 ** 2.0
            
            # Adjust for realizability
            if eta < (q ** 2 + 1):
                if abs(q) > verysmall:
                    slope = (eta - 3.0) / q
                    det = 8.0 + slope ** 2.0
                    qp = 0.5 * (slope + math.sqrt(det))
                    qm = 0.5 * (slope - math.sqrt(det))
                    if q > 0:
                        q = qp
                    else:
                        q = qm
                else:
                    q = 0
                eta = q ** 2 + 1

            # Compute moments for HyQMOM based on q and eta
            mu3 = q * mu2 ** (3.0 / 2.0)
            mu4 = eta * mu2 ** 2.0

        # Get y-quadrature for remaining moments
        M3 = np.array([1, 0, mu2, mu3, mu4])
        xp3, rh3 = hyqmom3(M3, max_skewness, checks)
        yp21 = xp3[0]
        yp22 = xp3[1]
        yp23 = xp3[2]
        rho21 = rh3[0]
        rho22 = rh3[1]
        rho23 = rh3[2]

    # Tensorize weights (3×3 grid)
    w[0] = rho[0] * rho21
    w[1] = rho[0] * rho22
    w[2] = rho[0] * rho23
    w[3] = rho[1] * rho21
    w[4] = rho[1] * rho22
    w[5] = rho[1] * rho23
    w[6] = rho[2] * rho21
    w[7] = rho[2] * rho22
    w[8] = rho[2] * rho23
    w = mom00 * w  # Scale by zeroth moment

    # x nodes (3 distinct x-positions repeated 3 times)
    x[0] = xp[0]
    x[1] = xp[0]
    x[2] = xp[0]
    x[3] = xp[1]
    x[4] = xp[1]
    x[5] = xp[1]
    x[6] = xp[2]
    x[7] = xp[2]
    x[8] = xp[2]
    x = bx + x  # Shift by mean

    # y nodes (conditional means plus deviations)
    y[0] = yf[0] + yp21
    y[1] = yf[0] + yp22
    y[2] = yf[0] + yp23
    y[3] = yf[1] + yp21
    y[4] = yf[1] + yp22
    y[5] = yf[1] + yp23
    y[6] = yf[2] + yp21
    y[7] = yf[2] + yp22
    y[8] = yf[2] + yp23
    y = by + y  # Shift by mean

    x = [x, y]
    return x, w


# Uncomment @jit if desired for performance; here left un-jitted for debugging.
def chyqmom27(moments, indices, max_skewness=30, checks=True):
    """
    (Non-jitted) Invert moments to obtain a 27-node quadrature rule for 3D distributions via CHyQMOM.
    
    Parameters:
        moments (array-like): Input moments (3D).
        indices (array-like): Indices corresponding to the moments.
        max_skewness (float): Maximum allowed skewness (default: 30).
        checks (bool): Flag for performing validity checks (default: True).
    
    Returns:
        Updates internal arrays. (This function uses many intermediate variables.)
    """
    # Indices used for calling chyqmom9
    RF_idx = np.array(
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [0, 3], [4, 0], [0, 4]]
    )

    # Extract raw moments
    m000 = moments[0]
    m100 = moments[1]
    m010 = moments[2]
    m001 = moments[3]
    m200 = moments[4]
    m110 = moments[5]
    m101 = moments[6]
    m020 = moments[7]
    m011 = moments[8]
    m002 = moments[9]
    m300 = moments[10]
    m030 = moments[11]
    m003 = moments[12]
    m400 = moments[13]
    m040 = moments[14]
    m004 = moments[15]

    # Numerical tolerance constants
    small = 1.0e-10
    isosmall = 1.0e-14
    csmall = 1.0e-10
    wsmall = 1.0e-4
    verysmall = 1.0e-14

    n = 27
    w = np.zeros(n)
    abscissas = np.zeros((n, 3))
    Yf = np.zeros(3)  # Conditional means for y|x
    Zf = np.zeros((3, 3))  # Conditional means for z|x,y
    W = np.zeros(n)

    # Edge case: zero total weight
    if m000 <= verysmall and checks:
        w[12] = m000
        return

    # Calculate means
    bx = m100 / m000
    by = m010 / m000
    bz = m001 / m000

    # Calculate central moments with checks for small values
    if checks and m000 <= isosmall:
        # Calculate normalized raw moments
        d200 = m200 / m000
        d020 = m020 / m000
        d002 = m002 / m000
        d300 = m300 / m000
        d030 = m030 / m000
        d003 = m003 / m000
        d400 = m400 / m000
        d040 = m040 / m000
        d004 = m004 / m000

        # Calculate central moments using raw moments and means
        c200 = d200 - bx ** 2
        c020 = d020 - by ** 2
        c002 = d002 - bz ** 2
        c300 = d300 - 3 * bx * d200 + 2 * bx ** 3
        c030 = d030 - 3 * by * d020 + 2 * by ** 3
        c003 = d003 - 3 * bz * d002 + 2 * bz ** 3
        c400 = d400 - 4 * bx * d300 + 6 * (bx ** 2) * d200 - 3 * bx ** 4
        c040 = d040 - 4 * by * d030 + 6 * (by ** 2) * d020 - 3 * by ** 4
        c004 = d004 - 4 * bz * d003 + 6 * (bz ** 2) * d002 - 3 * bz ** 4

        # Zero covariance terms for numerical stability
        c110 = 0
        c101 = 0
        c011 = 0
    else:
        # Calculate normalized raw moments
        d200 = m200 / m000
        d110 = m110 / m000
        d101 = m101 / m000
        d020 = m020 / m000
        d011 = m011 / m000
        d002 = m002 / m000
        d300 = m300 / m000
        d030 = m030 / m000
        d003 = m003 / m000
        d400 = m400 / m000
        d040 = m040 / m000
        d004 = m004 / m000

        # Calculate central moments
        c200 = d200 - bx ** 2
        c110 = d110 - bx * by
        c101 = d101 - bx * bz
        c020 = d020 - by ** 2
        c011 = d011 - by * bz
        c002 = d002 - bz ** 2
        c300 = d300 - 3 * bx * d200 + 2 * bx ** 3
        c030 = d030 - 3 * by * d020 + 2 * by ** 3
        c003 = d003 - 3 * bz * d002 + 2 * bz ** 3
        c400 = d400 - 4 * bx * d300 + 6 * bx ** 2 * d200 - 3 * bx ** 4
        c040 = d040 - 4 * by * d030 + 6 * by ** 2 * d020 - 3 * by ** 4
        c004 = d004 - 4 * bz * d003 + 6 * bz ** 2 * d002 - 3 * bz ** 4

    # Realizability checks and adjustments for x moments
    if c200 <= 0 and checks:
        c200 = 0
        c300 = 0
        c400 = 0

    if c200 * c400 < (c200 ** 3 + c300 ** 2) and checks:
        # Fix realizability by adjusting q and eta
        q = c300 / c200 ** (3.0 / 2.0)
        eta = c400 / c200 ** 2
        if abs(q) > verysmall:
            # Calculate new q that satisfies realizability
            slope = (eta - 3.0) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if q > 0:
                q = qp
            else:
                q = qm
        else:
            q = 0

        # Recompute moments based on adjusted q and eta
        eta = q ** 2 + 1
        c300 = q * c200 ** (3.0 / 2.0)
        c400 = eta * c200 ** 2.0

    # Realizability checks for y moments (similar to x)
    if c020 <= 0 and checks:
        c020 = 0
        c030 = 0
        c040 = 0

    if c200 * c400 < (c200 ** 3 + c300 ** 2) and checks:
        q = c300 / c200 ** (3 / 2)
        eta = c400 / c200 ** 2
        if abs(q) > verysmall:
            slope = (eta - 3) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if sign(q) == 1:
                q = qp
            else:
                q = qm
        else:
            q = 0
        eta = q ** 2 + 1
        c300 = q * c200 ** (3 / 2)
        c400 = eta * c200 ** 2

    if c020 <= 0 and checks:
        c020 = 0
        c030 = 0
        c040 = 0

    if c020 * c040 < (c020 ** 3 + c030 ** 2) and checks:
        q = c030 / c020 ** (3 / 2)
        eta = c040 / c020 ** 2
        if abs(q) > verysmall:
            slope = (eta - 3) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if sign(q) == 1:
                q = qp
            else:
                q = qm
        else:
            q = 0
        eta = q ** 2 + 1
        c030 = q * c020 ** (3 / 2)
        c040 = eta * c020 ** 2

    # Realizability checks for z moments (similar to x and y)
    if c002 <= 0 and checks:
        c002 = 0
        c003 = 0
        c004 = 0

    if c002 * c004 < (c002 ** 3 + c003 ** 2) and checks:
        q = c003 / c002 ** (3 / 2)
        eta = c004 / c002 ** 2
        if abs(q) > verysmall:
            slope = (eta - 3) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if sign(q) == 1:
                q = qp
            else:
                q = qm
        else:
            q = 0
        eta = q ** 2 + 1
        c003 = q * c002 ** (3 / 2)
        c004 = eta * c002 ** 2

    # Get x-quadrature using 3-node HyQMOM
    M1 = np.array([1, 0, c200, c300, c400])
    xp, rho = hyqmom3(M1, max_skewness, checks)

    # Initialize weights for the nodes
    # These will be updated below depending on variance conditions
    rho11 = 0
    rho12 = 1
    rho13 = 0
    rho21 = 0
    rho23 = 0
    rho31 = 0
    rho32 = 1
    rho33 = 0

    yp11 = 0
    yp12 = 0
    yp13 = 0
    yp21 = 0
    yp22 = 0
    yp23 = 0
    yp31 = 0
    yp32 = 0
    yp33 = 0

    rho111 = 0
    rho112 = 1
    rho113 = 0
    rho121 = 0
    rho122 = 1
    rho123 = 0
    rho131 = 0
    rho132 = 1
    rho133 = 0
    rho211 = 0
    rho212 = 1
    rho213 = 0
    rho221 = 0
    rho222 = 1
    rho223 = 0
    rho231 = 0
    rho232 = 1
    rho233 = 0
    rho311 = 0
    rho312 = 1
    rho313 = 0
    rho321 = 0
    rho322 = 1
    rho323 = 0
    rho331 = 0
    rho332 = 1
    rho333 = 0

    zp111 = 0
    zp112 = 0
    zp113 = 0
    zp121 = 0
    zp122 = 0
    zp123 = 0
    zp131 = 0
    zp132 = 0
    zp133 = 0
    zp211 = 0
    zp212 = 0
    zp213 = 0
    zp221 = 0
    zp222 = 0
    zp223 = 0
    zp231 = 0
    zp232 = 0
    zp233 = 0
    zp311 = 0
    zp312 = 0
    zp313 = 0
    zp321 = 0
    zp322 = 0
    zp323 = 0
    zp331 = 0
    zp332 = 0
    zp333 = 0

    # Special case handling for different variance conditions
    # Case 1: Near-zero variance in x direction
    if c200 <= csmall and checks:
        if c020 <= csmall:
            # Both x and y have near-zero variance - use 1D HyQMOM for z
            M0 = np.array([1, 0, c002, c003, c004])
            Z0, W0 = hyqmom3(M0, max_skewness, checks)

            rho[0] = 0
            rho[1] = 1
            rho[2] = 0
            rho22 = 1
            rho221 = W0[0]
            rho222 = W0[1]
            rho223 = W0[2]
            xp = 0 * xp
            zp221 = Z0[0]
            zp222 = Z0[1]
            zp223 = Z0[2]
        else:
            M1 = np.array([1, 0, 0, c020, c011, c002, c030, c003, c040, c004])
            Q1, W1 = chyqmom9(M1, RF_idx, max_skewness, checks)
            Y1 = Q1[0]
            Z1 = Q1[1]

            rho[0] = 0
            rho[1] = 1
            rho[2] = 0
            rho12 = 0
            rho21 = 1
            rho22 = 1
            rho23 = 1
            rho31 = 0
            rho211 = W1[0]
            rho212 = W1[1]
            rho213 = W1[2]
            rho221 = W1[3]
            rho222 = W1[4]
            rho223 = W1[5]
            rho231 = W1[6]
            rho232 = W1[7]
            rho233 = W1[8]

            xp = 0 * xp
            yp21 = Y1[0]
            yp22 = Y1[4]
            yp23 = Y1[8]
            zp211 = Z1[0]
            zp212 = Z1[1]
            zp213 = Z1[2]
            zp221 = Z1[3]
            zp222 = Z1[4]
            zp223 = Z1[5]
            zp231 = Z1[6]
            zp232 = Z1[7]
            zp233 = Z1[8]
    elif c020 <= csmall and checks:
        M2 = np.array([1, 0, 0, c200, c101, c002, c300, c003, c400, c004])
        Q2, W2 = chyqmom9(M2, RF_idx, max_skewness, checks)
        X2 = Q2[0]
        Z2 = Q2[1]

        rho[0] = 1
        rho[1] = 1
        rho[2] = 1
        rho12 = 1
        rho22 = 1
        rho32 = 1
        rho121 = W2[0]
        rho122 = W2[1]
        rho123 = W2[2]
        rho221 = W2[3]
        rho222 = W2[4]
        rho223 = W2[5]
        rho321 = W2[6]
        rho322 = W2[7]
        rho323 = W2[8]
        xp[0] = X2[0]
        xp[1] = X2[4]
        xp[2] = X2[8]
        zp121 = Z2[0]
        zp122 = Z2[1]
        zp123 = Z2[2]
        zp221 = Z2[3]
        zp222 = Z2[4]
        zp223 = Z2[5]
        zp321 = Z2[6]
        zp322 = Z2[7]
        zp323 = Z2[8]
    elif c002 <= csmall and checks:
        M3 = np.array([1, 0, 0, c200, c110, c020, c300, c030, c400, c040])
        Q3, W3 = chyqmom9(M3, RF_idx, max_skewness, checks)
        X3 = Q3[0]
        Y3 = Q3[1]

        rho[0] = 1
        rho[1] = 1
        rho[2] = 1
        rho11 = W3[0]
        rho12 = W3[1]
        rho13 = W3[2]
        rho21 = W3[3]
        rho22 = W3[4]
        rho23 = W3[5]
        rho31 = W3[6]
        rho32 = W3[7]
        rho33 = W3[8]
        xp[0] = X3[0]
        xp[1] = X3[4]
        xp[2] = X3[8]
        yp11 = Y3[0]
        yp12 = Y3[1]
        yp13 = Y3[2]
        yp21 = Y3[3]
        yp22 = Y3[4]
        yp23 = Y3[5]
        yp31 = Y3[6]
        yp32 = Y3[7]
        yp33 = Y3[8]
    else:
        M4 = np.array([1, 0, 0, c200, c110, c020, c300, c030, c400, c040])
        Q4, W4 = chyqmom9(M4, RF_idx, max_skewness, checks)
        X4 = Q4[0]
        Y4 = Q4[1]

        rho11 = W4[0] / (W4[0] + W4[1] + W4[2])
        rho12 = W4[1] / (W4[0] + W4[1] + W4[2])
        rho13 = W4[2] / (W4[0] + W4[1] + W4[2])
        rho21 = W4[3] / (W4[3] + W4[4] + W4[5])
        rho22 = W4[4] / (W4[3] + W4[4] + W4[5])
        rho23 = W4[5] / (W4[3] + W4[4] + W4[5])
        rho31 = W4[6] / (W4[6] + W4[7] + W4[8])
        rho32 = W4[7] / (W4[6] + W4[7] + W4[8])
        rho33 = W4[8] / (W4[6] + W4[7] + W4[8])

        Yf[0] = rho11 * Y4[0] + rho12 * Y4[1] + rho13 * Y4[2]
        Yf[1] = rho21 * Y4[3] + rho22 * Y4[4] + rho23 * Y4[5]
        Yf[2] = rho31 * Y4[6] + rho32 * Y4[7] + rho33 * Y4[8]

        yp11 = Y4[0] - Yf[0]
        yp12 = Y4[1] - Yf[0]
        yp13 = Y4[2] - Yf[0]
        yp21 = Y4[3] - Yf[1]
        yp22 = Y4[4] - Yf[1]
        yp23 = Y4[5] - Yf[1]
        yp31 = Y4[6] - Yf[2]
        yp32 = Y4[7] - Yf[2]
        yp33 = Y4[8] - Yf[2]
        scale1 = math.sqrt(c200)
        scale2 = math.sqrt(c020)
        Rho1 = np.diag(rho)
        Rho2 = np.array(
            [[rho11, rho12, rho13], [rho21, rho22, rho23], [rho31, rho32, rho33]]
        )
        Yp2 = np.array([[yp11, yp12, yp13], [yp21, yp22, yp23], [yp31, yp32, yp33]])
        Yp2s = Yp2 / scale2
        RAB = Rho1 * Rho2
        XAB = np.array(
            [[xp[0], xp[1], xp[2]], [xp[0], xp[1], xp[2]], [xp[0], xp[1], xp[2]]]
        )
        XABs = XAB / scale1
        YAB = Yp2 + np.diag(Yf) * np.ones(3)
        YABs = YAB / scale2
        C01 = np.multiply(RAB, YABs)
        Yc0 = np.ones(3)
        Yc1 = XABs
        Yc2 = Yp2s
        A1 = np.sum(np.multiply(C01, Yc1))
        A2 = np.sum(np.multiply(C01, Yc2))

        c101s = c101 / scale1
        c011s = c011 / scale2
        if c101s ** 2 >= c002 * (1 - small):
            c101s = sign(c101s) * math.sqrt(c002)
        elif c011s ** 2 >= c002 * (1 - small):
            c110s = c110 / scale1 / scale2
            c011s = sign(c011s) * math.sqrt(c002)
            c101s = c110s * c011s

        b0 = 0
        b1 = c101s
        b2 = 0
        if A2 < wsmall:
            b2 = (c011s - A1 * b1) / A2

        Zf = b0 * Yc0 + b1 * Yc1 + b2 * Yc2
        SUM002 = np.sum(np.multiply(RAB, Zf ** 2))
        mu2 = c002 - SUM002
        mu2 = max(0, mu2)
        q = 0
        eta = 1
        if mu2 > csmall:
            SUM1 = mu2 ** (3 / 2)
            SUM3 = np.sum(np.multiply(RAB, Zf ** 3))
            q = (c003 - SUM3) / SUM1
            SUM2 = mu2 ** 2
            SUM4 = np.sum(np.multiply(RAB, Zf ** 4)) + 6 * SUM002 * mu2
            eta = (c004 - SUM4) / SUM2
            if eta < (q ** 2 + 1):
                if abs(q) > verysmall:
                    slope = (eta - 3) / q
                    det = 8 + slope ** 2
                    qp = 0.5 * (slope + math.sqrt(det))
                    qm = 0.5 * (slope - math.sqrt(det))
                    if sign(q) == 1:
                        q = qp
                    else:
                        q = qm
                else:
                    q = 0
                eta = q ** 2 + 1
        mu3 = q * mu2 ** (3 / 2)
        mu4 = eta * mu2 ** 2
        M5 = np.array([1, 0, mu2, mu3, mu4])
        xp11, rh11 = hyqmom3(M5, max_skewness, checks)

        rho111 = rh11[0]
        rho112 = rh11[1]
        rho113 = rh11[2]

        zp111 = xp11[0]
        zp112 = xp11[1]
        zp113 = xp11[2]

        rh12 = rh11
        xp12 = xp11
        rho121 = rh12[0]
        rho122 = rh12[1]
        rho123 = rh12[2]

        zp121 = xp12[0]
        zp122 = xp12[1]
        zp123 = xp12[2]

        rh13 = rh11
        xp13 = xp11
        rho131 = rh13[0]
        rho132 = rh13[1]
        rho133 = rh13[2]

        zp131 = xp13[0]
        zp132 = xp13[1]
        zp133 = xp13[2]

        rh21 = rh11
        xp21 = xp11
        zp211 = xp21[0]
        zp212 = xp21[1]
        zp213 = xp21[2]

        rho211 = rh21[0]
        rho212 = rh21[1]
        rho213 = rh21[2]

        rh22 = rh11
        xp22 = xp11
        zp221 = xp22[0]
        zp222 = xp22[1]
        zp223 = xp22[2]

        rho221 = rh22[0]
        rho222 = rh22[1]
        rho223 = rh22[2]

        rh23 = rh11
        xp23 = xp11
        zp231 = xp23[0]
        zp232 = xp23[1]
        zp233 = xp23[2]

        rho231 = rh23[0]
        rho232 = rh23[1]
        rho233 = rh23[2]

        rh31 = rh11
        xp31 = xp11
        rho311 = rh31[0]
        rho312 = rh31[1]
        rho313 = rh31[2]

        zp311 = xp31[0]
        zp312 = xp31[1]
        zp313 = xp31[2]

        rh32 = rh11
        xp32 = xp11
        rho321 = rh32[0]
        rho322 = rh32[1]
        rho323 = rh32[2]

        zp321 = xp32[0]
        zp322 = xp32[1]
        zp323 = xp32[2]

        rh33 = rh11
        xp33 = xp11
        rho331 = rh33[0]
        rho332 = rh33[1]
        rho333 = rh33[2]

        zp331 = xp33[0]
        zp332 = xp33[1]
        zp333 = xp33[2]

    W[0] = rho[0] * rho11 * rho111
    W[1] = rho[0] * rho11 * rho112
    W[2] = rho[0] * rho11 * rho113
    W[3] = rho[0] * rho12 * rho121
    W[4] = rho[0] * rho12 * rho122
    W[5] = rho[0] * rho12 * rho123
    W[6] = rho[0] * rho13 * rho131
    W[7] = rho[0] * rho13 * rho132
    W[8] = rho[0] * rho13 * rho133
    W[9] = rho[1] * rho21 * rho211
    W[10] = rho[1] * rho21 * rho212
    W[11] = rho[1] * rho21 * rho213
    W[12] = rho[1] * rho22 * rho221
    W[13] = rho[1] * rho22 * rho222
    W[14] = rho[1] * rho22 * rho223
    W[15] = rho[1] * rho23 * rho231
    W[16] = rho[1] * rho23 * rho232
    W[17] = rho[1] * rho23 * rho233
    W[18] = rho[2] * rho31 * rho311
    W[19] = rho[2] * rho31 * rho312
    W[20] = rho[2] * rho31 * rho313
    W[21] = rho[2] * rho32 * rho321
    W[22] = rho[2] * rho32 * rho322
    W[23] = rho[2] * rho32 * rho323
    W[24] = rho[2] * rho33 * rho331
    W[25] = rho[2] * rho33 * rho332
    W[26] = rho[2] * rho33 * rho333
    W = m000 * W

    abscissas[0, 0] = xp[0]
    abscissas[1, 0] = xp[0]
    abscissas[2, 0] = xp[0]
    abscissas[3, 0] = xp[0]
    abscissas[4, 0] = xp[0]
    abscissas[5, 0] = xp[0]
    abscissas[6, 0] = xp[0]
    abscissas[7, 0] = xp[0]
    abscissas[8, 0] = xp[0]
    abscissas[9, 0] = xp[1]
    abscissas[10, 0] = xp[1]
    abscissas[11, 0] = xp[1]
    abscissas[12, 0] = xp[1]
    abscissas[13, 0] = xp[1]
    abscissas[14, 0] = xp[1]
    abscissas[15, 0] = xp[1]
    abscissas[16, 0] = xp[1]
    abscissas[17, 0] = xp[1]
    abscissas[18, 0] = xp[2]
    abscissas[19, 0] = xp[2]
    abscissas[20, 0] = xp[2]
    abscissas[21, 0] = xp[2]
    abscissas[22, 0] = xp[2]
    abscissas[23, 0] = xp[2]
    abscissas[24, 0] = xp[2]
    abscissas[25, 0] = xp[2]
    abscissas[26, 0] = xp[2]
    abscissas[:, 0] += bx

    abscissas[0, 1] = Yf[0] + yp11
    abscissas[1, 1] = Yf[0] + yp11
    abscissas[2, 1] = Yf[0] + yp11
    abscissas[3, 1] = Yf[0] + yp12
    abscissas[4, 1] = Yf[0] + yp12
    abscissas[5, 1] = Yf[0] + yp12
    abscissas[6, 1] = Yf[0] + yp13
    abscissas[7, 1] = Yf[0] + yp13
    abscissas[8, 1] = Yf[0] + yp13
    abscissas[9, 1] = Yf[1] + yp21
    abscissas[10, 1] = Yf[1] + yp21
    abscissas[11, 1] = Yf[1] + yp21
    abscissas[12, 1] = Yf[1] + yp22
    abscissas[13, 1] = Yf[1] + yp22
    abscissas[14, 1] = Yf[1] + yp22
    abscissas[15, 1] = Yf[1] + yp23
    abscissas[16, 1] = Yf[1] + yp23
    abscissas[17, 1] = Yf[1] + yp23
    abscissas[18, 1] = Yf[2] + yp31
    abscissas[19, 1] = Yf[2] + yp31
    abscissas[20, 1] = Yf[2] + yp31
    abscissas[21, 1] = Yf[2] + yp32
    abscissas[22, 1] = Yf[2] + yp32
    abscissas[23, 1] = Yf[2] + yp32
    abscissas[24, 1] = Yf[2] + yp33
    abscissas[25, 1] = Yf[2] + yp33
    abscissas[26, 1] = Yf[2] + yp33
    abscissas[:, 1] += by

    abscissas[0, 2] = Zf[0, 0] + zp111
    abscissas[1, 2] = Zf[0, 0] + zp112
    abscissas[2, 2] = Zf[0, 0] + zp113
    abscissas[3, 2] = Zf[0, 1] + zp121
    abscissas[4, 2] = Zf[0, 1] + zp122
    abscissas[5, 2] = Zf[0, 1] + zp123
    abscissas[6, 2] = Zf[0, 2] + zp131
    abscissas[7, 2] = Zf[0, 2] + zp132
    abscissas[8, 2] = Zf[0, 2] + zp133
    abscissas[9, 2] = Zf[1, 0] + zp211
    abscissas[10, 2] = Zf[1, 0] + zp212
    abscissas[11, 2] = Zf[1, 0] + zp213
    abscissas[12, 2] = Zf[1, 1] + zp221
    abscissas[13, 2] = Zf[1, 1] + zp222
    abscissas[14, 2] = Zf[1, 1] + zp223
    abscissas[15, 2] = Zf[1, 2] + zp231
    abscissas[16, 2] = Zf[1, 2] + zp232
    abscissas[17, 2] = Zf[1, 2] + zp233
    abscissas[18, 2] = Zf[2, 0] + zp311
    abscissas[19, 2] = Zf[2, 0] + zp312
    abscissas[20, 2] = Zf[2, 0] + zp313
    abscissas[21, 2] = Zf[2, 1] + zp321
    abscissas[22, 2] = Zf[2, 1] + zp322
    abscissas[23, 2] = Zf[2, 1] + zp323
    abscissas[24, 2] = Zf[2, 2] + zp331
    abscissas[25, 2] = Zf[2, 2] + zp332
    abscissas[26, 2] = Zf[2, 2] + zp333
    abscissas[:, 2] += bz

    return abscissas, W

@jit(nopython=True)
def quadrature_1d(weights, abscissas, moment_index):
    """
    Compute a unidimensional quadrature sum.
    
    Parameters:
        weights (array-like): Quadrature weights.
        abscissas (array-like): Abscissas corresponding to weights.
        moment_index (int): Power to which abscissas are raised.
    
    Returns:
        float: Approximated moment.
    """
    xi_to_idx = abscissas ** moment_index
    mu = np.dot(weights, xi_to_idx)
    return mu


@jit(nopython=True)
def quadrature_2d(weights, abscissas, moment_index):
    """
    Compute a two-dimensional quadrature sum.
    
    Parameters:
        weights (array-like): Quadrature weights.
        abscissas (list of arrays): List containing two arrays of abscissas (for x and y).
        moment_index (list or array): Powers for the two dimensions.
    
    Returns:
        float: Approximated moment.
    """
    mu = 0.0
    for i, weight in enumerate(weights):
        mu += (
            weights[i]
            * (abscissas[0][i] ** moment_index[0])
            * (abscissas[1][i] ** moment_index[1])
        )
    return mu

@jit(nopython=True)
def quadrature_3d(weights, abscissas, moment_index):
    """
    Compute a three-dimensional quadrature sum.
    
    Parameters:
        weights (array-like): Quadrature weights.
        abscissas (ndarray): 2D array where each row corresponds to one coordinate.
        moment_index (list or array): Powers for each of the three dimensions.
    
    Returns:
        float: Approximated moment.
    """
    mu = 0.0
    for i, weight in enumerate(weights):
        mu += (
            weights[i]
            * abscissas[0, i] ** moment_index[0]
            * abscissas[1, i] ** moment_index[1]
            * abscissas[2, i] ** moment_index[2]
        )
    return mu

@jit(nopython=True)
def compute_central_moments_2d(moments, indices):
    """
    Compute central moments for a 2D distribution.
    
    Parameters:
        moments (array-like): Array of raw moments.
        indices (array-like): 2D indices array such as [[0,0], [1,0], [0,1], ...].
    
    Returns:
        tuple: (mom00, bx, by, central_moments) where mom00 is the zeroth moment,
               bx and by are the centroids, and central_moments is an array of central moments.
    """
    mom00 = get_moment(moments, indices, 0, 0)
    mom10 = get_moment(moments, indices, 1, 0)
    mom01 = get_moment(moments, indices, 0, 1)
    bx = mom10 / mom00
    by = mom01 / mom00
    central_moments = np.zeros_like(moments)
    idx = 0
    for k, l in indices:
        if (k, l) == (0, 0):
            central_moments[idx] = 1.0
            idx += 1
            continue
        if (k, l) in [(1, 0), (0, 1)]:
            central_moments[idx] = 0.0
            idx += 1
            continue
        raw_moment = moments[idx] / mom00
        ckl = raw_moment
        for p in range(k + 1):
            for q in range(l + 1):
                if p == k and q == l:
                    continue
                binom_kp = comb(k, p)
                binom_lq = comb(l, q)
                moment_pq = get_moment(moments, indices, p, q)
                ckl += binom_kp * binom_lq * ((-bx) ** (k - p)) * ((-by) ** (l - q)) * moment_pq / mom00
        central_moments[idx] = ckl
        idx += 1
    return mom00, bx, by, central_moments

@jit(nopython=True)
def compute_central_moments_1d(moments):
    """
    Compute central moments for a 1D distribution.
    
    Parameters:
        moments (array-like): Array of raw moments.
    
    Returns:
        tuple: (bx, central_moments) where bx is the centroid and central_moments is an array of central moments.
    """
    mom0 = moments[0]
    mom1 = moments[1]
    bx = mom1 / mom0
    central_moments = np.zeros_like(moments)
    for k in range(len(moments)):
        if k == 0:
            central_moments[k] = 1.0
            continue
        if k == 1:
            central_moments[k] = 0.0
            continue
        
        raw_moment = moments[k] / mom0

        for p in range(k + 1):
                if p == k:
                    continue 
                binom_kp = comb(k, p)
                raw_moment += binom_kp * ((-bx) ** (k - p)) * moments[p] / mom0
        central_moments[k] = raw_moment

    return bx, central_moments

@jit(nopython=True)
def comb(n, k):
    """
    Compute the binomial coefficient "n choose k".
    
    Parameters:
        n (int): Total number.
        k (int): Number chosen.
    
    Returns:
        int: The binomial coefficient.
    """
    if k > n or k < 0:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

@jit(nopython=True)
def factorial(n):
    """
    Compute the factorial of n.
    
    Parameters:
        n (int): Non-negative integer.
    
    Returns:
        int: n!
    """
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

@jit(nopython=True)
def get_moment(moments, indices, p, q):
    """
    Retrieve the moment corresponding to orders (p, q) from the moments array.
    
    Parameters:
        moments (array-like): Array of moments.
        indices (array-like): Array of indices (2D).
        p (int): Order in first variable.
        q (int): Order in second variable.
    
    Returns:
        float: The moment value if found; otherwise 0.
    """
    for i in range(len(moments)):  # 线性搜索
        if indices[i, 0] == p and indices[i, 1] == q:
            return moments[i]
    return 0

def generalized_hyqmom(moments):
    """
    (Placeholder) Generalized HYQMOM for inverting moments.
    
    Parameters:
        moments (array-like): Input moments.
        
    Returns:
        int: Currently returns 0.
    """
    return 0
