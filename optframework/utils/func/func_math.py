# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:54:06 2023

@author: px2030
"""
import numpy as np

def float_equal(a, b, rel_tol=1e-6):
    return abs(a - b) <= rel_tol * max(abs(a), abs(b))

def float_in_list(target, float_list, rel_tol=1e-6):
    for item in float_list:
        if float_equal(target, item, rel_tol):
            return True
    return False

def isZero(value, tol=1e-20):
    return abs(value) <= tol

def ensure_integer_array(x):
    """
    Ensure the input is of integer type, supports both scalars and arrays.

    Parameters:
        x (scalar or numpy.ndarray): Input value or array.

    Returns:
        int or numpy.ndarray: Integer version of the input.
    """
    if np.isscalar(x):
        return int(x) if not isinstance(x, (int, np.integer)) else x
    elif isinstance(x, np.ndarray):
        return x if np.issubdtype(x.dtype, np.integer) else x.astype(int)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")