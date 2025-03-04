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

def ensure_integer_array(array):
    """
    Ensure the input array is of integer type.

    Parameters:
        array (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Integer array.
    """
    if not np.issubdtype(array.dtype, np.integer):
        return array.astype(int)
    return array