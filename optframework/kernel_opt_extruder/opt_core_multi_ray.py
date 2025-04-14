# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:05:42 2023

@author: px2030
"""
from .opt_core_multi import OptCoreMulti
from .opt_core_ray import OptCoreRay   

class OptCoreMultiRay(OptCoreRay, OptCoreMulti):
    """
    This class inherits from both OptCoreRay (for Ray Tune integration) and OptCoreMulti 
    (for using 1D data to assist in 2D data optimization). 
    """
    def __init__(self, *args, **kwargs):
       OptCoreRay.__init__(self, *args, **kwargs)