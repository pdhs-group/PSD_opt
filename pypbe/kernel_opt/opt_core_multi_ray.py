# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:05:42 2023

@author: px2030
"""
from .opt_core_multi import OptCoreMulti
from .opt_core_ray import OptCoreRay   

class OptCoreMultiRay(OptCoreRay, OptCoreMulti):
    def __init__(self, *args, **kwargs):
       OptCoreRay.__init__(self, *args, **kwargs)