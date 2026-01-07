# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:15:55 2025

@author: px2030
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GridMeta:
    H: int
    W: int
    A0: float
    N1: int
    N2: int
    R: Tuple[float, float]
    total_units: int
    aspect_ratio: float
    int_bre: float = 0.0
    int_bre_len: int = 1
