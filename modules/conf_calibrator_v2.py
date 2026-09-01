# -*- coding: utf-8 -*-
from __future__ import annotations
def calibrate(p:float, strength:float=0.15)->float:
    # simple shrinkage towards 0.5
    return max(0.0, min(1.0, 0.5 + (p-0.5)*(1.0-strength)))
