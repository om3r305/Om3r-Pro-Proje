# -*- coding: utf-8 -*-
from __future__ import annotations
def score(imbalance:float, wall_dist_bps:float, burst:float)->float:
    # normalize and combine
    im=max(0.0, min(1.0, imbalance))
    wd=max(0.0, min(1.0, wall_dist_bps/50.0))
    bu=max(0.0, min(1.0, burst))
    return 0.5*im + 0.3*(1.0-wd) + 0.2*bu
