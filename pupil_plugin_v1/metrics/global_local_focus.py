# metrics/global_local_focus.py

import numpy as np
from pupil_plugin_v1.utils.geometry import convex_hull_area
from pupil_plugin_v1.utils.normalization import norm


def compute_global_local_focus(fix, sacc):
    """
    Calcola:
    - GLI: Global–Local Index
    - FI : Focus Index
    + metriche di supporto (FDM, MFD, SDI)
    """
    FDM = float(fix["duration [ms]"].median())
    SAM_median = float(sacc["amplitude [px]"].median())

    GLI = norm(SAM_median) - norm(FDM)

    MFD = float(fix["duration [ms]"].mean())
    points = fix[["fixation x [px]", "fixation y [px]"]].dropna().values
    SDI = float(convex_hull_area(points))

    # 🔥 FIX IMPORTANTE: check SDI robusto
    if SDI is not None and not np.isnan(SDI) and SDI != 0:
        SCI = 1.0 / SDI
    else:
        SCI = np.nan

    FI = norm(MFD) + norm(SCI)

    return {
        "GLI": GLI,
        "FI": FI,
        "FDM": FDM,
        "MFD": MFD,
        "SDI_GL": SDI,
    }
