# metrics/exploration.py

from pupil_plugin_v1.utils.geometry import convex_hull_area
from pupil_plugin_v1.utils.normalization import norm



def compute_exploration(fix, sacc):
    """
    Calcola:
    - SAM: saccade amplitude mean
    - SDI: spatial dispersion index (area convex hull)
    - EI : Exploration Index
    """
    SAM = float(sacc["amplitude [px]"].mean())

    points = fix[["fixation x [px]", "fixation y [px]"]].dropna().values
    SDI = float(convex_hull_area(points))

    EI = norm(SAM) + norm(SDI)

    return {
        "SAM": SAM,
        "SDI": SDI,
        "EI": EI,
    }
