# metrics/systematicity.py

import numpy as np
from pupil_plugin_v1.utils.geometry import path_lengths_from_points


def compute_systematicity(fix):
    """
    Calcola:
    - LIN: linearity index (corr ordine vs x)
    - RR : regression rate (quanti passi tornano indietro in x)
    - PEI: path efficiency index
    """
    x = fix["fixation x [px]"].values
    y = fix["fixation y [px]"].values
    coords = np.column_stack([x, y])
    fix_order = np.arange(len(x))

    if len(x) > 1:
        LIN = float(np.corrcoef(fix_order, x)[0, 1])
        steps = np.diff(x)
        regressions = float(np.sum(steps < 0))
        RR = regressions / len(steps)
    else:
        LIN = np.nan
        RR = np.nan

    path_observed, path_ideal = path_lengths_from_points(coords)
    if path_observed and path_observed > 0:
        PEI = float(path_ideal / path_observed)
    else:
        PEI = np.nan

    return {
        "LIN": LIN,
        "RR": RR,
        "PEI": PEI,
    }
