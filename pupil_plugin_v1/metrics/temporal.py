# metrics/temporal.py

import numpy as np
from pupil_plugin_v1.utils.geometry import convex_hull_area
from pupil_plugin_v1.utils.normalization import norm


def compute_temporal_dynamics(fix, sacc):
    """
    Divide la sessione in 3 terzili temporali (T1,T2,T3)
    e calcola EI, GLI, FI per ogni fase.
    """
    fix_sorted = fix.sort_values("start timestamp [ns]")

    ts = fix_sorted["start timestamp [ns]"].values
    N = len(ts)
    if N < 6:
        # troppo pochi punti per una dinamica sensata
        return {k: np.nan for k in [
            "EI_T1", "EI_T2", "EI_T3",
            "GLI_T1", "GLI_T2", "GLI_T3",
            "FI_T1", "FI_T2", "FI_T3",
        ]}

    T1_end = ts[int(N/3)]
    T2_end = ts[int(2*N/3)]

    T1 = fix_sorted[fix_sorted["start timestamp [ns]"] <= T1_end]
    T2 = fix_sorted[(fix_sorted["start timestamp [ns]"] > T1_end) & (fix_sorted["start timestamp [ns]"] <= T2_end)]
    T3 = fix_sorted[fix_sorted["start timestamp [ns]"] > T2_end]

    def compute_seg(seg):
        if len(seg) < 5:
            return np.nan, np.nan, np.nan

        FDM = float(seg["duration [ms]"].median())
        pts = seg[["fixation x [px]", "fixation y [px]"]].dropna().values
        SDI_seg = float(convex_hull_area(pts))

        SAM = float(sacc["amplitude [px]"].mean())

        EI_seg = norm(SAM) + norm(SDI_seg)
        GLI_seg = norm(SAM) - norm(FDM)

        if SDI_seg not in [0, np.nan]:
            SCI_seg = 1.0 / SDI_seg
        else:
            SCI_seg = np.nan
        FI_seg = norm(seg["duration [ms]"].mean()) + norm(SCI_seg)

        return EI_seg, GLI_seg, FI_seg

    EI_T1, GLI_T1, FI_T1 = compute_seg(T1)
    EI_T2, GLI_T2, FI_T2 = compute_seg(T2)
    EI_T3, GLI_T3, FI_T3 = compute_seg(T3)

    return {
        "EI_T1": EI_T1, "EI_T2": EI_T2, "EI_T3": EI_T3,
        "GLI_T1": GLI_T1, "GLI_T2": GLI_T2, "GLI_T3": GLI_T3,
        "FI_T1": FI_T1, "FI_T2": FI_T2, "FI_T3": FI_T3,
    }
