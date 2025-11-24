# utils/geometry.py

import numpy as np
from scipy.spatial import ConvexHull


def convex_hull_area(points_xy):
    """
    Restituisce l'area del convex hull dato un array Nx2 di punti (x,y).
    Ritorna NaN se ci sono meno di 4 punti validi.
    """
    if points_xy is None:
        return np.nan

    points = np.asarray(points_xy)
    if points.shape[0] > 3:
        hull = ConvexHull(points)
        return hull.area
    return np.nan


def path_lengths_from_points(points_xy):
    """
    Calcola (path_observed, path_ideal) dati punti sequenziali (x,y).
    path_observed = somma delle distanze tra punti consecutivi
    path_ideal    = distanza diretta tra primo e ultimo punto
    """
    points = np.asarray(points_xy)
    if points.shape[0] <= 1:
        return np.nan, np.nan

    diffs = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    path_observed = float(np.sum(step_lengths))

    path_ideal = float(np.linalg.norm(points[-1] - points[0]))

    return path_observed, path_ideal
