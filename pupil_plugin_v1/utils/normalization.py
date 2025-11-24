# utils/normalization.py

import numpy as np


def norm(x):
    """
    Normalizza usando log(1+x) per evitare NaN e divisioni per zero.
    Funziona anche su valori scalari singoli.
    """
    if x is None:
        return np.nan

    # se è un array numpy, prendi un singolo valore medio
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return np.nan
        x = float(np.mean(x))

    if isinstance(x, float) and np.isnan(x):
        return np.nan

    if x < 0:
        return -np.log1p(abs(x))

    return np.log1p(x)
