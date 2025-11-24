# metrics/cog_indices.py

import numpy as np


def _stability_index(values):
    """Restituisce un indice di stabilità in [0,1] dato un array di valori."""
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    v_range = v_max - v_min
    if v_range == 0:
        return 1.0  # perfettamente stabili
    sd = float(np.std(vals))
    # più sd è piccolo rispetto al range, più stabilità
    stab = 1.0 - (sd / (v_range + 1e-8))
    return float(np.clip(stab, 0.0, 1.0))


def _delta_index(values):
    """Indice di switching/f-lessibilità dinamica in [0,1]."""
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan
    diffs = np.abs(np.diff(vals))
    mean_diff = float(np.mean(diffs))
    scale = float(np.max(np.abs(vals)) + 1e-8)
    # normalizza rispetto all'ampiezza tipica del segnale
    delta = mean_diff / (scale + 1e-8)
    return float(np.clip(delta, 0.0, 1.0))


def compute_cog_indices(metrics: dict) -> dict:
    """
    Calcola:
    - COG_X: indice di coerenza cognitiva (0-1, più alto = più coerente)
    - COG_DELTA: indice di switching/f-lessibilità (0-1, più alto = più dinamico/volatile)
    """
    temporal = metrics.get("temporal", {})

    EI_vals  = [temporal.get("EI_T1"),  temporal.get("EI_T2"),  temporal.get("EI_T3")]
    GLI_vals = [temporal.get("GLI_T1"), temporal.get("GLI_T2"), temporal.get("GLI_T3")]
    FI_vals  = [temporal.get("FI_T1"),  temporal.get("FI_T2"),  temporal.get("FI_T3")]

    # Stabilità esplorativa e globale-locale
    EI_stab  = _stability_index(EI_vals)
    GLI_stab = _stability_index(GLI_vals)

    # PEI normalizzato (0-1)
    PEI = metrics.get("PEI")
    if PEI is None or np.isnan(PEI):
        PEI_norm = np.nan
    else:
        # teoricamente PEI ∈ [0,1]
        PEI_norm = float(np.clip(PEI, 0.0, 1.0))

    # COG-X: combinazione pesata (coerenza cognitiva globale)
    components = []
    weights = []

    if not np.isnan(EI_stab):
        components.append(EI_stab)
        weights.append(0.4)
    if not np.isnan(GLI_stab):
        components.append(GLI_stab)
        weights.append(0.3)
    if not (PEI_norm is None or np.isnan(PEI_norm)):
        components.append(PEI_norm)
        weights.append(0.3)

    if components and weights:
        w = np.array(weights, dtype=float)
        w = w / np.sum(w)
        COG_X = float(np.sum(w * np.array(components, dtype=float)))
    else:
        COG_X = np.nan

    # COG-DELTA: dinamica di switching (global-local + focus)
    GLI_delta = _delta_index(GLI_vals)
    FI_delta  = _delta_index(FI_vals)

    sub = []
    if not np.isnan(GLI_delta):
        sub.append(GLI_delta)
    if not np.isnan(FI_delta):
        sub.append(FI_delta)

    if sub:
        COG_DELTA = float(np.mean(sub))  # già in [0,1]
    else:
        COG_DELTA = np.nan

    return {
        "COG_X": COG_X,
        "COG_DELTA": COG_DELTA,
        "COG_components": {
            "EI_stability": EI_stab,
            "GLI_stability": GLI_stab,
            "PEI_norm": PEI_norm,
            "GLI_delta": GLI_delta,
            "FI_delta": FI_delta,
        },
    }
