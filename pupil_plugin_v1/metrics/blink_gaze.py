# pupil_plugin_v1/metrics/blink_gaze.py

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# ---------------  BLINK METRICS  -------------------------
# ---------------------------------------------------------

def compute_blink_metrics(blinks_df: pd.DataFrame, session_duration_s: float):
    """
    Calcola metriche blink compatibili con i blinks.csv Pupil Labs:

    Colonne supportate:
    - 'start timestamp [ns]'
    - 'end timestamp [ns]'
    - 'duration [ms]'

    Mantiene tutte le metriche originali:
    - BR
    - MBD
    - BD_SD
    - IBI_mean
    - PERC_EC
    """

    if blinks_df is None or blinks_df.empty:
        return {
            "BR": np.nan,
            "MBD": np.nan,
            "BD_SD": np.nan,
            "IBI_mean": np.nan,
            "PERC_EC": np.nan
        }

    df = blinks_df.copy()

    # -----------------------------------------
    # 1) NORMALIZZAZIONE COLONNE
    # -----------------------------------------
    # I tuoi CSV hanno:
    # - start timestamp [ns]
    # - end timestamp [ns]
    # - duration [ms]

    if "duration [ms]" in df.columns:
        df["duration"] = df["duration [ms]"] / 1000.0   # → sec
    else:
        raise ValueError("Colonna 'duration [ms]' non trovata in blinks.csv")

    if "start timestamp [ns]" in df.columns:
        df["onset"] = df["start timestamp [ns]"] / 1e9  # → sec
    else:
        raise ValueError("Colonna 'start timestamp [ns]' non trovata in blinks.csv")

    if "end timestamp [ns]" in df.columns:
        df["offset"] = df["end timestamp [ns]"] / 1e9   # → sec
    else:
        raise ValueError("Colonna 'end timestamp [ns]' non trovata in blinks.csv")

    # -----------------------------------------
    # 2) CALCOLO METRICHE
    # -----------------------------------------

    # Numero totale blink
    n_blinks = len(df)

    # Blink Rate (blink/min)
    if session_duration_s > 0:
        BR = n_blinks / (session_duration_s / 60.0)
    else:
        BR = np.nan

    # Mean Blink Duration (sec)
    MBD = df["duration"].mean()

    # Blink Duration StdDev
    BD_SD = df["duration"].std()

    # Inter-Blink Interval (IBI)
    df_sorted = df.sort_values("onset")
    if len(df_sorted) >= 2:
        IBIs = np.diff(df_sorted["onset"].values)
        IBI_mean = float(np.mean(IBIs))
    else:
        IBI_mean = np.nan

    # Percent Eye Closure (proporzione sul totale)
    total_closure_s = df["duration"].sum()
    PERC_EC = total_closure_s / session_duration_s if session_duration_s > 0 else np.nan

    return {
        "BR": float(BR),
        "MBD": float(MBD),
        "BD_SD": float(BD_SD),
        "IBI_mean": float(IBI_mean),
        "PERC_EC": float(PERC_EC),
    }



# ---------------------------------------------------------
# ---------------  GAZE METRICS  --------------------------
# ---------------------------------------------------------

def compute_gaze_metrics(gaze_df: pd.DataFrame):
    """
    Calcola metriche derivate dal gaze point (x,y) compatibili con i gaze.csv Pupil Labs:

    Metriche restituite:
    - GS: Gaze Stability = Var(x)+Var(y)
    - GD: Gaze Dispersion (area ellisse di covarianza)
    - GV_mean: velocità media gaze (px/s)
    - SQI: Signal Quality Index (% campioni con confidence > 0.6)

    Supporta colonne:
    - 'gaze x [px]'
    - 'gaze y [px]'
    - 'confidence'
    - 'timestamp [ns]' oppure altre colonne time/ts
    """

    if gaze_df is None or gaze_df.empty:
        return {
            "GS": np.nan,
            "GD": np.nan,
            "GV_mean": np.nan,
            "SQI": np.nan,
        }

    df = gaze_df.copy()

    # ------------------------------------------------------
    # 1. Identificazione automatica delle colonne
    # ------------------------------------------------------
    # x / y
    possible_x = [c for c in df.columns if "x" in c.lower() and "gaze" in c.lower()]
    possible_y = [c for c in df.columns if "y" in c.lower() and "gaze" in c.lower()]
    # time
    possible_t = [c for c in df.columns if "time" in c.lower() or "ts" in c.lower() or "timestamp" in c.lower()]

    if not (possible_x and possible_y and possible_t):
        return {
            "GS": np.nan,
            "GD": np.nan,
            "GV_mean": np.nan,
            "SQI": np.nan,
        }

    xcol = possible_x[0]
    ycol = possible_y[0]
    tcol = possible_t[0]

    # ------------------------------------------------------
    # 2. Normalizzazione timestamp
    #   - Se è in nanosecondi → / 1e9
    #   - Se è in millisecondi → / 1000
    # ------------------------------------------------------
    tmax = df[tcol].max()

    if tmax > 1e12:                   # ns
        df[tcol] = df[tcol] / 1e9
    elif tmax > 2000:                 # ms
        df[tcol] = df[tcol] / 1000.0
    # else: già in secondi

    # ------------------------------------------------------
    # 3. SQI (Signal Quality Index) usando confidence
    # ------------------------------------------------------
    if "confidence" in df.columns:
        valid = df[df["confidence"] > 0.6]
        SQI = len(valid) / len(df)
        df = valid
    else:
        SQI = 1.0

    if df.empty:
        return {"GS": np.nan, "GD": np.nan, "GV_mean": np.nan, "SQI": float(SQI)}

    # ------------------------------------------------------
    # 4. GS: Gaze Stability
    # ------------------------------------------------------
    GS = float(np.var(df[xcol]) + np.var(df[ycol]))

    # ------------------------------------------------------
    # 5. GD: Gaze Dispersion (area ellisse covarianza)
    # ------------------------------------------------------
    coords = np.vstack([df[xcol].values, df[ycol].values])
    cov = np.cov(coords)

    eigvals = np.linalg.eigvals(cov)
    if np.all(eigvals > 0):
        a = np.sqrt(eigvals[0])
        b = np.sqrt(eigvals[1])
        GD = float(np.pi * a * b)
    else:
        GD = np.nan

    # ------------------------------------------------------
    # 6. GV_mean: gaze velocity
    # ------------------------------------------------------
    dx = np.diff(df[xcol].values)
    dy = np.diff(df[ycol].values)
    dt = np.diff(df[tcol].values)

    # evita divisioni per zero
    vel = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
    GV_mean = float(np.mean(vel))

    return {
        "GS": float(GS),
        "GD": float(GD),
        "GV_mean": float(GV_mean),
        "SQI": float(SQI),
    }

