# loader/csv_loader.py

import pandas as pd
from pupil_plugin_v1 import config


def load_timeseries_data():
    """Carica i CSV principali di una registrazione e li restituisce come dict di DataFrame."""
    print("Carico i CSV...")

    fix    = pd.read_csv(config.PATH_FIXATIONS)
    gaze   = pd.read_csv(config.PATH_GAZE)
    sacc   = pd.read_csv(config.PATH_SACCADES)
    blinks = pd.read_csv(config.PATH_BLINKS)

    print("CSV caricati correttamente!")
    print("Fixations shape =", fix.shape)
    print("Gaze shape      =", gaze.shape)
    print("Saccades shape  =", sacc.shape)
    print("Blinks shape    =", blinks.shape)

    return {
        "fix": fix,
        "gaze": gaze,
        "sacc": sacc,
        "blinks": blinks,
    }
