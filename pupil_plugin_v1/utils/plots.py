# utils/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_fixations(fix, save_path: str | None = None, show: bool = True):
    """
    Scatter plot delle fissazioni.
    Se save_path è fornito, salva il grafico come PNG.
    """
    x = fix["fixation x [px]"]
    y = fix["fixation y [px]"]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5, s=20)
    plt.title("Fixation Scatter Plot")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_heatmap(fix, save_path: str | None = None, show: bool = True):
    """
    Heatmap (KDE) delle fissazioni.
    Se save_path è fornito, salva il grafico come PNG.
    """
    x = fix["fixation x [px]"]
    y = fix["fixation y [px]"]

    plt.figure(figsize=(6, 6))
    sns.kdeplot(x=x, y=y, fill=True, thresh=0.05, levels=50, cmap="magma")
    plt.title("Fixation Heatmap")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_saccade_path(fix, save_path: str | None = None, show: bool = True):
    """
    Saccade path plot:
    - collega le fissazioni in ordine temporale
    - visualizza il percorso oculare come polilinea
    """
    # ordiniamo per timestamp per sicurezza
    fix_sorted = fix.sort_values("start timestamp [ns]")

    x = fix_sorted["fixation x [px]"].values
    y = fix_sorted["fixation y [px]"].values

    plt.figure(figsize=(6, 6))
    # linea del percorso
    plt.plot(x, y, linewidth=1, alpha=0.7)
    # fissazioni come punti
    plt.scatter(x, y, s=15, alpha=0.7)

    plt.title("Saccade Path Plot")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
