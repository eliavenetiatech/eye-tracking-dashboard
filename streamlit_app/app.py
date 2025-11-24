# streamlit_app/app.py

# --- PATH FIX: permette di importare pupil_plugin_v1 anche da Streamlit --- #
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- LIBRERIE DI BASE --- #
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# --- IMPORT DAL MOTORE pupil_plugin_v1 --- #
from pupil_plugin_v1.metrics.exploration import compute_exploration
from pupil_plugin_v1.metrics.global_local_focus import compute_global_local_focus
from pupil_plugin_v1.metrics.systematicity import compute_systematicity
from pupil_plugin_v1.metrics.temporal import compute_temporal_dynamics
from pupil_plugin_v1.metrics.cog_indices import compute_cog_indices
from pupil_plugin_v1.profiling.profile_builder import build_behavior_profile
from pupil_plugin_v1.metrics.blink_gaze import compute_blink_metrics, compute_gaze_metrics


# ------------------------ CONFIG STREAMLIT ------------------------------- #

st.set_page_config(
    page_title="Behavioral Eye-Tracking Dashboard",
    layout="wide",
)

# Font + tema Venetia Tech (Space Grotesk + neon)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --vt-blue: #00f6ff;
    --vt-purple: #bc00ff;
    --vt-magenta: #ff0099;
    --vt-dark: #0d1117;
    --vt-gray: #c7d0d9;
}

html, body, [class*="css"]  {
    background-color: var(--vt-dark) !important;
    color: var(--vt-gray) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

h1, h2, h3, h4, h5 {
    color: var(--vt-blue) !important;
    font-weight: 600;
}

.sidebar .sidebar-content {
    background-color: #0f1320 !important;
}

.block-container {
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------ HELPERS DATI ---------------------------------- #

def load_subjects_from_zip(uploaded_zip):
    subjects = {}

    bytes_data = uploaded_zip.getvalue()
    with zipfile.ZipFile(io.BytesIO(bytes_data)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if not name.lower().endswith(".csv"):
                continue

            parts = name.split("/")
            if len(parts) >= 2:
                subj = parts[-2]
                csv_name = parts[-1]
            else:
                subj = "default"
                csv_name = parts[0]

            with zf.open(name) as f:
                df = pd.read_csv(f)

            if subj not in subjects:
                subjects[subj] = {}
            subjects[subj][csv_name] = df

    return subjects


def extract_fix_sacc(file_dict):
    """
    Dato il dict {"fixations.csv": df, "saccades.csv": df, "blinks.csv": df, "gaze.csv": df, ...},
    restituisce (fix, sacc, blinks, gaze).
    """
    fix = None
    sacc = None
    blinks = None
    gaze = None

    for fname, df in file_dict.items():
        lower = fname.lower()
        if "fixation" in lower:
            fix = df
        elif "saccade" in lower:
            sacc = df
        elif "blink" in lower:
            blinks = df
        elif "gaze" in lower:
            gaze = df

    return fix, sacc, blinks, gaze


def estimate_session_duration_s(fix, sacc, gaze, blinks):
    """
    Stima la durata totale della registrazione (in secondi)
    gestendo automaticamente timestamp in:
    - nanosecondi (ns)
    - microsecondi (us)
    - millisecondi (ms)
    - secondi (s)
    """

    dfs = [fix, sacc, gaze, blinks]
    max_span = 0.0

    for df in dfs:
        if df is None or df.empty:
            continue

        # trova le colonne temporali
        time_cols = [
            c for c in df.columns
            if "time" in c.lower() or "ts" in c.lower()
        ]
        if not time_cols:
            continue

        tcol = time_cols[0]

        tmin = df[tcol].min()
        tmax = df[tcol].max()
        if pd.isna(tmin) or pd.isna(tmax):
            continue

        span_raw = float(tmax - tmin)

        # --- AUTO-DETECT UNITÀ TEMPO --- #
        # ns → valori dell'ordine 1e12–1e18
        if span_raw > 1e12:
            span_s = span_raw / 1e9   # converti ns → s

        # μs → 1e6
        elif span_raw > 1e6:
            span_s = span_raw / 1e6   # converti μs → s

        # ms → 2e3
        elif span_raw > 2000:
            span_s = span_raw / 1000  # converti ms → s

        else:
            span_s = span_raw  # già in secondi

        # tieni la durata massima tra fix, sacc, gaze, blink
        if span_s > max_span:
            max_span = span_s

    return max_span



def compute_metrics_for_subject(
    fix: pd.DataFrame,
    sacc: pd.DataFrame,
    blinks: pd.DataFrame | None = None,
    gaze: pd.DataFrame | None = None,
):
    """
    Pipeline completa per un singolo soggetto.
    Include:
    - metriche di esplorazione (EI)
    - global/local (GLI)
    - focus index (FI)
    - sistematicità (LIN, RR)
    - dinamiche temporali
    - metriche BLINK (BR, MBD, BD_SD, IBI_mean, PERC_EC)
    - metriche GAZE (GS, GD, GV_mean, SQI)
    - indici cognitivi (COG-X, COG-DELTA)
    """

    # metriche base
    m_exp = compute_exploration(fix, sacc)
    m_glf = compute_global_local_focus(fix, sacc)
    m_sys = compute_systematicity(fix)
    m_temp = compute_temporal_dynamics(fix, sacc)

    # durata registrazione (necessaria per BR)
    session_duration_s = estimate_session_duration_s(fix, sacc, gaze, blinks)

    # metriche blink
    blink_metrics = {}
    if blinks is not None and not blinks.empty:
        blink_metrics = compute_blink_metrics(blinks, session_duration_s)

    # metriche gaze
    gaze_metrics = {}
    if gaze is not None and not gaze.empty:
        gaze_metrics = compute_gaze_metrics(gaze)

    # merge metriche
    metrics = {
        **m_exp,
        **m_glf,
        **m_sys,
        **blink_metrics,
        **gaze_metrics,
        "temporal": m_temp,
    }

    # indici cognitivi
    cog = compute_cog_indices(metrics)
    metrics.update(cog)

    # profilo psicologico sintetico
    profile = build_behavior_profile(metrics)

    return {"metrics": metrics, "profile": profile}




# ------------------------ HELPERS PLOTLY -------------------------------- #

METRIC_RADAR_ORDER = ["EI", "GLI", "FI", "LIN", "RR", "PEI", "COG_X", "COG_DELTA"]

# Etichette cognitive per il radar (stesso ordine di METRIC_RADAR_ORDER)
COGNITIVE_LABELS = {
    "EI": "Exploration breadth (EI)",           # ampiezza esplorazione
    "GLI": "Global vs Local (GLI)",            # strategia globale vs locale
    "FI": "Attentional focus (FI)",            # stabilità del focus
    "LIN": "Scanpath linearity (LIN)",         # linearità percorso
    "RR": "Re-check / regressions (RR)",       # movimenti di ritorno
    "PEI": "Path efficiency (PEI)",            # efficienza complessiva
    "COG_X": "Cognitive coherence (COG-X)",    # coerenza cognitiva globale
    "COG_DELTA": "Flexibility / switching (COG-DELTA)",  # flessibilità strategica
}


def radar_for_subject(subj_name: str, metrics_row: dict):
    """Radar Plotly per un singolo soggetto, con etichette cognitive."""
    values = [metrics_row[m] for m in METRIC_RADAR_ORDER]

    categories = [COGNITIVE_LABELS[m] for m in METRIC_RADAR_ORDER]
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=subj_name,
            line=dict(color="rgba(0, 246, 255, 0.9)", width=2),
            fillcolor="rgba(188, 0, 255, 0.25)",  # viola neon soft
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c7d0d9"),
    )
    return fig



def radar_for_group(df_group: pd.DataFrame):
    """Radar medio di gruppo (+ overlay soggetti) con etichette cognitive."""
    mean_vals = df_group[METRIC_RADAR_ORDER].mean()

    categories = [COGNITIVE_LABELS[m] for m in METRIC_RADAR_ORDER]
    categories = categories + [categories[0]]
    mean_r = list(mean_vals.values) + [mean_vals.values[0]]

    fig = go.Figure()

    # Overlay soggetti (linee sottili semi-trasparenti)
    for _, row in df_group.iterrows():
        r = [row[m] for m in METRIC_RADAR_ORDER]
        r = r + [r[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=categories,
                fill="none",
                mode="lines",
                line=dict(color="rgba(150,150,150,0.18)", width=1),
                showlegend=False,
            )
        )

    # Profilo medio
    fig.add_trace(
        go.Scatterpolar(
            r=mean_r,
            theta=categories,
            fill="toself",
            name="Media gruppo",
            line=dict(color="rgba(0, 246, 255, 1.0)", width=3),
            fillcolor="rgba(188, 0, 255, 0.30)",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c7d0d9"),
    )
    return fig


TEMPORAL_METRICS = ["EI", "GLI", "FI"]


def temporal_line_for_subject(temporal: dict, metric: str, subj_name: str):
    """Line chart EI/GLI/FI T1–T3 per singolo soggetto."""
    x = ["T1", "T2", "T3"]
    y = [
        temporal.get(f"{metric}_T1", np.nan),
        temporal.get(f"{metric}_T2", np.nan),
        temporal.get(f"{metric}_T3", np.nan),
    ]

    color_map = {
        "EI": "#00f6ff",
        "GLI": "#bc00ff",
        "FI": "#ff0099",
    }
    col = color_map.get(metric, "#00f6ff")

    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=3, color=col),
            marker=dict(size=8),
            name=metric,
        )
    )
    fig.update_layout(
        title=f"Andamento {metric} nel tempo – {subj_name}",
        xaxis_title="Fase temporale",
        yaxis_title=metric,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c7d0d9"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def temporal_line_group(temporal_per_subject: dict, metric: str):
    """
    Line chart EI/GLI/FI T1–T3 per tutti i soggetti (una linea per soggetto).
    temporal_per_subject: {subj_name: temporal_dict}
    """
    x = ["T1", "T2", "T3"]
    fig = go.Figure()

    for subj_name, temporal in temporal_per_subject.items():
        y = [
            temporal.get(f"{metric}_T1", np.nan),
            temporal.get(f"{metric}_T2", np.nan),
            temporal.get(f"{metric}_T3", np.nan),
        ]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=subj_name,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title=f"Confronto di gruppo nel tempo – {metric}",
        xaxis_title="Fase temporale",
        yaxis_title=metric,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c7d0d9"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def scatter_fix_plotly(fix: pd.DataFrame, title="Fixations"):
    """Scatter Plotly delle fissazioni."""
    fig = px.scatter(
        fix,
        x="fixation x [px]",
        y="fixation y [px]",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=5, color="#00f6ff"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c7d0d9"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig



def heatmap_group_fix(fix_list):
    """Heatmap aggregata delle fissazioni di tutti i soggetti."""
    if not fix_list:
        return None

    all_fix = pd.concat(fix_list, ignore_index=True)
    fig = px.density_heatmap(
        all_fix,
        x="fixation x [px]",
        y="fixation y [px]",
        nbinsx=40,
        nbinsy=40,
        color_continuous_scale=[
            "#0d1117",  # background
            "#00f6ff",
            "#bc00ff",
            "#ff0099",
        ],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title="Heatmap aggregata fissazioni (gruppo)",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c7d0d9"),
    )
    return fig



# ------------------------ INTERPRETAZIONE GRUPPO ------------------------ #

def level_descriptor(value, strategy="abs"):
    if np.isnan(value):
        return "non interpretabile"

    x = value if strategy == "signed" else abs(value)

    if x < 0.5:
        return "molto basso"
    elif x < 1.5:
        return "basso"
    elif x < 3:
        return "moderato"
    elif x < 6:
        return "alto"
    else:
        return "molto alto"


def variability_descriptor(std_val):
    if np.isnan(std_val):
        return "non interpretabile"
    if std_val < 0.2:
        return "molto bassa (profilo omogeneo)"
    if std_val < 1:
        return "bassa (profilo relativamente simile tra i soggetti)"
    if std_val < 3:
        return "moderata (differenze individuali presenti)"
    return "alta (profilo molto eterogeneo)"


def interpret_group(df_group: pd.DataFrame, mean_vals: pd.Series, std_vals: pd.Series) -> str:
    n_subj = len(df_group)

    ei_level = level_descriptor(mean_vals["EI"])
    gli_level = level_descriptor(mean_vals["GLI"], strategy="signed")
    fi_level = level_descriptor(mean_vals["FI"])
    lin_level = level_descriptor(mean_vals["LIN"])
    rr_level = level_descriptor(mean_vals["RR"])
    pei_level = level_descriptor(mean_vals["PEI"])
    cogx_level = level_descriptor(mean_vals["COG_X"])
    cogd_level = level_descriptor(mean_vals["COG_DELTA"])

    ei_var = variability_descriptor(std_vals["EI"])
    gli_var = variability_descriptor(std_vals["GLI"])
    fi_var = variability_descriptor(std_vals["FI"])
    cogx_var = variability_descriptor(std_vals["COG_X"])
    cogd_var = variability_descriptor(std_vals["COG_DELTA"])

    lines = []

    lines.append(
        f"Il gruppo è composto da {n_subj} soggetti. "
        f"L’indice di esplorazione medio (EI = {mean_vals['EI']:.3f}) è {ei_level}, "
        f"indicando un livello di esplorazione visiva complessivamente {ei_level}."
    )

    if mean_vals["GLI"] < 0:
        lines.append(
            f"Il Global–Local Index medio (GLI = {mean_vals['GLI']:.3f}) è negativo, "
            "a indicare una tendenza prevalente verso strategie **locali/analitiche** "
            "più che globali."
        )
    elif mean_vals["GLI"] > 0:
        lines.append(
            f"Il Global–Local Index medio (GLI = {mean_vals['GLI']:.3f}) è positivo, "
            "suggerendo una preferenza per strategie **globali/panoramiche**."
        )
    else:
        lines.append(
            f"Il Global–Local Index medio (GLI = {mean_vals['GLI']:.3f}) è neutro, "
            "il che suggerisce un bilanciamento tra analisi globale e locale."
        )

    lines.append(
        f"Il Focus Index medio (FI = {mean_vals['FI']:.3f}) risulta {fi_level}, "
        "indicando una capacità di mantenere l’attenzione complessivamente "
        f"{fi_level}."
    )

    lines.append(
        f"Il Path Efficiency Index (PEI = {mean_vals['PEI']:.3f}) è {pei_level}, "
        "quindi il percorso oculare medio è "
        + ("piuttosto efficiente." if "alto" in pei_level else "relativamente poco efficiente.")
    )

    lines.append(
        f"L’indice COG-X medio ({mean_vals['COG_X']:.3f}) è {cogx_level}, "
        "che riflette un livello complessivo di coerenza cognitiva "
        f"{cogx_level}. "
        f"La variabilità di COG-X nel gruppo è {cogx_var}."
    )

    lines.append(
        f"L’indice COG-DELTA medio ({mean_vals['COG_DELTA']:.3f}) è {cogd_level}, "
        "indicando una flessibilità attentiva globale "
        f"{cogd_level}. La variabilità associata ({cogd_var}) suggerisce "
        "quanto i soggetti differiscono tra loro nella capacità di cambiare strategia visiva."
    )

    lines.append(
        f"La variabilità dell’esplorazione (EI std = {std_vals['EI']:.3f}) è {ei_var}, "
        f"mentre quella tra strategie globali/locali (GLI std = {std_vals['GLI']:.3f}) è {gli_var}. "
        "Questi valori indicano il grado di omogeneità del gruppo rispetto al modo di esplorare lo stimolo."
    )

    return "\n".join(lines)

# ------------------------ CLUSTERING PROFILI --------------------------- #
def summarize_cognitive_layer_subject(metrics: dict) -> str:
    """
    Restituisce un testo interpretativo compatto che combina
    blink e gaze in ottica di carico cognitivo / controllo attentivo.
    """
    br = metrics.get("BR")
    mbd = metrics.get("MBD")
    perc_ec = metrics.get("PERC_EC")
    gs = metrics.get("GS")
    gd = metrics.get("GD")
    gv = metrics.get("GV_mean")
    sqi = metrics.get("SQI")

    lines = []

    # Blink rate
    if not np.isnan(br):
        if br < 5:
            lines.append(f"- **Blink Rate (BR = {br:.2f} blink/min)**: molto basso, compatibile con elevato carico visivo o forte concentrazione.")
        elif br < 20:
            lines.append(f"- **Blink Rate (BR = {br:.2f} blink/min)**: nella norma per compiti di osservazione visiva.")
        else:
            lines.append(f"- **Blink Rate (BR = {br:.2f} blink/min)**: elevato; possibile affaticamento, secchezza o ridotto engagement visivo.")

    # Durata blink
    if not np.isnan(mbd):
        lines.append(f"- **Mean Blink Duration (MBD = {mbd:.3f} s)**: durata media del blink, valori più lunghi possono riflettere affaticamento o micro-pause più marcate.")

    # Percent Eye Closure
    if not np.isnan(perc_ec):
        lines.append(f"- **Percent Eye Closure (PERC_EC = {perc_ec:.3f})**: quota di tempo a occhi chiusi; valori più alti suggeriscono maggiore affaticamento o riduzione di vigilanza.")

    # Gaze stability / dispersion
    if not np.isnan(gs) and not np.isnan(gd):
        lines.append(
            f"- **Gaze Stability / Dispersion (GS = {gs:.1f}, GD = {gd:.1f})**: stabilità e ampiezza del gaze; "
            "valori molto alti di GD indicano esplorazione ampia, valori più bassi profili più focalizzati."
        )

    # Gaze velocity
    if not np.isnan(gv):
        lines.append(
            f"- **Gaze Velocity (GV_mean = {gv:.1f} px/s)**: velocità media dei movimenti del gaze; "
            "profili molto veloci possono riflettere scanning rapido, quelli lenti osservazione più approfondita."
        )

    # Signal quality
    if not np.isnan(sqi):
        lines.append(
            f"- **Signal Quality Index (SQI = {sqi:.2f})**: qualità del segnale; valori > 0.8 sono generalmente "
            "considerati ottimali per inferenze affidabili."
        )

    if not lines:
        return "Dati blink/gaze non disponibili o non interpretabili per questo soggetto."

    return "\n".join(lines)

def summarize_cognitive_layer_group(df_group: pd.DataFrame) -> str:
    """
    Interpreta BR, MBD, PERC_EC, GS, GD, GV_mean, SQI 
    a livello DI GRUPPO, integrando:
    - vigilanza / affaticamento (blink)
    - stabilità attentiva (gaze)
    - ampiezza della ricerca visiva (dispersione)
    - controllo oculomotorio (velocità gaze)
    - qualità del segnale
    """

    mean = df_group.mean(numeric_only=True)
    std  = df_group.std(numeric_only=True)

    BR      = mean.get("BR")
    MBD     = mean.get("MBD")
    PERC_EC = mean.get("PERC_EC")
    GS      = mean.get("GS")
    GD      = mean.get("GD")
    GV      = mean.get("GV_mean")
    SQI     = mean.get("SQI")

    lines = []
    lines.append("### **Sintesi del livello cognitivo del gruppo**\n")

    # ---------------------- BLINK RATE ----------------------
    if not np.isnan(BR):
        if BR < 5:
            lines.append(f"- **Blink Rate medio (BR = {BR:.2f} blink/min): basso** → gruppo altamente concentrato, possibile carico cognitivo elevato.")
        elif BR < 20:
            lines.append(f"- **Blink Rate medio (BR = {BR:.2f} blink/min): nella norma** → buon equilibrio tra concentrazione e comfort visivo.")
        else:
            lines.append(f"- **Blink Rate medio (BR = {BR:.2f} blink/min): alto** → possibile affaticamento / discomfort / minore engagement.")

    # ---------------------- BLINK DURATION ----------------------
    if not np.isnan(MBD):
        lines.append(f"- **Durata media del blink (MBD = {MBD:.3f}s)** → indica il tempo di “micro-disconnessione”; valori alti riflettono affaticamento o necessità di recupero.")

    # ---------------------- EYE CLOSURE ----------------------
    if not np.isnan(PERC_EC):
        lines.append(f"- **Percent Eye Closure (PERC_EC = {PERC_EC:.3f})** → quota totale di tempo a occhi chiusi; valori alti = minor vigilanza, stanchezza.")

    # ---------------------- GAZE STABILITY ----------------------
    if not np.isnan(GS):
        lines.append(
            f"- **Gaze Stability (GS = {GS:.1f})** → stabilità del controllo oculare: valori alti suggeriscono maggiore variabilità micro-saccadica "
            "o difficoltà a mantenere un focus stabile."
        )

    # ---------------------- GAZE DISPERSION ----------------------
    if not np.isnan(GD):
        lines.append(
            f"- **Gaze Dispersion (GD = {GD:.1f})** → ampiezza media dell’esplorazione. "
            "Valori alti = ricerca ampia; valori bassi = focalizzazione marcata."
        )

    # ---------------------- GAZE VELOCITY ----------------------
    if not np.isnan(GV):
        lines.append(
            f"- **Gaze Velocity (GV_mean = {GV:.1f} px/s)** → velocità media del gaze: "
            "alta = scanning rapido; bassa = ispezione più profonda."
        )

    # ---------------------- SIGNAL QUALITY ----------------------
    if not np.isnan(SQI):
        lines.append(
            f"- **Signal Quality Index (SQI = {SQI:.2f})** → qualità del segnale. "
            "Valori > 0.85 indicano misurazioni molto affidabili."
        )

    # ---------------------- VARIABILITÀ ----------------------
    lines.append("\n### **Omogeneità del gruppo (variabilità)**\n")

    for metric in ["BR", "MBD", "PERC_EC", "GS", "GD", "GV_mean"]:
        val = std.get(metric)
        if not np.isnan(val):
            if val < 0.15 * mean[metric]:
                desc = "profilo omogeneo"
            elif val < 0.5 * mean[metric]:
                desc = "eterogeneità moderata"
            else:
                desc = "profilo altamente diversificato"
            lines.append(f"- **{metric}: σ = {val:.3f}** → {desc}")

    return "\n".join(lines)


def compute_clusters(df_group: pd.DataFrame, n_clusters: int = 3):
    """
    KMeans su EI, GLI, FI, LIN, RR, PEI, COG_X, COG_DELTA.
    Ritorna la serie di cluster (0..k-1) o None se non fattibile.
    """
    numeric_cols = METRIC_RADAR_ORDER
    if len(df_group) < 3:
        return None

    k = min(n_clusters, len(df_group))
    X = df_group[numeric_cols].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    return clusters

# ---------------------------- UI PRINCIPALE ---------------------------- #

def main():
    # --- Sidebar con info --- #
    st.sidebar.title("Info")
    st.sidebar.markdown(
    """
**Behavioral Eye-Tracking Dashboard**

Questa dashboard elabora dati di eye-tracking (Pupil Labs) per:

- Profilare il comportamento visivo del singolo soggetto  
- Calcolare indici di esplorazione, focus e coerenza cognitiva  
- Analizzare il gruppo (medie, varianza, cluster)  
- Visualizzare scatterplot, radar, heatmap aggregate  
- Stimare indici di **carico cognitivo** e **stabilità del gaze** (blink & gaze)

---

Carica uno ZIP con una cartella per soggetto, contenente almeno:

- `fixations.csv`  *(fissazioni, necessario)*  
- `saccades.csv`   *(saccadi, necessario)*  

Opzionale ma consigliato:

- `blinks.csv`     *(blink oculo-fisiologici → BR, MBD, PERC_EC, ecc.)*  
- `gaze.csv`       *(stream continuo di gaze → GS, GD, GV_mean, SQI)*  

Se i file opzionali non sono presenti, le metriche corrispondenti verranno lasciate vuote (NaN) senza interrompere l’analisi.
"""
)


    st.markdown(
    "<h1 style='text-align:center; color:#bc00ff;'>Behavioral Eye-Tracking Dashboard</h1>",
    unsafe_allow_html=True,
    )


    uploaded_zip = st.file_uploader(
        "Carica lo ZIP della sessione (multi-soggetto o singolo soggetto)",
        type=["zip"],
    )

    if uploaded_zip is None:
        st.info("In attesa di uno ZIP...")
        return

    # 1) Carichiamo soggetti dallo ZIP
    subjects = load_subjects_from_zip(uploaded_zip)
    if not subjects:
        st.error("Nello ZIP non sono stati trovati CSV. Controlla la struttura.")
        return

    st.success(f"Trovati {len(subjects)} soggetti: {', '.join(subjects.keys())}")

    # --- Accordion descrittivi (dopo upload) --- #
    with st.expander("📘 Introduzione & Legenda delle metriche", expanded=False):
        st.markdown(
        """
**Introduzione**

Questo strumento analizza il comportamento visivo dei soggetti a partire da dati di eye-tracking,
integrando misure di esplorazione, focus, efficienza del percorso, coerenza cognitiva (COG-X, COG-DELTA),
e indici oculo-fisiologici derivati da **blink** e **gaze** continui.

---

### Static metrics principali (scanpath & strategia visiva)

- **EI – Exploration Index**  
  Ampiezza e intensità dell’esplorazione visiva. Valori più alti indicano esplorazione più ampia.

- **GLI – Global–Local Index**  
  Negativo → strategie **locali/analitiche** (attenzione ai dettagli).  
  Positivo → strategie **globali/panoramiche** (visione d’insieme).

- **FI – Focus Index**  
  Stabilità del focus attentivo (tendenza a mantenere fissazioni più stabili nel tempo).

- **LIN – Linearity Index**  
  Linearità del percorso oculare. Valori bassi = percorso più tortuoso/pendolare.

- **RR – Regression Rate**  
  Frequenza dei movimenti di ritorno. Può indicare ricontrollo, incertezza o revisione.

- **PEI – Path Efficiency Index**  
  Efficienza del percorso di sguardo, dato dal rapporto tra distanza ideale e percorso osservato.

---

### Blink metrics (carico / stato oculo-fisiologico)

Richiedono `blinks.csv`.

- **BR – Blink Rate (blink/min)**  
  Numero di blink per minuto. Può riflettere stato di attivazione/calore emotivo e carico cognitivo.

- **MBD – Mean Blink Duration**  
  Durata media dei blink. Blink molto lunghi possono indicare affaticamento o momenti di disingaggio.

- **PERC_EC – Percent Eye Closure**  
  Percentuale di tempo con occhi chiusi sul totale della registrazione.

*(La metrica IBI_mean e la variabilità interna dei blink sono usate nell’engine, ma non sono ancora mostrate
esplicitamente in questa vista.)*

---

### Gaze metrics (stabilità, dispersione, qualità segnale)

Richiedono `gaze.csv`.

- **GS – Gaze Stability**  
  Somma delle varianze di x e y. Più alto = traiettoria più instabile / “rumorosa”.

- **GD – Gaze Dispersion (ellisse)**  
  Area dell’ellisse di covarianza dei punti di sguardo: quanto è “larga” la nuvola di gaze.

- **GV_mean – Gaze Velocity (media)**  
  Velocità media dello spostamento del gaze (px/s). Può riflettere dinamica di scanning e arousal.

- **SQI – Signal Quality Index**  
  Percentuale di campioni validi (confidence > soglia). Vicino a 1.0 → segnale affidabile.

---

### COG-X & COG-DELTA (indici cognitivi compositi)

- **COG-X – Coherency Index**  
  Indice sintetico di coerenza cognitiva del profilo visivo.  
  Valori alti → strategie più stabili/consistenti rispetto al compito.

- **COG-DELTA – Flexibility Index**  
  Indice di flessibilità attentiva dinamica. Valori alti → maggiore capacità di cambiare pattern visivo nel tempo.

---

### Metrics di gruppo

- Le **medie** descrivono lo stile “tipico” del gruppo.  
- Le **deviazioni standard** indicano quanto i soggetti differiscono tra loro.  
- I **cluster (KMeans)** raggruppano soggetti con pattern visivi simili (non sono categorie cliniche).

"""
        )


    with st.expander("📚 Bibliografia essenziale", expanded=False):
        st.markdown(
            """
- Holmqvist, K. et al. (2011). *Eye Tracking: A Comprehensive Guide to Methods and Measures.* Oxford University Press.  
- Duchowski, A. T. (2017). *Eye Tracking Methodology – Theory and Practice.* Springer.  
- Yarbus, A. L. (1967). *Eye Movements and Vision.* Springer.  
- Tatler, B. W., Hayhoe, M., Land, M., & Ballard, D. (2011). Eye guidance in natural vision. *Journal of Vision*, 11(5).  
- Noton, D., & Stark, L. (1971). Scanpaths in eye movements during pattern perception. *Science*, 171(3968), 308–311.  
- Goldberg, J. H., & Kotval, X. P. (1999). Computer interface evaluation using eye movements. *Int. J. Industrial Ergonomics*, 24(6), 631–645.
"""
        )

    all_results = {}
    group_rows = []
    all_fix_for_group = []
    temporal_per_subject = {}

    # 2) Analisi per soggetto
    for subj_name, files_dict in subjects.items():
        st.markdown("---")
        st.subheader(f"Soggetto: **{subj_name}**")

        fix, sacc, blinks, gaze = extract_fix_sacc(files_dict)
        if fix is None or sacc is None:
            st.error("Mancano fixations.csv o saccades.csv per questo soggetto.")
            continue

        res = compute_metrics_for_subject(fix, sacc, blinks=blinks, gaze=gaze)
        metrics = res["metrics"]
        profile = res["profile"]
        all_results[subj_name] = res
        temporal_per_subject[subj_name] = metrics.get("temporal", {})


        st.markdown("## 📊 Static Metrics – Profilo Cognitivo")

        # Descrizioni leggibili delle metriche
        metric_labels = {
            "EI": "Exploration Index",
            "GLI": "Global–Local Index",
            "FI": "Focus Index",
            "LIN": "Linearity Index",
            "RR": "Regression Rate",
            "PEI": "Path Efficiency Index",

            "BR": "Blink Rate (blink/min)",
            "MBD": "Mean Blink Duration (s)",
            "IBI_mean": "Inter-Blink Interval (s)",
            "PERC_EC": "Percent Eye Closure",

            "GS": "Gaze Stability (VarX+VarY)",
            "GD": "Gaze Dispersion (Elliptical Area)",
            "GV_mean": "Mean Gaze Velocity (px/s)",
            "SQI": "Signal Quality Index",

            "COG-X": "Cognitive Coherency Index",
            "COG-DELTA": "Attentional Flexibility Index",
        }

        # --- MACROAREE COGNITIVE (strutturazione scientifico-cognitiva) --- #
        macroareas = {
            "🧠 Cognitive Control & Focus": [
                "FI", "LIN", "RR", "COG-X", "COG-DELTA"
            ],
            "🔍 Exploration & Visual Search": [
                "EI", "GLI", "PEI"
            ],
            "👁️ Oculo-Physiology (Blink Behavior)": [
                "BR", "MBD", "IBI_mean", "PERC_EC"
            ],
            "🎯 Gaze Stability & Processing Dynamics": [
                "GS", "GD", "GV_mean", "SQI"
            ],
        }

        from streamlit.components.v1 import html

        table_html = """
        <style>
        .table-scroll-container {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #333;
            border-radius: 8px;
        }

        .table-container {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 16px;
            color: #c7d0d9;
            width: 100%;
        }
        .table-container table {
            border-collapse: collapse;
            width: 100%;
        }
        .table-container th, .table-container td {
            border: 1px solid #444;
            padding: 10px;
            text-align: left;
        }
        .table-container th {
            background-color: #111827;
            color: #00f6ff;
        }
        .table-container tr.section-header td {
            background-color: #0f172a;
            color: #bc00ff;
            font-size: 18px;
            font-weight: bold;
            padding-top: 18px;
        }
        </style>
        <div class="table-scroll-container">
        <div class="table-container">
            <table>
            <tr>
                <th>Metrica</th>
                <th>Descrizione</th>
                <th>Valore</th>
            </tr>
        """

        for area, metric_list in macroareas.items():
            table_html += f"""
            <tr class="section-header">
                <td colspan="3">{area}</td>
            </tr>
            """
            for m in metric_list:
                if m in ["COG-X", "COG-DELTA"]:
                    key = m.replace("-", "_")
                    val = profile.get(key, np.nan)
                else:
                    val = metrics.get(m, np.nan)
                val_str = f"{round(val, 4)}" if pd.notna(val) else "-"
                table_html += f"""
                <tr>
                    <td>{m}</td>
                    <td>{metric_labels[m]}</td>
                    <td>{val_str}</td>
                </tr>
                """

        table_html += """
            </table>
        </div>
        </div>
        """

        # ✅ Scroll abilitato
        html(table_html, height=800, scrolling=True)



        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Profilo radar del soggetto**")
            row_for_radar = {
                "EI": metrics["EI"],
                "GLI": metrics["GLI"],
                "FI": metrics["FI"],
                "LIN": metrics["LIN"],
                "RR": metrics["RR"],
                "PEI": metrics["PEI"],
                "COG_X": profile["COG_X"],
                "COG_DELTA": profile["COG_DELTA"],
            }
            fig_radar = radar_for_subject(subj_name, row_for_radar)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.markdown("**Fixation Scatter Plot (soggetto)**")
            fig_scatter = scatter_fix_plotly(fix, title=f"Fixations – {subj_name}")
            st.plotly_chart(fig_scatter, use_container_width=True)

                # Grafico temporale EI/GLI/FI
        with st.expander("Andamento temporale (EI / GLI / FI)", expanded=False):
            temporal = metrics.get("temporal", {})
            if temporal:
                metric_sc = st.selectbox(
                    "Seleziona la metrica temporale da visualizzare",
                    TEMPORAL_METRICS,
                    key=f"{subj_name}_temp_metric",
                )
                fig_temp = temporal_line_for_subject(temporal, metric_sc, subj_name)
                st.plotly_chart(fig_temp, use_container_width=True)
                st.markdown(
                    """
*L’asse orizzontale indica la progressione temporale del compito (T1–T3).  
Un incremento di **EI** segnala esplorazione più ampia,  
un incremento di **GLI** uno shift verso strategie più globali,  
un incremento di **FI** maggiore stabilità del focus.*"""
                )
            else:
                st.info("Metriche temporali non disponibili per questo soggetto.")

        with st.expander("Interpretazione psicologica (dettagli soggetto)", expanded=False):
            st.text(profile["summary"])
        # Cognitive Performance Layer (blink + gaze)
        with st.expander("Cognitive Performance Layer (blink + gaze)", expanded=False):
            perf_text = summarize_cognitive_layer_subject(metrics)
            st.markdown(perf_text)

        all_fix_for_group.append(fix)

        row_group = {
            "subject": subj_name,
            "EI": metrics.get("EI"),
            "GLI": metrics.get("GLI"),
            "FI": metrics.get("FI"),
            "LIN": metrics.get("LIN"),
            "RR": metrics.get("RR"),
            "PEI": metrics.get("PEI"),
            "BR": metrics.get("BR"),
            "MBD": metrics.get("MBD"),
            "PERC_EC": metrics.get("PERC_EC"),
            "GS": metrics.get("GS"),
            "GD": metrics.get("GD"),
            "GV_mean": metrics.get("GV_mean"),
            "SQI": metrics.get("SQI"),
            "COG_X": profile["COG_X"],
            "COG_DELTA": profile["COG_DELTA"],
        }
        group_rows.append(row_group)

    if not group_rows:
        st.warning("Nessun soggetto analizzabile (mancavano i CSV richiesti).")
        return

    # ----------------------- ANALISI DI GRUPPO --------------------------- #
    st.markdown("---")
    st.header("Analisi di Gruppo")

    df_group = pd.DataFrame(group_rows)

    st.subheader("Metriche individuali (tabella gruppo)")
    st.dataframe(df_group)

    numeric_cols = METRIC_RADAR_ORDER
    group_mean = df_group[numeric_cols].mean()
    group_std = df_group[numeric_cols].std()

    col_mean, col_std = st.columns(2)

    with col_mean:
        st.subheader("Profilo medio di gruppo (mean)")
        st.table(group_mean.to_frame(name="mean"))

    with col_std:
        st.subheader("Variabilità di gruppo (std)")
        st.table(group_std.to_frame(name="std"))

    st.subheader("Profilo radar di gruppo")
    radar_group_fig = radar_for_group(df_group)
    st.plotly_chart(radar_group_fig, use_container_width=True)
    # ---------------------------------------------------------
    # Cognitive Performance Layer – GRUPPO
    # ---------------------------------------------------------
    st.subheader("Cognitive Performance Layer – Gruppo (Blink + Gaze)")

    group_cog_text = summarize_cognitive_layer_group(df_group)
    st.markdown(group_cog_text)

    # -----------------------------------------------------------
    # Confronto temporale di gruppo (EI / GLI / FI)
    # -----------------------------------------------------------
    st.subheader("Confronto temporale di gruppo (EI / GLI / FI)")

    # Dizionario "etichetta leggibile" -> metrica interna
    TEMPORAL_METRIC_LABELS = {
    "Esplorazione (EI)": "EI",
    "Strategia Globale/Locale (GLI)": "GLI",
    "Focus Attentivo (FI)": "FI",
    }


    selected_label = st.selectbox(
    "Seleziona la metrica temporale per il confronto tra soggetti",
    list(TEMPORAL_METRIC_LABELS.keys()),
    )

    selected_metric = TEMPORAL_METRIC_LABELS[selected_label]


    # Costruzione tabella temporale di gruppo
    temporal_rows = []
    for subj_name, res in all_results.items():
        t = res["metrics"]["temporal"]
        temporal_rows.append(
            {
                "subject": subj_name,
                "EI_T1": t["EI_T1"],
                "EI_T2": t["EI_T2"],
                "EI_T3": t["EI_T3"],
                "GLI_T1": t["GLI_T1"],
                "GLI_T2": t["GLI_T2"],
                "GLI_T3": t["GLI_T3"],
                "FI_T1": t["FI_T1"],
                "FI_T2": t["FI_T2"],
                "FI_T3": t["FI_T3"],
            }
        )

    df_temporal = pd.DataFrame(temporal_rows)

    # Mappo la metrica scelta su T1/T2/T3
    metric_cols = [f"{selected_metric}_T1", f"{selected_metric}_T2", f"{selected_metric}_T3"]

    fig_temporal = go.Figure()
    for _, row in df_temporal.iterrows():
        fig_temporal.add_trace(
            go.Scatter(
                x=["T1", "T2", "T3"],
                y=[row[metric_cols[0]], row[metric_cols[1]], row[metric_cols[2]]],
                mode="lines+markers",
                name=row["subject"],
            )
        )

    fig_temporal.update_layout(
        xaxis_title="Fase temporale",
        yaxis_title=selected_metric,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c7d0d9"),
    )
    st.plotly_chart(fig_temporal, use_container_width=True)

 

    st.subheader("Heatmap aggregata del gruppo")
    fig_heat = heatmap_group_fix(all_fix_for_group)
    if fig_heat is not None:
        st.plotly_chart(fig_heat, use_container_width=True)

    clusters = compute_clusters(df_group)
    if clusters is not None:
        df_group["cluster"] = clusters
        st.subheader("Cluster dei soggetti (KMeans)")
        st.dataframe(df_group[["subject", "cluster"] + numeric_cols])

    st.subheader("Interpretazione psicologica del gruppo")
    interpretation_text = interpret_group(df_group, group_mean, group_std)
    st.text(interpretation_text)

    st.info(
        "Le metriche sono normalizzate o derivate in modo da poter essere confrontate "
        "tra soggetti con durate di registrazione differenti. "
        "L’analisi di gruppo fornisce una fotografia media del comportamento visivo, "
        "integrando anche coerenza (COG-X) e flessibilità (COG-DELTA)."
    )


if __name__ == "__main__":
    main()