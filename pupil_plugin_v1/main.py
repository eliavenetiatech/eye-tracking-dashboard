# main.py

from pupil_plugin_v1.loader.csv_loader import load_timeseries_data

# Metriche
from pupil_plugin_v1.metrics.exploration import compute_exploration
from pupil_plugin_v1.metrics.global_local_focus import compute_global_local_focus
from pupil_plugin_v1.metrics.systematicity import compute_systematicity
from pupil_plugin_v1.metrics.temporal import compute_temporal_dynamics
from pupil_plugin_v1.metrics.cog_indices import compute_cog_indices
from pupil_plugin_v1.metrics.blink_gaze import compute_blink_metrics, compute_gaze_metrics


# Profilazione
from pupil_plugin_v1.profiling.profile_builder import build_behavior_profile

# Grafici
from pupil_plugin_v1.utils.plots import (
    plot_fixations,
    plot_heatmap,
    plot_saccade_path
)

import json
import pandas as pd


# --------------------------------------------------------------------
#                    HTML REPORT GENERATION
# --------------------------------------------------------------------

def generate_html_report(profile: dict):
    """
    Genera il report HTML completo con:
    - Introduzione e bibliografia
    - Profilo comportamentale
    - Tags
    - Metriche statiche + legenda
    - Metriche temporali + legenda
    - Interpretazione psicologica
    - COG-X & COG-DELTA
    - Legenda dei grafici
    - Visualizzazioni
    """

    metrics   = profile["metrics"]
    temporal  = metrics["temporal"]

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <title>Behavioral Eye-Tracking Report</title>

  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 1000px;
      margin: 30px auto;
      padding: 20px;
      line-height: 1.6;
      color: #222;
    }}

    h1, h2, h3, h4 {{
      font-weight: 600;
      margin-top: 25px;
    }}

    .tags span {{
      display: inline-block;
      background-color: #eee;
      padding: 5px 8px;
      margin: 3px;
      border-radius: 4px;
      font-size: 0.85rem;
    }}

    table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 25px;
    }}

    th, td {{
      border: 1px solid #ccc;
      padding: 8px;
      font-size: 0.92rem;
    }}

    th {{
      background: #f3f3f3;
    }}

    img {{
      max-width: 100%;
      height: auto;
      margin-bottom: 25px;
      border: 1px solid #ddd;
    }}

    pre {{
      background: #fafafa;
      border: 1px solid #ddd;
      padding: 15px;
      white-space: pre-wrap;
      border-radius: 6px;
      font-size: 0.92rem;
    }}

  </style>

</head>


<body>

  <!-- --------------------- INTRO + BIBLIO -------------------------------- -->

  <h2>Introduzione</h2>
  <pre>
Questo report analizza il comportamento visivo del soggetto attraverso un insieme di metriche di eye-tracking
scientificamente validate, derivate dalla letteratura sullo scanning visivo, sulle dinamiche saccadiche e sulla
distribuzione dell’attenzione.

Riferimenti scientifici principali:

- Holmqvist, K. et al. (2011). *Eye Tracking: A Comprehensive Guide to Methods and Measures.* Oxford University Press.
- Yarbus, A. L. (1967). *Eye Movements and Vision.* Springer.
- Duchowski, A. T. (2017). *Eye Tracking Methodology — Theory and Practice.* Springer.
- Noton, D., & Stark, L. (1971). Scanpaths in pattern perception. *Science*, 171(3968), 308–311.
- Velichkovsky, B. M. et al. (2005). Two visual systems and their eye movements. *Cognitive Processing*, 6(1), 55–67.
- Goldberg, J. & Kotval, X. (1999). Interface evaluation through eye movements. *IJIE*, 24(6), 631–645.
- Tatler, B. et al. (2011). Eye guidance in natural vision. *Journal of Vision*, 11(5).

Queste fonti costituiscono la base teorica delle metriche utilizzate nel report.
  </pre>


  <!-- --------------------- HOW TO READ -------------------------------- -->

  <h2>How to Read This Report</h2>
  <pre>
STATIC METRICS – Quick Guide
- EI: indice di esplorazione visiva (alto = esplorazione ampia).
- GLI: negativo = local; positivo = global.
- FI: stabilità attentiva (alto = focus più stabile).
- LIN: linearità dei movimenti oculari.
- RR: movimenti di ritorno (ricontrollo).
- PEI: efficienza del percorso visivo.
- COG-X: coerenza cognitiva globale.
- COG-DELTA: flessibilità attentiva dinamica.

TEMPORAL METRICS – Quick Guide
- EI crescente → esplorazione più ampia.
- GLI crescente → shift verso visione globale.
- FI crescente → aumento della stabilità attentiva.

Nota: nessuna metrica è “positiva” o “negativa” in senso assoluto: dipende dal compito e dal profilo individuale.
  </pre>



  <!-- --------------------- PROFILE ---------------------------- -->

  <h2>Profile</h2>
  <p><strong>{profile["label"]}</strong></p>

  <h3>Tags</h3>
  <div class="tags">
    {''.join(f'<span>{tag}</span>' for tag in profile["tags"])}
  </div>


  <!-- --------------------- STATIC METRICS ----------------------- -->

  <h3>Static Metrics</h3>

  <table>
    <tr><th>Metrica</th><th>Valore</th></tr>
    <tr><td>EI</td><td>{metrics["EI"]:.3f}</td></tr>
    <tr><td>GLI</td><td>{metrics["GLI"]:.3f}</td></tr>
    <tr><td>FI</td><td>{metrics["FI"]:.3f}</td></tr>
    <tr><td>LIN</td><td>{metrics["LIN"]:.3f}</td></tr>
    <tr><td>RR</td><td>{metrics["RR"]:.3f}</td></tr>
    <tr><td>PEI</td><td>{metrics["PEI"]:.3f}</td></tr>
    <tr><td>COG-X</td><td>{profile["COG_X"]:.3f} ({profile["cogx_level"]})</td></tr>
    <tr><td>COG-DELTA</td><td>{profile["COG_DELTA"]:.3f} ({profile["cogdelta_level"]})</td></tr>
  </table>

  <h4>Legenda — Static Metrics</h4>
  <pre>
EI: esplorazione visiva.
GLI: negativo = local (dettaglio); positivo = global (panoramica).
FI: stabilità del focus.
LIN: linearità del percorso oculare.
RR: movimenti di ritorno.
PEI: efficienza del percorso.
COG-X: coerenza cognitiva globale.
COG-DELTA: flessibilità attentiva.
  </pre>


  <!-- ---------------- TEMPORAL DYNAMICS ------------------------- -->

  <h3>Temporal Dynamics</h3>

  <table>
    <tr><th>Metrica</th><th>T1</th><th>T2</th><th>T3</th></tr>
    <tr><td>EI</td><td>{temporal["EI_T1"]:.3f}</td><td>{temporal["EI_T2"]:.3f}</td><td>{temporal["EI_T3"]:.3f}</td></tr>
    <tr><td>GLI</td><td>{temporal["GLI_T1"]:.3f}</td><td>{temporal["GLI_T2"]:.3f}</td><td>{temporal["GLI_T3"]:.3f}</td></tr>
    <tr><td>FI</td><td>{temporal["FI_T1"]:.3f}</td><td>{temporal["FI_T2"]:.3f}</td><td>{temporal["FI_T3"]:.3f}</td></tr>
  </table>

  <h4>Legenda — Temporal Metrics</h4>
  <pre>
EI (T1–T3): indica come cambia l’esplorazione nel tempo.
GLI (T1–T3): shift tra strategia “local” e “global”.
FI  (T1–T3): variazioni nella qualità del focus.
  </pre>


  <!-- ---------------- PSYCHOLOGICAL INTERPRETATION -------------- -->

  <h3>Psychological Interpretation</h3>
  <pre>{profile["psychology"]}</pre>


  <!-- ----------------- COG-X & COG-DELTA SECTION ----------------- -->

  <h3>COG-X & COG-DELTA</h3>

  <p><strong>COG-X Index:</strong> {profile["COG_X"]:.3f} ({profile["cogx_level"]})</p>
  <p><strong>COG-DELTA Index:</strong> {profile["COG_DELTA"]:.3f} ({profile["cogdelta_level"]})</p>

  <h4>Normative Ranges & Interpretation</h4>
  <pre>{profile["cog_section"]}</pre>



  <!-- --------------------- GRAPH LEGEND -------------------------- -->

  <h3>Reading the Visualizations</h3>
  <pre>
Fixation Scatter Plot:
  - Ogni punto è una fissazione.
  - La distribuzione indica le zone esplorate e la densità attentiva.

Heatmap delle fissazioni:
  - Aree più calde = maggiore attenzione.
  - Metodo KDE ampiamente utilizzato per analizzare salienze visive.

Saccade Path Plot:
  - Sequenza dei movimenti oculari.
  - Utile per osservare ritorni, pendolarità, zig-zag.
  - Non ha interpretazione psicologica diretta ma mostra lo stile di scanning.
  </pre>



  <!-- --------------------- VISUALIZATIONS ------------------------- -->

  <h3>Visualizations</h3>

  <h4>Fixation Scatter Plot</h4>
  <img src="fixations_scatter.png">

  <h4>Fixation Heatmap</h4>
  <img src="fixations_heatmap.png">

  <h4>Saccade Path Plot</h4>
  <img src="saccade_path.png">


</body>
</html>
"""

    with open("behavior_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Report HTML salvato in behavior_report.html")



# --------------------------------------------------------------------
#                            MAIN PIPELINE
# --------------------------------------------------------------------

def estimate_session_duration_s(fix, sacc, gaze, blinks):
    """
    Stima la durata totale della registrazione (in secondi)
    cercando colonne tempo nei vari DataFrame.
    Usa la massima estensione temporale trovata.
    """

    dfs = [fix, sacc, gaze, blinks]
    max_span = 0.0

    for df in dfs:
        if df is None or df.empty:
            continue

        # colonne candidate: timestamp, time, ts, ecc.
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

        span = float(tmax - tmin)

        # se sembra millisecondi, converto in secondi
        if span > 2000:
            span = span / 1000.0

        if span > max_span:
            max_span = span

    return max_span


def main():

    # 1. Carica dati
    data = load_timeseries_data()
    fix  = data["fix"]
    sacc = data["sacc"]
    gaze = data.get("gaze")
    blinks = data.get("blinks")

    # 2. Metriche statiche
    m_exp  = compute_exploration(fix, sacc)
    m_glf  = compute_global_local_focus(fix, sacc)
    m_sys  = compute_systematicity(fix)
    m_temp = compute_temporal_dynamics(fix, sacc)

    # 1) stima durata sessione (serve per i blink)
    session_duration_s = estimate_session_duration_s(fix, sacc, gaze, blinks)

    # 2) metriche blink (se disponibili)
    blink_metrics = {}
    if blinks is not None and not blinks.empty:
        blink_metrics = compute_blink_metrics(blinks, session_duration_s)

    # 3) metriche gaze (se disponibili)
    gaze_metrics = {}
    if gaze is not None and not gaze.empty:
        gaze_metrics = compute_gaze_metrics(gaze)

    # 4) aggregazione totale metriche
    metrics = {
        **m_exp,
        **m_glf,
        **m_sys,
        **blink_metrics,
        **gaze_metrics,
        "temporal": m_temp,
    }


    # 2b. COG-X e COG-DELTA
    cog = compute_cog_indices(metrics)
    metrics.update(cog)

    # 3. Profilo comportamentale
    profile = build_behavior_profile(metrics)

    # 4. Mostra a terminale
    print(profile["summary"])

    # 5. Salva JSON
    with open("behavior_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print("Profilo salvato in behavior_profile.json")

    # 6. Report HTML
    generate_html_report(profile)

    # 7. Grafici
    plot_fixations(fix, save_path="fixations_scatter.png", show=True)
    plot_heatmap(fix, save_path="fixations_heatmap.png", show=True)
    plot_saccade_path(fix, save_path="saccade_path.png", show=True)



if __name__ == "__main__":
    main()
