# profiling/profile_builder.py


def build_behavior_profile(metrics: dict) -> dict:
    """
    Costruisce un profilo comportamentale a partire dalle metriche.
    Ritorna:
    - label: stringa compatta
    - summary: testo descrittivo
    - tags: lista di etichette brevi
    - psychology: interpretazione psicologica generale
    - cog_section: interpretazione specifica di COG-X e COG-DELTA
    - COG_X, COG_DELTA: valori numerici
    - metrics: tutte le metriche originali
    """
    EI  = metrics.get("EI")
    GLI = metrics.get("GLI")
    FI  = metrics.get("FI")

    LIN = metrics.get("LIN")
    RR  = metrics.get("RR")
    PEI = metrics.get("PEI")

    T   = metrics.get("temporal", {})

    COG_X      = metrics.get("COG_X")
    COG_DELTA  = metrics.get("COG_DELTA")

    tags = []

    # Esplorazione
    if EI is not None and EI > 10:
        tags.append("high_exploration")
        expl_label = "Exploratory"
    else:
        tags.append("focused_exploration")
        expl_label = "Focused Exploration"

    # Global vs Local
    if GLI is not None and GLI < 0:
        tags.append("local_bias")
        gl_label = "Local"
    else:
        tags.append("global_bias")
        gl_label = "Global"

    # Focus
    if FI is not None and FI > 4:
        tags.append("moderate_focus")
        focus_label = "Moderate Focus"
    else:
        tags.append("diffuse_focus")
        focus_label = "Diffuse Focus"

    # Systematicity
    if (LIN is not None and LIN > 0.3) and (RR is not None and RR < 0.3):
        tags.append("systematic")
        scan_label = "Systematic"
    else:
        tags.append("opportunistic")
        scan_label = "Opportunistic"

    # Path efficiency
    if PEI is not None and PEI > 0.3:
        tags.append("efficient_path")
        path_label = "Efficient Path"
    else:
        tags.append("inefficient_path")
        path_label = "Inefficient Path"

    profile_label = " – ".join(
        [expl_label, gl_label, focus_label, scan_label, path_label]
    )

    # --- Interpretazione psicologica generale ---
    psych_lines = []

    # Curiosità / esplorazione
    if "high_exploration" in tags:
        psych_lines.append(
            "- Tendenza ad esplorare ampiamente lo spazio informativo, "
            "con un'elevata sensibilità agli stimoli presenti."
        )
    else:
        psych_lines.append(
            "- Esplorazione più mirata, verosimilmente guidata da obiettivi o aree di interesse specifiche."
        )

    # Local vs global
    if "local_bias" in tags:
        psych_lines.append(
            "- Bias verso il dettaglio: la persona sembra lavorare molto su elementi locali "
            "più che costruire subito una mappa globale."
        )
    else:
        psych_lines.append(
            "- Bias verso una visione d'insieme: tendenza a costruire rapidamente una mappa globale dello stimolo."
        )

    # Focus
    if "moderate_focus" in tags:
        psych_lines.append(
            "- Livello di focus moderato: capacità di mantenere l'attenzione su elementi rilevanti "
            "pur continuando ad esplorare."
        )
    else:
        psych_lines.append(
            "- Focus più diffuso: possibile ricerca di più alternative o difficoltà a stabilizzarsi su un singolo target."
        )

    # Stile di scanning
    if "systematic" in tags:
        psych_lines.append(
            "- Stile di scanning relativamente sistematico: pattern di lettura/scan coerente e ripetibile."
        )
    else:
        psych_lines.append(
            "- Stile di scanning opportunistico: il percorso sembra emergere 'dal momento', "
            "con numerosi ritorni e cambi di direzione."
        )

    # Efficienza
    if "efficient_path" in tags:
        psych_lines.append(
            "- Percorso oculare efficiente: poche ridondanze, buona ottimizzazione del movimento."
        )
    else:
        psych_lines.append(
            "- Percorso oculare poco efficiente: molte ridondanze e ritorni, compatibili con indecisione, "
            "ricontrollo o esplorazione ampia non pianificata."
        )

    psychology_text = "\n".join(psych_lines)

    # --- COG-X & COG-DELTA: livelli normativi ---
    def classify_cogx(x):
        if x is None or x != x:
            return "non disponibile"
        if x >= 0.80:
            return "alta coerenza cognitiva"
        if x >= 0.60:
            return "buona coerenza cognitiva"
        if x >= 0.40:
            return "coerenza cognitiva ridotta"
        if x >= 0.20:
            return "bassa coerenza cognitiva"
        return "marcata disorganizzazione cognitiva"

    def classify_cogdelta(d):
        if d is None or d != d:
            return "non disponibile"
        if d <= 0.20:
            return "stile rigido (bassa flessibilità)"
        if d <= 0.60:
            return "flessibilità attentiva adattiva"
        return "switching instabile / volatile"

    cogx_level = classify_cogx(COG_X)
    cogdelta_level = classify_cogdelta(COG_DELTA)

    # Testo descrittivo specifico per COG-X e COG-DELTA
    cog_lines = []

    cog_lines.append(
        f"COG-X Index (coerenza cognitiva globale) = {COG_X:.3f} → {cogx_level}."
        if COG_X == COG_X else
        "COG-X Index non disponibile (dati insufficienti)."
    )
    cog_lines.append(
        "Questo indice sintetizza la stabilità dell'esplorazione (EI), "
        "la coerenza tra strategie globali/locali (GLI) e l'efficienza del percorso (PEI)."
    )

    cog_lines.append("")
    cog_lines.append("Valori normativi COG-X:")
    cog_lines.append("  0.80 – 1.00 : alta coerenza cognitiva")
    cog_lines.append("  0.60 – 0.79 : buona coerenza cognitiva")
    cog_lines.append("  0.40 – 0.59 : coerenza ridotta")
    cog_lines.append("  0.20 – 0.39 : bassa coerenza")
    cog_lines.append("  0.00 – 0.19 : disorganizzazione cognitiva marcata")

    cog_lines.append("")
    cog_lines.append(
        f"COG-DELTA Index (dinamica di switching attentivo) = {COG_DELTA:.3f} → {cogdelta_level}."
        if COG_DELTA == COG_DELTA else
        "COG-DELTA Index non disponibile (dati insufficienti)."
    )
    cog_lines.append(
        "Questo indice riflette quanto il soggetto cambia strategia tra globale/locale e focus "
        "nel tempo: valori troppo bassi indicano rigidità, valori intermedi flessibilità adattiva, "
        "valori molto alti instabilità e volatilità."
    )

    cog_lines.append("")
    cog_lines.append("Valori normativi COG-DELTA:")
    cog_lines.append("  0.00 – 0.20 : bassa flessibilità (stile rigido)")
    cog_lines.append("  0.21 – 0.60 : flessibilità adattiva")
    cog_lines.append("  0.61 – 1.00 : switching instabile / volatile")

    cog_section = "\n".join(cog_lines)

    summary = f"""
Profilo Comportamentale:
  {profile_label}

Static metrics:
  EI  = {EI:.3f}
  GLI = {GLI:.3f}
  FI  = {FI:.3f}
  LIN = {LIN:.3f}
  RR  = {RR:.3f}
  PEI = {PEI:.3f}
  COG-X = {COG_X:.3f}
  COG-DELTA = {COG_DELTA:.3f}

Temporal dynamics:
  EI  T1→T2→T3 : {T.get('EI_T1'):.3f} → {T.get('EI_T2'):.3f} → {T.get('EI_T3'):.3f}
  GLI T1→T2→T3 : {T.get('GLI_T1'):.3f} → {T.get('GLI_T2'):.3f} → {T.get('GLI_T3'):.3f}
  FI  T1→T2→T3 : {T.get('FI_T1'):.3f} → {T.get('FI_T2'):.3f} → {T.get('FI_T3'):.3f}

Psychological interpretation:
{psychology_text}

COG-X & COG-DELTA:
{cog_section}
"""

    return {
        "label": profile_label,
        "summary": summary,
        "tags": tags,
        "psychology": psychology_text,
        "cog_section": cog_section,
        "COG_X": COG_X,
        "COG_DELTA": COG_DELTA,
        "metrics": metrics,
        "cogx_level": cogx_level,
        "cogdelta_level": cogdelta_level,
    }
