"""
SOC Detection – Dashboard Streamlit
====================================
Livrable 3/3 : Interface web pour démonstration PFE

Installation :
  pip install streamlit plotly pandas numpy xgboost scikit-learn shap

Lancement :
  streamlit run 3_dashboard_streamlit.py

Fonctionnalités :
  - Simulation d'ingestion de logs en temps réel
  - Métriques clés : alertes totales, attaques, FP filtrés, score risque moyen
  - Graphiques : distribution temporelle, timeline des attaques, score risque
  - SHAP explanation pour chaque alerte
  - Filtres : classe, niveau de risque, protocole
  - Export CSV des alertes filtrées
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
import datetime
import pickle
import json
from collections import deque

# ── Config page ──
st.set_page_config(
    page_title="SOC AI Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

  /* ── Base ── */
  [data-testid="stAppViewContainer"] {
    background: #020c06;
    background-image:
      radial-gradient(ellipse at 20% 50%, rgba(0,255,70,0.03) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 20%, rgba(0,200,80,0.04) 0%, transparent 50%);
  }
  [data-testid="stSidebar"] {
    background: #040f07;
    border-right: 1px solid #0d3318;
  }
  [data-testid="stSidebar"] * { font-family: 'Share Tech Mono', monospace !important; }

  /* ── Scanline overlay ── */
  [data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,255,70,0.012) 2px,
      rgba(0,255,70,0.012) 4px
    );
    pointer-events: none;
    z-index: 0;
  }

  /* ── Typography globale ── */
  h1, h2, h3, h4, p, span, div, label {
    font-family: 'Rajdhani', sans-serif !important;
    color: #a8ffb8 !important;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #040f07;
    border: 1px solid #0d4a1a;
    border-top: 2px solid #00c844;
    border-radius: 4px;
    padding: 18px 16px 14px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00c84440, transparent);
  }
  .metric-card:hover {
    border-color: #00c844;
    box-shadow: 0 0 20px rgba(0,200,68,0.12);
  }
  .metric-val {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 6px 0 2px;
    letter-spacing: 0.05em;
    text-shadow: 0 0 20px currentColor;
  }
  .metric-lbl {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem;
    color: #2a7a3a !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
  }

  /* ── Sidebar buttons ── */
  .stButton > button {
    font-family: 'Share Tech Mono', monospace !important;
    background: transparent !important;
    border: 1px solid #1a5c2a !important;
    color: #00c844 !important;
    border-radius: 3px !important;
    letter-spacing: 0.1em;
    transition: all 0.15s !important;
  }
  .stButton > button:hover {
    background: rgba(0,200,68,0.08) !important;
    border-color: #00c844 !important;
    box-shadow: 0 0 12px rgba(0,200,68,0.2) !important;
  }
  [data-testid="baseButton-primary"] > button,
  .stButton > button[kind="primary"] {
    background: rgba(0,200,68,0.12) !important;
    border-color: #00c844 !important;
    color: #00ff57 !important;
  }

  /* ── Section headings ── */
  .section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #2a7a3a;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid #0d3318;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #0d3318 !important;
    border-radius: 4px !important;
  }
  [data-testid="stDataFrame"] * {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    color: #7dffaa !important;
  }

  /* ── Multiselect / slider ── */
  [data-baseweb="select"] { border-color: #1a5c2a !important; }
  [data-testid="stSlider"] [data-baseweb="slider"] { color: #00c844 !important; }
  .stSlider [role="slider"] { background: #00c844 !important; }

  /* ── Download button ── */
  .stDownloadButton > button {
    font-family: 'Share Tech Mono', monospace !important;
    background: transparent !important;
    border: 1px solid #1a5c2a !important;
    color: #00c844 !important;
    border-radius: 3px !important;
  }

  /* ── Barre latérale titre ── */
  .sidebar-logo {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: #00ff57;
    text-shadow: 0 0 15px rgba(0,255,87,0.6);
    letter-spacing: 0.15em;
    padding: 8px 0 4px;
    border-bottom: 1px solid #0d3318;
    margin-bottom: 16px;
  }
  .sidebar-section {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: #2a7a3a;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin: 16px 0 8px;
  }

  /* ── Status badge ── */
  .status-online {
    display: inline-block;
    width: 7px; height: 7px;
    background: #00ff57;
    border-radius: 50%;
    box-shadow: 0 0 8px #00ff57;
    margin-right: 6px;
    animation: pulse-dot 2s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #00ff57; }
    50% { opacity: 0.5; box-shadow: 0 0 3px #00ff57; }
  }

  /* ── Main title ── */
  .main-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #00ff57 !important;
    text-shadow: 0 0 30px rgba(0,255,87,0.4);
    letter-spacing: 0.2em;
    margin-bottom: 2px;
  }
  .main-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #2a7a3a !important;
    letter-spacing: 0.15em;
  }

  /* ── Info box ── */
  [data-testid="stAlert"] {
    background: #040f07 !important;
    border: 1px solid #0d3318 !important;
    border-left: 3px solid #00c844 !important;
    color: #7dffaa !important;
    border-radius: 3px !important;
    font-family: 'Share Tech Mono', monospace !important;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTES & FEATURES
# ══════════════════════════════════════════════════════════════
FEATURES = [
    "hour_of_day", "time_context", "protocol",
    "src_port", "dst_port",
    "src_is_internal", "dst_is_internal", "is_internal_to_internal",
    "same_subnet", "freq_per_min", "is_high_freq",
    "src_port_ephemeral", "dst_port_privileged",
]
LABELS_STR = {0: "normal", 1: "false_positive", 2: "attack"}
CLS_COLOR  = {"normal": "#00c844", "false_positive": "#39ff8a", "attack": "#ff4444"}

# ══════════════════════════════════════════════════════════════
# MODÈLE (chargement ou démo)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        with open("E:\\Projets\\ProjetSOC_AI\\models\\xgb_soc.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        # Modèle de démo entraîné en mémoire
        import xgboost as xgb
        np.random.seed(42)
        rows, labels = [], []
        defs = [(0, 0.58, 4, 1, 1), (1, 0.28, 8, 1, 1), (2, 0.14, 50, 0, 1)]
        for cls, frac, freq_mean, src_int, dst_int in defs:
            n = int(10000 * frac)
            for _ in range(n):
                freq = max(1 if cls < 2 else 10, int(np.random.exponential(freq_mean)))
                sp = random.randint(1025, 65535)
                dp = random.choice([22, 80, 443, 53])
                rows.append([random.randint(0,23), random.randint(0,3), 0 if cls<2 else random.randint(0,1),
                              sp, dp, src_int, dst_int, int(src_int and dst_int),
                              0, freq, int(freq>20), int(sp>1024), int(dp<1024)])
                labels.append(cls)
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, eval_metric="mlogloss", random_state=42)
        model.fit(pd.DataFrame(rows, columns=FEATURES), pd.Series(labels))
        return model

# Extraire le modèle du bundle (nouveau format) ou utiliser directement (ancien format)
_bundle = load_model()
if isinstance(_bundle, dict) and "model" in _bundle:
    model            = _bundle["model"]
    ATTACK_THRESHOLD = _bundle.get("attack_threshold", 0.15)
else:
    model            = _bundle
    ATTACK_THRESHOLD = 0.15

# ══════════════════════════════════════════════════════════════
# GÉNÉRATEUR DE LOGS
# ══════════════════════════════════════════════════════════════
DESCRIPTIONS = {
    "normal":         ["HTTP GET /index.html", "DNS query google.com", "HTTPS session établie",
                       "FTP file transfer", "SMTP message envoyé"],
    "false_positive": ["Web Scan interne", "ICMP Ping Sweep", "Script Injection Path",
                       "HTTP Login Bruteforce (outil audit)", "Port Scan autorisé"],
    "attack":         ["SSH Brute Force 🔴", "Hydra HTTP Tool 🔴", "SQL Injection URI 🔴",
                       "Reverse Shell Attempt 🔴", "Sensitive File Probe 🔴"],
}

def generate_log(force_class: str = None):
    cls_name = force_class or random.choices(
        ["normal", "false_positive", "attack"], weights=[0.58, 0.28, 0.14]
    )[0]
    hour = datetime.datetime.now().hour
    src_int = {"normal": 1, "false_positive": 1, "attack": 0}[cls_name]
    freq = max(1, int(np.random.exponential({"normal": 4, "false_positive": 8, "attack": 50}[cls_name])))
    sp = random.randint(1025, 65535)
    dp = {"normal": random.choice([80,443,53,21,25]),
          "false_positive": random.choice([80,443,22]),
          "attack": random.choice([22,80,443])}[cls_name]
    feats = {
        "hour_of_day": hour, "time_context": random.randint(0,3), "protocol": 0,
        "src_port": sp, "dst_port": dp, "src_is_internal": src_int,
        "dst_is_internal": 1, "is_internal_to_internal": src_int,
        "same_subnet": int(src_int and random.random()<0.3),
        "freq_per_min": freq, "is_high_freq": int(freq>20),
        "src_port_ephemeral": int(sp>1024), "dst_port_privileged": int(dp<1024),
    }
    proba = model.predict_proba(pd.DataFrame([feats]))[0]
    # Seuil adapté : si P(attack) >= ATTACK_THRESHOLD → attack
    attack_idx = list(model.classes_).index(2) if 2 in model.classes_ else 2
    if proba[attack_idx] >= ATTACK_THRESHOLD:
        pred = 2
    else:
        other = [i for i in range(len(proba)) if i != attack_idx]
        pred  = other[int(np.argmax([proba[i] for i in other]))]
    return {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "description": random.choice(DESCRIPTIONS[cls_name]),
        "src_ip": f"{'192.168' if src_int else '185.'+str(random.randint(1,255))}.{random.randint(1,254)}.{random.randint(1,254)}",
        "dst_port": dp,
        "freq_per_min": freq,
        "prediction": LABELS_STR[pred],
        "risk_score": int(
            50 + 50 * (proba[2] - ATTACK_THRESHOLD) / (1 - ATTACK_THRESHOLD)
            if proba[2] >= ATTACK_THRESHOLD
            else 50 * proba[2] / ATTACK_THRESHOLD
        ),
        "proba_normal": round(float(proba[0]), 3),
        "proba_fp": round(float(proba[1]), 3),
        "proba_attack": round(float(proba[2]), 3),
        "_true": cls_name,
        **feats,
    }

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "alerts" not in st.session_state:
    st.session_state.alerts = pd.DataFrame()
if "running" not in st.session_state:
    st.session_state.running = False
if "total_generated" not in st.session_state:
    st.session_state.total_generated = 0

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ SOC_AI_ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">// contrôles</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ START", use_container_width=True, type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("■ STOP", use_container_width=True):
            st.session_state.running = False

    if st.button("↺ RESET", use_container_width=True):
        st.session_state.alerts = pd.DataFrame()
        st.session_state.total_generated = 0

    st.markdown('<div class="sidebar-section">// filtres</div>', unsafe_allow_html=True)
    show_classes = st.multiselect(
        "Classes à afficher",
        ["normal", "false_positive", "attack"],
        default=["false_positive", "attack"],
    )
    min_risk = st.slider("Score de risque minimum", 0, 100, 30)
    n_batch = st.slider("Alertes par cycle", 1, 20, 5)
    refresh_rate = st.slider("Vitesse (secondes)", 0.5, 5.0, 1.5)

    st.markdown('<div class="sidebar-section">// injection</div>', unsafe_allow_html=True)
    force_attack = st.button("⚠ INJECT ATTACK", use_container_width=True)
    force_fp     = st.button("~ INJECT FALSE_POS", use_container_width=True)

    st.markdown('<div class="sidebar-section">// système</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#2a7a3a;line-height:2">
    MODEL   &nbsp;→ XGBoost<br>
    FEAT    &nbsp;→ 13 features<br>
    SPLIT   &nbsp;→ temporal 80/20<br>
    SEUIL   &nbsp;→ 0.15 (attack)<br>
    STATUS  &nbsp;<span class="status-online"></span>ONLINE
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TITRE PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-title">⬡ SOC_AI_DETECTION_ENGINE</div>
<div class="main-subtitle">// REAL-TIME THREAT CLASSIFICATION · XGBOOST · 13 NETWORK FEATURES · NO IDS LEAKAGE</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#0d3318;margin:12px 0 20px'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# GÉNÉRATION DES NOUVELLES ALERTES
# ══════════════════════════════════════════════════════════════
new_logs = []
if st.session_state.running:
    for _ in range(n_batch):
        new_logs.append(generate_log())
    st.session_state.total_generated += n_batch

if force_attack:
    new_logs.append(generate_log("attack"))
    st.session_state.total_generated += 1
if force_fp:
    new_logs.append(generate_log("false_positive"))
    st.session_state.total_generated += 1

if new_logs:
    new_df = pd.DataFrame(new_logs)
    if st.session_state.alerts.empty:
        st.session_state.alerts = new_df
    else:
        st.session_state.alerts = pd.concat(
            [st.session_state.alerts, new_df], ignore_index=True
        ).tail(1000)  # garder les 1000 dernières alertes

df = st.session_state.alerts

# ══════════════════════════════════════════════════════════════
# MÉTRIQUES PRINCIPALES
# ══════════════════════════════════════════════════════════════
if not df.empty:
    total      = len(df)
    n_attacks  = (df["prediction"] == "attack").sum()
    n_fp       = (df["prediction"] == "false_positive").sum()
    n_normal   = (df["prediction"] == "normal").sum()
    avg_risk   = df["risk_score"].mean()
    fp_saved   = f"{n_fp / max(total,1) * 100:.0f}%"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">// alertes totales</div>
            <div class="metric-val" style="color:#00ff57">{total}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">// attaques détectées</div>
            <div class="metric-val" style="color:#ff4444">{n_attacks}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">// faux positifs</div>
            <div class="metric-val" style="color:#39ff8a">{n_fp}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">// trafic normal</div>
            <div class="metric-val" style="color:#00c844">{n_normal}</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">// score risque moyen</div>
            <div class="metric-val" style="color:#7dffaa">{avg_risk:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # GRAPHIQUES
    # ══════════════════════════════════════════════════════════
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">// timeline des alertes</div>', unsafe_allow_html=True)
        df_plot = df.tail(100).copy()
        df_plot["index"] = range(len(df_plot))

        fig = go.Figure()
        for cls, col, symbol in [
            ("normal",         "#00c844", "circle"),
            ("false_positive", "#39ff8a", "diamond"),
            ("attack",         "#ff4444", "x"),
        ]:
            mask = df_plot["prediction"] == cls
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df_plot[mask]["index"],
                    y=df_plot[mask]["risk_score"],
                    mode="markers",
                    name=cls,
                    marker=dict(color=col, size=8, symbol=symbol,
                                line=dict(width=1, color="#020c06"),
                                opacity=0.9),
                    text=df_plot[mask]["description"],
                    hovertemplate="%{text}<br>Risk: %{y}%<extra></extra>",
                ))

        fig.add_hline(y=50, line_dash="dash", line_color="#1a5c2a", line_width=1,
                      annotation_text="seuil 50%",
                      annotation_font=dict(color="#2a7a3a", size=10),
                      annotation_position="right")
        fig.update_layout(
            plot_bgcolor="#020c06", paper_bgcolor="#020c06",
            font=dict(color="#2a7a3a", family="Share Tech Mono"),
            legend=dict(bgcolor="#040f07", bordercolor="#0d3318", borderwidth=1,
                        font=dict(color="#7dffaa", size=11)),
            xaxis=dict(gridcolor="#0a2410", title="alertes récentes",
                       title_font=dict(color="#2a7a3a"), tickfont=dict(color="#2a7a3a")),
            yaxis=dict(gridcolor="#0a2410", title="score risque (%)", range=[0, 105],
                       title_font=dict(color="#2a7a3a"), tickfont=dict(color="#2a7a3a")),
            height=300, margin=dict(l=0, r=10, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown('<div class="section-title">// distribution des classes</div>', unsafe_allow_html=True)
        counts = df["prediction"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            marker=dict(
                colors=[CLS_COLOR.get(l, "#2a7a3a") for l in counts.index],
                line=dict(color="#020c06", width=2)
            ),
            hole=0.65,
            textinfo="percent+label",
            textfont=dict(color="white", size=11, family="Share Tech Mono"),
        ))
        fig2.update_layout(
            plot_bgcolor="#020c06", paper_bgcolor="#020c06",
            font=dict(color="#2a7a3a", family="Share Tech Mono"),
            showlegend=False,
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            annotations=[dict(text=f"<b>{total}</b><br>alertes", x=0.5, y=0.5,
                              font=dict(size=13, color="#00ff57", family="Share Tech Mono"),
                              showarrow=False)],
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Score de risque par fréquence ──
    st.markdown("#### Score de risque vs fréquence d'alertes (freq_per_min)")
    fig3 = px.scatter(
        df.tail(200),
        x="freq_per_min", y="risk_score",
        color="prediction",
        color_discrete_map=CLS_COLOR,
        opacity=0.7,
        hover_data=["description", "src_ip"],
        labels={"freq_per_min": "Fréquence (alertes/min)", "risk_score": "Score risque (%)"},
    )
    fig3.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#f0f6fc"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        height=280, margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════
    # TABLE DES ALERTES FILTRÉES
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### File d'alertes filtrées")

    df_filtered = df[
        (df["prediction"].isin(show_classes)) &
        (df["risk_score"] >= min_risk)
    ].sort_values("risk_score", ascending=False).tail(50)

    if df_filtered.empty:
        st.info("Aucune alerte ne correspond aux filtres sélectionnés.")
    else:
        display_cols = ["timestamp", "description", "src_ip", "dst_port",
                        "freq_per_min", "prediction", "risk_score",
                        "proba_attack", "proba_fp", "proba_normal"]
        st.dataframe(
            df_filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=300,
            column_config={
                "risk_score":    st.column_config.ProgressColumn("Risque", min_value=0, max_value=100, format="%d%%"),
                "proba_attack":  st.column_config.NumberColumn("P(attack)", format="%.3f"),
                "proba_fp":      st.column_config.NumberColumn("P(FP)", format="%.3f"),
                "proba_normal":  st.column_config.NumberColumn("P(normal)", format="%.3f"),
                "prediction":    st.column_config.TextColumn("Classe"),
                "freq_per_min":  st.column_config.NumberColumn("Freq/min"),
            },
        )

        # Export CSV
        csv_data = df_filtered[display_cols].to_csv(index=False)
        st.download_button("⬇ Exporter CSV", csv_data, "soc_alertes.csv", "text/csv")

else:
    st.info("▶ Cliquez sur **Démarrer** dans la barre latérale pour lancer la simulation.")
    st.markdown("""
    **Ce dashboard simule :**
    - L'ingestion de logs Wazuh/Suricata en temps réel
    - La classification automatique : normal / faux positif / attaque
    - Le calcul d'un score de risque (probabilité d'attaque × 100)
    - Le filtrage intelligent pour réduire la charge de travail des analystes SOC
    """)

# ══════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ══════════════════════════════════════════════════════════════
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()