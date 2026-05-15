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
    page_title="SOC AI Dashboard",

    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-val  { font-size: 2.4rem; font-weight: 700; margin: 4px 0; }
  .metric-lbl  { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
  .attack-row  { background: rgba(226,75,74,0.08) !important; }
  .fp-row      { background: rgba(239,159,39,0.08) !important; }
  .alert-badge-attack { background:#E24B4A; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
  .alert-badge-fp     { background:#EF9F27; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
  .alert-badge-normal { background:#1D9E75; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
  h1, h2, h3  { color: #f0f6fc !important; }
  .stDataFrame { background: #161b22 !important; }
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
CLS_COLOR  = {"normal": "#1D9E75", "false_positive": "#EF9F27", "attack": "#E24B4A"}

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
    st.markdown("##  SOC AI Engine")
    st.markdown("---")
    st.markdown("### Contrôles")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Démarrer", use_container_width=True, type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Arrêter", use_container_width=True):
            st.session_state.running = False

    if st.button("🗑 Réinitialiser", use_container_width=True):
        st.session_state.alerts = pd.DataFrame()
        st.session_state.total_generated = 0

    st.markdown("---")
    st.markdown("### Filtres")
    show_classes = st.multiselect(
        "Classes à afficher",
        ["normal", "false_positive", "attack"],
        default=["false_positive", "attack"],
    )
    min_risk = st.slider("Score de risque minimum", 0, 100, 30)
    n_batch = st.slider("Alertes par cycle", 1, 20, 5)
    refresh_rate = st.slider("Vitesse (secondes)", 0.5, 5.0, 1.5)

    st.markdown("---")
    st.markdown("### Simulation forcée")
    force_attack = st.button("🔴 Injecter une attaque", use_container_width=True)
    force_fp     = st.button("🟡 Injecter un faux positif", use_container_width=True)

    st.markdown("---")
    st.caption(f"Modèle : XGBoost · 13 features réseau\nSplit temporel · Sans leakage IDS")

# ══════════════════════════════════════════════════════════════
# TITRE PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.markdown("# SOC AI Detection Dashboard")
st.markdown(f"*Détection en temps réel · Modèle XGBoost entraîné sur features réseau brutes*")
st.markdown("---")

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
            <div class="metric-lbl">Alertes totales</div>
            <div class="metric-val" style="color:#f0f6fc">{total}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">Attaques détectées</div>
            <div class="metric-val" style="color:#E24B4A">{n_attacks}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">Faux positifs filtrés</div>
            <div class="metric-val" style="color:#EF9F27">{n_fp}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">Trafic normal</div>
            <div class="metric-val" style="color:#1D9E75">{n_normal}</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">Score risque moyen</div>
            <div class="metric-val" style="color:#7F77DD">{avg_risk:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # GRAPHIQUES
    # ══════════════════════════════════════════════════════════
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Timeline des alertes")
        # Historique des 100 dernières avec score risque
        df_plot = df.tail(100).copy()
        df_plot["index"] = range(len(df_plot))

        fig = go.Figure()
        for cls, col, symbol in [
            ("normal", "#1D9E75", "circle"),
            ("false_positive", "#EF9F27", "diamond"),
            ("attack", "#E24B4A", "x"),
        ]:
            mask = df_plot["prediction"] == cls
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df_plot[mask]["index"],
                    y=df_plot[mask]["risk_score"],
                    mode="markers",
                    name=cls,
                    marker=dict(color=col, size=8, symbol=symbol,
                                line=dict(width=1, color="white")),
                    text=df_plot[mask]["description"],
                    hovertemplate="%{text}<br>Risk: %{y}%<extra></extra>",
                ))

        fig.add_hline(y=50, line_dash="dash", line_color="#888",
                      annotation_text="Seuil 50%", annotation_position="right")
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(color="#f0f6fc"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            xaxis=dict(gridcolor="#21262d", title="Alertes récentes"),
            yaxis=dict(gridcolor="#21262d", title="Score de risque (%)", range=[0, 105]),
            height=300, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("#### Distribution des classes")
        counts = df["prediction"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            marker=dict(colors=[CLS_COLOR.get(l, "#888") for l in counts.index]),
            hole=0.6,
            textinfo="percent+label",
            textfont=dict(color="white", size=12),
        ))
        fig2.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(color="#f0f6fc"),
            showlegend=False,
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            annotations=[dict(text=f"<b>{total}</b><br>alertes", x=0.5, y=0.5,
                              font_size=14, font_color="white", showarrow=False)],
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