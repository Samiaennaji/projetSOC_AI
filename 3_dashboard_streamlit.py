"""
SOC Detection – Dashboard Streamlit (Version Finale)
=====================================================
Intégration complète avec 4_llm_advisor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import datetime
import pickle

# ── Import du module LLM amélioré ────────────────────────────────────────────
from llm_advisor import analyser_alerte, render_llm_panel, badge_severity
# ─────────────────────────────────────────────────────────────────────────────

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

# Mapping port → nom de service (pour enrichir l'alerte envoyée au LLM)
PORT_SERVICE = {22: "ssh", 80: "http", 443: "https", 53: "dns",
                21: "ftp", 25: "smtp", 3306: "mysql", 3389: "rdp"}

# Mapping time_context int → string (pour le prompt LLM)
TIME_CTX_MAP = {0: "business_hours", 1: "evening", 2: "night", 3: "morning"}

# ══════════════════════════════════════════════════════════════
# MODÈLE
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        with open("E:\\Projets\\ProjetSOC_AI\\models\\xgb_soc.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
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
                rows.append([random.randint(0,23), random.randint(0,3),
                              0 if cls<2 else random.randint(0,1),
                              sp, dp, src_int, dst_int, int(src_int and dst_int),
                              0, freq, int(freq>20), int(sp>1024), int(dp<1024)])
                labels.append(cls)
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                   eval_metric="mlogloss", random_state=42)
        model.fit(pd.DataFrame(rows, columns=FEATURES), pd.Series(labels))
        return model

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
ALERT_NAMES = {
    "normal":         ["HTTP GET /index.html", "DNS query google.com",
                       "HTTPS session établie", "FTP file transfer", "SMTP message envoyé"],
    "false_positive": ["Web Scan interne", "ICMP Ping Sweep", "Script Injection Path",
                       "HTTP Login Bruteforce (outil audit)", "Port Scan autorisé"],
    "attack":         ["SSH Brute Force", "Hydra HTTP Tool", "SQL Injection URI",
                       "Reverse Shell Attempt", "Sensitive File Probe"],
}
SEVERITY_MAP = {"normal": "LOW", "false_positive": "MEDIUM", "attack": "HIGH"}

def generate_log(force_class: str = None):
    cls_name = force_class or random.choices(
        ["normal", "false_positive", "attack"], weights=[0.58, 0.28, 0.14]
    )[0]
    hour = datetime.datetime.now().hour
    src_int = {"normal": 1, "false_positive": 1, "attack": 0}[cls_name]
    freq = max(1, int(np.random.exponential(
        {"normal": 4, "false_positive": 8, "attack": 50}[cls_name]
    )))
    sp = random.randint(1025, 65535)
    dp = {"normal":         random.choice([80, 443, 53, 21, 25]),
          "false_positive": random.choice([80, 443, 22]),
          "attack":         random.choice([22, 80, 443])}[cls_name]

    feats = {
        "hour_of_day": hour, "time_context": random.randint(0, 3), "protocol": 0,
        "src_port": sp, "dst_port": dp, "src_is_internal": src_int,
        "dst_is_internal": 1, "is_internal_to_internal": src_int,
        "same_subnet": int(src_int and random.random() < 0.3),
        "freq_per_min": freq, "is_high_freq": int(freq > 20),
        "src_port_ephemeral": int(sp > 1024), "dst_port_privileged": int(dp < 1024),
    }
    proba = model.predict_proba(pd.DataFrame([feats]))[0]
    attack_idx = list(model.classes_).index(2) if 2 in model.classes_ else 2
    if proba[attack_idx] >= ATTACK_THRESHOLD:
        pred = 2
    else:
        other = [i for i in range(len(proba)) if i != attack_idx]
        pred  = other[int(np.argmax([proba[i] for i in other]))]

    src_ip  = (f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
               if src_int else
               f"185.{random.randint(1,255)}.{random.randint(1,254)}.{random.randint(1,254)}")
    dst_ip  = f"10.0.{random.randint(1,10)}.{random.randint(1,50)}"
    alert_name = random.choice(ALERT_NAMES[cls_name])

    return {
        # ── Champs affichage dashboard ──
        "timestamp":    datetime.datetime.now().strftime("%H:%M:%S"),
        "alert_name":   alert_name,
        "description":  alert_name + (" 🔴" if cls_name == "attack" else ""),
        "src_ip":       src_ip,
        "dst_ip":       dst_ip,
        "dst_port":     dp,
        "freq_per_min": freq,
        "prediction":   LABELS_STR[pred],
        "risk_score":   int(
            50 + 50 * (proba[2] - ATTACK_THRESHOLD) / (1 - ATTACK_THRESHOLD)
            if proba[2] >= ATTACK_THRESHOLD
            else 50 * proba[2] / ATTACK_THRESHOLD
        ),
        "proba_normal":  round(float(proba[0]), 3),
        "proba_fp":      round(float(proba[1]), 3),
        "proba_attack":  round(float(proba[2]), 3),
        "_true": cls_name,
        # ── Champs enrichis pour le LLM ──
        "severity":      SEVERITY_MAP[cls_name],
        "severity_score": {"normal": 1, "false_positive": 2, "attack": 3}[cls_name],
        "protocol":      "TCP",
        "dst_service":   PORT_SERVICE.get(dp, str(dp)),
        "src_port":      sp,
        **feats,
    }

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "alerts"          not in st.session_state: st.session_state.alerts = pd.DataFrame()
if "running"         not in st.session_state: st.session_state.running = False
if "total_generated" not in st.session_state: st.session_state.total_generated = 0
if "llm_results"     not in st.session_state: st.session_state.llm_results = {}
# llm_results : clé = index DataFrame → LLMAnalysis
# (le cache interne de 4_llm_advisor évite les re-appels API)

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
        st.session_state.llm_results = {}

    st.markdown("---")
    st.markdown("### Filtres")
    show_classes = st.multiselect(
        "Classes à afficher",
        ["normal", "false_positive", "attack"],
        default=["false_positive", "attack"],
    )
    min_risk     = st.slider("Score de risque minimum", 0, 100, 30)
    n_batch      = st.slider("Alertes par cycle", 1, 20, 5)
    refresh_rate = st.slider("Vitesse (secondes)", 0.5, 5.0, 1.5)

    st.markdown("---")
    st.markdown("### Simulation forcée")
    force_attack = st.button("🔴 Injecter une attaque",     use_container_width=True)
    force_fp     = st.button("🟡 Injecter un faux positif", use_container_width=True)

    st.markdown("---")
    st.caption("Modèle : XGBoost · 13 features réseau\nLLM : LLaMA-3.3-70b via Groq")

# ══════════════════════════════════════════════════════════════
# TITRE
# ══════════════════════════════════════════════════════════════
st.markdown("# SOC AI Detection Dashboard")
st.markdown("*Détection temps réel · XGBoost + Analyse LLM (LLaMA-3.3-70b via Groq)*")
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
    st.session_state.alerts = (
        new_df if st.session_state.alerts.empty
        else pd.concat([st.session_state.alerts, new_df], ignore_index=True).tail(1000)
    )

df = st.session_state.alerts

# ══════════════════════════════════════════════════════════════
# MÉTRIQUES PRINCIPALES
# ══════════════════════════════════════════════════════════════
if not df.empty:
    total     = len(df)
    n_attacks = (df["prediction"] == "attack").sum()
    n_fp      = (df["prediction"] == "false_positive").sum()
    n_normal  = (df["prediction"] == "normal").sum()
    avg_risk  = df["risk_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, color in [
        (c1, "Alertes totales",    total,     "#f0f6fc"),
        (c2, "Attaques détectées", n_attacks, "#E24B4A"),
        (c3, "Faux positifs",      n_fp,      "#EF9F27"),
        (c4, "Trafic normal",      n_normal,  "#1D9E75"),
        (c5, "Risque moyen",       f"{avg_risk:.0f}%", "#7F77DD"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-lbl">{label}</div>
                <div class="metric-val" style="color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # GRAPHIQUES
    # ══════════════════════════════════════════════════════════
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Timeline des alertes")
        df_plot = df.tail(100).copy()
        df_plot["index"] = range(len(df_plot))
        fig = go.Figure()
        for cls, color, symbol in [("normal","#1D9E75","circle"),
                                    ("false_positive","#EF9F27","diamond"),
                                    ("attack","#E24B4A","x")]:
            m = df_plot["prediction"] == cls
            if m.any():
                fig.add_trace(go.Scatter(
                    x=df_plot[m]["index"], y=df_plot[m]["risk_score"],
                    mode="markers", name=cls,
                    marker=dict(color=color, size=8, symbol=symbol,
                                line=dict(width=1, color="white")),
                    text=df_plot[m]["alert_name"],
                    hovertemplate="%{text}<br>Risk: %{y}%<extra></extra>",
                ))
        fig.add_hline(y=50, line_dash="dash", line_color="#888",
                      annotation_text="Seuil 50%", annotation_position="right")
        fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                          font=dict(color="#f0f6fc"),
                          legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
                          xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d", range=[0,105]),
                          height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("#### Distribution des classes")
        counts = df["prediction"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            marker=dict(colors=[CLS_COLOR.get(l,"#888") for l in counts.index]),
            hole=0.6, textinfo="percent+label", textfont=dict(color="white", size=12),
        ))
        fig2.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           font=dict(color="#f0f6fc"), showlegend=False,
                           height=300, margin=dict(l=0,r=0,t=20,b=0),
                           annotations=[dict(text=f"<b>{total}</b><br>alertes",
                                             x=0.5, y=0.5, font_size=14,
                                             font_color="white", showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### Score de risque vs fréquence")
    fig3 = px.scatter(df.tail(200), x="freq_per_min", y="risk_score",
                      color="prediction", color_discrete_map=CLS_COLOR, opacity=0.7,
                      hover_data=["alert_name", "src_ip"],
                      labels={"freq_per_min":"Fréquence/min","risk_score":"Score risque (%)"})
    fig3.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                       font=dict(color="#f0f6fc"),
                       legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
                       xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
                       height=280, margin=dict(l=0,r=0,t=20,b=0))
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
        st.info("Aucune alerte ne correspond aux filtres.")
    else:
        display_cols = ["timestamp", "alert_name", "src_ip", "dst_port",
                        "freq_per_min", "prediction", "risk_score",
                        "proba_attack", "proba_fp", "proba_normal"]
        st.dataframe(
            df_filtered[display_cols].reset_index(drop=True),
            use_container_width=True, height=300,
            column_config={
                "risk_score":   st.column_config.ProgressColumn("Risque", min_value=0, max_value=100, format="%d%%"),
                "proba_attack": st.column_config.NumberColumn("P(attack)", format="%.3f"),
                "proba_fp":     st.column_config.NumberColumn("P(FP)", format="%.3f"),
                "proba_normal": st.column_config.NumberColumn("P(normal)", format="%.3f"),
                "prediction":   st.column_config.TextColumn("Classe"),
                "freq_per_min": st.column_config.NumberColumn("Freq/min"),
            },
        )
        st.download_button("⬇ Exporter CSV",
                           df_filtered[display_cols].to_csv(index=False),
                           "soc_alertes.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # PANNEAU LLM — Analyse intelligente par alerte
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 🤖 Analyse LLM — Recommandations SOC (LLaMA-3.3-70b)")

    df_attacks = df[df["prediction"] == "attack"].sort_values("risk_score", ascending=False).head(10)

    if df_attacks.empty:
        st.info("Aucune attaque détectée. Cliquez sur **🔴 Injecter une attaque** pour tester.")
    else:
        # Sélecteur d'alerte
        alert_options = {
            f"[{row['timestamp']}] {row['alert_name']} — {row['src_ip']}:{row['dst_port']} "
            f"| risque {row['risk_score']}% | P(atk)={row['proba_attack']:.3f}": idx
            for idx, row in df_attacks.iterrows()
        }
        selected_label = st.selectbox(
            "Sélectionnez une alerte à analyser :",
            list(alert_options.keys()),
            key="llm_selector",
        )
        selected_idx = alert_options[selected_label]
        selected_row = df.loc[selected_idx]

        # Bouton d'analyse
        col_btn, col_status = st.columns([2, 3])
        with col_btn:
            run_llm = st.button("🔍 Analyser avec le LLM", type="primary", use_container_width=True)
        with col_status:
            if selected_idx in st.session_state.llm_results:
                prev = st.session_state.llm_results[selected_idx]
                cache_info = " *(depuis cache)*" if prev.from_cache else ""
                st.markdown(
                    f"Dernier résultat : {badge_severity(prev.niveau_risque)}{cache_info}",
                    unsafe_allow_html=True,
                )

        if run_llm:
            # Construction du dict enrichi pour 4_llm_advisor
            alert_dict = {
                "timestamp":               selected_row["timestamp"],
                "alert_name":              selected_row["alert_name"],
                "severity":                selected_row.get("severity", "HIGH"),
                "severity_score":          int(selected_row.get("severity_score", 3)),
                "protocol":                selected_row.get("protocol", "TCP"),
                "src_ip":                  selected_row["src_ip"],
                "src_port":                int(selected_row.get("src_port", 0)),
                "dst_ip":                  selected_row.get("dst_ip", "?"),
                "dst_port":                int(selected_row["dst_port"]),
                "dst_service":             selected_row.get("dst_service",
                                               PORT_SERVICE.get(int(selected_row["dst_port"]), "?")),
                "src_is_internal":         int(selected_row["src_is_internal"]),
                "dst_is_internal":         int(selected_row.get("dst_is_internal", 1)),
                "is_internal_to_internal": int(selected_row["is_internal_to_internal"]),
                "same_subnet":             int(selected_row["same_subnet"]),
                "freq_per_min":            int(selected_row["freq_per_min"]),
                "is_high_freq":            int(selected_row["is_high_freq"]),
                "time_context":            TIME_CTX_MAP.get(int(selected_row.get("time_context", 0)),
                                                             "business_hours"),
                "prediction":              selected_row["prediction"],
                "risk_score":              int(selected_row["risk_score"]),
                "proba_attack":            float(selected_row["proba_attack"]),
                "proba_fp":                float(selected_row["proba_fp"]),
                "proba_normal":            float(selected_row["proba_normal"]),
            }

            with st.spinner("⏳ LLaMA-3.3-70b analyse l'alerte..."):
                result = analyser_alerte(alert_dict)

            if result:
                st.session_state.llm_results[selected_idx] = result
            else:
                st.error("❌ Analyse échouée. Vérifiez votre clé GROQ_API_KEY.")

        # Affichage du résultat via render_llm_panel()
        if selected_idx in st.session_state.llm_results:
            with st.container():
                render_llm_panel(st.session_state.llm_results[selected_idx])

else:
    st.info("▶ Cliquez sur **Démarrer** dans la barre latérale pour lancer la simulation.")
    st.markdown("""
    **Ce dashboard combine :**
    - Classification temps réel des logs : XGBoost (13 features réseau)
    - Analyse intelligente de chaque attaque : LLaMA-3.3-70b via Groq
    - Résumé · Niveau de risque · Actions SOC · IOCs · Mapping MITRE ATT&CK
    """)

# ══════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ══════════════════════════════════════════════════════════════
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()