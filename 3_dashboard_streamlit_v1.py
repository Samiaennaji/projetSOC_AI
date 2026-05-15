import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import time
from sklearn.preprocessing import LabelEncoder

# ═════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════

CSV_REAL_LOGS = "E:/Projets/ProjetSOC_AI/soc_realtime_output.csv"
MODEL_PATH    = "E:/Projets/ProjetSOC_AI/models/xgb_soc.pkl"

FEATURES = [
    "hour_of_day",
    "time_context",
    "protocol",
    "src_port",
    "dst_port",
    "src_is_internal",
    "dst_is_internal",
    "is_internal_to_internal",
    "same_subnet",
    "freq_per_min",
    "is_high_freq",
    "src_port_ephemeral",
    "dst_port_privileged",
]

LABELS_STR = {
    0: "normal",
    1: "false_positive",
    2: "attack"
}

CLS_COLOR = {
    "normal": "#1D9E75",
    "false_positive": "#EF9F27",
    "attack": "#E24B4A"
}

# ═════════════════════════════════════════════════════
# PAGE
# ═════════════════════════════════════════════════════

st.set_page_config(
    page_title="SOC AI Dashboard",
    layout="wide"
)

st.title("🛡️ SOC AI Detection Dashboard")
st.markdown("Analyse de vraies alertes Wazuh / Suricata")

# ═════════════════════════════════════════════════════
# LOAD MODEL
# ═════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ═════════════════════════════════════════════════════
# LOAD REAL LOGS
# ═════════════════════════════════════════════════════

@st.cache_data
def load_logs():

    df = pd.read_csv(CSV_REAL_LOGS)

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Hour
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Time context
    def get_time_context(h):
        if 6 <= h < 12:
            return "morning"
        elif 12 <= h < 18:
            return "afternoon"
        elif 18 <= h < 24:
            return "evening"
        return "night"

    df["time_context"] = df["hour_of_day"].apply(get_time_context)

    # Protocol
    if "protocol" not in df.columns:
        df["protocol"] = "tcp"

    # Ports
    if "src_port" not in df.columns:
        df["src_port"] = 12345

    if "dst_port" not in df.columns:
        df["dst_port"] = 80

    # Internal IP detection
    def is_internal(ip):
        if pd.isna(ip):
            return 0

        ip = str(ip)

        return int(
            ip.startswith("192.168.")
            or ip.startswith("10.")
            or ip.startswith("172.")
        )

    df["src_is_internal"] = df["src_ip"].apply(is_internal)

    if "dst_ip" in df.columns:
        df["dst_is_internal"] = df["dst_ip"].apply(is_internal)
    else:
        df["dst_is_internal"] = 1

    # Internal to internal
    df["is_internal_to_internal"] = (
        (df["src_is_internal"] == 1)
        & (df["dst_is_internal"] == 1)
    ).astype(int)

    # Same subnet
    df["same_subnet"] = 0

    # Frequency
    df["freq_per_min"] = 1

    # High frequency
    df["is_high_freq"] = (
        df["freq_per_min"] > 20
    ).astype(int)

    # Port flags
    df["src_port_ephemeral"] = (
        df["src_port"] > 1024
    ).astype(int)

    df["dst_port_privileged"] = (
        df["dst_port"] < 1024
    ).astype(int)

    # Encode categorical
    for col in ["time_context", "protocol"]:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

    return df

df = load_logs()

# ═════════════════════════════════════════════════════
# MODEL PREDICTION
# ═════════════════════════════════════════════════════

X = df[FEATURES]

proba = model.predict_proba(X)

pred = np.argmax(proba, axis=1)

df["prediction"] = [LABELS_STR[p] for p in pred]

df["risk_score"] = (proba[:, 2] * 100).astype(int)

df["proba_attack"] = proba[:, 2]
df["proba_fp"] = proba[:, 1]
df["proba_normal"] = proba[:, 0]

# ═════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════

st.sidebar.header("Filtres")

show_classes = st.sidebar.multiselect(
    "Classes",
    ["normal", "false_positive", "attack"],
    default=["attack", "false_positive"]
)

min_risk = st.sidebar.slider(
    "Risk score minimum",
    0,
    100,
    30
)

# ═════════════════════════════════════════════════════
# FILTERING
# ═════════════════════════════════════════════════════

df_filtered = df[
    (df["prediction"].isin(show_classes))
    & (df["risk_score"] >= min_risk)
]

# ═════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════

total = len(df)

n_attack = (df["prediction"] == "attack").sum()

n_fp = (df["prediction"] == "false_positive").sum()

n_normal = (df["prediction"] == "normal").sum()

avg_risk = df["risk_score"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Alertes", total)

col2.metric("Attaques", n_attack)

col3.metric("False Positives", n_fp)

col4.metric("Risk Moyen", f"{avg_risk:.0f}%")

# ═════════════════════════════════════════════════════
# TIMELINE
# ═════════════════════════════════════════════════════

st.subheader("Timeline des alertes")

df_plot = df_filtered.tail(200).copy()

df_plot["index"] = range(len(df_plot))

fig = px.scatter(
    df_plot,
    x="index",
    y="risk_score",
    color="prediction",
    color_discrete_map=CLS_COLOR,
    hover_data=["src_ip", "dst_port"]
)

fig.update_layout(height=400)

st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════
# PIE CHART
# ═════════════════════════════════════════════════════

st.subheader("Distribution")

counts = df["prediction"].value_counts()

fig2 = px.pie(
    names=counts.index,
    values=counts.values,
    hole=0.5,
    color=counts.index,
    color_discrete_map=CLS_COLOR
)

st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════
# TABLE
# ═════════════════════════════════════════════════════

st.subheader("Alertes détectées")

display_cols = [
    "timestamp",
    "src_ip",
    "dst_port",
    "prediction",
    "risk_score",
    "proba_attack",
    "proba_fp",
    "proba_normal"
]

st.dataframe(
    df_filtered[display_cols],
    use_container_width=True,
    height=500
)

# ═════════════════════════════════════════════════════
# EXPORT CSV
# ═════════════════════════════════════════════════════

csv = df_filtered.to_csv(index=False)

st.download_button(
    "⬇ Export CSV",
    csv,
    "soc_alerts.csv",
    "text/csv"
)