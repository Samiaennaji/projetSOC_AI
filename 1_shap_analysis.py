"""
SOC Detection – Analyse SHAP & Feature Importance
==================================================
Livrable 1/3 : Explicabilité du modèle XGBoost

Génère :
  - fig_fi.png          : Feature importance globale (XGBoost gain)
  - fig_shap_beeswarm.png : SHAP beeswarm plot (vue globale)
  - fig_shap_per_class.png : SHAP |mean| par classe
  - fig_shap_waterfall.png : Waterfall pour 3 cas concrets (normal/FP/attack)
  - shap_report.json    : Rapport complet des contributions
"""
import os
import platform
import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CSV_PATH   = "E:/Projets/ProjetSOC_AI/soc_detection_dataset_v3 (2).csv"
OUT_DIR    = "E:/Projets/ProjetSOC_AI/outputs/"
LABELS_STR = ["normal", "false_positive", "attack"]
FEATURES   = [
    "hour_of_day", "time_context", "protocol",
    "src_port", "dst_port",
    "src_is_internal", "dst_is_internal", "is_internal_to_internal",
    "same_subnet", "freq_per_min", "is_high_freq",
    "src_port_ephemeral", "dst_port_privileged",
]
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. DONNÉES & MODÈLE
# ══════════════════════════════════════════════════════════════
print("Chargement des données...")
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

for col in ["time_context", "protocol"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df[FEATURES]
y = df["label"]

# Split temporel
split = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Entraînement XGBoost (avec pondération des classes)...")

# ── Pondération des classes pour corriger le déséquilibre ──────────────────
# En production SOC réelle, les attaques sont < 1% du trafic.
# On sur-pondère la classe "attack" (index 2) pour que le modèle paie
# beaucoup plus cher quand il rate une vraie attaque.
#
# Calcul automatique : poids inversement proportionnel à la fréquence
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# On multiplie le poids des attaques par un facteur supplémentaire
# pour simuler leur vraie rareté en production (14% → ~1%)
ATTACK_BOOST = 5.0   # ← ajustez si vous voulez plus/moins de sensibilité
attack_mask = (y_train == 2)
sample_weights[attack_mask] *= ATTACK_BOOST

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", tree_method="hist",
    random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train, sample_weight=sample_weights)

# ── Seuil de décision adapté (au lieu de 0.5 par défaut) ──────────────────
# Avec un seuil bas sur la classe attack, on préfère les faux positifs
# aux fausses négatives — essentiel pour un SOC.
ATTACK_THRESHOLD = 0.15   # si P(attack) > 15% → on classe en attack
print(f"  ✓ Seuil de décision attaque : {ATTACK_THRESHOLD} (au lieu de 0.5 par défaut)")

# ══════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE (XGBoost gain)
# ══════════════════════════════════════════════════════════════
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#7F77DD" if v > 0.05 else "#CECBF6" for v in fi.values]
bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
ax.set_title("Feature Importance – XGBoost (gain normalisé)\nSans features IDS leakées",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance relative")

for bar, val in zip(bars, fi.values[::-1]):
    if val > 0.003:
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#3C3489")

# Annotation explicative
top3 = fi.head(3).index.tolist()
ax.axvline(fi.iloc[2], color="#D85A30", linestyle="--", linewidth=1, alpha=0.5)
ax.text(fi.iloc[2] + 0.01, 0.5, f"Top 3 :\n" + "\n".join(top3),
        transform=ax.get_yaxis_transform(), fontsize=8.5, color="#D85A30",
        va="center", ha="left")

plt.tight_layout()
fig.savefig(f"{OUT_DIR}fig_fi.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig_fi.png")

# ══════════════════════════════════════════════════════════════
# 3. SHAP VALUES
# ══════════════════════════════════════════════════════════════
print("Calcul SHAP (TreeExplainer)...")

# On utilise un échantillon représentatif de 600 points (200 par classe)
idx_per_class = []
for cls in [0, 1, 2]:
    idx = X_test[y_test == cls].index[:200]
    idx_per_class.extend(idx.tolist())

X_shap = X_test.loc[idx_per_class]
y_shap = y_test.loc[idx_per_class]

explainer = shap.TreeExplainer(model)
sv = explainer(X_shap)   # shape: (600, 13, 3)

# ── 3a. SHAP |mean| par feature par classe ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("SHAP – Contribution moyenne absolue par feature et par classe\n"
             "(Plus la valeur est haute, plus la feature influence cette décision)",
             fontsize=13, fontweight="bold")

colors_cls = ["#378ADD", "#EF9F27", "#E24B4A"]
for ci, (cls_name, col) in enumerate(zip(LABELS_STR, colors_cls)):
    ax = axes[ci]
    mean_shap = np.abs(sv.values[:, :, ci]).mean(axis=0)
    shap_df = pd.Series(mean_shap, index=FEATURES).sort_values(ascending=True)

    light = col + "55"  # transparence approximative via hex
    bar_colors = [col if v == shap_df.max() else "#D3D1C7" for v in shap_df.values]
    bars = ax.barh(shap_df.index, shap_df.values, color=bar_colors, edgecolor="white", height=0.65)

    ax.set_title(f"Classe : {cls_name}", fontsize=12, fontweight="bold", color=col)
    ax.set_xlabel("|SHAP| moyen")

    for bar, val in zip(bars, shap_df.values):
        if val > 0.005:
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8.5)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}fig_shap_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig_shap_per_class.png")

# ── 3b. SHAP Beeswarm global (classe attack) ──
fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.beeswarm(
    sv[:, :, 2],   # classe attack
    max_display=13,
    show=False,
    color_bar_label="Valeur de la feature",
    plot_size=None,
)
ax = plt.gca()
ax.set_title("SHAP Beeswarm – Classe ATTACK\nChaque point = une alerte du jeu de test",
             fontsize=12, fontweight="bold", color="#A32D2D", pad=12)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}fig_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig_shap_beeswarm.png")

# ── 3c. Waterfall pour 3 cas concrets ──
# Trouver un exemple typique de chaque classe
# On utilise le seuil adapté (ATTACK_THRESHOLD) pour les prédictions
examples = {}
probas_test = model.predict_proba(X_test)

def predict_with_threshold(probas, threshold=ATTACK_THRESHOLD):
    """Applique le seuil bas sur la classe attack avant argmax."""
    preds = []
    for p in probas:
        if p[2] >= threshold:   # attack (index 2) au-dessus du seuil → ALERT
            preds.append(2)
        else:
            preds.append(np.argmax(p[:2]))  # entre normal et FP seulement
    return np.array(preds)

y_pred_all = predict_with_threshold(probas_test, ATTACK_THRESHOLD)
for cls in [0, 1, 2]:
    # Prendre un exemple bien classifié et représentatif
    mask = (y_test.values == cls) & (y_pred_all == cls)
    idx_correct = np.where(mask)[0]
    if len(idx_correct) > 0:
        # Prendre celui avec la plus haute probabilité dans sa classe
        proba = model.predict_proba(X_test.iloc[idx_correct])[:, cls]
        best = idx_correct[np.argmax(proba)]
        examples[LABELS_STR[cls]] = best

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("SHAP Waterfall – Explication de 3 décisions individuelles\n"
             "(Comment le modèle justifie chaque classification)",
             fontsize=13, fontweight="bold")

titles_col = {"normal": "#185FA5", "false_positive": "#854F0B", "attack": "#A32D2D"}
for ci, (cls_name, col) in enumerate(zip(LABELS_STR, colors_cls)):
    plt.sca(axes[ci])
    if cls_name in examples:
        local_idx = list(X_shap.index).index(X_test.index[examples[cls_name]]) \
            if X_test.index[examples[cls_name]] in list(X_shap.index) else 0
        shap.plots.waterfall(sv[local_idx, :, ci], max_display=10, show=False)
        axes[ci].set_title(f"Décision : {cls_name}", fontsize=12, fontweight="bold",
                           color=titles_col[cls_name])
    else:
        axes[ci].text(0.5, 0.5, "Pas d'exemple\ndisponible", ha="center", va="center",
                      transform=axes[ci].transAxes)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}fig_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig_shap_waterfall.png")

# ══════════════════════════════════════════════════════════════
# 4. RAPPORT JSON
# ══════════════════════════════════════════════════════════════
report = {
    "feature_importance": {k: round(float(v), 5) for k, v in fi.items()},
    "shap_mean_abs_per_class": {},
    "class_imbalance_correction": {
        "method": "sample_weight balanced + attack boost",
        "attack_boost_factor": ATTACK_BOOST,
        "attack_decision_threshold": ATTACK_THRESHOLD,
        "rationale": (
            "Les attaques réelles en SOC sont < 1% du trafic (vs 14.4% dans ce dataset). "
            "La sur-pondération compense le déséquilibre à l'entraînement. "
            f"Le seuil {ATTACK_THRESHOLD} remplace le seuil par défaut 0.5 : "
            "si P(attack) > 15%, on préfère une fausse alarme à une attaque manquée."
        ),
    },
    "interpretation": {
        "src_is_internal": "Feature la plus discriminante. Une source externe = signal fort d'attaque ou d'alerte légitime.",
        "is_internal_to_internal": "Flux latéraux entre hôtes internes = mouvement latéral post-compromission.",
        "freq_per_min": "Fréquence d'alertes : haute fréquence = scan, brute force, DoS.",
        "is_high_freq": "Version binaire de freq_per_min, capte le régime de haute fréquence.",
        "same_subnet": "Même sous-réseau + haute fréquence = reconnaissance interne.",
    }
}

for ci, cls_name in enumerate(LABELS_STR):
    mean_shap = np.abs(sv.values[:, :, ci]).mean(axis=0)
    report["shap_mean_abs_per_class"][cls_name] = {
        f: round(float(v), 5)
        for f, v in sorted(zip(FEATURES, mean_shap), key=lambda x: -x[1])
    }

with open(f"{OUT_DIR}shap_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("✓ shap_report.json")
print("\n=== TOP 3 FEATURES SHAP PAR CLASSE ===")
for cls_name, vals in report["shap_mean_abs_per_class"].items():
    top3 = list(vals.items())[:3]
    print(f"  {cls_name:15s} → {top3}")

print("\n✓ Analyse SHAP complète terminée.")