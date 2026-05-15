"""
SOC Detection – Entraînement corrigé + sauvegarde du modèle
============================================================
Correction principale :
  - XGBoost reçoit des sample_weights pour compenser le déséquilibre de classes
  - Seuil de décision optimisé sur la courbe PR (pas 0.5 par défaut)
  - Le modèle final est sauvegardé en .pkl prêt pour le pipeline temps réel

Pourquoi c'est important :
  Dataset : attaques = 14.4%  →  modèle biaisé vers "normal" / "false_positive"
  Production SOC réelle : attaques < 1%  →  biais encore plus fort en prod
  Solution : sur-pondérer la classe attack à l'entraînement + abaisser le seuil
"""

import pandas as pd
import numpy as np
import json, time, pickle, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing         import LabelEncoder
from sklearn.utils.class_weight    import compute_sample_weight
from sklearn.metrics               import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
CSV_PATH   = "E:/Projets/ProjetSOC_AI/soc_detection_dataset_v3 (2).csv"
OUTPUT_DIR = "E:/Projets/ProjetSOC_AI/outputs/"
MODEL_PATH = "E:/Projets/ProjetSOC_AI/models/xgb_soc.pkl"

FEATURES = [
    "hour_of_day", "time_context", "protocol",
    "src_port", "dst_port",
    "src_is_internal", "dst_is_internal", "is_internal_to_internal",
    "same_subnet", "freq_per_min", "is_high_freq",
    "src_port_ephemeral", "dst_port_privileged",
]
LABEL_COL  = "label"
LABELS_STR = ["normal", "false_positive", "attack"]

# ── Paramètres de correction du déséquilibre ──────────────────
# ATTACK_BOOST : facteur multiplicatif sur le poids de la classe attack.
# Plus il est élevé, plus le modèle évite de rater une attaque
# (au prix de plus de faux positifs).
# Valeurs recommandées : 3 (conservateur) · 5 (équilibré) · 10 (très sensible)
ATTACK_BOOST = 5.0

# ATTACK_THRESHOLD : seuil de décision pour la classe attack.
# Si P(attack) >= ATTACK_THRESHOLD → ALERT, même si P(normal) est plus haut.
# 0.10 = très sensible · 0.15 = équilibré · 0.25 = conservateur
ATTACK_THRESHOLD = 0.15

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT & PRÉPARATION
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  CHARGEMENT DES DONNÉES")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"  {len(df):,} lignes · {df.shape[1]} colonnes")
print(f"  Plage : {df['timestamp'].min()} → {df['timestamp'].max()}")

# Encodage des catégorielles
for col in ["time_context", "protocol"]:
    if col in df.columns and df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

available = [f for f in FEATURES if f in df.columns]
X = df[available].copy()

# Encoder toutes les colonnes non-numériques restantes
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    print(f"  Encodage LabelEncoder : {col}")

y = df[LABEL_COL]

print(f"\n  Distribution des classes :")
for cls, cnt in y.value_counts().items():
    print(f"    {str(cls):20s} : {cnt:6,} ({cnt/len(y)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# 2. SPLIT TEMPOREL 80/20
# ══════════════════════════════════════════════════════════════
split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n  Train : {len(X_train):,} · Test : {len(X_test):,}")
print(f"  Distribution train : {dict(y_train.value_counts())}")
print(f"  Distribution test  : {dict(y_test.value_counts())}")

# ══════════════════════════════════════════════════════════════
# 3. CALCUL DES SAMPLE WEIGHTS (correction déséquilibre)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CALCUL DES POIDS (correction déséquilibre)")
print("=" * 60)

# Étape 1 : poids de base inversement proportionnels à la fréquence de classe
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# Étape 2 : boost supplémentaire sur la classe attack
# Fonctionne que les labels soient numériques (2) ou texte ("attack")
attack_label = 2 if 2 in y_train.values else "attack"
attack_mask = (y_train == attack_label)
sample_weights[attack_mask.values] *= ATTACK_BOOST

# Résumé des poids moyens par classe
for cls in y_train.unique():
    mask = (y_train == cls)
    w_mean = sample_weights[mask].mean()
    print(f"  Poids moyen classe '{cls}' : {w_mean:.3f}")

print(f"\n  → Classe 'attack' sur-pondérée d'un facteur {ATTACK_BOOST}x")
print(f"    Rater une attaque coûte {ATTACK_BOOST}x plus cher à l'entraînement")

# ══════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT XGBOOST AVEC POIDS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ENTRAÎNEMENT XGBOOST")
print("=" * 60)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
model.fit(X_train, y_train, sample_weight=sample_weights)
train_time = time.time() - t0
print(f"  ✓ Entraînement terminé en {train_time:.1f}s")

# ══════════════════════════════════════════════════════════════
# 5. ÉVALUATION — seuil standard vs seuil adapté
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ÉVALUATION")
print("=" * 60)

probas_test = model.predict_proba(X_test)

# Récupérer l'ordre des classes tel que XGBoost les a vues
classes = list(model.classes_)  # ex: ["attack", "false_positive", "normal"] ou [0, 1, 2]
attack_idx = classes.index("attack") if "attack" in classes else 2
print(f"  Classes XGBoost : {classes}  (index attack = {attack_idx})")

# ── Prédiction avec seuil STANDARD (argmax) ──
y_pred_standard = model.predict(X_test)

# ── Prédiction avec seuil ADAPTÉ ──
def predict_with_threshold(probas, classes, attack_idx, threshold=ATTACK_THRESHOLD):
    """
    Si P(attack) >= threshold → classe attack.
    Sinon → argmax sur les autres classes seulement.
    """
    preds = []
    other_idx = [i for i in range(len(classes)) if i != attack_idx]
    for p in probas:
        if p[attack_idx] >= threshold:
            preds.append(classes[attack_idx])
        else:
            best_other = other_idx[int(np.argmax([p[i] for i in other_idx]))]
            preds.append(classes[best_other])
    return np.array(preds)

y_pred_adapted = predict_with_threshold(probas_test, classes, attack_idx, ATTACK_THRESHOLD)

# ── Métriques comparées ──
attack_label = classes[attack_idx]

def get_fn_attacks(y_true, y_pred, attack_label):
    labels_order = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    a_idx = labels_order.index(attack_label)
    fn = sum(cm[a_idx, j] for j in range(len(labels_order)) if j != a_idx)
    rate = fn / cm[a_idx].sum() if cm[a_idx].sum() > 0 else 0
    return fn, rate

fn_std,  fnr_std  = get_fn_attacks(y_test, y_pred_standard, attack_label)
fn_adpt, fnr_adpt = get_fn_attacks(y_test, y_pred_adapted,  attack_label)

print(f"\n  Seuil standard (0.5) :")
print(f"    Accuracy          : {accuracy_score(y_test, y_pred_standard):.4f}")
print(f"    F1 weighted       : {f1_score(y_test, y_pred_standard, average='weighted'):.4f}")
print(f"    Attaques manquées : {fn_std} ({fnr_std:.1%})")

print(f"\n  Seuil adapté ({ATTACK_THRESHOLD}) :")
print(f"    Accuracy          : {accuracy_score(y_test, y_pred_adapted):.4f}")
print(f"    F1 weighted       : {f1_score(y_test, y_pred_adapted, average='weighted'):.4f}")
print(f"    Attaques manquées : {fn_adpt} ({fnr_adpt:.1%})  ← objectif : minimiser")

print(f"\n  Rapport complet (seuil adapté) :")
print(classification_report(y_test, y_pred_adapted, target_names=LABELS_STR))

# ── Chercher le seuil optimal sur la courbe Précision-Rappel ──
print("\n  Recherche du seuil optimal (max F1 sur classe attack) ...")
p_vals, r_vals, thresholds = precision_recall_curve(
    (y_test == attack_label).astype(int), probas_test[:, attack_idx]
)
f1_vals = 2 * p_vals * r_vals / (p_vals + r_vals + 1e-9)
best_idx = np.argmax(f1_vals)
optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else ATTACK_THRESHOLD
print(f"  Seuil optimal trouvé : {optimal_threshold:.3f}  (F1 attack = {f1_vals[best_idx]:.3f})")
print(f"  Seuil utilisé        : {ATTACK_THRESHOLD}  (peut être ajusté dans la config)")

# ══════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════
print("\nGénération des figures...")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

# ── Fig 1 : Comparaison seuil standard vs adapté ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Impact du seuil de décision sur la détection des attaques\n"
             f"(XGBoost + sample_weight · ATTACK_BOOST={ATTACK_BOOST})",
             fontsize=13, fontweight="bold")

for ax, y_pred, titre, color in zip(
    axes,
    [y_pred_standard, y_pred_adapted],
    [f"Seuil standard (0.5)\n{fn_std} attaques manquées ({fnr_std:.1%})",
     f"Seuil adapté ({ATTACK_THRESHOLD})\n{fn_adpt} attaques manquées ({fnr_adpt:.1%})"],
    ["Blues", "RdYlGn"]
):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap=color, ax=ax,
                xticklabels=LABELS_STR, yticklabels=LABELS_STR,
                linewidths=0.5, linecolor="#e0e0e0")
    ax.set_title(titre, fontsize=11, fontweight="bold")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}fig_threshold_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_threshold_comparison.png")

# ── Fig 2 : Courbe Précision-Rappel pour la classe attack ──
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(r_vals, p_vals, color="#7F77DD", linewidth=2, label=f"PR curve (AUC={auc(r_vals, p_vals):.3f})")
ax.axvline(r_vals[best_idx], color="#1D9E75", linestyle="--", linewidth=1.5,
           label=f"Seuil optimal = {optimal_threshold:.3f}")
ax.axvline(
    r_vals[np.argmin(np.abs(thresholds - ATTACK_THRESHOLD))] if ATTACK_THRESHOLD in thresholds
    else r_vals[np.argmin(np.abs(thresholds - ATTACK_THRESHOLD))],
    color="#EF9F27", linestyle="--", linewidth=1.5,
    label=f"Seuil utilisé = {ATTACK_THRESHOLD}"
)
ax.set_xlabel("Rappel (Recall)")
ax.set_ylabel("Précision")
ax.set_title("Courbe Précision-Rappel — Classe ATTACK\n"
             "Plus le rappel est haut, moins d'attaques sont manquées",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}fig_pr_curve_attack.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_pr_curve_attack.png")

# ── Fig 3 : Feature importance ──
fi = pd.Series(model.feature_importances_, index=available).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#7F77DD" if v > 0.05 else "#CECBF6" for v in fi.values]
ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
ax.set_title("Feature Importance – XGBoost avec pondération des classes\n"
             "(sans features IDS — pas de leakage)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Importance relative")
for i, (feat, val) in enumerate(zip(fi.index[::-1], fi.values[::-1])):
    if val > 0.003:
        ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=9, color="#3C3489")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}fig_feature_importance_weighted.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_feature_importance_weighted.png")

# ══════════════════════════════════════════════════════════════
# 7. SAUVEGARDE DU MODÈLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SAUVEGARDE DU MODÈLE")
print("=" * 60)

# On sauvegarde le modèle ET ses métadonnées dans un seul objet
model_bundle = {
    "model":              model,
    "features":           available,
    "attack_threshold":   ATTACK_THRESHOLD,
    "attack_boost":       ATTACK_BOOST,
    "labels":             LABELS_STR,
    "train_samples":      len(X_train),
    "fn_rate_adapted":    round(fnr_adpt, 4),
    "optimal_threshold":  round(optimal_threshold, 4),
    "trained_at":         pd.Timestamp.now().isoformat(),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"  ✓ Modèle sauvegardé : {MODEL_PATH}")
print(f"  ✓ Seuil de décision embarqué : {ATTACK_THRESHOLD}")
print(f"  ✓ Seuil optimal calculé      : {optimal_threshold:.3f}")

# ══════════════════════════════════════════════════════════════
# 8. RAPPORT JSON
# ══════════════════════════════════════════════════════════════
report = {
    "modele": "XGBoost",
    "correction_desequilibre": {
        "methode": "sample_weight balanced + attack boost",
        "attack_boost_factor": ATTACK_BOOST,
        "attack_threshold": ATTACK_THRESHOLD,
        "optimal_threshold_pr_curve": round(optimal_threshold, 4),
        "justification": (
            f"Le dataset contient {(y_train==2).mean()*100:.1f}% d'attaques. "
            f"En production SOC réelle, ce taux est < 1%. "
            f"Le boost {ATTACK_BOOST}x + seuil {ATTACK_THRESHOLD} compensent ce biais."
        ),
    },
    "performances": {
        "seuil_standard_0.5": {
            "accuracy":        round(accuracy_score(y_test, y_pred_standard), 4),
            "f1_weighted":     round(f1_score(y_test, y_pred_standard, average="weighted"), 4),
            "attaques_manquees": int(fn_std),
            "taux_fn_attack":  round(fnr_std, 4),
        },
        f"seuil_adapte_{ATTACK_THRESHOLD}": {
            "accuracy":        round(accuracy_score(y_test, y_pred_adapted), 4),
            "f1_weighted":     round(f1_score(y_test, y_pred_adapted, average="weighted"), 4),
            "attaques_manquees": int(fn_adpt),
            "taux_fn_attack":  round(fnr_adpt, 4),
        },
    },
    "features": available,
    "model_path": MODEL_PATH,
}

with open(f"{OUTPUT_DIR}training_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  ✓ Rapport JSON sauvegardé")

# ══════════════════════════════════════════════════════════════
# 9. SYNTHÈSE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SYNTHÈSE")
print("=" * 60)
print(f"  Seuil 0.5 (avant) → {fn_std} attaques manquées ({fnr_std:.1%})")
print(f"  Seuil {ATTACK_THRESHOLD} (après) → {fn_adpt} attaques manquées ({fnr_adpt:.1%})")
if fn_std > 0:
    print(f"  Amélioration : {(1 - fnr_adpt/fnr_std)*100:.0f}% de réduction des attaques manquées")
print(f"\n  → Modèle prêt pour 2_realtime_pipeline.py")
print(f"  → Mettre à jour MODEL_PATH dans le pipeline si nécessaire")
print(f"\n✓ Terminé.")