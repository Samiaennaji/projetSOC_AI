"""
SOC Detection – Pipeline temps réel (simulation Wazuh)
=======================================================
Livrable 2/3 : Simulation d'ingestion de logs + prédiction en temps réel

Architecture simulée :
  Logs Wazuh (JSON)
      ↓ parse_wazuh_log()
  Extraction features réseau
      ↓ extract_features()
  Modèle XGBoost
      ↓ predict()
  Classification + score de risque
      ↓ filter_alerts()
  Dashboard SOC (console + CSV)

Usage :
  python 2_realtime_pipeline.py              # simulation avec logs synthétiques
  python 2_realtime_pipeline.py --csv logs.csv   # depuis un fichier CSV existant
"""

import argparse
import time
import json
import pickle
import queue
import threading
import datetime
import random
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
MODEL_PATH  = "E:\\Projets\\ProjetSOC_AI\\models\\xgb_soc.pkl"   # ← chemin vers votre modèle sauvegardé
OUT_CSV     = "E:\\Projets\\ProjetSOC_AI\\soc_realtime_output.csv"
BATCH_SIZE  = 10      # alertes traitées par batch
SLEEP_SEC   = 0.5     # délai entre batches (simulation temps réel)
N_LOGS_SIM  = 500     # nombre de logs simulés au total

FEATURES = [
    "hour_of_day", "time_context", "protocol",
    "src_port", "dst_port",
    "src_is_internal", "dst_is_internal", "is_internal_to_internal",
    "same_subnet", "freq_per_min", "is_high_freq",
    "src_port_ephemeral", "dst_port_privileged",
]
LABELS_STR  = {0: "normal", 1: "false_positive", 2: "attack"}
RISK_COLOR  = {0: "\033[32m", 1: "\033[33m", 2: "\033[31m"}  # vert/jaune/rouge
RESET       = "\033[0m"
BOLD        = "\033[1m"

# ── Correction du déséquilibre de classes ─────────────────────────────────
# En production SOC réelle, les attaques représentent souvent < 1% du trafic.
# Le modèle entraîné sur 14% d'attaques sera trop confiant sur "normal" en prod.
#
# Solution : on abaisse le seuil de décision pour la classe attack.
# Si P(attack) >= ATTACK_THRESHOLD → on classe en ALERT, même si ce n'est pas
# la classe la plus probable. On préfère une fausse alarme à une attaque manquée.
ATTACK_THRESHOLD = 0.15   # ← ajustez selon votre tolérance aux faux positifs
                           #   0.10 = très sensible · 0.20 = modéré · 0.30 = conservateur

# Facteur de boost pour l'entraînement du modèle de démo
ATTACK_BOOST = 5.0

# ══════════════════════════════════════════════════════════════
# 1. GÉNÉRATEUR DE LOGS WAZUH SIMULÉS
# ══════════════════════════════════════════════════════════════
WAZUH_TEMPLATES = {
    "normal": [
        {"rule.description": "Web traffic HTTPS", "rule.level": 3,
         "data.protocol": "TCP", "data.srcport": None, "data.dstport": 443,
         "data.srcip_internal": True, "data.dstip_internal": True},
        {"rule.description": "DNS query", "rule.level": 2,
         "data.protocol": "TCP", "data.srcport": None, "data.dstport": 53,
         "data.srcip_internal": True, "data.dstip_internal": False},
        {"rule.description": "FTP transfer", "rule.level": 3,
         "data.protocol": "TCP", "data.srcport": None, "data.dstport": 21,
         "data.srcip_internal": True, "data.dstip_internal": False},
    ],
    "false_positive": [
        {"rule.description": "Web Scan detected", "rule.level": 6,
         "data.protocol": "TCP", "data.srcport": None, "data.dstport": 80,
         "data.srcip_internal": True, "data.dstip_internal": True},
        {"rule.description": "ICMP Ping Sweep", "rule.level": 5,
         "data.protocol": "ICMP", "data.srcport": None, "data.dstport": 0,
         "data.srcip_internal": True, "data.dstip_internal": True},
    ],
    "attack": [
        {"rule.description": "SSH Brute Force attempt", "rule.level": 12,
         "data.protocol": "TCP", "data.srcport": None, "data.dstport": 22,
         "data.srcip_internal": False, "data.dstip_internal": True},
        {"rule.description": "Hydra HTTP Tool detected", "rule.level": 14,
         "data.protocol": "TCP", "data.srcport": None, "data.dstport": 80,
         "data.srcip_internal": False, "data.dstip_internal": True},
        {"rule.description": "SQL Injection pattern", "rule.level": 13,
         "data.protocol": "TCP", "data.srcport": None, "data.dstport": 443,
         "data.srcip_internal": False, "data.dstip_internal": True},
    ],
}

def generate_wazuh_log(true_class: str = None) -> dict:
    """Simule un log JSON tel que Wazuh le produit."""
    if true_class is None:
        true_class = random.choices(
            ["normal", "false_positive", "attack"],
            weights=[0.58, 0.28, 0.14]
        )[0]

    template = random.choice(WAZUH_TEMPLATES[true_class])
    now = datetime.datetime.now()
    hour = now.hour
    src_port = random.randint(1025, 65535)
    freq = {
        "normal": max(1, int(np.random.exponential(4))),
        "false_positive": max(1, int(np.random.exponential(8))),
        "attack": max(10, int(np.random.exponential(50))),
    }[true_class]

    return {
        "timestamp": now.isoformat(),
        "rule.description": template["rule.description"],
        "rule.level": template["rule.level"],
        "data.protocol": template["data.protocol"],
        "data.srcport": src_port,
        "data.dstport": template["data.dstport"],
        "data.srcip_internal": template["data.srcip_internal"],
        "data.dstip_internal": template["data.dstip_internal"],
        "_true_class": true_class,   # pour évaluation – pas utilisé par le modèle
        "_freq": freq,
        "_hour": hour,
    }

# ══════════════════════════════════════════════════════════════
# 2. EXTRACTION DE FEATURES DEPUIS UN LOG WAZUH
# ══════════════════════════════════════════════════════════════
TIME_CONTEXT_MAP = {
    range(9, 18): 0,   # business hours
    range(18, 22): 1,  # soirée
    range(0, 6): 2,    # nuit profonde
    range(6, 9): 3,    # matin
    range(22, 24): 3,
}

def get_time_context(hour: int) -> int:
    for r, v in TIME_CONTEXT_MAP.items():
        if hour in r:
            return v
    return 3

def extract_features(log: dict, freq_window: deque) -> pd.Series:
    """
    Transforme un log Wazuh brut en vecteur de features attendu par le modèle.
    Dans un vrai déploiement, freq_window est maintenu sur une fenêtre glissante.
    """
    hour        = log.get("_hour", datetime.datetime.now().hour)
    freq        = log.get("_freq", len(freq_window))
    src_int     = int(log.get("data.srcip_internal", False))
    dst_int     = int(log.get("data.dstip_internal", True))
    proto_str   = log.get("data.protocol", "TCP")
    proto       = 0 if proto_str == "TCP" else 1
    src_port    = log.get("data.srcport", 50000)
    dst_port    = log.get("data.dstport", 80)

    return pd.Series({
        "hour_of_day":              hour,
        "time_context":             get_time_context(hour),
        "protocol":                 proto,
        "src_port":                 src_port,
        "dst_port":                 dst_port,
        "src_is_internal":          src_int,
        "dst_is_internal":          dst_int,
        "is_internal_to_internal":  int(src_int and dst_int),
        "same_subnet":              int(src_int and dst_int and random.random() < 0.3),
        "freq_per_min":             freq,
        "is_high_freq":             int(freq > 20),
        "src_port_ephemeral":       int(src_port > 1024),
        "dst_port_privileged":      int(dst_port < 1024),
    })

# ══════════════════════════════════════════════════════════════
# 3. MOTEUR DE PRÉDICTION
# ══════════════════════════════════════════════════════════════
class SOCPredictor:
    def __init__(self, model_path: str):
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)

            # Support des deux formats : modèle seul (ancien) ou bundle (nouveau)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model = bundle["model"]
                # Lire le seuil embarqué dans le bundle si disponible
                global ATTACK_THRESHOLD
                ATTACK_THRESHOLD = bundle.get("attack_threshold", ATTACK_THRESHOLD)
                print(f"  ✓ Modèle chargé (bundle) : {model_path}")
                print(f"  ✓ Seuil de décision embarqué : {ATTACK_THRESHOLD}")
                print(f"  ✓ Entraîné le : {bundle.get('trained_at', 'inconnu')}")
                print(f"  ✓ Taux FN attack à l'entraînement : {bundle.get('fn_rate_adapted', 'N/A')}")
            else:
                self.model = bundle
                print(f"  ✓ Modèle chargé (format ancien) : {model_path}")
                print(f"  ⚠ Seuil de décision utilisé : {ATTACK_THRESHOLD} (config locale)")

        except FileNotFoundError:
            print("  ⚠ Modèle non trouvé – entraînement d'un modèle de démo...")
            self._train_demo_model()

        self.freq_window = deque(maxlen=60)   # fenêtre 60 secondes
        self.stats = {"normal": 0, "false_positive": 0, "attack": 0, "total": 0}

    def _train_demo_model(self):
        """Entraîne un petit modèle de démonstration si le modèle principal n'existe pas."""
        import xgboost as xgb
        np.random.seed(42)
        N = 10000
        rows, labels = [], []
        for _ in range(int(N * 0.58)):
            rows.append([random.randint(8,18), 0, 0, random.randint(1025,65535),
                         random.choice([80,443,53]), 1, 1, 1, 0, random.randint(1,8), 0, 1, 1])
            labels.append(0)
        for _ in range(int(N * 0.28)):
            rows.append([random.randint(0,23), random.randint(0,3), 0,
                         random.randint(1025,65535), random.choice([80,443,22]),
                         1, 1, 1, 1, random.randint(5,25), 1, 1, 1])
            labels.append(1)
        for _ in range(int(N * 0.14)):
            rows.append([random.randint(0,23), 2, 0, random.randint(30000,65535),
                         random.choice([22,80,443]), 0, 1, 0, 0,
                         random.randint(30,200), 1, 1, 1])
            labels.append(2)

        X = pd.DataFrame(rows, columns=FEATURES)
        y = pd.Series(labels)

        # Pondération pour corriger le déséquilibre même sur le modèle de démo
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y)
        attack_mask = (y == 2)
        sw[attack_mask] *= ATTACK_BOOST

        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                        eval_metric="mlogloss", random_state=42)
        self.model.fit(X, y, sample_weight=sw)
        print("  ✓ Modèle de démo entraîné (avec pondération classes)")

    def predict_batch(self, logs: list) -> list:
        """Prédit la classe pour un batch de logs bruts avec seuil adapté."""
        self.freq_window.extend(logs)
        features = pd.DataFrame([
            extract_features(log, self.freq_window) for log in logs
        ])
        probas = self.model.predict_proba(features)

        results = []
        for i, (log, proba) in enumerate(zip(logs, probas)):
            # ── Seuil adapté au lieu de argmax brut ──────────────────────
            # Si P(attack) >= ATTACK_THRESHOLD → forcer la classe attack,
            # même si P(normal) ou P(false_positive) est plus élevé.
            # Cela compense la sous-représentation des attaques en prod réelle.
            if proba[2] >= ATTACK_THRESHOLD:
                pred = 2
            else:
                pred = int(np.argmax(proba[:2]))  # choisir entre normal et FP

            cls_name   = LABELS_STR[pred]

            # ── Risk score calibré ────────────────────────────────────────
            # On ne prend plus P(attack) brut (biaisé vers 0 à cause du déséquilibre).
            # On normalise par rapport au seuil : score 50 = exactement au seuil.
            raw_proba_attack = float(proba[2])
            if raw_proba_attack >= ATTACK_THRESHOLD:
                # Au-dessus du seuil : score entre 50 et 100
                risk_score = int(50 + 50 * (raw_proba_attack - ATTACK_THRESHOLD) / (1 - ATTACK_THRESHOLD))
            else:
                # En dessous du seuil : score entre 0 et 50
                risk_score = int(50 * raw_proba_attack / ATTACK_THRESHOLD)

            result = {
                "timestamp":   log["timestamp"],
                "description": log.get("rule.description", ""),
                "prediction":  cls_name,
                "risk_score":  risk_score,
                "proba_normal":    round(float(proba[0]), 3),
                "proba_fp":        round(float(proba[1]), 3),
                "proba_attack":    round(float(proba[2]), 3),
                "attack_threshold_used": ATTACK_THRESHOLD,
                "true_class":  log.get("_true_class", "unknown"),
            }
            self.stats[cls_name] += 1
            self.stats["total"]  += 1
            results.append(result)

        return results

# ══════════════════════════════════════════════════════════════
# 4. AFFICHAGE CONSOLE (simulation dashboard terminal)
# ══════════════════════════════════════════════════════════════
def print_header():
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  SOC AI DETECTION ENGINE – Temps réel{RESET}")
    print(f"  Modèle : XGBoost · Features réseau brutes (sans leakage IDS)")
    print(f"{'═'*70}{RESET}\n")

def print_result(result: dict):
    cls = result["prediction"]
    col = RISK_COLOR.get({"normal":0,"false_positive":1,"attack":2}.get(cls,0), "")
    risk = result["risk_score"]
    bar  = "█" * (risk // 10) + "░" * (10 - risk // 10)

    status = "🔴 ATTAQUE" if cls == "attack" else "🟡 FAUX POS" if cls == "false_positive" else "🟢 NORMAL"
    print(f"  {result['timestamp'][11:19]}  │  {status:12s}  │  "
          f"Risk: {col}{bar}{RESET} {risk:3d}%  │  {result['description'][:45]}")

def print_stats(stats: dict, elapsed: float):
    total = stats["total"]
    if total == 0:
        return
    pct = lambda k: f"{stats[k]/total*100:.1f}%"
    print(f"\n{BOLD}  ── STATISTIQUES ({total} alertes en {elapsed:.0f}s) ──{RESET}")
    print(f"  \033[32m✓ Normal       : {stats['normal']:5d} ({pct('normal')}){RESET}")
    print(f"  \033[33m◐ Faux positifs: {stats['false_positive']:5d} ({pct('false_positive')}){RESET}")
    print(f"  \033[31m✗ Attaques     : {stats['attack']:5d} ({pct('attack')}){RESET}")
    print(f"  {'─'*40}")
    fp_reduction = stats["false_positive"] / max(total - stats["normal"], 1)
    print(f"  Taux FP filtrés de la file analyste : {fp_reduction:.1%}")
    print(f"  Attaques détectées / totales : {stats['attack']} / {total}")
    print(f"  \033[33m⚙  Seuil de décision attack : {ATTACK_THRESHOLD} (défaut = 0.5){RESET}\n")

# ══════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════
def run_pipeline(source: str = "sim", csv_path: str = None):
    print_header()
    predictor = SOCPredictor(MODEL_PATH)
    output_rows = []
    t_start = time.time()

    if source == "csv" and csv_path:
        df_logs = pd.read_csv(csv_path)
        print(f"  Source : {csv_path} ({len(df_logs)} lignes)\n")
        # Convertir le CSV en format log dict
        all_logs = df_logs.to_dict("records")
    else:
        print(f"  Source : simulation Wazuh ({N_LOGS_SIM} événements)\n")
        all_logs = [generate_wazuh_log() for _ in range(N_LOGS_SIM)]

    # Traitement par batch
    for i in range(0, len(all_logs), BATCH_SIZE):
        batch = all_logs[i : i + BATCH_SIZE]
        results = predictor.predict_batch(batch)

        for r in results:
            print_result(r)
            output_rows.append(r)

        # Afficher les stats toutes les 100 alertes
        if (i + BATCH_SIZE) % 100 == 0:
            print_stats(predictor.stats, time.time() - t_start)

        time.sleep(SLEEP_SEC)

    # Stats finales
    print_stats(predictor.stats, time.time() - t_start)

    # Sauvegarde CSV
    pd.DataFrame(output_rows).to_csv(OUT_CSV, index=False)
    print(f"  ✓ Résultats sauvegardés : {OUT_CSV}")
    return predictor.stats, output_rows

# ══════════════════════════════════════════════════════════════
# 6. ENTRÉE PRINCIPALE
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOC Realtime Pipeline")
    parser.add_argument("--csv", type=str, default=None,
                        help="Chemin vers un CSV de logs (optionnel, sinon simulation)")
    args = parser.parse_args()

    source = "csv" if args.csv else "sim"
    stats, results = run_pipeline(source=source, csv_path=args.csv)