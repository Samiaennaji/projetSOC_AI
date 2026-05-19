"""
SOC Detection – Pipeline temps réel CORRIGÉ
============================================
Correction principale : simulation alignée sur les distributions
réelles du dataset soc_detection_dataset_v3.

Problème initial :
  - generate_wazuh_log() générait des features aléatoires sans lien
    avec la distribution d'entraînement → proba_attack ≈ 0 sur 35/48
    vraies attaques → FNR = 72.9% (au lieu de ~15% attendu)

Corrections appliquées :
  1. Chaque classe utilise les vraies distributions observées :
       normal       → src_is_internal=1, freq médiane=6,  severity=INFO
       false_positive → src_is_internal=0|1, freq médiane=4, severity=LOW..CRITICAL
       attack       → src_is_internal=1, same_subnet=1, freq médiane=152, severity=HIGH|CRITICAL

  2. Les IPs sont tirées des pools réels du dataset (pas random.randint)

  3. Le résultat CSV est enrichi avec toutes les features d'entrée
     (severity, src_ip, dst_service, freq_per_min…) pour que le LLM
     4_llm_advisor.py reçoive un contexte complet

  4. Le mode --csv charge directement le dataset réel ligne par ligne
     → métriques identiques à l'évaluation d'entraînement

Usage :
  python 2_realtime_pipeline_fixed.py              # simulation fidèle
  python 2_realtime_pipeline_fixed.py --csv soc_detection_dataset_v3.csv
"""

import argparse
import time
import json
import pickle
import random
import datetime
import numpy as np
import pandas as pd
from collections import deque

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
MODEL_PATH  = "E:\\Projets\\ProjetSOC_AI\\models\\xgb_soc.pkl"
OUT_CSV     = "E:\\Projets\\ProjetSOC_AI\\soc_realtime_output.csv"
BATCH_SIZE  = 10
SLEEP_SEC   = 0.5
N_LOGS_SIM  = 500

FEATURES = [
    "hour_of_day", "time_context", "protocol",
    "src_port", "dst_port",
    "src_is_internal", "dst_is_internal", "is_internal_to_internal",
    "same_subnet", "freq_per_min", "is_high_freq",
    "src_port_ephemeral", "dst_port_privileged",
]
LABELS_STR = {0: "normal", 1: "false_positive", 2: "attack"}
RISK_COLOR = {0: "\033[32m", 1: "\033[33m", 2: "\033[31m"}
RESET = "\033[0m"
BOLD  = "\033[1m"

ATTACK_THRESHOLD = 0.15

# ══════════════════════════════════════════════════════════════
# DISTRIBUTIONS RÉELLES PAR CLASSE
# Extraites du dataset soc_detection_dataset_v3 (52 000 lignes)
# Vérifiées sur les 3 classes — à NE PAS modifier
# ══════════════════════════════════════════════════════════════

# ── Pools d'IPs réelles ───────────────────────────────────────
IPS = {
    "normal_src":  ["192.168.1.50", "10.0.0.15", "10.10.0.5",
                    "192.168.10.5", "192.168.1.20"],
    "normal_dst":  ["10.0.0.1", "192.168.1.1", "192.168.126.139"],
    "fp_src":      ["192.168.10.5", "172.16.0.5", "192.168.1.20",
                    "192.168.1.10", "10.0.0.100"],
    "fp_dst":      ["192.168.1.1", "192.168.126.139", "192.168.126.128"],
    "attack_src":  ["192.168.126.133", "192.168.126.1"],
    "attack_dst":  ["192.168.126.139", "192.168.126.128"],
}

# ── Noms d'alertes réels par classe ───────────────────────────
ALERT_NAMES = {
    "normal":         ["https", "http", "dns", "ssh", "smtp",
                       "ftp", "http_alt", "icmp"],
    "false_positive": ["Web Scan", "Basic Injection Pattern",
                       "HTTP Login Bruteforce", "ICMP Ping Sweep",
                       "Sensitive File Probe", "Script Injection Path",
                       "Traversal Path Attempt"],
    "attack":         ["Web Scan", "Hydra HTTP Tool", "SSH Scan",
                       "Traversal Path Attempt", "Sensitive File Probe",
                       "SSH Brute Force", "SQL Injection pattern",
                       "Hydra HTTP Tool detected"],
}

# ── Ports de destination réels par classe ────────────────────
DST_PORTS = {
    "normal":         [443, 80, 53, 22, 25, 21, 8080],
    "false_positive": [80, 22, 443, 0, 8080],
    "attack":         [80, 22],          # 6690/7500 → 80 ; 808/7500 → 22
}

DST_SERVICE_MAP = {
    443: "https", 80: "http", 53: "dns", 22: "ssh",
    25: "smtp",  21: "ftp",  8080: "http_alt", 0: "icmp",
}

# ── Severity réelle par classe ────────────────────────────────
SEVERITY = {
    "normal":         [("INFO", 0)],                         # 100 % INFO
    "false_positive": [("LOW", 1), ("HIGH", 3), ("CRITICAL", 4),
                       ("INFO", 0), ("MEDIUM", 2)],          # distribution mixte
    "attack":         [("HIGH", 3), ("LOW", 1),
                       ("CRITICAL", 4), ("MEDIUM", 2)],      # HIGH+CRITICAL dominant
}
SEVERITY_WEIGHTS = {
    "normal":         [1.0],
    "false_positive": [0.346, 0.300, 0.172, 0.138, 0.044],
    "attack":         [0.451, 0.426, 0.119, 0.005],
}

# ── Contexte horaire réel par classe ─────────────────────────
TIME_CONTEXTS = {
    "normal":         ["business_hours", "night", "after_hours"],
    "false_positive": ["business_hours"],                    # 100 % business_hours
    "attack":         ["business_hours", "after_hours"],     # 75 % / 25 %
}
TIME_CONTEXT_WEIGHTS = {
    "normal":         [0.458, 0.377, 0.165],
    "false_positive": [1.0],
    "attack":         [0.751, 0.249],
}

# Encodage numérique du time_context (identique à l'entraînement)
TIME_CTX_ENCODE = {"business_hours": 0, "night": 1, "after_hours": 2}


# ══════════════════════════════════════════════════════════════
# GÉNÉRATEUR DE LOGS WAZUH SIMULÉS — FIDÈLE AU DATASET
# ══════════════════════════════════════════════════════════════

def generate_wazuh_log(true_class: str = None) -> dict:
    """
    Génère un log Wazuh simulé dont les features respectent EXACTEMENT
    les distributions observées dans soc_detection_dataset_v3.

    Clé : chaque champ est tiré de la distribution réelle par classe,
    pas d'un random générique — c'est ce qui garantit que le modèle
    XGBoost reconnaît les patterns sur lesquels il a été entraîné.
    """
    if true_class is None:
        # Distribution réelle du dataset : 57.7% / 27.9% / 14.4%
        true_class = random.choices(
            ["normal", "false_positive", "attack"],
            weights=[0.577, 0.279, 0.144]
        )[0]

    cls = true_class

    # ── Contexte horaire ──────────────────────────────────────
    time_ctx_str = random.choices(
        TIME_CONTEXTS[cls],
        weights=TIME_CONTEXT_WEIGHTS[cls]
    )[0]
    time_ctx_enc = TIME_CTX_ENCODE[time_ctx_str]

    # Heure cohérente avec le contexte
    hour_map = {
        "business_hours": random.randint(9, 17),
        "after_hours":    random.choice(list(range(18, 22)) + list(range(6, 9))),
        "night":          random.randint(0, 5),
    }
    hour = hour_map[time_ctx_str]

    # ── Ports ─────────────────────────────────────────────────
    # dst_port : tiré du pool réel avec pondération implicite
    dst_port_weights = {
        "normal":         [0.22, 0.22, 0.15, 0.14, 0.07, 0.08, 0.12],
        "false_positive": [0.66, 0.10, 0.06, 0.08, 0.10],
        "attack":         [0.891, 0.109],
    }
    dst_port = random.choices(
        DST_PORTS[cls],
        weights=dst_port_weights[cls]
    )[0]
    dst_service = DST_SERVICE_MAP.get(dst_port, "unknown")

    # src_port : distribution réelle (éphémère > 1024 pour toutes les classes)
    if cls == "attack":
        src_port = random.randint(30000, 65194)   # mean=46775 dans le dataset
    else:
        src_port = random.randint(1025, 64998)

    # ── Flags réseau ──────────────────────────────────────────
    # CRITIQUE : ces valeurs doivent correspondre exactement
    # à ce que le modèle a appris
    if cls == "normal":
        src_int = 1                            # 100 % interne
        dst_int = 1
        int_to_int = 1
        same_subnet = 1 if random.random() < 0.187 else 0   # 5599/30000
        protocol = "TCP" if random.random() < 0.934 else "ICMP"

    elif cls == "false_positive":
        src_int = 1 if random.random() < 0.734 else 0        # 10637/14500
        dst_int = 1
        int_to_int = src_int                   # = src_int dans le dataset
        same_subnet = 1 if random.random() < 0.202 else 0    # 2934/14500
        protocol = "TCP" if random.random() < 0.940 else "ICMP"

    else:   # attack
        # CORRECTION CRITIQUE : dans le dataset, les attaques sont
        # TOUTES src_is_internal=1 (mouvement latéral interne)
        # La simulation précédente avait src_is_internal=0 → modèle trompé
        src_int = 1
        dst_int = 1
        int_to_int = 1
        same_subnet = 1                        # 100 % dans le dataset
        protocol = "TCP"                       # 7498/7500 TCP

    # ── Fréquence ─────────────────────────────────────────────
    # Distribution réelle validée :
    #   normal       → médiane=6,   is_high_freq=6.8%,  p95=34
    #   false_positive → médiane=4, is_high_freq=0%,    p95=16
    #   attack       → médiane=152, is_high_freq=84.4%
    if cls == "normal":
        freq = max(1, int(np.random.choice(
            [1, 2, 3, 4, 6, 8, 11, 16, 22, 30, 38],
            p=[0.05, 0.10, 0.13, 0.12, 0.18, 0.13, 0.12, 0.10, 0.04, 0.02, 0.01]
        )))
    elif cls == "false_positive":
        # is_high_freq = 0 dans le dataset réel → toutes les valeurs < 20
        freq = max(1, int(np.random.choice(
            [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
            p=[0.10, 0.14, 0.13, 0.14, 0.13, 0.12, 0.10, 0.08, 0.04, 0.02]
        )))
    else:   # attack — distribution bimodale (petite queue basse + masse haute)
        if random.random() < 0.155:   # ~1167/7500 is_high_freq=0
            freq = random.randint(1, 20)
        else:
            # Masse principale : percentile 25=140, médiane=152, p95=203
            freq = max(21, int(np.random.normal(loc=140, scale=40)))
            freq = min(freq, 230)

    is_high_freq = 1 if freq > 20 else 0

    # ── Severity ──────────────────────────────────────────────
    sev_pair = random.choices(
        SEVERITY[cls],
        weights=SEVERITY_WEIGHTS[cls]
    )[0]
    severity, severity_score = sev_pair

    # ── IPs ───────────────────────────────────────────────────
    src_ip = random.choice(IPS[f"{cls[:6] if cls!='false_positive' else 'fp'}_src"])
    dst_ip = random.choice(IPS[f"{cls[:6] if cls!='false_positive' else 'fp'}_dst"])

    # ── Nom de l'alerte ───────────────────────────────────────
    alert_name = random.choice(ALERT_NAMES[cls])

    return {
        # Champs de sortie (existants)
        "timestamp":    datetime.datetime.now().isoformat(),
        "description":  alert_name,
        "_true_class":  true_class,

        # Champs d'entrée enrichis (NOUVEAUTÉ — pour le LLM)
        "_alert_name":       alert_name,
        "_severity":         severity,
        "_severity_score":   severity_score,
        "_src_ip":           src_ip,
        "_dst_ip":           dst_ip,
        "_src_port":         src_port,
        "_dst_port":         dst_port,
        "_dst_service":      dst_service,
        "_time_context_str": time_ctx_str,
        "_freq":             freq,

        # Features Wazuh (utilisées par extract_features)
        "data.protocol":       protocol,
        "data.srcport":        src_port,
        "data.dstport":        dst_port,
        "data.srcip_internal": bool(src_int),
        "data.dstip_internal": bool(dst_int),
        "_hour":               hour,
        "_same_subnet":        same_subnet,
        "_int_to_int":         int_to_int,
        "_is_high_freq":       is_high_freq,
    }


# ══════════════════════════════════════════════════════════════
# EXTRACTION DE FEATURES
# ══════════════════════════════════════════════════════════════

TIME_CONTEXT_MAP = {
    range(9, 18): 0,   # business_hours
    range(18, 22): 2,  # after_hours
    range(0,  6):  1,  # night
    range(6,  9):  2,  # after_hours (matin)
    range(22, 24): 1,
}

def get_time_context(hour: int) -> int:
    for r, v in TIME_CONTEXT_MAP.items():
        if hour in r:
            return v
    return 0


def extract_features(log: dict) -> pd.Series:
    """Extrait exactement les 13 features attendues par le modèle."""
    hour     = log.get("_hour", datetime.datetime.now().hour)
    freq     = log.get("_freq", 1)
    src_int  = int(log.get("data.srcip_internal", False))
    dst_int  = int(log.get("data.dstip_internal", True))
    proto    = 0 if log.get("data.protocol", "TCP") == "TCP" else 1
    src_port = log.get("data.srcport", 50000)
    dst_port = log.get("data.dstport", 80)

    # Utiliser les valeurs pré-calculées si disponibles (simulation fidèle)
    same_subnet  = log.get("_same_subnet",  int(src_int and dst_int and random.random() < 0.3))
    int_to_int   = log.get("_int_to_int",   int(src_int and dst_int))
    is_high_freq = log.get("_is_high_freq", int(freq > 20))

    return pd.Series({
        "hour_of_day":             hour,
        "time_context":            get_time_context(hour),
        "protocol":                proto,
        "src_port":                src_port,
        "dst_port":                dst_port,
        "src_is_internal":         src_int,
        "dst_is_internal":         dst_int,
        "is_internal_to_internal": int_to_int,
        "same_subnet":             same_subnet,
        "freq_per_min":            freq,
        "is_high_freq":            is_high_freq,
        "src_port_ephemeral":      int(src_port > 1024),
        "dst_port_privileged":     int(dst_port < 1024),
    })


# ══════════════════════════════════════════════════════════════
# MOTEUR DE PRÉDICTION
# ══════════════════════════════════════════════════════════════

class SOCPredictor:
    def __init__(self, model_path: str):
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model = bundle["model"]
                global ATTACK_THRESHOLD
                ATTACK_THRESHOLD = bundle.get("attack_threshold", ATTACK_THRESHOLD)
                print(f"  ✓ Modèle chargé · seuil embarqué : {ATTACK_THRESHOLD}")
                print(f"  ✓ Entraîné le : {bundle.get('trained_at', 'inconnu')}")
            else:
                self.model = bundle
                print(f"  ✓ Modèle chargé (ancien format) · seuil config : {ATTACK_THRESHOLD}")
        except FileNotFoundError:
            print("  ⚠ Modèle non trouvé — entraînement d'un modèle de démo...")
            self._train_demo_model()

        self.stats = {"normal": 0, "false_positive": 0, "attack": 0, "total": 0}

    def _train_demo_model(self):
        import xgboost as xgb
        from sklearn.utils.class_weight import compute_sample_weight
        np.random.seed(42)
        rows, labels = [], []

        for _ in range(17310):   # normal 57.7%
            f  = random.randint(1025, 64998)
            s  = max(1, int(np.random.choice([1,2,3,4,6,8,11,16,22,30,38], p=[0.05,0.10,0.13,0.12,0.18,0.13,0.12,0.10,0.04,0.02,0.01])))
            rows.append([random.randint(9,17), 0, 0, f, random.choice([443,80,53,22,25]),
                         1, 1, 1, 1 if random.random()<0.187 else 0, s, int(s>20), 1, 1])
            labels.append(0)
        for _ in range(8370):    # false_positive 27.9%
            f  = random.randint(1025, 64995)
            s  = max(1, int(np.random.choice([1,2,3,4,5,6,8,10,13,16], p=[0.10,0.14,0.13,0.14,0.13,0.12,0.10,0.08,0.04,0.02])))
            si = 1 if random.random() < 0.734 else 0
            rows.append([random.randint(9,17), 0, 0, f, random.choice([80,22,443,0,8080]),
                         si, 1, si, 1 if random.random()<0.202 else 0, s, 0, 1, 1])
            labels.append(1)
        for _ in range(4320):    # attack 14.4%
            f = max(21, int(np.random.normal(140, 40)))
            rows.append([random.randint(9,22), 0, 0, random.randint(30000,65194),
                         random.choice([80,22]), 1, 1, 1, 1,
                         min(f, 230), 1, 1, 1])
            labels.append(2)

        X = pd.DataFrame(rows, columns=FEATURES)
        y = pd.Series(labels)
        sw = compute_sample_weight("balanced", y)
        sw[y == 2] *= 5.0
        self.model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        )
        self.model.fit(X, y, sample_weight=sw)
        print("  ✓ Modèle de démo entraîné (distribution alignée dataset)")

    def predict_batch(self, logs: list) -> list:
        features = pd.DataFrame([extract_features(log) for log in logs])
        probas   = self.model.predict_proba(features)
        results  = []

        for log, proba in zip(logs, probas):
            # Seuil adapté
            if proba[2] >= ATTACK_THRESHOLD:
                pred = 2
            else:
                pred = int(np.argmax(proba[:2]))

            cls_name = LABELS_STR[pred]

            # Risk score calibré
            p_atk = float(proba[2])
            if p_atk >= ATTACK_THRESHOLD:
                risk_score = int(50 + 50 * (p_atk - ATTACK_THRESHOLD) / (1 - ATTACK_THRESHOLD))
            else:
                risk_score = int(50 * p_atk / ATTACK_THRESHOLD)

            result = {
                # Champs de sortie existants
                "timestamp":              log["timestamp"],
                "description":            log.get("description", log.get("_alert_name", "")),
                "prediction":             cls_name,
                "risk_score":             risk_score,
                "proba_normal":           round(float(proba[0]), 3),
                "proba_fp":               round(float(proba[1]), 3),
                "proba_attack":           round(float(proba[2]), 3),
                "attack_threshold_used":  ATTACK_THRESHOLD,
                "true_class":             log.get("_true_class", "unknown"),

                # NOUVEAU — champs enrichis pour le LLM (4_llm_advisor.py)
                "alert_name":             log.get("_alert_name", ""),
                "severity":               log.get("_severity", ""),
                "severity_score":         log.get("_severity_score", 0),
                "src_ip":                 log.get("_src_ip", ""),
                "dst_ip":                 log.get("_dst_ip", ""),
                "src_port":               log.get("_src_port", 0),
                "dst_port":               log.get("_dst_port", 0),
                "dst_service":            log.get("_dst_service", ""),
                "freq_per_min":           log.get("_freq", 0),
                "is_high_freq":           log.get("_is_high_freq", 0),
                "time_context":           log.get("_time_context_str", ""),
                "src_is_internal":        int(log.get("data.srcip_internal", True)),
                "dst_is_internal":        int(log.get("data.dstip_internal", True)),
                "is_internal_to_internal": log.get("_int_to_int", 1),
                "same_subnet":             log.get("_same_subnet", 0),
                "protocol":               log.get("data.protocol", "TCP"),
            }
            self.stats[cls_name] += 1
            self.stats["total"]  += 1
            results.append(result)

        return results


# ══════════════════════════════════════════════════════════════
# AFFICHAGE CONSOLE
# ══════════════════════════════════════════════════════════════

def print_header():
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  SOC AI DETECTION ENGINE – Pipeline corrigé{RESET}")
    print(f"  Features alignées sur soc_detection_dataset_v3")
    print(f"{'═'*70}{RESET}\n")


def print_result(result: dict):
    cls = result["prediction"]
    col = RISK_COLOR.get({"normal":0,"false_positive":1,"attack":2}.get(cls, 0), "")
    risk = result["risk_score"]
    bar  = "█" * (risk // 10) + "░" * (10 - risk // 10)
    status = "🔴 ATTAQUE" if cls == "attack" else "🟡 FAUX POS" if cls == "false_positive" else "🟢 NORMAL"
    true_c = result.get("true_class", "?")
    match  = "✓" if cls == true_c else "✗"
    print(f"  {result['timestamp'][11:19]}  │  {status:12s}  │  "
          f"Risk: {col}{bar}{RESET} {risk:3d}%  │  "
          f"{result.get('description','')[:35]:35s}  │  [{match}] réel:{true_c}")


def print_stats(stats: dict, elapsed: float):
    total = stats["total"]
    if total == 0:
        return
    pct = lambda k: f"{stats[k]/total*100:.1f}%"
    print(f"\n{BOLD}  ── STATISTIQUES ({total} alertes en {elapsed:.0f}s) ──{RESET}")
    print(f"  \033[32m✓ Normal        : {stats['normal']:5d} ({pct('normal')}){RESET}")
    print(f"  \033[33m◐ Faux positifs : {stats['false_positive']:5d} ({pct('false_positive')}){RESET}")
    print(f"  \033[31m✗ Attaques      : {stats['attack']:5d} ({pct('attack')}){RESET}")
    print(f"  {'─'*40}")
    print(f"  Seuil de décision attack : {ATTACK_THRESHOLD}\n")


# ══════════════════════════════════════════════════════════════
# MODE CSV — lecture du dataset réel ligne par ligne
# ══════════════════════════════════════════════════════════════

def csv_row_to_log(row: dict) -> dict:
    """
    Convertit une ligne du dataset réel en format log dict
    compatible avec extract_features() et predict_batch().
    """
    time_ctx_str = str(row.get("time_context", "business_hours"))
    return {
        "timestamp":    str(row.get("timestamp", datetime.datetime.now().isoformat())),
        "description":  str(row.get("alert_name", "")),
        "_true_class":  {0: "normal", 1: "false_positive", 2: "attack"}.get(
                            int(row.get("label", 0)), "unknown"),
        "_alert_name":       str(row.get("alert_name", "")),
        "_severity":         str(row.get("severity", "")),
        "_severity_score":   int(row.get("severity_score", 0)),
        "_src_ip":           str(row.get("src_ip", "")),
        "_dst_ip":           str(row.get("dst_ip", "")),
        "_src_port":         int(row.get("src_port", 0)),
        "_dst_port":         int(row.get("dst_port", 0)),
        "_dst_service":      str(row.get("dst_service", "")),
        "_time_context_str": time_ctx_str,
        "_freq":             int(row.get("freq_per_min", 1)),
        "_is_high_freq":     int(row.get("is_high_freq", 0)),
        "_same_subnet":      int(row.get("same_subnet", 0)),
        "_int_to_int":       int(row.get("is_internal_to_internal", 0)),
        "_hour":             int(row.get("hour_of_day", 12)),
        "data.protocol":     str(row.get("protocol", "TCP")),
        "data.srcport":      int(row.get("src_port", 0)),
        "data.dstport":      int(row.get("dst_port", 0)),
        "data.srcip_internal": bool(int(row.get("src_is_internal", 1))),
        "data.dstip_internal": bool(int(row.get("dst_is_internal", 1))),
    }


# ══════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def run_pipeline(source: str = "sim", csv_path: str = None):
    print_header()
    predictor   = SOCPredictor(MODEL_PATH)
    output_rows = []
    t_start     = time.time()

    if source == "csv" and csv_path:
        df_logs = pd.read_csv(csv_path).head(N_LOGS_SIM)
        print(f"  Source : {csv_path} ({len(df_logs)} lignes)\n")
        all_logs = [csv_row_to_log(row) for row in df_logs.to_dict("records")]
    else:
        print(f"  Source : simulation fidèle ({N_LOGS_SIM} événements)\n")
        print(f"  Distribution cible : 57.7% normal / 27.9% FP / 14.4% attack\n")
        all_logs = [generate_wazuh_log() for _ in range(N_LOGS_SIM)]

    for i in range(0, len(all_logs), BATCH_SIZE):
        batch   = all_logs[i: i + BATCH_SIZE]
        results = predictor.predict_batch(batch)
        for r in results:
            print_result(r)
            output_rows.append(r)
        if (i + BATCH_SIZE) % 100 == 0:
            print_stats(predictor.stats, time.time() - t_start)
        time.sleep(SLEEP_SEC)

    print_stats(predictor.stats, time.time() - t_start)

    # Évaluation recall sur les attaques
    df_out = pd.DataFrame(output_rows)
    if "true_class" in df_out.columns:
        true_attacks = df_out[df_out["true_class"] == "attack"]
        if len(true_attacks) > 0:
            detected = (true_attacks["prediction"] == "attack").sum()
            recall   = detected / len(true_attacks) * 100
            fnr      = 100 - recall
            print(f"  Attack Recall : {detected}/{len(true_attacks)} = {recall:.1f}%")
            print(f"  FNR attack    : {fnr:.1f}%  (avant correction : 72.9%)\n")

    pd.DataFrame(output_rows).to_csv(OUT_CSV, index=False)
    print(f"  ✓ Résultats sauvegardés : {OUT_CSV}")
    print(f"  ✓ {len(output_rows)} lignes · {len(df_out.columns)} colonnes")
    print(f"    → colonnes LLM incluses : alert_name, severity, src_ip,")
    print(f"      dst_service, freq_per_min, time_context, protocol")
    return predictor.stats, output_rows


# ══════════════════════════════════════════════════════════════
# ENTRÉE PRINCIPALE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOC Realtime Pipeline — version corrigée")
    parser.add_argument("--csv", type=str, default=None,
                        help="Chemin vers le dataset CSV (optionnel)")
    parser.add_argument("--n", type=int, default=N_LOGS_SIM,
                        help=f"Nombre de logs à simuler (défaut: {N_LOGS_SIM})")
    args = parser.parse_args()

    N_LOGS_SIM = args.n

    source = "csv" if args.csv else "sim"
    stats, results = run_pipeline(source=source, csv_path=args.csv)