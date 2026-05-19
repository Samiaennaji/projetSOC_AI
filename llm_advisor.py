"""
SOC LLM Advisor — Module d'analyse intelligente des alertes
============================================================
Intégration Groq + LLaMA-3.3-70b dans le pipeline XGBoost Wazuh

Corrections apportées vs version initiale :
  - Prompt enrichi avec tous les champs du dataset (alert_name, severity,
    dst_service, time_context, freq_per_min, risk_score, proba_attack)
  - Cache LRU pour éviter les re-appels API sur la même alerte
  - max_tokens porté à 800 (évite le JSON tronqué)
  - Fallback propre si GROQ_API_KEY absent (mode demo)
  - Fonction render_llm_panel() prête à coller dans dashboard Streamlit
  - Validation JSON robuste avec tentative de repair automatique

Usage dans le dashboard :
  from 4_llm_advisor import analyser_alerte, render_llm_panel
  analyse = analyser_alerte(alert_dict)
  render_llm_panel(analyse)
"""

import os
import json
import hashlib
import functools
from dataclasses import dataclass, field
from typing import List, Optional

# ══════════════════════════════════════════════════════════════
# CONFIGURATION GROQ
# ══════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# Initialisation du client Groq avec gestion d'absence de clé
_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    if not GROQ_API_KEY:
        return None          # mode demo : pas de vraie API
    try:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
        return _client
    except ImportError:
        print("  ⚠ groq non installé : pip install groq")
        return None

# ══════════════════════════════════════════════════════════════
# DATACLASS RÉSULTAT
# ══════════════════════════════════════════════════════════════

@dataclass
class LLMAnalysis:
    resume:        str
    niveau_risque: str
    type_attaque:  str
    mitre:         str
    actions:       List[str] = field(default_factory=list)
    iocs:          List[str] = field(default_factory=list)
    from_cache:    bool = False     # True si réponse récupérée depuis le cache

# ── Analyse factice pour le mode démo (sans clé API) ──────────
_DEMO_ANALYSIS = LLMAnalysis(
    resume="[Mode démo] Tentative de brute-force SSH détectée depuis une IP externe. "
           "Fréquence élevée de connexions sur le port 22 en dehors des horaires bureaux.",
    niveau_risque="HIGH",
    type_attaque="SSH Brute Force",
    mitre="T1110.001 – Password Guessing",
    actions=[
        "Bloquer l'IP source sur le pare-feu périmétrique immédiatement",
        "Vérifier les logs d'authentification SSH sur la cible (dst_ip)",
        "Activer l'authentification par clé SSH et désactiver les mots de passe",
    ],
    iocs=["185.x.x.x (IP externe)", "Port dst 22", "Fréquence > 100 req/min"],
)

# ══════════════════════════════════════════════════════════════
# PROMPT SYSTÈME SOC
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are a senior SOC analyst (Tier-2) specialized in:
- Wazuh SIEM alert triage and incident response
- MITRE ATT&CK framework mapping
- Network threat detection (brute-force, scanning, lateral movement, injection)

Your task: analyze the structured security alert below and return a SOC report.

STRICT OUTPUT RULES:
- Return ONLY a valid JSON object, no markdown, no preamble, no explanation.
- All text fields must be in FRENCH.
- Use exactly this structure:

{
  "summary": "2-3 sentences describing what happened and why it is dangerous",
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "attack_type": "precise attack category (e.g. SSH Brute Force, SQL Injection, Port Scan)",
  "mitre": "TXXXX.XXX – Technique Name",
  "recommendations": [
    "Immediate action 1 (blocking, isolation)",
    "Investigation action 2 (log review, forensics)",
    "Hardening action 3 (config, policy)"
  ],
  "iocs": [
    "indicator 1 (IP, port, pattern)",
    "indicator 2"
  ]
}
"""

# ══════════════════════════════════════════════════════════════
# CONSTRUCTION DU PROMPT UTILISATEUR
# Exploite tous les champs riches du dataset Wazuh
# ══════════════════════════════════════════════════════════════

def _build_user_prompt(alert: dict) -> str:
    """
    Construit un prompt structuré à partir des champs du dataset.
    Beaucoup plus efficace qu'un json.dumps() brut : le LLM comprend
    mieux quand les champs sont nommés et contextualisés.
    """
    # Traduction des valeurs encodées en labels lisibles
    time_ctx_map = {
        "business_hours": "heures bureau (9h-18h)",
        "evening":        "soirée (18h-22h)",
        "night":          "nuit profonde (0h-6h)",
        "morning":        "matin tôt (6h-9h)",
        0: "heures bureau", 1: "soirée", 2: "nuit profonde", 3: "matin tôt",
    }
    src_type = "EXTERNE (hors réseau)" if not alert.get("src_is_internal", 1) else "INTERNE"
    dst_type = "INTERNE" if alert.get("dst_is_internal", 1) else "EXTERNE"
    lateral  = "OUI — mouvement latéral possible" if alert.get("is_internal_to_internal") else "NON"
    time_ctx = time_ctx_map.get(alert.get("time_context", ""), str(alert.get("time_context", "?")))
    freq     = alert.get("freq_per_min", alert.get("freq", "?"))
    high_f   = "OUI — régime haute fréquence (scan / brute-force)" if alert.get("is_high_freq") else "NON"

    return f"""
Analyze this Wazuh security alert:

[IDENTIFICATION]
- Nom de l'alerte  : {alert.get("alert_name", alert.get("description", "Inconnu"))}
- Sévérité Wazuh   : {alert.get("severity", "?")} (score {alert.get("severity_score", "?")})
- Timestamp        : {alert.get("timestamp", "?")}
- Contexte horaire : {time_ctx}

[RÉSEAU]
- IP source        : {alert.get("src_ip", "?")} [{src_type}]
- Port source      : {alert.get("src_port", "?")}
- IP destination   : {alert.get("dst_ip", "?")} [{dst_type}]
- Port destination : {alert.get("dst_port", "?")} / service : {alert.get("dst_service", "?")}
- Protocole        : {alert.get("protocol", "?")}
- Même sous-réseau : {"OUI" if alert.get("same_subnet") else "NON"}
- Flux interne→int.: {lateral}

[COMPORTEMENT]
- Fréquence        : {freq} événements/minute
- Haute fréquence  : {high_f}

[CLASSIFICATION XGBOOST]
- Prédiction       : {alert.get("prediction", "attack").upper()}
- Score de risque  : {alert.get("risk_score", "?")}%
- P(attack)        : {alert.get("proba_attack", "?")}
- P(false_positive): {alert.get("proba_fp", "?")}
- P(normal)        : {alert.get("proba_normal", "?")}

Provide the JSON SOC analysis now.
"""

# ══════════════════════════════════════════════════════════════
# CACHE LRU — évite les re-appels API sur la même alerte
# ══════════════════════════════════════════════════════════════

_analysis_cache: dict = {}   # clé = hash des champs clés de l'alerte

def _cache_key(alert: dict) -> str:
    """Hash des champs discriminants pour identifier une alerte unique."""
    key_fields = {
        "alert_name":  alert.get("alert_name", alert.get("description", "")),
        "src_ip":      alert.get("src_ip", ""),
        "dst_port":    alert.get("dst_port", ""),
        "freq":        alert.get("freq_per_min", alert.get("freq", 0)),
        "risk_score":  alert.get("risk_score", 0),
    }
    return hashlib.md5(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:12]

# ══════════════════════════════════════════════════════════════
# VALIDATION ET RÉPARATION JSON
# ══════════════════════════════════════════════════════════════

def _parse_llm_json(raw: str) -> Optional[dict]:
    """
    Tente de parser le JSON du LLM.
    Gère les cas où le modèle ajoute des backticks ou du texte en préambule.
    """
    # Nettoyage des balises markdown
    text = raw.replace("```json", "").replace("```", "").strip()

    # Tentative directe
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extraction du premier bloc JSON valide dans la réponse
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None   # échec de parsing

# ══════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE : analyser_alerte()
# ══════════════════════════════════════════════════════════════

def analyser_alerte(alert: dict) -> Optional[LLMAnalysis]:
    """
    Analyse une alerte SOC via LLM et retourne un LLMAnalysis.

    Paramètres
    ----------
    alert : dict
        Dictionnaire issu de SOCPredictor.predict_batch() ou d'une ligne
        du dashboard Streamlit. Champs attendus (au moins) :
          alert_name / description, src_ip, dst_port, freq_per_min,
          risk_score, proba_attack, proba_fp, proba_normal, prediction

    Retourne
    --------
    LLMAnalysis ou None si erreur fatale
    """
    # ── 1. Vérification du cache ──────────────────────────────
    ck = _cache_key(alert)
    if ck in _analysis_cache:
        cached = _analysis_cache[ck]
        cached.from_cache = True
        return cached

    # ── 2. Mode démo si pas de clé API ───────────────────────
    client = _get_client()
    if client is None:
        print("  ⚠ GROQ_API_KEY non définie — réponse de démo retournée")
        demo = LLMAnalysis(**{k: v for k, v in vars(_DEMO_ANALYSIS).items()})
        _analysis_cache[ck] = demo
        return demo

    # ── 3. Appel API Groq ─────────────────────────────────────
    try:
        user_prompt = _build_user_prompt(alert)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.1,      # faible température = réponses stables et factuelles
            max_tokens=800,       # suffisant pour le JSON complet avec 3 reco + IOCs
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]
        )

        raw_content = response.choices[0].message.content

    except Exception as e:
        print(f"  ✗ Erreur API Groq : {e}")
        return None

    # ── 4. Parsing JSON ───────────────────────────────────────
    data = _parse_llm_json(raw_content)
    if data is None:
        print(f"  ✗ JSON invalide reçu du LLM :\n{raw_content[:300]}")
        return None

    # ── 5. Construction du résultat ───────────────────────────
    analysis = LLMAnalysis(
        resume        = data.get("summary",         "Aucun résumé disponible"),
        niveau_risque = data.get("risk_level",       "MEDIUM"),
        type_attaque  = data.get("attack_type",      "Inconnu"),
        mitre         = data.get("mitre",            "Non identifié"),
        actions       = data.get("recommendations", [])[:5],   # max 5 actions
        iocs          = data.get("iocs",             [])[:5],   # max 5 IOCs
        from_cache    = False,
    )

    # ── 6. Mise en cache ──────────────────────────────────────
    _analysis_cache[ck] = analysis
    return analysis


# ══════════════════════════════════════════════════════════════
# BADGE SEVERITY HTML (inchangé, compatible Streamlit unsafe_html)
# ══════════════════════════════════════════════════════════════

def badge_severity(level: str) -> str:
    colors = {
        "LOW":      "#1D9E75",
        "MEDIUM":   "#EF9F27",
        "HIGH":     "#E24B4A",
        "CRITICAL": "#B00020",
    }
    color = colors.get(level.upper(), "#888")
    return (
        f'<span style="background:{color};color:white;padding:4px 10px;'
        f'border-radius:6px;font-size:12px;font-weight:600">{level.upper()}</span>'
    )


# ══════════════════════════════════════════════════════════════
# RENDER_LLM_PANEL() — composant Streamlit prêt à l'emploi
# ══════════════════════════════════════════════════════════════

def render_llm_panel(analysis: LLMAnalysis) -> None:
    """
    Affiche le panneau d'analyse LLM dans Streamlit.
    À appeler après un st.expander() ou directement dans une colonne.

    Exemple d'utilisation dans 3_dashboard_streamlit.py :
    -------------------------------------------------------
        from 4_llm_advisor import analyser_alerte, render_llm_panel
        ...
        if st.button("🤖 Analyser via LLM", key=f"llm_{i}"):
            with st.spinner("Analyse en cours..."):
                analysis = analyser_alerte(row.to_dict())
            if analysis:
                render_llm_panel(analysis)
    """
    import streamlit as st

    cache_tag = " *(cache)*" if analysis.from_cache else ""

    # ── En-tête ──
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.markdown(f"**Type d'attaque :** {analysis.type_attaque}")
    with col2:
        st.markdown(f"**Niveau de risque :**")
        st.markdown(badge_severity(analysis.niveau_risque), unsafe_allow_html=True)
    with col3:
        st.caption(f"MITRE : `{analysis.mitre}`{cache_tag}")

    st.markdown("---")

    # ── Résumé ──
    st.markdown("**📋 Résumé de l'incident**")
    st.info(analysis.resume)

    # ── Recommandations ──
    if analysis.actions:
        st.markdown("**🛡️ Recommandations SOC**")
        priorities = ["🔴 Immédiat", "🟡 Investigation", "🔵 Durcissement"]
        for i, action in enumerate(analysis.actions):
            prefix = priorities[i] if i < len(priorities) else f"**{i+1}.**"
            st.markdown(f"{prefix} — {action}")

    # ── IOCs ──
    if analysis.iocs:
        st.markdown("**🔍 Indicateurs de compromission (IOC)**")
        ioc_cols = st.columns(min(len(analysis.iocs), 3))
        for i, ioc in enumerate(analysis.iocs):
            with ioc_cols[i % 3]:
                st.code(ioc, language=None)


# ══════════════════════════════════════════════════════════════
# TEST STANDALONE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Alerte exemple tirée du dataset soc_detection_dataset_v3
    test_alert = {
        "timestamp":      "2026-05-02 18:43:48",
        "alert_name":     "SSH Brute Force",
        "severity":       "HIGH",
        "severity_score": 3,
        "protocol":       "TCP",
        "src_ip":         "192.168.126.133",
        "src_port":       40464,
        "dst_ip":         "192.168.126.139",
        "dst_port":       22,
        "dst_service":    "ssh",
        "src_is_internal":         1,
        "dst_is_internal":         1,
        "is_internal_to_internal": 1,
        "same_subnet":             1,
        "freq_per_min":            159,
        "is_high_freq":            1,
        "time_context":   "business_hours",
        "prediction":     "attack",
        "risk_score":     94,
        "proba_attack":   0.961,
        "proba_fp":       0.028,
        "proba_normal":   0.011,
    }

    print("=" * 60)
    print("  TEST analyser_alerte()")
    print("=" * 60)

    result = analyser_alerte(test_alert)

    if result:
        print(f"\n  Type       : {result.type_attaque}")
        print(f"  Risque     : {result.niveau_risque}")
        print(f"  MITRE      : {result.mitre}")
        print(f"  Résumé     : {result.resume[:100]}...")
        print(f"  Actions    : {len(result.actions)}")
        print(f"  IOCs       : {len(result.iocs)}")
        print(f"  Cache      : {result.from_cache}")
        print("\n  Actions :")
        for i, a in enumerate(result.actions, 1):
            print(f"    {i}. {a}")
        print("\n  IOCs :")
        for ioc in result.iocs:
            print(f"    - {ioc}")
    else:
        print("  ✗ Aucun résultat — vérifiez GROQ_API_KEY")

    print("\n✓ Test terminé.")