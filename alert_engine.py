
import requests
import json
from risk_engine import get_risk_score, get_all_risk_scores, HIGHWAYS

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


def call_ollama(prompt: str) -> str:
    """
    Try Ollama first. If unavailable, use rule-based fallback.
    This makes the app work both locally and on cloud.
    """
    try:
        payload = {
            "model":  MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 200}
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()["response"].strip()

    except Exception:
        # Cloud fallback — rule-based bilingual alert
        return generate_rule_based_alert(prompt)


def generate_rule_based_alert(prompt: str) -> str:
    """
    Rule-based alert generator when Ollama is not available.
    Reads risk level from the prompt and generates appropriate alert.
    """
    if "HIGH" in prompt:
        return (
            "ALERT: High accident risk detected on this highway. "
            "Reduce speed immediately and maintain safe distance.\n"
            "Avoid overtaking. Stay alert for sudden stops ahead.\n"
            "चेतावनी: इस राजमार्ग पर दुर्घटना का उच्च खतरा है। "
            "तुरंत गति कम करें और सुरक्षित दूरी बनाए रखें।\n"
            "ओवरटेकिंग से बचें। आगे अचानक रुकने के लिए सतर्क रहें।"
        )
    elif "MEDIUM" in prompt:
        return (
            "ALERT: Moderate risk on this route. "
            "Drive carefully and follow speed limits.\n"
            "Stay cautious — weather or traffic conditions may change.\n"
            "चेतावनी: इस मार्ग पर मध्यम खतरा है। "
            "सावधानी से चलाएं और गति सीमा का पालन करें।\n"
            "सतर्क रहें — मौसम या यातायात की स्थिति बदल सकती है।"
        )
    else:
        return (
            "ALERT: Road conditions are currently safe. "
            "Maintain standard speed limits and stay alert.\n"
            "Drive responsibly and watch for pedestrians.\n"
            "चेतावनी: सड़क की स्थिति वर्तमान में सुरक्षित है। "
            "मानक गति सीमा बनाए रखें और सतर्क रहें।\n"
            "जिम्मेदारी से गाड़ी चलाएं और पैदल यात्रियों का ध्यान रखें।"
        )


# ── Build the prompt from risk data ─────────────────────────────────────
def build_prompt(risk_data: dict) -> str:
    hw  = risk_data["highway"]
    sc  = risk_data["risk_score"]
    lvl = risk_data["risk_level"]
    spd = risk_data["speed_kmh"]
    den = risk_data["density_veh"]
    inc = "Yes" if risk_data["incident"] else "No"
    w   = risk_data["weather"]

    prompt = f"""
You are RoadSense India, an AI road safety assistant.
Generate a SHORT road safety alert for drivers.

Highway data:
- Highway     : {hw}
- Risk Score  : {sc}/100 ({lvl} RISK)
- Avg Speed   : {spd} km/h
- Traffic     : {den} vehicles/km
- Incident    : {inc}
- Weather     : {w['condition']}
- Rain        : {w['rain_mm']} mm
- Visibility  : {w['visibility_m']} m
- Wind        : {w['wind_kmh']} km/h

Instructions:
1. Write exactly 2 lines in English first.
2. Then write the same 2 lines in Hindi.
3. Be direct and actionable — tell drivers what to DO.
4. Do not use bullet points. No extra explanation.
5. Start English with "ALERT:" and Hindi with "चेतावनी:"

Example format:
ALERT: High accident risk on NH-48. Reduce speed to 40 km/h, fog lights on.
Avoid overtaking in low visibility conditions.
चेतावनी: NH-48 पर दुर्घटना का खतरा अधिक है। गति 40 km/h तक कम करें, फॉग लाइट चालू करें।
कम दृश्यता में ओवरटेकिंग से बचें।
"""
    return prompt.strip()


# ── Generate alert for one highway ───────────────────────────────────────
def generate_alert(highway: str) -> dict:
    print(f"Generating alert for {highway}...")
    risk_data = get_risk_score(highway)
    prompt    = build_prompt(risk_data)
    alert_text = call_ollama(prompt)

    # Split English and Hindi parts
    lines  = alert_text.strip().split("\n")
    lines  = [l.strip() for l in lines if l.strip()]

    english_lines = [l for l in lines if not any(
        ord(c) > 2304 for c in l)]   # no Devanagari chars
    hindi_lines   = [l for l in lines if any(
        ord(c) > 2304 for c in l)]   # has Devanagari chars

    english_alert = " ".join(english_lines) if english_lines else alert_text
    hindi_alert   = " ".join(hindi_lines)   if hindi_lines   else "हिंदी अलर्ट उत्पन्न हो रहा है..."

    return {
        "highway":       highway,
        "risk_score":    risk_data["risk_score"],
        "risk_level":    risk_data["risk_level"],
        "risk_color":    risk_data["risk_color"],
        "english_alert": english_alert,
        "hindi_alert":   hindi_alert,
        "full_alert":    alert_text,
        "weather":       risk_data["weather"],
        "speed_kmh":     risk_data["speed_kmh"],
        "density_veh":   risk_data["density_veh"],
    }


# ── Generate alerts for ALL highways ────────────────────────────────────
def generate_all_alerts() -> list:
    print("Generating alerts for all highways...\n")
    alerts = []
    for hw in HIGHWAYS:
        alert = generate_alert(hw)
        alerts.append(alert)
        print(f"Highway : {alert['highway']}")
        print(f"Risk    : {alert['risk_score']}/100 ({alert['risk_level']})")
        print(f"English : {alert['english_alert'][:120]}...")
        print(f"Hindi   : {alert['hindi_alert'][:80]}...")
        print("-" * 60)
    return alerts


# ── Quick single highway test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Testing Alert Engine ===\n")

    # Make sure Ollama is running first
    test = call_ollama("Say OK if you are working.")
    if "ERROR" in test:
        print(test)
        print("\nFix: Open a NEW terminal and run: ollama serve")
        print("Then run this file again.")
    else:
        print("Ollama connection: OK\n")
        alert = generate_alert("NH-48 Jaipur-Delhi")
        print("\n=== GENERATED ALERT ===")
        print(alert["full_alert"])