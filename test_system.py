print("Testing all modules...\n")

from iot_simulator import simulate_reading
from weather_api import get_weather
from risk_engine import get_risk_score
from hotspot_detector import load_accident_data
from alert_engine import call_ollama

hw = "NH-8 Mumbai-Pune"

# IoT
iot = simulate_reading(hw)
print(f"IoT     : speed={iot['speed_kmh']} density={iot['density_veh']}")

# Weather
w = get_weather(hw)
print(f"Weather : {w['condition']} rain={w['rain_mm']}mm fog={w['fog_flag']}")

# Risk
r = get_risk_score(hw)
print(f"Risk    : {r['risk_score']}/100 ({r['risk_level']})")

# Ollama
llm = call_ollama("Say READY in one word.")
print(f"Ollama  : {llm}")

print("\nAll systems GO!")