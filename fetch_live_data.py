import requests
import pandas as pd
from typing import Optional

EU_PLUS = {
    "AUT": "Austria",
    "BEL": "Belgium",
    "BGR": "Bulgaria",
    "HRV": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DNK": "Denmark",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "DEU": "Germany",
    "GRC": "Greece",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LVA": "Latvia",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MLT": "Malta",
    "NLD": "Netherlands",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "ESP": "Spain",
    "SWE": "Sweden",
    "IND": "India",
    "CHN": "China"
}

TRADE_INDICATOR = "NE.TRD.GNFS.ZS"
ENERGY_INDICATOR = "EG.IMP.CONS.ZS"

def fetch_latest_value(country_code: str, indicator_code: str) -> Optional[float]:
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=100"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        return None

    for row in data[1]:
        if row.get("value") is not None:
            return float(row["value"])

    return None

def dependency_level(energy_imports_pct: float) -> str:
    if energy_imports_pct >= 70:
        return "High"
    elif energy_imports_pct >= 40:
        return "Medium"
    else:
        return "Low"

def classify_risk(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

rows = []

for iso3, country_name in EU_PLUS.items():
    trade_value = fetch_latest_value(iso3, TRADE_INDICATOR)
    energy_value = fetch_latest_value(iso3, ENERGY_INDICATOR)

    if trade_value is None or energy_value is None:
        print(f"Skipping {country_name} ({iso3}) due to missing data")
        continue

    total_risk_score = round(0.4 * trade_value + 0.6 * energy_value, 1)
    dep_level = dependency_level(energy_value)
    risk_cat = classify_risk(total_risk_score)

    rows.append({
        "ISO3": iso3,
        "Country": country_name,
        "Trade_Risk": round(trade_value, 1),
        "Energy_Risk": round(energy_value, 1),
        "Dependency_Level": dep_level,
        "Total_Risk_Score": total_risk_score,
        "Risk_Category": risk_cat
    })

df = pd.DataFrame(rows).sort_values("Total_Risk_Score", ascending=False)

output_path = "data/live_country_risk_data.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} rows to {output_path}")
print(df.head(10))