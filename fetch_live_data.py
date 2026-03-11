import requests
import pandas as pd

# --------------------------------
# Countries: EU + India + China
# --------------------------------
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

TRADE_INDICATOR = "NE.TRD.GNFS.ZS"      # Trade (% of GDP)
ENERGY_INDICATOR = "EG.IMP.CONS.ZS"     # Energy imports, net (% of energy use)

# --------------------------------
# Fetch latest available value
# --------------------------------
def fetch_latest_value(country_code, indicator_code):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=100"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        return None

    for row in data[1]:
        if row.get("value") is not None:
            return float(row["value"])

    return None

# --------------------------------
# Min-max normalization to 0-100
# --------------------------------
def min_max_scale(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([50.0] * len(series), index=series.index)

    return ((series - min_val) / (max_val - min_val) * 100).round(1)

# --------------------------------
# Category logic
# --------------------------------
def classify_risk(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

def dependency_level(energy_score):
    if energy_score >= 70:
        return "High"
    elif energy_score >= 40:
        return "Medium"
    else:
        return "Low"

# --------------------------------
# Collect raw data
# --------------------------------
rows = []

for iso3, country_name in EU_PLUS.items():
    print(f"Fetching {country_name}...")

    try:
        trade_value = fetch_latest_value(iso3, TRADE_INDICATOR)
        energy_value = fetch_latest_value(iso3, ENERGY_INDICATOR)

        if trade_value is None or energy_value is None:
            print(f"Skipping {country_name} due to missing data")
            continue

        rows.append({
            "ISO3": iso3,
            "Country": country_name,
            "Trade_Raw": round(trade_value, 2),
            "Energy_Raw": round(energy_value, 2)
        })

    except Exception as e:
        print(f"Error for {country_name}: {e}")

df = pd.DataFrame(rows)

# Stop if fetch failed badly
if df.empty:
    raise ValueError("No data was fetched from the API.")

# --------------------------------
# Normalize to 0-100
# --------------------------------
df["Trade_Risk"] = min_max_scale(df["Trade_Raw"])
df["Energy_Risk"] = min_max_scale(df["Energy_Raw"])

# --------------------------------
# Total risk score
# --------------------------------
trade_weight = 0.4
energy_weight = 0.6

df["Total_Risk_Score"] = (
    trade_weight * df["Trade_Risk"] +
    energy_weight * df["Energy_Risk"]
).round(1)

# Safety clamp
df["Total_Risk_Score"] = df["Total_Risk_Score"].clip(lower=0, upper=100)

# --------------------------------
# Labels
# --------------------------------
df["Dependency_Level"] = df["Energy_Risk"].apply(dependency_level)
df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)

# --------------------------------
# Final dataset
# --------------------------------
df = df[[
    "ISO3",
    "Country",
    "Trade_Risk",
    "Energy_Risk",
    "Dependency_Level",
    "Total_Risk_Score",
    "Risk_Category"
]].sort_values("Total_Risk_Score", ascending=False)

# --------------------------------
# Save file
# --------------------------------
output_path = "data/live_country_risk_data.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved normalized dataset to {output_path}")
print(f"Rows saved: {len(df)}")
print(df.head(10))