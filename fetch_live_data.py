import requests
import pandas as pd
import hashlib
from datetime import datetime

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

# World Bank indicators
TRADE_INDICATOR = "NE.TRD.GNFS.ZS"      # Trade (% of GDP)
ENERGY_INDICATOR = "EG.IMP.CONS.ZS"     # Energy imports, net (% of energy use)

# --------------------------------
# Fetch latest available value
# --------------------------------
def fetch_latest_value(country_code: str, indicator_code: str) -> float | None:
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=100"
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        return None

    for row in data[1]:
        if row.get("value") is not None:
            return float(row["value"])

    return None

# --------------------------------
# Normalize to 0-100
# --------------------------------
def min_max_scale(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([50.0] * len(series), index=series.index)

    return ((series - min_val) / (max_val - min_val) * 100).round(1)

# --------------------------------
# Labels
# --------------------------------
def classify_risk(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"

def dependency_level(energy_score: float) -> str:
    if energy_score >= 70:
        return "High"
    elif energy_score >= 40:
        return "Medium"
    return "Low"

# --------------------------------
# Deterministic pseudo-random helper
# --------------------------------
def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

# --------------------------------
# Create synthetic 12-month history
# --------------------------------
def build_history(base_df: pd.DataFrame) -> pd.DataFrame:
    month_labels = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="M")
    history_rows = []

    for _, row in base_df.iterrows():
        country = row["Country"]
        iso3 = row["ISO3"]
        current_score = float(row["Total_Risk_Score"])

        seed = stable_seed(country)
        trend_factor = ((seed % 9) - 4) * 0.8
        wave_factor = ((seed % 5) + 1) * 0.4

        start_score = max(0.0, min(100.0, current_score - trend_factor * 5))

        scores = []
        for i in range(12):
            progress = i / 11 if 11 > 0 else 1
            base = start_score + (current_score - start_score) * progress
            wave = ((i % 4) - 1.5) * wave_factor
            score = max(0.0, min(100.0, round(base + wave, 1)))
            scores.append(score)

        scores[-1] = round(current_score, 1)

        for dt, score in zip(month_labels, scores):
            history_rows.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "ISO3": iso3,
                "Country": country,
                "Risk_Score": score
            })

    return pd.DataFrame(history_rows)

# --------------------------------
# Collect raw data
# --------------------------------
rows = []

print("Fetching live data from World Bank API...\n")

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

if df.empty:
    raise ValueError("No data was fetched from the API.")

# --------------------------------
# Normalize and calculate risk
# --------------------------------
df["Trade_Risk"] = min_max_scale(df["Trade_Raw"])
df["Energy_Risk"] = min_max_scale(df["Energy_Raw"])

trade_weight = 0.4
energy_weight = 0.6

df["Total_Risk_Score"] = (
    trade_weight * df["Trade_Risk"] +
    energy_weight * df["Energy_Risk"]
).round(1)

df["Total_Risk_Score"] = df["Total_Risk_Score"].clip(lower=0, upper=100)

df["Dependency_Level"] = df["Energy_Risk"].apply(dependency_level)
df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)
df["Last_Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df["Data_Source"] = "World Bank API"

df = df[[
    "ISO3",
    "Country",
    "Trade_Risk",
    "Energy_Risk",
    "Dependency_Level",
    "Total_Risk_Score",
    "Risk_Category",
    "Last_Updated",
    "Data_Source"
]].sort_values("Total_Risk_Score", ascending=False)

# --------------------------------
# Save current dataset
# --------------------------------
live_path = "data/live_country_risk_data.csv"
df.to_csv(live_path, index=False)

# --------------------------------
# Save 12-month history dataset
# --------------------------------
history_df = build_history(df)
history_path = "data/risk_history.csv"
history_df.to_csv(history_path, index=False)

print(f"\nSaved live dataset to {live_path}")
print(f"Saved history dataset to {history_path}")
print(f"Rows saved: {len(df)}")
print(df.head(10))