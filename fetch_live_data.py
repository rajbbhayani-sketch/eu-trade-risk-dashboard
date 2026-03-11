import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# World Bank indicators
# -----------------------------
INDICATORS = {
    "trade": "NE.TRD.GNFS.ZS",     # Trade % of GDP
    "energy": "EG.USE.PCAP.KG.OE"  # Energy use per capita
}

# Countries (EU + China + India)
COUNTRIES = [
    "DEU","FRA","ITA","ESP","NLD","BEL","POL","AUT",
    "PRT","GRC","FIN","IRL","DNK","SWE","CZE","HUN",
    "SVK","SVN","HRV","BGR","ROU","EST","LVA","LTU",
    "LUX","MLT","CYP","CHN","IND"
]

# -----------------------------
# Fetch indicator data
# -----------------------------
def fetch_indicator(indicator):

    url = f"https://api.worldbank.org/v2/country/{';'.join(COUNTRIES)}/indicator/{indicator}?format=json&per_page=100"

    r = requests.get(url)
    data = r.json()[1]

    records = []

    for d in data:
        if d["value"] is not None:
            records.append({
                "ISO3": d["country"]["id"],
                "Country": d["country"]["value"],
                "value": d["value"]
            })

    return pd.DataFrame(records).drop_duplicates("ISO3")


trade_df = fetch_indicator(INDICATORS["trade"])
energy_df = fetch_indicator(INDICATORS["energy"])

# rename columns
trade_df = trade_df.rename(columns={"value": "Trade_Risk"})
energy_df = energy_df.rename(columns={"value": "Energy_Risk"})

# merge datasets
df = pd.merge(trade_df, energy_df, on=["ISO3","Country"])

# -----------------------------
# Normalize scores (0-100)
# -----------------------------
scaler = MinMaxScaler(feature_range=(0,100))

df[["Trade_Risk","Energy_Risk"]] = scaler.fit_transform(
    df[["Trade_Risk","Energy_Risk"]]
)

# -----------------------------
# Dependency level
# -----------------------------
df["Dependency_Level"] = pd.cut(
    df["Energy_Risk"],
    bins=[0,33,66,100],
    labels=["Low","Medium","High"]
)

# -----------------------------
# Total risk score
# -----------------------------
trade_weight = 0.6
energy_weight = 0.4

df["Total_Risk_Score"] = (
    df["Trade_Risk"]*trade_weight +
    df["Energy_Risk"]*energy_weight
)

# -----------------------------
# Risk category
# -----------------------------
df["Risk_Category"] = pd.cut(
    df["Total_Risk_Score"],
    bins=[0,40,70,100],
    labels=["Low","Medium","High"]
)

# -----------------------------
# Save dataset
# -----------------------------
df = df.sort_values("Total_Risk_Score",ascending=False)

df.to_csv("data/live_country_risk_data.csv",index=False)

print("Saved dataset to data/live_country_risk_data.csv")
print(df.head(10))