import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------
# Page setup
# ---------------------------------
st.set_page_config(
    page_title="EU Trade & Energy Risk Dashboard",
    layout="wide"
)

# ---------------------------------
# Load data
# ---------------------------------
df = pd.read_csv("data/live_country_risk_data.csv")

if df.empty:
    st.error("Dataset is empty. Please run fetch_live_data.py first.")
    st.stop()

# ---------------------------------
# Sidebar controls
# ---------------------------------
st.sidebar.header("Scenario Controls")

trade_weight = st.sidebar.slider(
    "Trade Risk Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.1
)

energy_weight = st.sidebar.slider(
    "Energy Risk Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.1
)

forecast_growth = st.sidebar.slider(
    "Future Risk Growth %",
    min_value=-20,
    max_value=20,
    value=5,
    step=1
)

country_options = sorted(df["Country"].dropna().tolist())
selected_explain_country = st.sidebar.selectbox(
    "Risk Driver Focus",
    country_options,
    index=country_options.index("Germany") if "Germany" in country_options else 0
)

if trade_weight + energy_weight == 0:
    trade_weight = 0.4
    energy_weight = 0.6

# ---------------------------------
# Risk model
# ---------------------------------
df["Total_Risk_Score"] = (
    trade_weight * df["Trade_Risk"] +
    energy_weight * df["Energy_Risk"]
).round(1)

def classify_risk(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)

# ---------------------------------
# Future projection
# ---------------------------------
df["Projected_Risk_Score"] = (
    df["Total_Risk_Score"] * (1 + forecast_growth / 100)
).round(1)

df["Projected_Risk_Score"] = df["Projected_Risk_Score"].clip(lower=0, upper=100)

def risk_trend(current, projected):
    if projected > current:
        return "Rising"
    elif projected < current:
        return "Falling"
    return "Stable"

df["Risk_Trend"] = df.apply(
    lambda row: risk_trend(row["Total_Risk_Score"], row["Projected_Risk_Score"]),
    axis=1
)

# ---------------------------------
# Explainability
# ---------------------------------
df["Trade_Contribution"] = (trade_weight * df["Trade_Risk"]).round(1)
df["Energy_Contribution"] = (energy_weight * df["Energy_Risk"]).round(1)

def main_driver(row):
    if row["Energy_Contribution"] > row["Trade_Contribution"]:
        return "Energy Dependency"
    elif row["Trade_Contribution"] > row["Energy_Contribution"]:
        return "Trade Exposure"
    return "Balanced"

df["Main_Risk_Driver"] = df.apply(main_driver, axis=1)

# ---------------------------------
# AI model
# ---------------------------------
dependency_map = {"Low": 0, "Medium": 1, "High": 2}
df["Dependency_Level_Encoded"] = df["Dependency_Level"].map(dependency_map).fillna(1)

features = df[["Trade_Risk", "Energy_Risk", "Dependency_Level_Encoded"]]
target = df["Risk_Category"]

df["Predicted_Risk_Category"] = df["Risk_Category"]
model_accuracy = None

if target.nunique() > 1 and len(df) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    df["Predicted_Risk_Category"] = model.predict(features)

    feature_importance = pd.DataFrame({
        "Feature": features.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
else:
    feature_importance = pd.DataFrame({
        "Feature": ["Trade_Risk", "Energy_Risk", "Dependency_Level_Encoded"],
        "Importance": [0.0, 0.0, 0.0]
    })

# ---------------------------------
# Summary values
# ---------------------------------
df_sorted = df.sort_values("Total_Risk_Score", ascending=False).reset_index(drop=True)

top_country = df_sorted.loc[0, "Country"]
top_score = float(df_sorted.loc[0, "Total_Risk_Score"])
top_driver = df_sorted.loc[0, "Main_Risk_Driver"]

avg_risk = round(df["Total_Risk_Score"].mean(), 1)
avg_projected_risk = round(df["Projected_Risk_Score"].mean(), 1)
high_risk_count = int((df["Risk_Category"] == "High").sum())
total_countries = len(df)

avg_trade = round(df["Trade_Risk"].mean(), 1)
avg_energy = round(df["Energy_Risk"].mean(), 1)
portfolio_driver = "Energy Dependency" if avg_energy > avg_trade else "Trade Exposure"

top5 = df_sorted.head(5).copy()
focus_row = df[df["Country"] == selected_explain_country].iloc[0]

# ---------------------------------
# Header
# ---------------------------------
st.title("EU Trade & Energy Risk Intelligence Dashboard")
st.markdown("### Executive monitoring of trade exposure, energy dependency, and projected country risk")

st.info(
    "This dashboard combines live country data, AI classification, scenario analysis, "
    "and forward-looking risk projection to support strategic decision-making."
)

# ---------------------------------
# Executive KPI row
# ---------------------------------
st.subheader("Executive Summary")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Average Risk Score", avg_risk)
k2.metric("High Risk Countries", high_risk_count)
k3.metric("Projected Average Risk", avg_projected_risk)
k4.metric("Top Risk Country", top_country)

# ---------------------------------
# Executive alert
# ---------------------------------
st.subheader("Critical Alert")

if high_risk_count > 0:
    st.warning(
        f"Alert: **{top_country}** currently has the highest total risk score (**{top_score:.1f}**). "
        f"The main risk driver is **{top_driver}**."
    )
else:
    st.success("No countries are currently classified as High Risk under the selected scenario.")

st.divider()

# ---------------------------------
# Forecast section
# ---------------------------------
st.subheader("Risk Outlook")

o1, o2, o3 = st.columns(3)
o1.metric("Current Scenario", f"Trade {trade_weight:.1f} / Energy {energy_weight:.1f}")
o2.metric("Forecast Growth Assumption", f"{forecast_growth}%")
o3.metric("Portfolio Risk Driver", portfolio_driver)

forecast_chart_df = df_sorted.head(10)[
    ["Country", "Total_Risk_Score", "Projected_Risk_Score"]
].copy()

fig_forecast = px.bar(
    forecast_chart_df,
    x="Country",
    y=["Total_Risk_Score", "Projected_Risk_Score"],
    barmode="group",
    title="Current vs Projected Risk Score"
)
st.plotly_chart(fig_forecast, width="stretch")

st.divider()

# ---------------------------------
# Risk driver analysis
# ---------------------------------
st.subheader("Risk Driver Analysis")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Selected Country", selected_explain_country)
r2.metric("Current Risk", f"{focus_row['Total_Risk_Score']:.1f}")
r3.metric("Projected Risk", f"{focus_row['Projected_Risk_Score']:.1f}")
r4.metric("Primary Driver", focus_row["Main_Risk_Driver"])

st.write(
    f"**{selected_explain_country}** is currently classified as **{focus_row['Risk_Category']}** risk. "
    f"Its risk structure is primarily driven by **{focus_row['Main_Risk_Driver']}**."
)

contrib_df = pd.DataFrame({
    "Component": ["Trade Contribution", "Energy Contribution"],
    "Value": [focus_row["Trade_Contribution"], focus_row["Energy_Contribution"]]
})

fig_driver = px.bar(
    contrib_df,
    x="Component",
    y="Value",
    title=f"Risk Contribution Breakdown: {selected_explain_country}"
)
st.plotly_chart(fig_driver, width="stretch")

st.divider()

# ---------------------------------
# Country benchmark table
# ---------------------------------
st.subheader("Top Risk Countries")

benchmark_df = top5[[
    "Country",
    "Total_Risk_Score",
    "Projected_Risk_Score",
    "Risk_Category",
    "Risk_Trend",
    "Main_Risk_Driver"
]].copy()

st.dataframe(
    benchmark_df.style.format({
        "Total_Risk_Score": "{:.1f}",
        "Projected_Risk_Score": "{:.1f}"
    }),
    width="stretch"
)

st.divider()

# ---------------------------------
# AI monitoring section
# ---------------------------------
st.subheader("AI Monitoring")

a1, a2 = st.columns(2)

with a1:
    st.metric("Model Accuracy", f"{model_accuracy:.2f}" if model_accuracy is not None else "N/A")
    st.dataframe(
        df_sorted[["Country", "Risk_Category", "Predicted_Risk_Category"]].head(10),
        width="stretch"
    )

with a2:
    fig_importance = px.bar(
        feature_importance,
        x="Feature",
        y="Importance",
        title="Model Feature Importance"
    )
    st.plotly_chart(fig_importance, width="stretch")

st.divider()

# ---------------------------------
# Portfolio comparison chart
# ---------------------------------
st.subheader("Portfolio Risk Comparison")

fig_compare = px.bar(
    df_sorted,
    x="Country",
    y=["Trade_Risk", "Energy_Risk"],
    barmode="group",
    title="Trade Risk vs Energy Risk by Country"
)
st.plotly_chart(fig_compare, width="stretch")

st.divider()

# ---------------------------------
# Strategic recommendations
# ---------------------------------
st.subheader("Management Recommendations")

if portfolio_driver == "Energy Dependency":
    st.write("**Primary Action:** Prioritize energy diversification and reduce structural import dependency.")
else:
    st.write("**Primary Action:** Strengthen trade resilience and reduce supply-chain concentration risk.")

if high_risk_count > 0:
    st.write(f"**Immediate Response:** Intensify monitoring and contingency planning for **{top_country}**.")
else:
    st.write("**Immediate Response:** Maintain routine monitoring under the current scenario.")

st.write("**Medium-Term Priorities:**")
st.write("- Improve cross-border resilience planning")
st.write("- Expand alternative sourcing strategies")
st.write("- Increase scenario-based stress testing")
st.write("- Review exposure to geopolitical disruption")

st.divider()

# ---------------------------------
# Global map
# ---------------------------------
st.subheader("Geographic Risk View")

fig_map = px.choropleth(
    df_sorted,
    locations="ISO3",
    color="Total_Risk_Score",
    hover_name="Country",
    color_continuous_scale="Reds",
    title="Global Trade & Energy Risk Map"
)
st.plotly_chart(fig_map, width="stretch")

st.divider()

# ---------------------------------
# Detailed dataset
# ---------------------------------
st.subheader("Detailed Dataset")

detailed_columns = [
    "ISO3",
    "Country",
    "Trade_Risk",
    "Energy_Risk",
    "Dependency_Level",
    "Total_Risk_Score",
    "Risk_Category",
    "Predicted_Risk_Category",
    "Projected_Risk_Score",
    "Risk_Trend",
    "Trade_Contribution",
    "Energy_Contribution",
    "Main_Risk_Driver"
]

st.dataframe(
    df_sorted[detailed_columns].style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}",
        "Projected_Risk_Score": "{:.1f}",
        "Trade_Contribution": "{:.1f}",
        "Energy_Contribution": "{:.1f}"
    }),
    width="stretch"
)