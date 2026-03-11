import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="EU Trade Risk AI", layout="wide")

# -------------------------------
# Load LIVE data
# -------------------------------
df = pd.read_csv("data/live_country_risk_data.csv")

# -------------------------------
# Sidebar scenario analysis
# -------------------------------
st.sidebar.header("Scenario Analysis")

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

if trade_weight + energy_weight == 0:
    trade_weight = 0.4
    energy_weight = 0.6

# -------------------------------
# Dynamic risk model
# -------------------------------
df["Total_Risk_Score"] = (
    (trade_weight * df["Trade_Risk"]) +
    (energy_weight * df["Energy_Risk"])
).round(1)

def classify_risk(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)

# -------------------------------
# Future projection
# -------------------------------
df["Projected_Risk_Score"] = (
    df["Total_Risk_Score"] * (1 + forecast_growth / 100)
).round(1)

df["Projected_Risk_Score"] = df["Projected_Risk_Score"].clip(lower=0, upper=100)

def risk_trend(current, projected):
    if projected > current:
        return "Rising"
    elif projected < current:
        return "Falling"
    else:
        return "Stable"

df["Risk_Trend"] = df.apply(
    lambda row: risk_trend(row["Total_Risk_Score"], row["Projected_Risk_Score"]),
    axis=1
)

# -------------------------------
# Encode dependency level
# -------------------------------
dependency_map = {"Low": 0, "Medium": 1, "High": 2}
df["Dependency_Level_Encoded"] = df["Dependency_Level"].map(dependency_map).fillna(1)

# -------------------------------
# AI model
# -------------------------------
features = df[["Trade_Risk", "Energy_Risk", "Dependency_Level_Encoded"]]
target = df["Risk_Category"]

model_accuracy = None
df["Predicted_Risk_Category"] = df["Risk_Category"]

if target.nunique() > 1 and len(df) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
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

# -------------------------------
# Sort
# -------------------------------
df_sorted = df.sort_values("Total_Risk_Score", ascending=False)

# -------------------------------
# KPI values
# -------------------------------
average_risk = round(df["Total_Risk_Score"].mean(), 1)
average_projected_risk = round(df["Projected_Risk_Score"].mean(), 1)
high_risk_count = (df["Risk_Category"] == "High").sum()
total_countries = len(df)
top_country = df_sorted.iloc[0]["Country"]
top_score = df_sorted.iloc[0]["Total_Risk_Score"]

avg_trade = round(df["Trade_Risk"].mean(), 1)
avg_energy = round(df["Energy_Risk"].mean(), 1)

if avg_energy > avg_trade:
    main_driver = "Energy Dependency"
else:
    main_driver = "Trade Disruption"

top3 = df_sorted.head(3)

# -------------------------------
# Title
# -------------------------------
st.title("EU Trade & Energy Risk Business Intelligence Dashboard")
st.markdown(
    "### Business Intelligence platform for monitoring EU trade disruption and energy dependency risk"
)

st.info(
    "This dashboard uses live country data to identify high-risk countries using trade risk, "
    "energy risk, dynamic risk scoring, AI-based prediction, and future risk projection."
)

# -------------------------------
# Executive Summary
# -------------------------------
st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Risk Score", average_risk)
col2.metric("High Risk Countries", high_risk_count)
col3.metric("Total Countries", total_countries)
col4.metric("Top Risk Country", top_country)

st.divider()

# -------------------------------
# AI Risk Prediction
# -------------------------------
st.subheader("AI Risk Prediction")

a1, a2 = st.columns(2)

with a1:
    if model_accuracy is not None:
        st.metric("Model Accuracy", f"{model_accuracy:.2f}")
    else:
        st.metric("Model Accuracy", "N/A")

with a2:
    st.metric("Top Predicted Risk", df_sorted.iloc[0]["Predicted_Risk_Category"])

st.warning(
    "This prediction model is for portfolio demonstration. Results should be interpreted cautiously."
)

st.subheader("Predicted Risk Categories")
st.dataframe(
    df_sorted[["Country", "Risk_Category", "Predicted_Risk_Category"]]
)

st.subheader("Feature Importance")
fig_importance = px.bar(
    feature_importance,
    x="Feature",
    y="Importance",
    title="AI Model Feature Importance"
)
st.plotly_chart(fig_importance, use_container_width=True)

st.divider()

# -------------------------------
# Future Risk Projection
# -------------------------------
st.subheader("Future Risk Projection")

f1, f2, f3 = st.columns(3)
f1.metric("Average Current Risk", average_risk)
f2.metric("Average Projected Risk", average_projected_risk)
f3.metric("Scenario Growth %", f"{forecast_growth}%")

forecast_df = df_sorted[["Country", "Total_Risk_Score", "Projected_Risk_Score", "Risk_Trend"]].copy()
st.dataframe(
    forecast_df.style.format({
        "Total_Risk_Score": "{:.1f}",
        "Projected_Risk_Score": "{:.1f}"
    })
)

fig_forecast = px.bar(
    forecast_df.head(10),
    x="Country",
    y=["Total_Risk_Score", "Projected_Risk_Score"],
    barmode="group",
    title="Current vs Projected Risk Score (Top 10 Countries)"
)
st.plotly_chart(fig_forecast, use_container_width=True)

st.divider()

# -------------------------------
# Executive Insights
# -------------------------------
st.subheader("Executive Insights")

e1, e2, e3 = st.columns(3)

with e1:
    st.metric("Top Risk Score", top_score)

with e2:
    st.metric("Primary Risk Driver", main_driver)

with e3:
    st.metric("Overall Risk Level", "Moderate" if average_risk < 70 else "High")

st.write(
    f"**Key Finding:** {top_country} currently has the highest combined trade and energy risk "
    f"in the live dataset, with a total risk score of **{top_score}**."
)

st.divider()

# -------------------------------
# Risk Alert System
# -------------------------------
st.subheader("Risk Alert System")

if high_risk_count > 0:
    st.warning(
        f"⚠ Alert: {high_risk_count} country/countries are currently classified as High Risk. "
        f"Immediate monitoring is recommended for {top_country}."
    )
else:
    st.success("No countries are currently classified as High Risk.")

if avg_energy > avg_trade:
    st.error("Energy dependency is currently the dominant structural risk across the dataset.")
else:
    st.error("Trade disruption is currently the dominant structural risk across the dataset.")

st.divider()

# -------------------------------
# Scenario Summary
# -------------------------------
st.subheader("Scenario Summary")

st.write(
    f"Current scenario uses **Trade Risk Weight = {trade_weight:.1f}** and "
    f"**Energy Risk Weight = {energy_weight:.1f}**."
)

st.write(
    f"Future projection scenario applies **{forecast_growth}%** growth to the current total risk score."
)

st.divider()

# -------------------------------
# Business Problem
# -------------------------------
st.subheader("Business Problem")

st.write(
    "Countries face exposure to trade disruption and energy import dependency. "
    "Decision-makers need a simple way to identify vulnerable countries, compare risk levels, "
    "prioritize strategic responses, and anticipate future risk developments."
)

st.divider()

# -------------------------------
# Risk Drivers Analysis
# -------------------------------
st.subheader("Risk Drivers Analysis")

d1, d2 = st.columns(2)

with d1:
    st.metric("Average Trade Risk", avg_trade)

with d2:
    st.metric("Average Energy Risk", avg_energy)

if avg_energy > avg_trade:
    st.write("**Insight:** Energy dependency is currently the main risk driver across countries.")
else:
    st.write("**Insight:** Trade disruption is currently the main risk driver across countries.")

st.divider()

# -------------------------------
# Country Risk Comparison
# -------------------------------
st.subheader("Country Risk Comparison")

fig_compare = px.bar(
    df_sorted,
    x="Country",
    y=["Trade_Risk", "Energy_Risk"],
    barmode="group",
    title="Trade Risk vs Energy Risk by Country"
)

st.plotly_chart(fig_compare, use_container_width=True)

st.divider()

# -------------------------------
# Key Insights
# -------------------------------
st.subheader("Key Insights")

st.write(f"- **{top3.iloc[0]['Country']}** has the highest combined risk exposure in the live dataset.")
st.write(f"- The **average risk score is {average_risk}**, indicating the overall regional risk level.")
st.write(f"- **{main_driver}** appears to be the dominant driver of total risk.")
st.write(f"- **{high_risk_count} country/countries** are currently classified as High Risk.")
st.write(f"- The projection scenario currently suggests an average future risk score of **{average_projected_risk}**.")

st.divider()

# -------------------------------
# Top 3 Risk Countries
# -------------------------------
st.subheader("Top 3 Risk Countries")

top3_display = top3[["Country", "Total_Risk_Score", "Risk_Category", "Projected_Risk_Score", "Risk_Trend"]]
st.dataframe(top3_display.style.format({
    "Total_Risk_Score": "{:.1f}",
    "Projected_Risk_Score": "{:.1f}"
}))

st.divider()

# -------------------------------
# Strategic Recommendations
# -------------------------------
st.subheader("Strategic Recommendations")

if main_driver == "Energy Dependency":
    st.write("### Primary Strategic Action")
    st.write("- Prioritize energy diversification and reduce import dependency.")
else:
    st.write("### Primary Strategic Action")
    st.write("- Strengthen trade partnerships and reduce supply chain disruption exposure.")

if high_risk_count > 0:
    st.write("### Immediate Executive Recommendation")
    st.write(f"- Conduct targeted monitoring and contingency planning for **{top_country}**.")
    st.write("- Review strategic reserves, supply agreements, and emergency trade alternatives.")
else:
    st.write("### Immediate Executive Recommendation")
    st.write("- Maintain current resilience strategy and continue routine monitoring.")

st.write("### Medium-Term Recommendations")
st.write("- Increase renewable energy investment")
st.write("- Strengthen cross-border trade resilience")
st.write("- Improve geopolitical scenario planning")

st.divider()

# -------------------------------
# Dataset
# -------------------------------
st.subheader("Live Country Risk Overview")
st.dataframe(
    df_sorted[[
        "ISO3",
        "Country",
        "Trade_Risk",
        "Energy_Risk",
        "Dependency_Level",
        "Total_Risk_Score",
        "Risk_Category",
        "Predicted_Risk_Category",
        "Projected_Risk_Score",
        "Risk_Trend"
    ]].style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}",
        "Projected_Risk_Score": "{:.1f}"
    })
)

st.divider()

# -------------------------------
# Risk Distribution
# -------------------------------
st.subheader("Risk Category Distribution")

risk_counts = df["Risk_Category"].value_counts()

fig_dist = px.bar(
    x=risk_counts.index,
    y=risk_counts.values,
    labels={"x": "Risk Category", "y": "Number of Countries"},
    title="Distribution of Risk Levels"
)

st.plotly_chart(fig_dist, use_container_width=True)