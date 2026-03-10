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
# Load data
# -------------------------------
df = pd.read_csv("data/eu_trade_energy_risk.csv")

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
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)

# -------------------------------
# Encode dependency level
# -------------------------------
dependency_map = {"Low": 0, "Medium": 1, "High": 2}
df["Dependency_Level_Encoded"] = df["Dependency_Level"].map(dependency_map)

# -------------------------------
# Machine Learning Model
# -------------------------------
features = df[["Trade_Risk", "Energy_Risk", "Dependency_Level_Encoded"]]
target = df["Risk_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=42
)

model = RandomForestClassifier(random_state=42, n_estimators=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

df["Predicted_Risk_Category"] = model.predict(features)

feature_importance = pd.DataFrame({
    "Feature": features.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

# -------------------------------
# Sort
# -------------------------------
df_sorted = df.sort_values("Total_Risk_Score", ascending=False)

# -------------------------------
# KPI values
# -------------------------------
average_risk = round(df["Total_Risk_Score"].mean(), 1)
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
    "This dashboard helps identify high-risk European countries using trade risk, "
    "energy risk, a dynamic calculated total risk score, and AI-based risk prediction."
)

# -------------------------------
# Executive Summary
# -------------------------------
st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average EU Risk", average_risk)
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
    st.metric("Model Accuracy", f"{model_accuracy:.2f}")

with a2:
    st.metric("Top Predicted Risk", df_sorted.iloc[0]["Predicted_Risk_Category"])

st.warning(
    "This prediction model is a portfolio demonstration only. The dataset is very small, "
    "so results should be treated as illustrative rather than production-grade."
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
# Executive Insights
# -------------------------------
st.subheader("Executive Insights")

e1, e2, e3 = st.columns(3)

with e1:
    st.metric("Top Risk Score", top_score)

with e2:
    st.metric("Primary Risk Driver", main_driver)

with e3:
    st.metric("EU Risk Level", "Moderate" if average_risk < 70 else "High")

st.write(
    f"**Key Finding:** {top_country} currently represents the highest combined trade "
    f"and energy risk in the dataset, with a total risk score of **{top_score}**."
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
    st.error("Energy dependency is currently the dominant structural risk across the EU dataset.")
else:
    st.error("Trade disruption is currently the dominant structural risk across the EU dataset.")

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
    "Adjusting these weights changes the total risk score and helps simulate "
    "different policy or strategic priorities."
)

st.divider()

# -------------------------------
# Business Problem
# -------------------------------
st.subheader("Business Problem")

st.write(
    "European economies face rising exposure to trade disruption and energy dependency. "
    "Decision-makers need a simple way to identify vulnerable countries, compare risk levels, "
    "and prioritize strategic responses."
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
    st.write("**Insight:** Energy dependency is currently the main risk driver across EU countries.")
else:
    st.write("**Insight:** Trade disruption is currently the main risk driver across EU countries.")

st.divider()

# -------------------------------
# Country Risk Comparison
# -------------------------------
st.subheader("Country Risk Comparison")

fig_compare = px.bar(
    df,
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

st.write(f"- **{top3.iloc[0]['Country']}** has the highest combined risk exposure in the dataset.")
st.write(f"- The **average EU risk score is {average_risk}**, indicating a moderate overall risk environment.")
st.write(f"- **{main_driver}** appears to be the dominant driver of total risk.")
st.write(f"- **{high_risk_count} country/countries** are currently classified as High Risk.")

st.divider()

# -------------------------------
# Top 3 Risk Countries
# -------------------------------
st.subheader("Top 3 Risk Countries")

top3_display = top3[["Country", "Total_Risk_Score", "Risk_Category"]]
st.dataframe(top3_display.style.format({"Total_Risk_Score": "{:.1f}"}))

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
st.subheader("EU Trade Risk Overview")
st.dataframe(
    df_sorted[[
        "Country",
        "Trade_Risk",
        "Energy_Risk",
        "Dependency_Level",
        "Total_Risk_Score",
        "Risk_Category",
        "Predicted_Risk_Category"
    ]].style.format({"Total_Risk_Score": "{:.1f}"})
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