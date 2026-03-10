import streamlit as st
import pandas as pd

st.set_page_config(page_title="EU Trade Risk AI", layout="wide")

# Load dataset
df = pd.read_csv("data/eu_trade_energy_risk.csv")

# Calculate total risk score
df["Total_Risk_Score"] = (0.4 * df["Trade_Risk"] + 0.6 * df["Energy_Risk"]).round(1)

# Risk category
def classify_risk(score):
    if score >= 70:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)

# KPI values
average_risk = round(df["Total_Risk_Score"].mean(), 1)
high_risk_count = (df["Risk_Category"] == "High").sum()
total_countries = len(df)
top_country = df.sort_values("Total_Risk_Score", ascending=False).iloc[0]["Country"]

# Sort table
df_sorted = df.sort_values("Total_Risk_Score", ascending=False)

# Page title
st.title("AI Decision Support for EU Trade & Energy Risk")
st.markdown("### Monitoring trade disruption and energy risk across Europe")

st.info(
    "This dashboard helps identify high-risk European countries using trade risk, "
    "energy risk, and a calculated total risk score."
)

# KPI section
st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average EU Risk", average_risk)
col2.metric("High Risk Countries", high_risk_count)
col3.metric("Total Countries", total_countries)
col4.metric("Top Risk Country", top_country)

st.divider()

# Dataset section
st.subheader("EU Trade Risk Overview")
st.dataframe(df_sorted.style.format({"Total_Risk_Score": "{:.1f}"}))

st.divider()

# Risk distribution
st.subheader("Risk Category Distribution")
risk_counts = df["Risk_Category"].value_counts()
st.bar_chart(risk_counts)