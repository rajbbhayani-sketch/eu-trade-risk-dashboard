import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EU Trade Risk AI", layout="wide")

# Load data
df = pd.read_csv("data/eu_trade_energy_risk.csv")

# Calculate Total Risk Score
df["Total_Risk_Score"] = (0.4 * df["Trade_Risk"] + 0.6 * df["Energy_Risk"]).round(1)

# Risk Category
def classify_risk(score):
    if score >= 70:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)

# KPI calculations
average_risk = round(df["Total_Risk_Score"].mean(),1)
high_risk_count = (df["Risk_Category"] == "High").sum()
total_countries = len(df)
top_country = df.sort_values("Total_Risk_Score",ascending=False).iloc[0]["Country"]

df_sorted = df.sort_values("Total_Risk_Score",ascending=False)

# Title
st.title("AI Decision Support for EU Trade & Energy Risk")
st.markdown("### Monitoring trade disruption and energy risk across Europe")

st.info(
"This dashboard helps identify high-risk European countries using trade risk, energy risk and a calculated total risk score."
)

# Executive summary
st.subheader("Executive Summary")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Average EU Risk",average_risk)
col2.metric("High Risk Countries",high_risk_count)
col3.metric("Total Countries",total_countries)
col4.metric("Top Risk Country",top_country)

st.divider()

# Risk Drivers Analysis
st.subheader("Risk Drivers Analysis")

avg_trade = df["Trade_Risk"].mean()
avg_energy = df["Energy_Risk"].mean()

c1,c2 = st.columns(2)

c1.metric("Average Trade Risk",round(avg_trade,1))
c2.metric("Average Energy Risk",round(avg_energy,1))

if avg_energy > avg_trade:
    st.warning("Energy dependency is the main risk driver in the EU.")
else:
    st.warning("Trade disruption is the main risk driver in the EU.")

st.divider()

# Country comparison chart
st.subheader("Country Risk Comparison")

fig = px.bar(
    df,
    x="Country",
    y=["Trade_Risk","Energy_Risk"],
    barmode="group",
    title="Trade Risk vs Energy Risk by Country"
)

st.plotly_chart(fig,use_container_width=True)

st.divider()

# Top risk countries
st.subheader("Top 3 Risk Countries")

top3 = df_sorted.head(3)

st.dataframe(
    top3[["Country","Total_Risk_Score","Risk_Category"]]
)

st.divider()

# Dataset
st.subheader("EU Trade Risk Overview")

st.dataframe(df_sorted)

st.divider()

# Risk distribution
st.subheader("Risk Category Distribution")

risk_counts = df["Risk_Category"].value_counts()

fig2 = px.bar(
    risk_counts,
    title="Distribution of Risk Levels",
)

st.plotly_chart(fig2,use_container_width=True)