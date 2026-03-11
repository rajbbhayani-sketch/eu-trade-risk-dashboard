import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Country Risk Analysis", layout="wide")

# -------------------------------
# Load live data
# -------------------------------
df = pd.read_csv("data/live_country_risk_data.csv")

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Country Risk Controls")

country_list = sorted(df["Country"].unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    country_list,
    index=country_list.index("Germany") if "Germany" in country_list else 0
)

selected_categories = st.sidebar.multiselect(
    "Filter by Risk Category",
    options=sorted(df["Risk_Category"].unique()),
    default=sorted(df["Risk_Category"].unique())
)

# New compare section
st.sidebar.header("Compare Two Countries")

country_a = st.sidebar.selectbox(
    "Country A",
    country_list,
    index=country_list.index("Germany") if "Germany" in country_list else 0,
    key="country_a"
)

default_b_index = country_list.index("China") if "China" in country_list else min(1, len(country_list)-1)

country_b = st.sidebar.selectbox(
    "Country B",
    country_list,
    index=default_b_index,
    key="country_b"
)

# -------------------------------
# Filtered data
# -------------------------------
filtered_df = df[df["Risk_Category"].isin(selected_categories)]

selected_df = filtered_df[filtered_df["Country"] == selected_country]

country_a_df = df[df["Country"] == country_a]
country_b_df = df[df["Country"] == country_b]

# -------------------------------
# Page title
# -------------------------------
st.title("Country Risk Analysis")

# -------------------------------
# Summary KPIs
# -------------------------------
top_country = df.sort_values("Total_Risk_Score", ascending=False).iloc[0]["Country"]
highest_score = df["Total_Risk_Score"].max()
average_risk = round(df["Total_Risk_Score"].mean(), 1)
high_risk_count = (df["Risk_Category"] == "High").sum()

st.subheader("Risk Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Top Risk Country", top_country)
c2.metric("Highest Risk Score", round(highest_score, 1))
c3.metric("Average Risk Score", average_risk)

st.metric("Number of High Risk Countries", high_risk_count)

if high_risk_count > 0:
    st.warning(f"High Risk Alert: {high_risk_count} country/countries are currently classified as High Risk.")

st.divider()

# -------------------------------
# Full dataset
# -------------------------------
st.subheader("Live Country Risk Dataset")
st.dataframe(
    filtered_df.style.format({"Total_Risk_Score": "{:.1f}"})
)

st.divider()

# -------------------------------
# Selected country detail
# -------------------------------
st.subheader("Selected Country Risk")
st.dataframe(
    selected_df.style.format({"Total_Risk_Score": "{:.1f}"})
)

st.divider()

# -------------------------------
# Country comparison section
# -------------------------------
st.subheader("Compare Two Countries")

compare_df = pd.concat([country_a_df, country_b_df])

st.write(f"### {country_a} vs {country_b}")

comparison_table = compare_df[[
    "Country",
    "Trade_Risk",
    "Energy_Risk",
    "Dependency_Level",
    "Total_Risk_Score",
    "Risk_Category"
]]

st.dataframe(
    comparison_table.style.format({"Total_Risk_Score": "{:.1f}"})
)

# Comparison chart
fig_compare = px.bar(
    compare_df,
    x="Country",
    y=["Trade_Risk", "Energy_Risk", "Total_Risk_Score"],
    barmode="group",
    title=f"Risk Comparison: {country_a} vs {country_b}"
)

st.plotly_chart(fig_compare, use_container_width=True)

# -------------------------------
# Business insight text
# -------------------------------
score_a = float(country_a_df.iloc[0]["Total_Risk_Score"])
score_b = float(country_b_df.iloc[0]["Total_Risk_Score"])

trade_a = float(country_a_df.iloc[0]["Trade_Risk"])
trade_b = float(country_b_df.iloc[0]["Trade_Risk"])

energy_a = float(country_a_df.iloc[0]["Energy_Risk"])
energy_b = float(country_b_df.iloc[0]["Energy_Risk"])

risk_a = country_a_df.iloc[0]["Risk_Category"]
risk_b = country_b_df.iloc[0]["Risk_Category"]

st.subheader("Comparison Insight")

if score_a > score_b:
    st.write(
        f"**{country_a}** has a higher total risk score (**{score_a:.1f}**) than **{country_b}** (**{score_b:.1f}**)."
    )
elif score_b > score_a:
    st.write(
        f"**{country_b}** has a higher total risk score (**{score_b:.1f}**) than **{country_a}** (**{score_a:.1f}**)."
    )
else:
    st.write(
        f"**{country_a}** and **{country_b}** currently have the same total risk score (**{score_a:.1f}**)."
    )

if energy_a > energy_b:
    st.write(f"- {country_a} has higher energy-related risk than {country_b}.")
elif energy_b > energy_a:
    st.write(f"- {country_b} has higher energy-related risk than {country_a}.")
else:
    st.write(f"- {country_a} and {country_b} have the same energy-related risk.")

if trade_a > trade_b:
    st.write(f"- {country_a} has higher trade exposure than {country_b}.")
elif trade_b > trade_a:
    st.write(f"- {country_b} has higher trade exposure than {country_a}.")
else:
    st.write(f"- {country_a} and {country_b} have the same trade exposure.")

st.write(f"- Current risk categories: **{country_a}: {risk_a}**, **{country_b}: {risk_b}**.")

st.divider()

# -------------------------------
# Multi-country risk comparison
# -------------------------------
st.subheader("All-Country Risk Comparison")

fig_all = px.bar(
    filtered_df.sort_values("Total_Risk_Score", ascending=False),
    x="Country",
    y=["Trade_Risk", "Energy_Risk"],
    barmode="group",
    title="Trade Risk vs Energy Risk by Country"
)

st.plotly_chart(fig_all, use_container_width=True)

st.divider()

# -------------------------------
# EU + external comparison map substitute note
# -------------------------------
st.subheader("Analysis Note")

st.info(
    "This page compares country-level trade risk, energy risk, and total risk score "
    "across all countries in the live dataset, including EU members, India, and China."
)