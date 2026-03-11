import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Country Risk Analysis", layout="wide")

# Load dataset
df = pd.read_csv("data/live_country_risk_data.csv")

# Remove missing rows
df = df.dropna()

# Sidebar
st.sidebar.header("Country Risk Controls")

countries = sorted(df["Country"].unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    countries
)

risk_options = sorted(df["Risk_Category"].unique())

selected_categories = st.sidebar.multiselect(
    "Filter by Risk Category",
    risk_options,
    default=risk_options
)

st.sidebar.header("Compare Two Countries")

country_a = st.sidebar.selectbox(
    "Country A",
    countries,
    key="a"
)

country_b = st.sidebar.selectbox(
    "Country B",
    countries,
    key="b"
)

# Filter dataset
filtered_df = df[df["Risk_Category"].isin(selected_categories)]

# Page title
st.title("Country Risk Analysis")

# Summary metrics
top_country = df.sort_values("Total_Risk_Score", ascending=False).iloc[0]["Country"]
highest_score = df["Total_Risk_Score"].max()
avg_score = round(df["Total_Risk_Score"].mean(), 1)
high_risk_count = (df["Risk_Category"] == "High").sum()

c1, c2, c3 = st.columns(3)

c1.metric("Top Risk Country", top_country)
c2.metric("Highest Risk Score", round(highest_score,1))
c3.metric("Average Risk Score", avg_score)

st.metric("Number of High Risk Countries", high_risk_count)

st.divider()

# Dataset table
st.subheader("Live Country Risk Dataset")

st.dataframe(
    filtered_df.style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}"
    })
)

st.divider()

# Compare section
st.subheader("Compare Two Countries")

compare_df = df[df["Country"].isin([country_a, country_b])]

st.write(f"### {country_a} vs {country_b}")

st.dataframe(compare_df)

fig = px.bar(
    compare_df,
    x="Country",
    y=["Trade_Risk","Energy_Risk","Total_Risk_Score"],
    barmode="group",
    title="Risk Comparison"
)

st.plotly_chart(fig, use_container_width=True)