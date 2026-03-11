import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Country Risk Analysis", layout="wide")

df = pd.read_csv("data/live_country_risk_data.csv")
df = df.dropna(subset=["Country", "Trade_Risk", "Energy_Risk", "Total_Risk_Score"])
df["Risk_Category"] = df["Risk_Category"].fillna("Unknown")
df["Dependency_Level"] = df["Dependency_Level"].fillna("Unknown")

st.title("Country Risk Analysis")

if df.empty:
    st.error("The live dataset is empty. Please run fetch_live_data.py again.")
    st.stop()

st.sidebar.header("Country Risk Controls")

countries = sorted(df["Country"].astype(str).unique().tolist())

selected_country = st.sidebar.selectbox(
    "Select Country",
    countries,
    index=countries.index("Germany") if "Germany" in countries else 0
)

risk_options = sorted(df["Risk_Category"].astype(str).unique().tolist())

selected_categories = st.sidebar.multiselect(
    "Filter by Risk Category",
    risk_options,
    default=risk_options
)

st.sidebar.header("Compare Two Countries")

country_a = st.sidebar.selectbox(
    "Country A",
    countries,
    index=countries.index("Germany") if "Germany" in countries else 0,
    key="country_a"
)

default_country_b = "China" if "China" in countries else countries[min(1, len(countries) - 1)]

country_b = st.sidebar.selectbox(
    "Country B",
    countries,
    index=countries.index(default_country_b),
    key="country_b"
)

filtered_df = df[df["Risk_Category"].isin(selected_categories)].copy()

if filtered_df.empty:
    st.warning("No countries match the selected risk categories. Showing the full dataset instead.")
    filtered_df = df.copy()

selected_df = filtered_df[filtered_df["Country"] == selected_country].copy()
if selected_df.empty:
    selected_df = df[df["Country"] == selected_country].copy()

country_a_df = df[df["Country"] == country_a].copy()
country_b_df = df[df["Country"] == country_b].copy()

top_row = df.sort_values("Total_Risk_Score", ascending=False).iloc[0]
top_country = top_row["Country"]
highest_score = float(top_row["Total_Risk_Score"])
avg_score = round(df["Total_Risk_Score"].mean(), 1)
high_risk_count = int((df["Risk_Category"] == "High").sum())

st.subheader("Risk Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Top Risk Country", top_country)
c2.metric("Highest Risk Score", f"{highest_score:.1f}")
c3.metric("Average Risk Score", avg_score)

st.metric("Number of High Risk Countries", high_risk_count)

if high_risk_count > 0:
    st.warning(f"High Risk Alert: {high_risk_count} country/countries are currently classified as High Risk.")

st.divider()

st.subheader("Live Country Risk Dataset")
st.dataframe(
    filtered_df.style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}"
    })
)

st.divider()

st.subheader("Selected Country Risk")
if selected_df.empty:
    st.info("No selected-country data available for the current filters.")
else:
    st.dataframe(
        selected_df.style.format({
            "Trade_Risk": "{:.1f}",
            "Energy_Risk": "{:.1f}",
            "Total_Risk_Score": "{:.1f}"
        })
    )

st.divider()

st.subheader("Compare Two Countries")
st.write(f"### {country_a} vs {country_b}")

compare_df = pd.concat([country_a_df, country_b_df], ignore_index=True)

if compare_df.empty:
    st.info("Comparison data is not available.")
else:
    comparison_table = compare_df[
        ["Country", "Trade_Risk", "Energy_Risk", "Dependency_Level", "Total_Risk_Score", "Risk_Category"]
    ]

    st.dataframe(
        comparison_table.style.format({
            "Trade_Risk": "{:.1f}",
            "Energy_Risk": "{:.1f}",
            "Total_Risk_Score": "{:.1f}"
        })
    )

    fig_compare = px.bar(
        compare_df,
        x="Country",
        y=["Trade_Risk", "Energy_Risk", "Total_Risk_Score"],
        barmode="group",
        title="Risk Comparison"
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    if len(country_a_df) > 0 and len(country_b_df) > 0:
        score_a = float(country_a_df.iloc[0]["Total_Risk_Score"])
        score_b = float(country_b_df.iloc[0]["Total_Risk_Score"])
        trade_a = float(country_a_df.iloc[0]["Trade_Risk"])
        trade_b = float(country_b_df.iloc[0]["Trade_Risk"])
        energy_a = float(country_a_df.iloc[0]["Energy_Risk"])
        energy_b = float(country_b_df.iloc[0]["Energy_Risk"])
        risk_a = str(country_a_df.iloc[0]["Risk_Category"])
        risk_b = str(country_b_df.iloc[0]["Risk_Category"])

        st.subheader("Comparison Insight")

        if score_a > score_b:
            st.write(f"**{country_a}** has a higher total risk score (**{score_a:.1f}**) than **{country_b}** (**{score_b:.1f}**).")
        elif score_b > score_a:
            st.write(f"**{country_b}** has a higher total risk score (**{score_b:.1f}**) than **{country_a}** (**{score_a:.1f}**).")
        else:
            st.write(f"**{country_a}** and **{country_b}** currently have the same total risk score (**{score_a:.1f}**).")

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

st.subheader("Global Risk Map")
fig_map = px.choropleth(
    filtered_df,
    locations="ISO3",
    color="Total_Risk_Score",
    hover_name="Country",
    color_continuous_scale="Reds",
    title="Global Trade & Energy Risk Map"
)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

st.subheader("Analysis Note")
st.info("This page compares trade risk, energy risk, and total risk score across all countries in the live dataset, including EU members, India, and China.")