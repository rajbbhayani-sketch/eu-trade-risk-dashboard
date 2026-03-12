import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Country Risk Analysis",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM STYLE
# -------------------------------------------------
st.markdown("""
<style>
    .main {
        padding-top: 1.2rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #1f2a44;
        font-weight: 700;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #4b5563;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    .info-banner {
        background-color: #eef4ff;
        border-left: 6px solid #4f46e5;
        padding: 16px 18px;
        border-radius: 10px;
        color: #1f2937;
        margin-bottom: 18px;
    }

    .alert-banner {
        background-color: #fff7e6;
        border-left: 6px solid #d97706;
        padding: 16px 18px;
        border-radius: 10px;
        color: #3f3f46;
        margin-bottom: 18px;
    }

    .section-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("data/live_country_risk_data.csv")

df = df.dropna(subset=["Country", "Trade_Risk", "Energy_Risk", "Total_Risk_Score"])
df["Risk_Category"] = df["Risk_Category"].fillna("Unknown")
df["Dependency_Level"] = df["Dependency_Level"].fillna("Unknown")

if df.empty:
    st.error("The live dataset is empty. Please run fetch_live_data.py again.")
    st.stop()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Country Analysis Controls")

countries = sorted(df["Country"].astype(str).unique().tolist())

selected_country = st.sidebar.selectbox(
    "Focus Country",
    countries,
    index=countries.index("Germany") if "Germany" in countries else 0
)

risk_options = sorted(df["Risk_Category"].astype(str).unique().tolist())

selected_categories = st.sidebar.multiselect(
    "Filter by Risk Category",
    risk_options,
    default=risk_options
)

st.sidebar.header("Country Benchmarking")

country_a = st.sidebar.selectbox(
    "Country A",
    countries,
    index=countries.index("Germany") if "Germany" in countries else 0,
    key="country_a"
)

default_country_b = "China" if "China" in countries else countries[min(1, len(countries)-1)]

country_b = st.sidebar.selectbox(
    "Country B",
    countries,
    index=countries.index(default_country_b),
    key="country_b"
)

# -------------------------------------------------
# FILTERING
# -------------------------------------------------
filtered_df = df[df["Risk_Category"].isin(selected_categories)].copy()

if filtered_df.empty:
    st.warning("No countries match the selected filter. Showing full dataset instead.")
    filtered_df = df.copy()

selected_df = df[df["Country"] == selected_country].copy()
country_a_df = df[df["Country"] == country_a].copy()
country_b_df = df[df["Country"] == country_b].copy()

top_row = df.sort_values("Total_Risk_Score", ascending=False).iloc[0]
top_country = top_row["Country"]
top_score = float(top_row["Total_Risk_Score"])
avg_score = round(df["Total_Risk_Score"].mean(), 1)
high_risk_count = int((df["Risk_Category"] == "High").sum())

selected_row = selected_df.iloc[0]
compare_df = pd.concat([country_a_df, country_b_df], ignore_index=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Country Risk Analysis")
st.markdown(
    "<div class='subtitle'>Country-level benchmarking of trade exposure, energy dependency, and total risk</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='info-banner'>"
    "This page supports country-level analysis, peer comparison, and risk benchmarking using the live dataset."
    "</div>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
st.subheader("Country Overview")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Top Risk Country", top_country)
s2.metric("Highest Risk Score", f"{top_score:.1f}")
s3.metric("Average Portfolio Risk", avg_score)
s4.metric("High Risk Countries", high_risk_count)

if high_risk_count > 0:
    st.markdown(
        f"<div class='alert-banner'><b>Country Alert:</b> {top_country} currently has the highest country-level risk "
        f"with a score of <b>{top_score:.1f}</b>.</div>",
        unsafe_allow_html=True
    )

st.divider()

# -------------------------------------------------
# FOCUS COUNTRY
# -------------------------------------------------
st.subheader("Focus Country Profile")

f1, f2, f3, f4 = st.columns(4)
f1.metric("Country", selected_country)
f2.metric("Trade Risk", f"{selected_row['Trade_Risk']:.1f}")
f3.metric("Energy Risk", f"{selected_row['Energy_Risk']:.1f}")
f4.metric("Total Risk", f"{selected_row['Total_Risk_Score']:.1f}")

st.markdown(
    f"<div class='section-card'>"
    f"<b>{selected_country}</b> is currently classified as <b>{selected_row['Risk_Category']}</b> risk, "
    f"with <b>{selected_row['Dependency_Level']}</b> energy dependency."
    f"</div>",
    unsafe_allow_html=True
)

selected_display = selected_df[[
    "Country",
    "Trade_Risk",
    "Energy_Risk",
    "Dependency_Level",
    "Total_Risk_Score",
    "Risk_Category"
]]

st.dataframe(
    selected_display.style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}"
    }),
    width="stretch"
)

st.divider()

# -------------------------------------------------
# COUNTRY BENCHMARKING
# -------------------------------------------------
st.subheader("Country Benchmarking")
st.markdown(f"### {country_a} vs {country_b}")

benchmark_table = compare_df[[
    "Country",
    "Trade_Risk",
    "Energy_Risk",
    "Dependency_Level",
    "Total_Risk_Score",
    "Risk_Category"
]]

st.dataframe(
    benchmark_table.style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}"
    }),
    width="stretch"
)

fig_compare = px.bar(
    compare_df,
    x="Country",
    y=["Trade_Risk", "Energy_Risk", "Total_Risk_Score"],
    barmode="group",
    title="Benchmark Comparison"
)
fig_compare.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=50, b=20),
    legend_title_text=""
)
st.plotly_chart(fig_compare, width="stretch")

score_a = float(country_a_df.iloc[0]["Total_Risk_Score"])
score_b = float(country_b_df.iloc[0]["Total_Risk_Score"])

risk_a = str(country_a_df.iloc[0]["Risk_Category"])
risk_b = str(country_b_df.iloc[0]["Risk_Category"])

if score_a > score_b:
    compare_text = f"{country_a} currently has a higher total risk profile than {country_b}."
elif score_b > score_a:
    compare_text = f"{country_b} currently has a higher total risk profile than {country_a}."
else:
    compare_text = f"{country_a} and {country_b} currently have the same total risk score."

st.markdown(
    f"<div class='section-card'>"
    f"<b>Benchmark Insight:</b> {compare_text} "
    f"Current categories: <b>{country_a}: {risk_a}</b>, <b>{country_b}: {risk_b}</b>."
    f"</div>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# PORTFOLIO COMPARISON
# -------------------------------------------------
st.subheader("Portfolio Benchmarking")

fig_all = px.bar(
    filtered_df.sort_values("Total_Risk_Score", ascending=False),
    x="Country",
    y=["Trade_Risk", "Energy_Risk"],
    barmode="group",
    title="Trade Risk vs Energy Risk by Country"
)
fig_all.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=50, b=20),
    legend_title_text=""
)
st.plotly_chart(fig_all, width="stretch")

st.divider()

# -------------------------------------------------
# GLOBAL MAP
# -------------------------------------------------
st.subheader("Geographic Risk View")

fig_map = px.choropleth(
    filtered_df,
    locations="ISO3",
    color="Total_Risk_Score",
    hover_name="Country",
    color_continuous_scale="Reds",
    title="Global Trade & Energy Risk Map"
)
fig_map.update_layout(
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_map, width="stretch")

st.divider()

# -------------------------------------------------
# DATASET
# -------------------------------------------------
st.subheader("Detailed Country Dataset")

detail_cols = [
    "ISO3",
    "Country",
    "Trade_Risk",
    "Energy_Risk",
    "Dependency_Level",
    "Total_Risk_Score",
    "Risk_Category"
]

st.dataframe(
    filtered_df[detail_cols].style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}"
    }),
    width="stretch"
)

st.divider()

# -------------------------------------------------
# NOTE
# -------------------------------------------------
st.subheader("Analysis Note")
st.markdown(
    "<div class='section-card'>"
    "This page supports country-level comparison across EU members and selected global partners, "
    "including India and China. It is designed for benchmarking, monitoring, and strategic screening."
    "</div>",
    unsafe_allow_html=True
)