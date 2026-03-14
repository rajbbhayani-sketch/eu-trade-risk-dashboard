import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="EU Trade & Energy Risk Dashboard",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM STYLE
# -------------------------------------------------
st.markdown("""
<style>
    .main { padding-top: 1.2rem; }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 { color: #1f2a44; font-weight: 700; }
    .subtitle {
        font-size: 1.15rem;
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
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
live_path = Path("data/live_country_risk_data.csv")
history_path = Path("data/risk_history.csv")

df = pd.read_csv(live_path)

if df.empty:
    st.error("Dataset empty. Run fetch_live_data.py first.")
    st.stop()

if history_path.exists():
    history_df = pd.read_csv(history_path)
else:
    history_df = pd.DataFrame(columns=["Date", "ISO3", "Country", "Risk_Score"])

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Scenario Controls")

trade_weight = st.sidebar.slider("Trade Risk Weight", 0.0, 1.0, 0.4, 0.1)
energy_weight = st.sidebar.slider("Energy Risk Weight", 0.0, 1.0, 0.6, 0.1)
growth = st.sidebar.slider("Future Risk Growth %", -20, 20, 5)

country_list = sorted(df["Country"].dropna().tolist())
focus_country = st.sidebar.selectbox("Risk Driver Focus", country_list)

if trade_weight + energy_weight == 0:
    trade_weight = 0.4
    energy_weight = 0.6

# -------------------------------------------------
# RISK MODEL
# -------------------------------------------------
df["Total_Risk_Score"] = (
    trade_weight * df["Trade_Risk"] +
    energy_weight * df["Energy_Risk"]
).round(1)

def classify(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"

df["Risk_Category"] = df["Total_Risk_Score"].apply(classify)

# -------------------------------------------------
# FUTURE PROJECTION
# -------------------------------------------------
df["Projected_Risk_Score"] = (
    df["Total_Risk_Score"] * (1 + growth / 100)
).round(1)
df["Projected_Risk_Score"] = df["Projected_Risk_Score"].clip(0, 100)

def trend(row):
    if row["Projected_Risk_Score"] > row["Total_Risk_Score"]:
        return "Rising"
    elif row["Projected_Risk_Score"] < row["Total_Risk_Score"]:
        return "Falling"
    return "Stable"

df["Risk_Trend"] = df.apply(trend, axis=1)

# -------------------------------------------------
# DRIVER ANALYSIS
# -------------------------------------------------
df["Trade_Contribution"] = (trade_weight * df["Trade_Risk"]).round(1)
df["Energy_Contribution"] = (energy_weight * df["Energy_Risk"]).round(1)

def driver(row):
    if row["Energy_Contribution"] > row["Trade_Contribution"]:
        return "Energy Dependency"
    elif row["Trade_Contribution"] > row["Energy_Contribution"]:
        return "Trade Exposure"
    return "Balanced"

df["Main_Risk_Driver"] = df.apply(driver, axis=1)

def top_3_drivers(row):
    items = [
        ("Energy Dependency", row["Energy_Contribution"]),
        ("Trade Exposure", row["Trade_Contribution"]),
        ("Baseline Structural Risk", max(0.0, 100 - row["Total_Risk_Score"]) / 10)
    ]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    return [items[0][0], items[1][0], items[2][0]]

# -------------------------------------------------
# AI MODEL
# -------------------------------------------------
dep_map = {"Low": 0, "Medium": 1, "High": 2}
df["Dependency_Level_Encoded"] = df["Dependency_Level"].map(dep_map).fillna(1)

X = df[["Trade_Risk", "Energy_Risk", "Dependency_Level_Encoded"]]
y = df["Risk_Category"]

accuracy = None

if len(df) > 10 and y.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    df["Predicted_Risk_Category"] = model.predict(X)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
else:
    df["Predicted_Risk_Category"] = df["Risk_Category"]
    importance = pd.DataFrame({
        "Feature": ["Trade_Risk", "Energy_Risk", "Dependency_Level_Encoded"],
        "Importance": [0, 0, 0]
    })

# -------------------------------------------------
# SUMMARY VALUES
# -------------------------------------------------
df = df.sort_values("Total_Risk_Score", ascending=False).reset_index(drop=True)

top_country = df.loc[0, "Country"]
top_score = float(df.loc[0, "Total_Risk_Score"])
top_driver = df.loc[0, "Main_Risk_Driver"]

avg_risk = round(df["Total_Risk_Score"].mean(), 1)
avg_future = round(df["Projected_Risk_Score"].mean(), 1)
high_count = int((df["Risk_Category"] == "High").sum())
total_countries = len(df)

avg_trade = round(df["Trade_Risk"].mean(), 1)
avg_energy = round(df["Energy_Risk"].mean(), 1)
portfolio_driver = "Energy Dependency" if avg_energy > avg_trade else "Trade Exposure"

focus_row = df[df["Country"] == focus_country].iloc[0]
focus_drivers = top_3_drivers(focus_row)

last_updated = df["Last_Updated"].iloc[0] if "Last_Updated" in df.columns else "N/A"
data_source = df["Data_Source"].iloc[0] if "Data_Source" in df.columns else "N/A"

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("EU Trade & Energy Risk Intelligence Dashboard")
st.markdown(
    "<div class='subtitle'>Strategic monitoring of trade exposure, energy dependency, and projected geopolitical risk</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='info-banner'>"
    "This platform combines live macro-risk data, AI classification, scenario simulation, forward-looking projection, "
    "and explainability to support executive monitoring and strategic response planning."
    "</div>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# REAL-TIME SOURCE INFO
# -------------------------------------------------
st.subheader("Data Status")

d1, d2, d3 = st.columns(3)
d1.metric("Data Source", data_source)
d2.metric("Last Update", last_updated[:16] if isinstance(last_updated, str) else "N/A")
d3.metric("Countries Monitored", total_countries)

st.divider()

# -------------------------------------------------
# EXECUTIVE SUMMARY
# -------------------------------------------------
st.subheader("Executive Summary")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Average Portfolio Risk", avg_risk)
k2.metric("High Risk Countries", high_count)
k3.metric("Projected Portfolio Risk", avg_future)
k4.metric("Highest Risk Country", top_country)

st.markdown(
    f"<div class='alert-banner'><b>Strategic Alert:</b> {top_country} currently represents the highest portfolio risk "
    f"with a score of <b>{top_score:.1f}</b>. Primary driver: <b>{top_driver}</b>.</div>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# RISK OUTLOOK
# -------------------------------------------------
st.subheader("Risk Outlook")

o1, o2, o3 = st.columns(3)
o1.metric("Current Scenario", f"Trade {trade_weight:.1f} / Energy {energy_weight:.1f}")
o2.metric("Forecast Growth Assumption", f"{growth}%")
o3.metric("Portfolio Risk Driver", portfolio_driver)

fig_forecast = px.bar(
    df.head(10),
    x="Country",
    y=["Total_Risk_Score", "Projected_Risk_Score"],
    barmode="group",
    title="Top Countries – Current vs Projected Risk"
)
fig_forecast.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=50, b=20),
    legend_title_text=""
)
st.plotly_chart(fig_forecast, width="stretch")

st.divider()

# -------------------------------------------------
# RISK TREND VISUALIZATION
# -------------------------------------------------
st.subheader("Risk Trend Over Time")

if not history_df.empty:
    country_history = history_df[history_df["Country"] == focus_country].copy()
    country_history["Date"] = pd.to_datetime(country_history["Date"])

    fig_trend = px.line(
        country_history,
        x="Date",
        y="Risk_Score",
        markers=True,
        title=f"12-Month Risk Trend – {focus_country}"
    )
    fig_trend.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_trend, width="stretch")
else:
    st.info("Risk history file not found. Run fetch_live_data.py to generate trend data.")

st.divider()

# -------------------------------------------------
# RISK DRIVER ANALYSIS
# -------------------------------------------------
st.subheader("Risk Driver Analysis")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Focus Country", focus_country)
r2.metric("Current Risk", f"{focus_row['Total_Risk_Score']:.1f}")
r3.metric("Projected Risk", f"{focus_row['Projected_Risk_Score']:.1f}")
r4.metric("Primary Driver", focus_row["Main_Risk_Driver"])

st.markdown(
    f"<div class='section-card'>"
    f"<b>{focus_country}</b> is currently classified as <b>{focus_row['Risk_Category']}</b> risk. "
    f"The country’s risk profile is primarily driven by <b>{focus_row['Main_Risk_Driver']}</b>.<br><br>"
    f"<b>Top 3 Drivers:</b><br>"
    f"1. {focus_drivers[0]}<br>"
    f"2. {focus_drivers[1]}<br>"
    f"3. {focus_drivers[2]}<br><br>"
    f"Trade contribution: <b>{focus_row['Trade_Contribution']:.1f}</b> | "
    f"Energy contribution: <b>{focus_row['Energy_Contribution']:.1f}</b> | "
    f"Trend: <b>{focus_row['Risk_Trend']}</b>."
    f"</div>",
    unsafe_allow_html=True
)

driver_df = pd.DataFrame({
    "Component": ["Trade Contribution", "Energy Contribution"],
    "Value": [focus_row["Trade_Contribution"], focus_row["Energy_Contribution"]]
})

fig_driver = px.bar(
    driver_df,
    x="Component",
    y="Value",
    title=f"Contribution Breakdown – {focus_country}"
)
fig_driver.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_driver, width="stretch")

st.divider()

# -------------------------------------------------
# TOP RISK COUNTRIES
# -------------------------------------------------
st.subheader("Top Risk Countries")

top_table = df[[
    "Country",
    "Total_Risk_Score",
    "Projected_Risk_Score",
    "Risk_Category",
    "Risk_Trend",
    "Main_Risk_Driver"
]].head(10)

st.dataframe(
    top_table.style.format({
        "Total_Risk_Score": "{:.1f}",
        "Projected_Risk_Score": "{:.1f}"
    }),
    width="stretch"
)

st.divider()

# -------------------------------------------------
# AI MONITORING
# -------------------------------------------------
st.subheader("AI Monitoring")

a1, a2 = st.columns(2)

with a1:
    st.metric("Model Accuracy", f"{accuracy:.2f}" if accuracy is not None else "N/A")
    st.dataframe(
        df[["Country", "Risk_Category", "Predicted_Risk_Category"]].head(10),
        width="stretch"
    )

with a2:
    fig_imp = px.bar(
        importance,
        x="Feature",
        y="Importance",
        title="AI Model Feature Importance"
    )
    fig_imp.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_imp, width="stretch")

st.divider()

# -------------------------------------------------
# HEATMAP
# -------------------------------------------------
st.subheader("Risk Heatmap")

heatmap_df = df[["Country", "Trade_Risk", "Energy_Risk", "Total_Risk_Score"]].set_index("Country")
fig_heatmap = px.imshow(
    heatmap_df,
    text_auto=".1f",
    aspect="auto",
    color_continuous_scale="Reds",
    title="Country vs Risk Factors Heatmap"
)
fig_heatmap.update_layout(
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_heatmap, width="stretch")

st.divider()

# -------------------------------------------------
# PORTFOLIO COMPARISON
# -------------------------------------------------
st.subheader("Portfolio Risk Comparison")

fig_compare = px.bar(
    df,
    x="Country",
    y=["Trade_Risk", "Energy_Risk"],
    barmode="group",
    title="Trade Risk vs Energy Risk by Country"
)
fig_compare.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_text="",
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_compare, width="stretch")

st.divider()

# -------------------------------------------------
# MANAGEMENT RECOMMENDATIONS
# -------------------------------------------------
st.subheader("Management Recommendations")

st.markdown("<div class='section-card'>", unsafe_allow_html=True)

if portfolio_driver == "Energy Dependency":
    st.write("**Primary Action:** Prioritize energy diversification and reduce structural import dependency.")
else:
    st.write("**Primary Action:** Strengthen trade resilience and reduce concentration exposure.")

if high_count > 0:
    st.write(f"**Immediate Response:** Intensify monitoring and contingency planning for **{top_country}**.")
else:
    st.write("**Immediate Response:** Maintain routine monitoring under the current scenario.")

st.write("**Medium-Term Priorities:**")
st.write("- Improve cross-border resilience planning")
st.write("- Expand alternative sourcing strategies")
st.write("- Increase scenario-based stress testing")
st.write("- Review exposure to geopolitical disruption")

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# GEOGRAPHIC VIEW
# -------------------------------------------------
st.subheader("Geographic Risk View")

fig_map = px.choropleth(
    df,
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
# DOWNLOADABLE REPORTS
# -------------------------------------------------
st.subheader("Download Reports")

csv_data = df.to_csv(index=False).encode("utf-8")
top10_csv = top_table.to_csv(index=False).encode("utf-8")

b1, b2 = st.columns(2)
with b1:
    st.download_button(
        label="Download Full Risk Dataset (CSV)",
        data=csv_data,
        file_name="eu_trade_energy_risk_report.csv",
        mime="text/csv"
    )

with b2:
    st.download_button(
        label="Download Top Risk Countries Report (CSV)",
        data=top10_csv,
        file_name="top_risk_countries_report.csv",
        mime="text/csv"
    )

st.divider()

# -------------------------------------------------
# DETAILED DATA
# -------------------------------------------------
st.subheader("Detailed Dataset")

detail_cols = [
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
    "Main_Risk_Driver",
    "Last_Updated",
    "Data_Source"
]

st.dataframe(
    df[detail_cols].style.format({
        "Trade_Risk": "{:.1f}",
        "Energy_Risk": "{:.1f}",
        "Total_Risk_Score": "{:.1f}",
        "Projected_Risk_Score": "{:.1f}",
        "Trade_Contribution": "{:.1f}",
        "Energy_Contribution": "{:.1f}"
    }),
    width="stretch"
)