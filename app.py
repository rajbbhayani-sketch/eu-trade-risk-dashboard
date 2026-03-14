import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="EU Risk Intelligence Platform",
    layout="wide"
)

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #132238;
        font-weight: 700;
    }

   .hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #111827;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #5b6472;
        margin-bottom: 1rem;
    }

    .company-banner {
    background: #ffffff;
    border: 1px solid #e8edf3;
    padding: 22px 24px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .section-card {
        background: white;
        border: 1px solid #e8edf3;
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .metric-card {
        background: white;
        border: 1px solid #e8edf3;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        min-height: 120px;
    }

    .metric-label {
        color: #667085;
        font-size: 0.92rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #132238;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-note {
        color: #667085;
        font-size: 0.85rem;
        margin-top: 8px;
    }

    .alert-box {
        background: #fff4e8;
        border-left: 6px solid #d97706;
        padding: 16px 18px;
        border-radius: 12px;
        color: #5b3b00;
        margin-top: 8px;
    }

    .insight-box {
        background: #eff6ff;
        border-left: 6px solid #2563eb;
        padding: 16px 18px;
        border-radius: 12px;
        color: #1e3a5f;
    }

    .action-box {
        background: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 16px 18px;
        border-radius: 12px;
        color: #14532d;
    }

    .small-muted {
        color: #667085;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8edf3;
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
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
    st.error("Dataset is empty. Run fetch_live_data.py first.")
    st.stop()

if history_path.exists():
    history_df = pd.read_csv(history_path)
else:
    history_df = pd.DataFrame(columns=["Date", "ISO3", "Country", "Risk_Score"])

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("Scenario Controls")

trade_weight = st.sidebar.slider("Trade Risk Weight", 0.0, 1.0, 0.4, 0.1)
energy_weight = st.sidebar.slider("Energy Risk Weight", 0.0, 1.0, 0.6, 0.1)
growth = st.sidebar.slider("Future Risk Growth %", -20, 20, 5)

country_list = sorted(df["Country"].dropna().tolist())
focus_country = st.sidebar.selectbox("Country Focus", country_list)

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
).round(1).clip(0, 100)

def trend(row):
    if row["Projected_Risk_Score"] > row["Total_Risk_Score"]:
        return "Rising"
    elif row["Projected_Risk_Score"] < row["Total_Risk_Score"]:
        return "Falling"
    return "Stable"

df["Risk_Trend"] = df.apply(trend, axis=1)

# -------------------------------------------------
# EXPLAINABILITY
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
avg_projected = round(df["Projected_Risk_Score"].mean(), 1)
high_count = int((df["Risk_Category"] == "High").sum())
countries_count = len(df)

avg_trade = round(df["Trade_Risk"].mean(), 1)
avg_energy = round(df["Energy_Risk"].mean(), 1)
portfolio_driver = "Energy Dependency" if avg_energy > avg_trade else "Trade Exposure"

focus_row = df[df["Country"] == focus_country].iloc[0]

last_updated = df["Last_Updated"].iloc[0] if "Last_Updated" in df.columns else "N/A"
data_source = df["Data_Source"].iloc[0] if "Data_Source" in df.columns else "N/A"

# -------------------------------------------------
# HEADER / HERO
# -------------------------------------------------
st.markdown("""
<div class="company-banner">
    <div class="hero-title">EU Risk Intelligence Platform</div>
    <div class="hero-subtitle">
        Executive monitoring of trade exposure, energy dependency, and projected country risk across Europe and key global partners.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# KPI ROW
# -------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Average Risk", f"{avg_risk:.1f}")
with c2:
    st.metric("Projected Risk", f"{avg_projected:.1f}")
with c3:
    st.metric("High-Risk Markets", f"{high_count}")
with c4:
    st.metric("Top Risk Country", top_country)
with c5:
    st.metric("Data Coverage", f"{countries_count} Countries")

st.markdown(
    f"""
<div class="alert-box">
<b>Strategic Alert:</b> {top_country} currently represents the highest portfolio exposure with a total risk score of <b>{top_score:.1f}</b>.
The primary structural driver is <b>{top_driver}</b>. Immediate monitoring and contingency review are recommended.
</div>
""",
    unsafe_allow_html=True
)

# -------------------------------------------------
# DATA STATUS
# -------------------------------------------------
left, right = st.columns([2, 3])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data Status")
    st.write(f"**Source:** {data_source}")
    st.write(f"**Last Update:** {last_updated}")
    st.write(f"**Current Scenario:** Trade {trade_weight:.1f} / Energy {energy_weight:.1f}")
    st.write(f"**Forecast Assumption:** {growth}%")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Executive Assessment")
    st.markdown(
        f"""
<div class="insight-box">
The current portfolio remains primarily influenced by <b>{portfolio_driver}</b>. 
Average risk stands at <b>{avg_risk:.1f}</b>, while projected risk under the current scenario rises to <b>{avg_projected:.1f}</b>. 
Among all monitored countries, <b>{top_country}</b> requires the highest level of management attention.
</div>
""",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive View",
    "Forecast & Trends",
    "Risk Drivers",
    "Reporting"
])

# -------------------------------------------------
# TAB 1: EXECUTIVE VIEW
# -------------------------------------------------
with tab1:
    st.subheader("Top Risk Markets")

    top10 = df.head(10).copy()

    fig_rank = px.bar(
        top10,
        x="Country",
        y="Total_Risk_Score",
        color="Risk_Category",
        title="Top 10 Countries by Risk Score",
        color_discrete_map={
            "High": "#dc2626",
            "Medium": "#f59e0b",
            "Low": "#16a34a"
        }
    )
    fig_rank.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text=""
    )
    st.plotly_chart(fig_rank, width="stretch")

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader("Risk Heatmap")
        heatmap_df = df[["Country", "Trade_Risk", "Energy_Risk", "Total_Risk_Score"]].set_index("Country")
        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Reds",
            title="Country vs Risk Factors"
        )
        fig_heatmap.update_layout(
            margin=dict(l=20, r=20, t=55, b=20)
        )
        st.plotly_chart(fig_heatmap, width="stretch")

    with col_b:
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
            margin=dict(l=10, r=10, t=55, b=10)
        )
        st.plotly_chart(fig_map, width="stretch")

# -------------------------------------------------
# TAB 2: FORECAST & TRENDS
# -------------------------------------------------
with tab2:
    st.subheader("Risk Projection")

    fig_proj = px.bar(
        df.head(10),
        x="Country",
        y=["Total_Risk_Score", "Projected_Risk_Score"],
        barmode="group",
        title="Current vs Projected Risk – Top 10 Countries"
    )
    fig_proj.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text=""
    )
    st.plotly_chart(fig_proj, width="stretch")

    st.subheader("Country Trend View")

    if not history_df.empty:
        focus_history = history_df[history_df["Country"] == focus_country].copy()
        focus_history["Date"] = pd.to_datetime(focus_history["Date"])

        fig_trend = px.line(
            focus_history,
            x="Date",
            y="Risk_Score",
            markers=True,
            title=f"12-Month Risk Trend – {focus_country}"
        )
        fig_trend.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=55, b=20)
        )
        st.plotly_chart(fig_trend, width="stretch")
    else:
        st.info("Risk history file not found. Run fetch_live_data.py to generate trend data.")

# -------------------------------------------------
# TAB 3: RISK DRIVERS
# -------------------------------------------------
with tab3:
    st.subheader("Country Driver Summary")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Focus Country", focus_country)
    k2.metric("Current Risk", f"{focus_row['Total_Risk_Score']:.1f}")
    k3.metric("Projected Risk", f"{focus_row['Projected_Risk_Score']:.1f}")
    k4.metric("Primary Driver", focus_row["Main_Risk_Driver"])

    st.markdown(
        f"""
<div class="section-card">
<b>{focus_country}</b> is currently classified as <b>{focus_row['Risk_Category']}</b> risk.
The strongest structural driver is <b>{focus_row['Main_Risk_Driver']}</b>.

<br><br>
<b>Contribution Breakdown</b><br>
Trade Contribution: <b>{focus_row['Trade_Contribution']:.1f}</b><br>
Energy Contribution: <b>{focus_row['Energy_Contribution']:.1f}</b><br>
Projected Trend: <b>{focus_row['Risk_Trend']}</b>
</div>
""",
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        contrib_df = pd.DataFrame({
            "Component": ["Trade Contribution", "Energy Contribution"],
            "Value": [focus_row["Trade_Contribution"], focus_row["Energy_Contribution"]]
        })

        fig_driver = px.bar(
            contrib_df,
            x="Component",
            y="Value",
            title=f"Risk Contribution Breakdown – {focus_country}",
            color="Component"
        )
        fig_driver.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=55, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_driver, width="stretch")

    with right:
        st.subheader("AI Monitoring")
        st.metric("Model Accuracy", f"{accuracy:.2f}" if accuracy is not None else "N/A")

        fig_imp = px.bar(
            importance,
            x="Feature",
            y="Importance",
            title="Model Feature Importance"
        )
        fig_imp.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=55, b=20)
        )
        st.plotly_chart(fig_imp, width="stretch")

# -------------------------------------------------
# TAB 4: REPORTING
# -------------------------------------------------
with tab4:
    st.subheader("Management Recommendations")

    st.markdown(
        f"""
<div class="action-box">
<b>Primary Action:</b> {"Prioritise energy diversification and reduce structural import dependency." if portfolio_driver == "Energy Dependency" else "Strengthen trade resilience and reduce concentration exposure."}
<br><br>
<b>Immediate Response:</b> {"Intensify monitoring and contingency planning for " + top_country + "." if high_count > 0 else "Maintain routine monitoring under the current scenario."}
<br><br>
<b>Medium-Term Priorities:</b><br>
• Improve cross-border resilience planning<br>
• Expand alternative sourcing strategies<br>
• Increase scenario-based stress testing<br>
• Review exposure to geopolitical disruption
</div>
""",
        unsafe_allow_html=True
    )

    st.subheader("Download Reports")

    full_csv = df.to_csv(index=False).encode("utf-8")
    top10_csv = df.head(10).to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download Full Risk Dataset (CSV)",
            data=full_csv,
            file_name="eu_trade_energy_risk_report.csv",
            mime="text/csv"
        )
    with d2:
        st.download_button(
            label="Download Top 10 Risk Report (CSV)",
            data=top10_csv,
            file_name="top_10_risk_report.csv",
            mime="text/csv"
        )

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