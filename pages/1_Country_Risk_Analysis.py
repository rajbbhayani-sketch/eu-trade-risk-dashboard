import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Country Risk Analysis")

# Load dataset
df = pd.read_csv("data/eu_trade_energy_risk.csv")

# Create total risk score
df["Total_Risk_Score"] = (0.4 * df["Trade_Risk"] + 0.6 * df["Energy_Risk"]).round(1)

# Create risk category
def classify_risk(score):
    if score >= 70:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

df["Risk_Category"] = df["Total_Risk_Score"].apply(classify_risk)

# Sort by highest risk
df = df.sort_values(by="Total_Risk_Score", ascending=False)

# Sidebar controls
st.sidebar.header("Country Risk Controls")
selected_country = st.sidebar.selectbox("Select Country", df["Country"])
selected_category = st.sidebar.multiselect(
    "Filter by Risk Category",
    options=df["Risk_Category"].unique(),
    default=df["Risk_Category"].unique()
)

# Apply filters
filtered_df = df[df["Risk_Category"].isin(selected_category)]
selected_df = df[df["Country"] == selected_country]

# KPI summary
highest_risk_country = filtered_df.iloc[0]["Country"]
highest_risk_score = filtered_df.iloc[0]["Total_Risk_Score"]
average_risk = round(filtered_df["Total_Risk_Score"].mean(), 1)
high_risk_count = (filtered_df["Risk_Category"] == "High").sum()

st.subheader("Risk Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Top Risk Country", highest_risk_country)
col2.metric("Highest Risk Score", highest_risk_score)
col3.metric("Average EU Risk", average_risk)

st.metric("Number of High Risk Countries", high_risk_count)

# Alert box
if high_risk_count > 0:
    st.warning(f"High Risk Alert: {high_risk_count} country/countries currently classified as High Risk.")
else:
    st.success("No countries are currently classified as High Risk.")

st.divider()

# Styled risk table
st.subheader("EU Country Risk Dataset")

def color_risk(val):
    if val == "High":
        return "background-color: #ffcccc"
    elif val == "Medium":
        return "background-color: #fff3cd"
    elif val == "Low":
        return "background-color: #d4edda"
    return ""

styled_df = filtered_df.style.map(color_risk, subset=["Risk_Category"]).format(
    {"Total_Risk_Score": "{:.1f}"}
)

st.dataframe(styled_df)

# Selected country section
st.subheader("Selected Country Risk")
st.dataframe(selected_df.style.format({"Total_Risk_Score": "{:.1f}"}))

# Two-column layout for chart and map
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk Comparison")
    st.bar_chart(filtered_df.set_index("Country")[["Trade_Risk", "Energy_Risk"]])

with col_right:
    st.subheader("EU Risk Map")
    fig = px.choropleth(
        filtered_df,
        locations="Country",
        locationmode="country names",
        color="Total_Risk_Score",
        hover_name="Country",
        color_continuous_scale="Reds",
        scope="europe",
        title="EU Trade & Energy Risk Map"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)