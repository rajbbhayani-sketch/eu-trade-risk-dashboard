import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="EU Trade & Energy Risk Dashboard",
    layout="wide"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("data/live_country_risk_data.csv")

if df.empty:
    st.error("Dataset empty. Run fetch_live_data.py first.")
    st.stop()

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("Scenario Controls")

trade_weight = st.sidebar.slider(
    "Trade Risk Weight",
    0.0, 1.0, 0.4, 0.1
)

energy_weight = st.sidebar.slider(
    "Energy Risk Weight",
    0.0, 1.0, 0.6, 0.1
)

growth = st.sidebar.slider(
    "Future Risk Growth %",
    -20, 20, 5
)

country_list = sorted(df["Country"].tolist())

focus_country = st.sidebar.selectbox(
    "Risk Driver Focus",
    country_list
)

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

df["Projected_Risk_Score"] = df["Projected_Risk_Score"].clip(0,100)

def trend(row):
    if row["Projected_Risk_Score"] > row["Total_Risk_Score"]:
        return "Rising"
    if row["Projected_Risk_Score"] < row["Total_Risk_Score"]:
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
    return "Trade Exposure"

df["Main_Risk_Driver"] = df.apply(driver, axis=1)

# -------------------------------------------------
# AI MODEL
# -------------------------------------------------
dep_map = {"Low":0,"Medium":1,"High":2}
df["Dependency_Level_Encoded"] = df["Dependency_Level"].map(dep_map)

X = df[["Trade_Risk","Energy_Risk","Dependency_Level_Encoded"]]
y = df["Risk_Category"]

accuracy = None

if len(df) > 10 and y.nunique() > 1:

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.25,random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,pred)

    df["Predicted_Risk_Category"] = model.predict(X)

    importance = pd.DataFrame({
        "Feature":X.columns,
        "Importance":model.feature_importances_
    })

else:
    df["Predicted_Risk_Category"] = df["Risk_Category"]

    importance = pd.DataFrame({
        "Feature":["Trade_Risk","Energy_Risk","Dependency_Level_Encoded"],
        "Importance":[0,0,0]
    })

# -------------------------------------------------
# SUMMARY VALUES
# -------------------------------------------------
df = df.sort_values("Total_Risk_Score",ascending=False)

top_country = df.iloc[0]["Country"]
top_score = df.iloc[0]["Total_Risk_Score"]
top_driver = df.iloc[0]["Main_Risk_Driver"]

avg_risk = round(df["Total_Risk_Score"].mean(),1)
avg_future = round(df["Projected_Risk_Score"].mean(),1)

high_count = len(df[df["Risk_Category"]=="High"])

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("EU Trade & Energy Risk Intelligence Dashboard")

st.markdown(
"### Strategic monitoring of trade exposure, energy dependency, and geopolitical risk"
)

st.info(
"This platform analyzes trade risk and energy dependency across EU countries and key global partners. "
"It combines AI classification, scenario simulation, and future risk projection to support decision making."
)

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
st.subheader("Executive Summary")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Average Portfolio Risk",avg_risk)
c2.metric("High Risk Countries",high_count)
c3.metric("Projected Portfolio Risk",avg_future)
c4.metric("Highest Risk Country",top_country)

# -------------------------------------------------
# ALERT
# -------------------------------------------------
st.subheader("Strategic Alert")

st.warning(
f"{top_country} currently represents the highest portfolio risk "
f"with a score of {top_score}. Primary driver: {top_driver}."
)

st.divider()

# -------------------------------------------------
# FORECAST CHART
# -------------------------------------------------
st.subheader("Top Countries – Current vs Projected Risk")

top10 = df.head(10)

fig = px.bar(
    top10,
    x="Country",
    y=["Total_Risk_Score","Projected_Risk_Score"],
    barmode="group"
)

st.plotly_chart(fig,width="stretch")

# -------------------------------------------------
# DRIVER ANALYSIS
# -------------------------------------------------
st.subheader("Risk Driver Analysis")

row = df[df["Country"]==focus_country].iloc[0]

d1,d2,d3,d4 = st.columns(4)

d1.metric("Country",focus_country)
d2.metric("Current Risk",row["Total_Risk_Score"])
d3.metric("Projected Risk",row["Projected_Risk_Score"])
d4.metric("Primary Driver",row["Main_Risk_Driver"])

driver_df = pd.DataFrame({
    "Component":["Trade Contribution","Energy Contribution"],
    "Value":[row["Trade_Contribution"],row["Energy_Contribution"]]
})

fig_driver = px.bar(
    driver_df,
    x="Component",
    y="Value"
)

st.plotly_chart(fig_driver,width="stretch")

st.divider()

# -------------------------------------------------
# TOP RISK COUNTRIES
# -------------------------------------------------
st.subheader("Top Risk Countries")

st.dataframe(
    df[[
        "Country",
        "Total_Risk_Score",
        "Projected_Risk_Score",
        "Risk_Category",
        "Risk_Trend",
        "Main_Risk_Driver"
    ]].head(10),
    width="stretch"
)

st.divider()

# -------------------------------------------------
# AI MONITORING
# -------------------------------------------------
st.subheader("AI Monitoring")

a1,a2 = st.columns(2)

with a1:

    st.metric("Model Accuracy",
    round(accuracy,2) if accuracy else "N/A")

    st.dataframe(
        df[[
            "Country",
            "Risk_Category",
            "Predicted_Risk_Category"
        ]].head(10),
        width="stretch"
    )

with a2:

    fig_imp = px.bar(
        importance,
        x="Feature",
        y="Importance"
    )

    st.plotly_chart(fig_imp,width="stretch")

st.divider()

# -------------------------------------------------
# GLOBAL MAP
# -------------------------------------------------
st.subheader("Global Risk Map")

fig_map = px.choropleth(
    df,
    locations="ISO3",
    color="Total_Risk_Score",
    hover_name="Country",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig_map,width="stretch")

st.divider()

# -------------------------------------------------
# DATA TABLE
# -------------------------------------------------
st.subheader("Full Dataset")

st.dataframe(df,width="stretch")