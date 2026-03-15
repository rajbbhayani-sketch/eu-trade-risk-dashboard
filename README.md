EU Trade Risk Intelligence Platform

An interactive risk intelligence dashboard designed to monitor trade exposure, energy dependency, and projected country risk across Europe and key global partners.

The platform provides executives and analysts with a real-time view of macro-economic risk drivers, enabling scenario analysis and strategic monitoring of high-risk markets.

Overview

The EU Trade Risk Intelligence Platform combines trade exposure indicators with energy dependency data to estimate a country-level risk score.

Users can adjust risk weights and simulate future risk scenarios to understand how changes in global trade or energy markets could impact country exposure.

The dashboard includes:

Executive risk monitoring

Scenario-based risk simulations

Country risk rankings

Risk heatmaps

Geographic risk visualization

Key Features
Executive Risk Dashboard

Provides high-level KPIs including:

Average portfolio risk

Projected future risk

Number of high-risk markets

Highest risk country

Data coverage

Designed to give decision makers a quick strategic overview of risk exposure.

Scenario Risk Controls

Interactive sliders allow users to adjust:

Trade risk weight

Energy dependency weight

Future risk growth assumptions

Country-specific analysis

This enables what-if simulations to test how different economic conditions affect risk levels.

Risk Market Ranking

A visual comparison of top risk countries based on total risk score.

Helps identify markets requiring:

strategic monitoring

contingency planning

risk mitigation.

Risk Heatmap

A heatmap visualization highlighting how trade risk and energy risk contribute to total country risk.

This helps analysts identify structural risk drivers across countries.

Geographic Risk View

A global map showing the geographic distribution of trade and energy risk exposure.

Useful for understanding regional concentration of economic vulnerabilities.

Risk Model

The platform estimates total country risk using a weighted scoring approach.

Total Risk Score =
(Trade Risk × Trade Weight)
+ (Energy Risk × Energy Weight)

Users can dynamically adjust the weights to simulate different strategic priorities.

Example:

Trade Weight = 0.40
Energy Weight = 0.60

Future risk projections are calculated using a configurable growth assumption.

Data Sources

The platform uses publicly available macroeconomic indicators including:

World Bank trade data

Energy dependency indicators

Country-level economic metrics

Data is processed and aggregated into a structured dataset used by the dashboard.

Technology Stack

Python

Streamlit

Pandas

NumPy

Plotly

Scikit-learn

Project Structure
eu-trade-risk-dashboard
│
├── app.py                       # Main Streamlit dashboard
├── pages/
│   └── Country_Risk_Analysis.py # Country-level risk insights
│
├── fetch_live_data.py           # Data generation script
│
├── data/
│   ├── live_country_risk_data.csv
│   └── risk_history.csv
│
├── requirements.txt
└── README.md
Installation

Clone the repository

git clone https://github.com/rajbbhayani-sketch/eu-trade-risk-dashboard.git
cd eu-trade-risk-dashboard

Install dependencies

pip install -r requirements.txt

Run the Streamlit application

streamlit run app.py
Live Dashboard

Access the deployed application:

https://eu-trade-risk-dashboard-opdttpup9zzh7opacsqmki.streamlit.app/

Future Improvements

Potential enhancements include:

Integration with additional economic indicators

Time-series forecasting of risk trends

Expanded geopolitical risk factors

Automated data pipeline for real-time updates

Advanced machine learning risk prediction models

License

This project is shared for educational and demonstration purposes.
All rights reserved by the author.
