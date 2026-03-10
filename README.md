## Live Demo
[Open the dashboard](https://eu-trade-risk-dashboard-opdttpup9zzh7opacsqmki.streamlit.app/country_risk)
## GitHub Repository
[View the code](https://github.com/rajbhayani-sketch/eu-trade-risk-dashboard)
# AI Decision Support for EU Trade & Energy Risk

## Overview
This project is an interactive analytics dashboard built with Python and Streamlit to monitor European trade disruption and energy risk.

## Main Features
- EU country risk scoring
- Trade risk and energy risk analysis
- Total risk score calculation
- High / Medium / Low risk classification
- Executive KPI dashboard
- Country risk filtering
- Europe risk map
- Risk comparison charts

## Tools Used
- Python
- Streamlit
- Pandas
- Plotly

## Risk Model
The dashboard calculates:

Total Risk Score = 0.4 × Trade Risk + 0.6 × Energy Risk

Risk categories:
- High: score >= 70
- Medium: score >= 60 and < 70
- Low: score < 60

## Project Structure
- `app.py` → main dashboard
- `pages/country_risk.py` → country analysis page
- `data/eu_trade_energy_risk.csv` → dataset
- `requirements.txt` → Python packages

## How to Run
Activate environment:

`venv\Scripts\activate`

Run app:

`streamlit run app.py`

## Current Status
Working prototype with interactive dashboard, KPI summary, risk map, and filtering.