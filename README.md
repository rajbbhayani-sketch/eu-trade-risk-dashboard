AI Decision Support for EU Trade & Energy Risk

An interactive business intelligence dashboard built with Python and Streamlit to monitor trade disruption and energy dependency risk across European countries and key global partners.

The platform combines data analytics, machine learning, scenario simulation, and geographic visualization to support strategic monitoring of geopolitical and economic risk.

Live Dashboard

Access the live application:
https://eu-trade-risk-dashboard-opdttpup9zzh7opacsqmki.streamlit.app/

Project Overview

Global supply chains and energy dependencies expose countries to geopolitical and economic disruption.
This dashboard provides a decision-support tool to monitor risk exposure across countries and evaluate how trade and energy dependency influence national risk profiles.

The application enables users to:

Monitor country-level trade and energy risk indicators

Simulate different risk scenarios using adjustable weights

Project future risk levels

Identify dominant risk drivers

Compare countries across multiple risk dimensions

Visualize global risk exposure on an interactive map

Use machine learning to classify risk categories

The goal is to provide strategic insight for policy analysts, business planners, and risk managers.

Key Features
Executive Risk Dashboard

Portfolio risk summary

Strategic alert for highest-risk country

Top risk country ranking

Scenario simulation controls

Country Risk Analysis

Country-level risk breakdown

Trade vs energy risk contribution

Country benchmarking and comparison

Peer country analysis

Risk Projection

Future risk simulation using adjustable growth scenarios

Identification of rising or declining risk trends

AI Risk Classification

Machine learning model using Random Forest to classify country risk levels.

Model features include:

Trade risk

Energy risk

Energy dependency level

Geographic Risk Visualization

Interactive world map showing risk exposure by country.

Strategic Decision Support

The dashboard highlights:

dominant risk drivers

portfolio risk exposure

strategic recommendations

Technologies Used

Python
Streamlit
Pandas
Plotly
Scikit-learn

Additional tools:

GitHub for version control

Streamlit Cloud for deployment

Project Structure
eu-trade-risk-dashboard
│
├── app.py
│   Main executive dashboard
│
├── pages
│   └── 1_Country_Risk_Analysis.py
│       Country-level risk benchmarking
│
├── data
│   ├── eu_trade_energy_risk.csv
│   └── live_country_risk_data.csv
│
├── fetch_live_data.py
│   Script to generate/update the live dataset
│
├── requirements.txt
│   Python dependencies
│
└── README.md
How to Run Locally

Clone the repository:

git clone https://github.com/your-username/eu-trade-risk-dashboard.git
cd eu-trade-risk-dashboard

Install dependencies:

pip install -r requirements.txt

Run the dashboard:

streamlit run app.py
Example Use Cases

This dashboard can support:

Risk analysts evaluating geopolitical exposure

Business strategy teams monitoring trade disruption risk

Energy policy researchers studying dependency patterns

Supply chain planners assessing country-level vulnerability

Future Improvements

Potential enhancements include:

Integration with real-time economic data APIs

Advanced forecasting models

Automated risk alerts

Exportable analytics reports

Additional macroeconomic indicators

Author

Developed as a data analytics and decision-support project demonstrating:

data analysis

machine learning

dashboard development

business intelligence visualization

License

This project is for educational and portfolio purposes.
