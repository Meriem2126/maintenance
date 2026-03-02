# factory Predictive Maintenance Dashboard

## Overview
The Henkel Predictive Maintenance Dashboard is an AI-powered web application built with Streamlit to monitor industrial machinery health and predict potential failures before they occur.

The system leverages Machine Learning models to:

Predict Remaining Useful Life (RUL)

Detect maintenance requirements

Identify anomalies in sensor data

Estimate carbon emissions and financial impact

This solution helps reduce downtime, optimize maintenance schedules, and support sustainability goals
## Key Features
1. **Home:** Dashboard overview

System health status

Key performance metrics

Summary statistics (records, average RUL, carbon levels)
2. **Historical Data:** View complete machinery dataset

Display recent records

Statistical summary

Dataset information (rows, columns, memory usage)
3. **Input Data:** SUsers can:

Generate random sensor values

Manually adjust sensor readings using sliders

Input features:

Sensor 1

Sensor 2

Sensor 3

Operational Hours

Carbon Sensor
4. **Results:** After submitting input data, the system provides:

⏱️ Remaining Useful Life Prediction

⚙️ Maintenance Status (Needs Maintenance / Normal)

🔍 Anomaly Detection

🌱 Estimated CO₂ Emission

💰 Estimated Financial Impact

Smart alerts are displayed when:

Maintenance is required

An anomaly is detected

Carbon emissions are high
## 5.Visualizations: Interactive data visualizations including:

Histograms of sensor readings

Scatter plots (Operational Hours vs Sensors)

Line chart of RUL over time

Overlay of user input on charts
## 6.Machine Status

Cluster-based health monitoring (KMeans clustering)

List of machines requiring maintenance
## 7.Reports

Users can generate downloadable CSV reports:

Full dataset

Maintenance-required machines

Cluster overview
## Machine Learning Models Used

The application trains three models:

Random Forest Regressor

Predicts Remaining Useful Life (RUL)

Random Forest Classifier

Predicts maintenance requirement

KMeans Clustering

Detects anomalies and machine health clusters

All models are trained dynamically when the app runs
## Installation
1. Clone the repository from GitHub: `git clone https://github.com/your_username/predictive-maintenance-dashboard.git`
2. Install the required Python packages: `pip install -r requirements.txt`
3. Run the Streamlit application: `streamlit run app.py`

## Usage
1. Launch the application using the provided installation instructions.
2. Navigate through the different sections using the sidebar menu:
   - **Home:** Provides a brief introduction to the application.
   - **Historical Data:** Displays historical sensor data and operational hours.
   - **Input Data:** Allows users to submit input data for prediction.
   - **Results:** Shows predictions for RUL, maintenance status, and anomaly detection based on the input data.
   - **Visualizations:** Visualizes historical data and optionally overlays generated input values.
3. Follow the on-screen instructions to interact with the application, submit input data, and view predictions.






