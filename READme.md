# AI-Powered Customer Analytics Project

## Overview
This project is an end-to-end **customer analytics system** designed for SaaS CRM platforms (like Salesforce or Zoho).  
It combines machine learning models with generative AI to deliver actionable insights for customer success teams.

The pipeline predicts churn, forecasts revenue, segments customers into behavioral clusters, and fuses everything into unified profiles.  
Finally, a language model generates **executive-level insights** that are displayed in a Streamlit dashboard.

## Objectives
- **Churn Prediction (Classification)** → Identify customers at risk of leaving
- **Revenue Forecasting (Regression)** → Estimate next-quarter revenue
- **Customer Segmentation (Clustering)** → Group customers by behavior
- **Fusion Layer** → Build unified customer profiles
- **Executive Insights (LLM Integration)** → Generate summaries and recommended actions

## Dataset
The dataset is structured like a CRM warehouse:
- **Fact tables**: customers, transactions, usage, engagement
- **Dimension tables**: geography, industry, product

Features include:
- Demographics
- Subscription details
- Usage frequency
- Payment history
- Engagement scores
- Revenue metrics

## Tools & Skills
- **Python** (Pandas, NumPy)
- **Scikit-learn** (preprocessing, clustering)
- **XGBoost** (classification & regression)
- **KMeans** (segmentation)
- **SHAP** (model explainability)
- **Streamlit** (interactive dashboard)
- **Joblib** (model persistence)
- **Generative AI (Gemma-2B, Mistral, LLaMA-3)** via Ollama for executive insights

## Project Structure
finaalproject/
│
├── DataSet/                # Raw CSV files (dimensions and facts)
├── Models/                 # Trained ML models (XGBoost, KMeans, encoders, scalers)
├── Outputs/                # JSON outputs for dashboard
│   ├── fusion_profiles.json
│   ├── revenue_forecast.json
│   └── executive_insights.json
│
├── preprocess.py           # Data preprocessing, model training, JSON generation
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## Workflow
1. **Preprocess Data**  
   Run `preprocess.py` to:
   - Merge dimensions and facts
   - Train churn, revenue, and segmentation models
   - Save fusion profiles, forecasts, and insights into `Outputs`

