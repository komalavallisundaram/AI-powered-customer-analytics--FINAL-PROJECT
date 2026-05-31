# AI-Powered Customer Analytics Project
## Overview
This project is about building an end-to-end customer analytics system for SaaS CRM platforms (like Salesforce or Zoho).  
The idea is to use machine learning to predict churn, forecast revenue, segment customers, and then combine everything into a single profile.  
Finally, we use a language model to generate executive-level insights that can be shown in a dashboard.

## Objectives
- Predict customer churn (classification)
- Forecast next-quarter revenue (regression)
- Segment customers into behavioral clusters (clustering)
- Build unified customer profiles (fusion layer)
- Generate automated insights with GPT (LLM integration)

## Dataset
The dataset is structured like a CRM warehouse:
- **Fact tables**: customers, transactions, usage, engagement
- **Dimension tables**: geography, industry, product
- Features include demographics, subscription details, usage frequency, payment history, engagement scores, and revenue.

## Tools & Skills
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- KMeans clustering
- SHAP for explainability
- Streamlit for dashboard
- OpenAI API for LLM insights
