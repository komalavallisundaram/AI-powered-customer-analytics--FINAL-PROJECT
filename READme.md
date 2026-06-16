# Enterprise AI-Powered Customer Analytics & Strategic Insight Framework

## Overview
This project develops an **AI-powered framework** for CRM platforms such as **Salesforce** and **Zoho**, enabling organizations to:
- Predict customer churn with actionable precision.
- Forecast quarterly revenue for strategic planning.
- Segment customers into meaningful behavioral clusters.
- Generate executive-level insights using Generative AI.

The framework bridges the gap between **technical analytics** and **business decision-making**, delivering measurable impact in retention, revenue growth, and personalized marketing.

## Dataset Description
The framework integrates multiple structured datasets:
- `dim_geography.csv` – Customer location data.
- `dim_industry.csv` – Industry classification.
- `dim_product.csv` – Product categories.
- `fact_customers.csv` – Customer master records.
- `fact_engagement_events.csv` – Interaction logs.
- `fact_transactions.csv` – Transactional purchase history.
- `fact_usage_monthly.csv` – Monthly usage metrics.

### Preprocessing
- Missing value imputation  
- Encoding categorical variables  
- Scaling numerical features  
- Feature engineering (e.g., CLV, engagement score)  

## Methodology
The framework consists of four analytical modules:

1. **Churn Prediction**  
   - Model: XGBoost  
   - Techniques: SMOTE for class imbalance  
   - Features: Engagement frequency, support tickets, product usage  

2. **Revenue Forecasting**  
   - Models: Random Forest, XGBoost  
   - Features: Lag variables, rolling averages  
   - Goal: Predict next-quarter revenue  

3. **Customer Segmentation**  
   - Model: KMeans clustering  
   - Validation: Silhouette Score, Davies-Bouldin Index  
   - Outcome: 4 distinct customer clusters  

4. **Executive Insights**  
   - GPT-based LLM generates automated summaries and recommendations  

## Model Evaluation
**Churn Prediction**
- Accuracy: 87%  
- Precision: 0.82  
- Recall: 0.79  
- ROC-AUC: 0.91  

**Revenue Forecasting**
- MAE: 1.25  
- RMSE: 2.10  
- R²: 0.89  

**Customer Segmentation**
- Optimal clusters: 4  
- Silhouette Score: 0.62  

## Interpretations
- High churn risk concentrated among low-engagement customers.  
- Forecasts align closely with actuals, supporting proactive planning.  
- Segmentation reveals distinct groups: loyalists, churners, growth-potential, inactive accounts.  

## Business Impact
- **Retention**: Targeted campaigns reduce churn.  
- **Revenue Growth**: Forecasting enables upsell/cross-sell strategies.  
- **Segmentation**: Personalized marketing improves conversion rates.  
- **Executive Insights**: Automated reporting accelerates decision-making.  

## Future Improvements
- Incorporate deep learning for sequential engagement data.  
- Expand datasets with external market signals and sentiment.  
- Enhance LLM outputs with reinforcement learning.  
- Deploy as a cloud-native microservice integrated with Salesforce/Zoho APIs.  

## References
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*.  
- Breiman, L. (2001). *Random forests*.  
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data*.  
- Salesforce CRM Analytics Documentation.  
- Zoho CRM Analytics Documentation.  

## Author
**Komalavalli Sundaram**  
Data Annotator | Aspiring Data Analyst / Business Analyst / Security Analyst  
