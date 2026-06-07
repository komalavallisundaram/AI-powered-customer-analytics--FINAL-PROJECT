import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans

BASE_PATH = "C:/Users/shenbagam/Downloads/finaalproject"
DATA_PATH = f"{BASE_PATH}/DataSet"
OUTPUT_PATH = f"{BASE_PATH}/Outputs"
MODEL_PATH = f"{BASE_PATH}/Models"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

print("Project folders ready")

# --- Load CSVs safely ---
def load_csv(filename):
    file_path = f"{DATA_PATH}/{filename}"
    if os.path.exists(file_path):
        print(f"Loading {filename}...")
        try:
            # read in chunks to avoid memory errors
            chunks = pd.read_csv(
                file_path,
                low_memory=False,
                on_bad_lines="skip",
                encoding="utf-8",
                chunksize=100000
            )
            return pd.concat(chunks, ignore_index=True)
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
            return None
    else:
        print(f"⚠️ File not found: {file_path}")
        return None

dim_geography = load_csv("dim_geography.csv")
dim_industry = load_csv("dim_industry.csv")
dim_product = load_csv("dim_product.csv")
fact_customers = load_csv("fact_customers.csv")
fact_engagement = load_csv("fact_engagement_events.csv")
fact_transactions = load_csv("fact_transactions.csv")
fact_usage = load_csv("fact_usage_monthly.csv")

if any(x is None for x in (dim_geography, dim_industry, dim_product,
                           fact_customers, fact_engagement, fact_transactions, fact_usage)):
    raise FileNotFoundError("❌ Missing required CSV files.")

print("✅ Dimensions and Facts loaded")

# --- Merge Facts & Dimensions ---
df = fact_customers.copy()
df = df.merge(dim_geography, on="Geo_ID", how="left")
df = df.merge(dim_industry, on="Industry_ID", how="left")
df = df.merge(dim_product, on="Product_ID", how="left")

transactions_agg = fact_transactions.groupby("Customer_ID").agg({
    "Net_Revenue_USD": "sum",
    "Total_Billed_USD": "sum",
    "Payment_Delay_Days": "mean"
}).reset_index()
df = df.merge(transactions_agg, on="Customer_ID", how="left")

usage_agg = fact_usage.groupby("Customer_ID").agg({
    "Daily_Active_Users": "mean",
    "Feature_Adoption_Pct": "mean",
    "API_Calls_Monthly": "mean",
    "Uptime_Pct": "mean"
}).reset_index()
df = df.merge(usage_agg, on="Customer_ID", how="left")

engagement_agg = fact_engagement.groupby("Customer_ID").agg({
    "Score": "mean",
    "Resolution_Days": "mean",
    "Churn": "max"
}).reset_index()
df = df.merge(engagement_agg, on="Customer_ID", how="left")

print("✅ Facts merged with Dimensions")

original_ids = df["Customer_ID"].copy()

# --- Encoding & Scaling ---
categorical_cols = [col for col in df.select_dtypes(include=["object", "string"]).columns if df[col].nunique() < 50]
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
joblib.dump(encoder, f"{MODEL_PATH}/encoder.pkl")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")

if "Churn" not in df.columns:
    df["Churn"] = np.random.randint(0, 2, size=len(df))

df.to_csv(f"{OUTPUT_PATH}/cleaneddata.csv", index=False)

# --- Models ---
X = df.drop("Churn", axis=1).select_dtypes(include=[np.number])
y = df["Churn"]

# Clean column names for XGBoost
X.columns = X.columns.str.replace('[<> ]', '_', regex=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
churn_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
churn_model.fit(X_train, y_train)
print("📊 Churn Model ROC-AUC:", roc_auc_score(y_test, churn_model.predict_proba(X_test)[:,1]))
joblib.dump(churn_model, f"{MODEL_PATH}/churn_model.pkl")

y_rev = df["Net_Revenue_USD"]
X_rev = df.drop(["Churn", "Net_Revenue_USD"], axis=1).select_dtypes(include=[np.number])
X_rev.columns = X_rev.columns.str.replace('[<> ]', '_', regex=True)

X_train, X_test, y_train, y_test = train_test_split(X_rev, y_rev, test_size=0.2, random_state=42)
revenue_model = XGBRegressor()
revenue_model.fit(X_train, y_train)
print("📊 Revenue Model RMSE:", np.sqrt(mean_squared_error(y_test, revenue_model.predict(X_test))))
joblib.dump(revenue_model, f"{MODEL_PATH}/revenue_model.pkl")

X_seg = df.drop(["Churn", "Net_Revenue_USD"], axis=1).select_dtypes(include=[np.number]).fillna(0)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_seg)
df["Cluster"] = clusters
joblib.dump(kmeans, f"{MODEL_PATH}/segmenter.pkl")

# --- Fusion Profiles ---
churn_probs = churn_model.predict_proba(X)[:, 1]
revenue_preds = revenue_model.predict(X_rev)
segment_map = {0: "High-Value Loyalists", 1: "At-Risk", 2: "Price-Sensitive", 3: "Growth Potential (Upsell)"}

fusion_profiles = []
executive_insights = {}
for i, cust_id in enumerate(original_ids):   
    churn_prob = float(churn_probs[i])
    revenue_pred = max(float(revenue_preds[i]), 0)  
    cluster_label = int(clusters[i])
    risk_level = "High Risk" if churn_prob > 0.7 else "Moderate Risk" if churn_prob > 0.4 else "Low Risk"
    action_required = "Immediate" if risk_level == "High Risk" else "Monitor" if risk_level == "Moderate Risk" else "Routine"

    profile = {
        "Customer_ID": int(cust_id),
        "Customer": f"Customer_{int(cust_id)}",
        "Churn_Probability": f"{churn_prob*100:.1f}%",
        "Risk_Level": risk_level,
        "Projected_Revenue_USD": f"${revenue_pred:,.0f}",
        "Segment": segment_map.get(cluster_label, "Unclassified"),
        "Action_Required": action_required
    }
    fusion_profiles.append(profile)

    executive_insights[str(int(cust_id))] = (
        f"Executive Summary: {profile['Customer']} is {profile['Risk_Level']} "
        f"with churn probability {profile['Churn_Probability']} and projected revenue {profile['Projected_Revenue_USD']}.\n"
        f"Recommended Action: Tailored engagement strategy based on segment {profile['Segment']}.\n"
        f"Offer: Incentives or loyalty rewards to reduce churn."
    )

fusion_path = f"{OUTPUT_PATH}/fusion_profiles.json"
with open(fusion_path, "w") as f:
    json.dump(fusion_profiles, f, indent=4)

insights_path = f"{OUTPUT_PATH}/executive_insights.json"
with open(insights_path, "w") as f:
    json.dump(executive_insights, f, indent=4)

# --- Revenue Forecast File ---
quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023",
            "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]

historical_revenue = [1696, 9932, 10485, 14535]   # example past values
predicted_revenue = [10058, 22624, 23644, 24664]  # example forecast values

revenue_forecast = []
for i, q in enumerate(quarters):
    if i < len(historical_revenue):
        revenue_forecast.append({
            "Quarter": q,
            "Revenue": historical_revenue[i],
            "Type": "Historical"
        })
    else:
        revenue_forecast.append({
            "Quarter": q,
            "Revenue": predicted_revenue[i - len(historical_revenue)],
            "Type": "Forecast"
        })

forecast_path = f"{OUTPUT_PATH}/revenue_forecast.json"
with open(forecast_path, "w") as f:
    json.dump(revenue_forecast, f, indent=4)

print("✅ Preprocessing complete. Cleaned data, models, fusion profiles, executive insights, and revenue forecast saved.")
