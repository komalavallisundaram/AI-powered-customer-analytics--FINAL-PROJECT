import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, silhouette_score
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

def load_csv(filename):
    file_path = f"{DATA_PATH}/{filename}"
    if os.path.exists(file_path):
        print(f"Loading {filename}...")
        return pd.read_csv(file_path)
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

categorical_cols = [col for col in df.select_dtypes(include=["object", "string"]).columns if df[col].nunique() < 50]
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
joblib.dump(encoder, f"{MODEL_PATH}/encoder.pkl")
print("✅ Categorical encoding done")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
print("✅ Scaling applied")

if "Churn" not in df.columns:
    df["Churn"] = np.random.randint(0, 2, size=len(df))
    print("⚠️ Dummy churn target created")

df.to_csv(f"{OUTPUT_PATH}/cleaneddata.csv", index=False)
print(f" Cleaned data saved to {OUTPUT_PATH}/cleaneddata.csv")

# 1️⃣ Churn Classifier
X = df.drop("Churn", axis=1).select_dtypes(include=[np.number])
y = df["Churn"]
X.columns = X.columns.str.replace('[', '', regex=False)\
                     .str.replace(']', '', regex=False)\
                     .str.replace('<', '', regex=False)\
                     .str.replace('>', '', regex=False)\
                     .str.replace(' ', '_', regex=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
churn_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
churn_model.fit(X_train, y_train)
print("📊 Churn Model ROC-AUC:", roc_auc_score(y_test, churn_model.predict_proba(X_test)[:,1]))
joblib.dump(churn_model, f"{MODEL_PATH}/churn_model.pkl")

# 2️⃣ Revenue Forecaster
y_rev = df["Net_Revenue_USD"]
X_rev = df.drop(["Churn", "Net_Revenue_USD"], axis=1).select_dtypes(include=[np.number])

X_rev.columns = X_rev.columns.str.replace('[', '', regex=False)\
                             .str.replace(']', '', regex=False)\
                             .str.replace('<', '', regex=False)\
                             .str.replace('>', '', regex=False)\
                             .str.replace(' ', '_', regex=False)

X_train, X_test, y_train, y_test = train_test_split(X_rev, y_rev, test_size=0.2, random_state=42)
revenue_model = XGBRegressor()
revenue_model.fit(X_train, y_train)
print("📊 Revenue Model RMSE:", np.sqrt(mean_squared_error(y_test, revenue_model.predict(X_test))))
joblib.dump(revenue_model, f"{MODEL_PATH}/revenue_model.pkl")

# 3️⃣ Behavioral Segmentation
X_seg = df.drop(["Churn", "Net_Revenue_USD"], axis=1).select_dtypes(include=[np.number]).fillna(0)

X_seg.columns = X_seg.columns.str.replace('[', '', regex=False)\
                             .str.replace(']', '', regex=False)\
                             .str.replace('<', '', regex=False)\
                             .str.replace('>', '', regex=False)\
                             .str.replace(' ', '_', regex=False)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_seg)
df["Cluster"] = clusters
joblib.dump(kmeans, f"{MODEL_PATH}/segmenter.pkl")
print("📊 Segmentation Silhouette Score:", silhouette_score(X_seg, clusters))

churn_probs = churn_model.predict_proba(X)[:, 1]
revenue_preds = revenue_model.predict(X_rev)
segment_map = {
    0: "High-Value Loyalists",
    1: "At-Risk",
    2: "Price-Sensitive",
    3: "Growth Potential (Upsell)"
}

fusion_profiles = []
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

fusion_path = f"{OUTPUT_PATH}/fusion_profiles.json"
with open(fusion_path, "w") as f:
    json.dump(fusion_profiles, f, indent=4)


joblib.dump(X.columns.tolist(), f"{MODEL_PATH}/feature_names.pkl")
print(f"✅ Fusion layer complete. Profiles saved to {fusion_path}")
