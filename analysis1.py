import pandas as pd
import pickle

revenue_file = r"C:\Users\shenbagam\Downloads\FINALPROJECT\revenue\cleaned\Customer_Total_Revenue_USD_cleaned.csv"
churn_file = r"C:\Users\shenbagam\Downloads\FINALPROJECT\Outputs\Phase4_Churn_Predictions.csv"
cluster_file = r"C:\Users\shenbagam\Downloads\FINALPROJECT\Segment\clustered_features.csv"
output_pkl = r"C:\Users\shenbagam\Downloads\FINALPROJECT\Outputs\customer_insights.pkl"

df_revenue = pd.read_csv(revenue_file)
df_churn = pd.read_csv(churn_file)
df_cluster = pd.read_csv(cluster_file)

print("Revenue Data Sample:\n", df_revenue.head())
print("Churn Predictions Sample:\n", df_churn.head())
print("Clustered Features Sample:\n", df_cluster.head())

merged_df = df_revenue.merge(
    df_churn[["Customer_Id", "Churn_Prediction", "Churn_Probability"]],
    on="Customer_Id",
    how="inner"
).merge(
    df_cluster[["Customer_Id", "Cluster_KMeans"]],
    on="Customer_Id",
    how="inner"
)

print("Merged Data Sample:\n", merged_df.head())

agg_df = merged_df.groupby("Customer_Id").agg({
    "Revenue_USD": "sum",
    "Churn_Probability": "mean",
    "Churn_Prediction": lambda x: x.mode()[0] if not x.mode().empty else None,
    "Cluster_KMeans": lambda x: x.mode()[0] if not x.mode().empty else None
}).reset_index()

print("Aggregated Data Sample:\n", agg_df.head())

customer_dict = agg_df.set_index("Customer_Id").to_dict(orient="index")

with open(output_pkl, "wb") as f:
    pickle.dump(customer_dict, f)

print(f"📂 Dictionary saved as pickle at {output_pkl}")
