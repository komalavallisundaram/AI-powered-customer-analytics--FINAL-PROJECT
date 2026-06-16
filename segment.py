import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

input_file = r"C:\Users\shenbagam\Downloads\FINALPROJECT\Segment\scaled_features.csv"
df = pd.read_csv(input_file)

if "customer_id" in df.columns:
    df.rename(columns={"customer_id": "Customer_ID"}, inplace=True)

X = df.drop(columns=["Customer_ID"], errors="ignore")

X_elbow = X.sample(frac=0.2, random_state=42)  
X_silhouette = X.sample(n=min(1000, len(X)), random_state=42) 

inertia, silhouette_scores = [], []
K_range = range(2, 7)

for k in K_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=500)
    kmeans.fit(X_elbow)
    inertia.append(kmeans.inertia_)
    labels = kmeans.predict(X_silhouette)
    silhouette_scores.append(silhouette_score(X_silhouette, labels))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method (sample)")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(K_range, silhouette_scores, marker="o", color="green")
plt.title("Silhouette Scores (sample)")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"✅ Best k chosen: {optimal_k}")

kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=500)
df["Cluster_KMeans"] = kmeans.fit_predict(X)

print("Davies-Bouldin Index:", davies_bouldin_score(X, df["Cluster_KMeans"]))
print("Calinski-Harabasz Score:", calinski_harabasz_score(X, df["Cluster_KMeans"]))

try:
    X_hier = X.sample(n=min(200, len(X)), random_state=42)  
    linked = linkage(X_hier, method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(linked, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram (sample of 200)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()
except Exception as e:
    print(f"⚠️ Skipping hierarchical clustering due to error: {e}")

dbscan = DBSCAN(eps=0.5, min_samples=10)
labels_dbscan = dbscan.fit_predict(X_silhouette)
print("DBSCAN sample cluster counts:", pd.Series(labels_dbscan).value_counts())

interpret_cols = [
    "engagement_score", "revenue_level", "churn_risk", "support_dependency",
    "daily_active_users", "feature_adoption_pct", "lifetime_revenue_usd",
    "renewal_probability", "support_tickets"
]
available_cols = [col for col in interpret_cols if col in df.columns]

if available_cols:
    cluster_summary = df.groupby("Cluster_KMeans")[available_cols].mean().reset_index()
    print("\nCluster Interpretation Summary:")
    print(cluster_summary)
else:
    print("\n⚠️ No interpretation columns available in scaled dataset.")

output_file = r"C:\Users\shenbagam\Downloads\FINALPROJECT\Segment\clustered_features.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ Full clustered dataset saved at: {output_file} with shape {df.shape}")
