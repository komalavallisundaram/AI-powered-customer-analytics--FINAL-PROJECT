import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import shap

data_path = r"C:/Users/shenbagam/Downloads/FINALPROJECT/Outputs/smote_churn.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "CV Mean Accuracy": np.mean(cv_scores)
    }

print("📊 Model Evaluation Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

best_model = models["XGBoost"]
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap_df = pd.DataFrame({
    "Customer_Id": X_test.index,
    "Churn_Prediction": best_model.predict(X_test),
    "Churn_Probability": best_model.predict_proba(X_test)[:, 1],
    "SHAP_Explanation": [
        f"Churn influenced by {X_test.columns[np.argmax(np.abs(shap_values[i]))]} (synthetic SMOTE sample)"
        for i in range(len(X_test))
    ]
})

output_path = r"C:/Users/shenbagam/Downloads/FINALPROJECT/Outputs/Phase4_Churn_Predictions.csv"
shap_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ SHAP explanations saved at: {output_path} with shape {shap_df.shape}")
