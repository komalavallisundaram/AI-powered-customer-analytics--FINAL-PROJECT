import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv(r"C:\Users\shenbagam\Downloads\FINALPROJECT\revenue\fiscal_revenue_usd_summary.csv")

df["Year"] = df["Fiscal_Quarter"].str[-4:].astype(int)
df["Quarter"] = df["Fiscal_Quarter"].str[1].astype(int)

train_df = df[df["Year"] == 2023]
test_df = df[df["Year"] == 2024]

X_train = train_df[["Year", "Quarter"]]
y_train = train_df["Revenue_USD"]
X_test = test_df[["Year", "Quarter"]]

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
pred_gbr = gbr.predict(X_test)

xgb = XGBRegressor(random_state=42, n_estimators=200)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)

y_pred_train = gbr.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae = mean_absolute_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)

print(f"📊 Gradient Boosting Model Performance:")
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

pred_df = test_df.copy()
pred_df["Predicted_Revenue_GBR"] = pred_gbr
pred_df["Predicted_Revenue_XGB"] = pred_xgb
pred_df.to_csv(r"C:\Users\shenbagam\Downloads\FINALPROJECT\revenue\predicted_revenue_2024.csv", index=False, encoding="utf-8-sig")

print("✅ Predictions saved for 2024 quarters.")
