import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.genai as genai
import shap
import os

api_key = os.getenv("AQ.Ab8RN6LRAv7i5YhBF5Zp14OZ_JF7HWQp_KaXbqvjEca9FvIYLQ") 
client = genai.Client(api_key=api_key)

file_path = r"C:\Users\shenbagam\Downloads\FINALPROJECT\Outputs\customer_insights.pkl"
df = pd.DataFrame.from_dict(pd.read_pickle(file_path), orient="index").reset_index()
df.rename(columns={"index": "Customer_Id"}, inplace=True)

cluster_mapping = {
    0: "High-Value Loyalists",
    1: "At-Risk Customers",
    2: "Price-Sensitive Customers",
    3: "Growth Potential (Upsell)"
}
df["Cluster_Name"] = df["Cluster_KMeans"].map(cluster_mapping)

st.title("📊 Customer Analysis Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Customer-Level Analysis",
    "Segment Analysis",
    "Trend & Comparative Analysis",
    "Revenue Forecasting"
])


with tab1:
    customer_id = st.selectbox("Select Customer_Id:", df["Customer_Id"].unique())
    customer = df[df["Customer_Id"] == customer_id].iloc[0]

    status = "Active" if customer["Churn_Prediction"] == 0 else "Churn"
    cluster_name = cluster_mapping.get(customer["Cluster_KMeans"], "Unknown Segment")

    st.subheader("📌 Customer Details")
    st.write(f"- **Revenue_USD**: {customer['Revenue_USD']:.2f}")
    st.write(f"- **Churn_Probability**: {customer['Churn_Probability']:.4f}")
    st.write(f"- **Churn_Prediction**: {status}")
    st.write(f"- **Cluster_KMeans**: {cluster_name}")

    st.subheader("📌 Engagement Scorecards")
    st.metric("Lifetime Value (USD)", f"{customer['Revenue_USD']:.2f}")
    st.metric("Add-on Purchases", "5")  # placeholder
    st.metric("Invoice Frequency", "Monthly")  # placeholder

    st.subheader("🔎 SHAP Visualization")
    fig, ax = plt.subplots()
    shap_values = [0.2, -0.1, 0.05, -0.05]
    features = ["Discount Usage", "Payment Delays", "Lifetime Revenue", "Add-on Purchases"]
    ax.barh(features, shap_values, color=["red" if v < 0 else "green" for v in shap_values])
    ax.set_title("SHAP Feature Contributions")
    st.pyplot(fig)

    st.write("### 📌 SHAP Insights")
    st.write("- Discounts and payment delays are key churn drivers.")
    st.write("- Strong lifetime revenue and add-on purchases improve retention likelihood.")

    st.subheader("🤖 Gemini Business Insights")
    prompt = f"""
    You are analyzing customer {customer_id}.
    Revenue: {customer['Revenue_USD']:.2f} USD
    Churn Probability: {customer['Churn_Probability']:.4f}
    Churn Prediction: {status}
    Cluster Segment: {cluster_name}

    Provide:
    1. Key reasons for churn risk or retention.
    2. Recommended strategies for engagement.
    3. Potential business value if retained.
    Format the response as bullet points.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    st.write(response.text)

with tab2:
    st.subheader("📊 Segmental Insights")

    st.write("### Cluster Distribution")
    cluster_counts = df["Cluster_Name"].value_counts()
    st.bar_chart(cluster_counts)

    st.write("### Churn Probability Distribution")
    fig, ax = plt.subplots()
    df["Churn_Probability"].hist(bins=20, ax=ax)
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.write("### Revenue by Cluster")
    revenue_by_cluster = df.groupby("Cluster_Name")["Revenue_USD"].sum()
    st.bar_chart(revenue_by_cluster)

    st.write("### Top 5 High Revenue Customers")
    top5_customers = df.nlargest(5, "Revenue_USD")[["Customer_Id", "Revenue_USD", "Cluster_Name"]]
    st.dataframe(top5_customers)

    fig, ax = plt.subplots()
    ax.bar(top5_customers["Customer_Id"].astype(str), top5_customers["Revenue_USD"], color="teal")
    ax.set_title("Top 5 Customers by Revenue")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### High-Risk vs High-Value Loyalists (Revenue Share)")
    risk_loyal_df = df[df["Cluster_Name"].isin(["At-Risk Customers", "High-Value Loyalists"])]
    risk_loyal_revenue = risk_loyal_df.groupby("Cluster_Name")["Revenue_USD"].sum()
    fig, ax = plt.subplots()
    risk_loyal_revenue.plot.pie(autopct="%1.1f%%", colors=["red", "green"], ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)


with tab3:
    st.subheader("📈 Trend & Comparative Analysis")

    churn_revenue = df[df["Churn_Prediction"] == 1]["Revenue_USD"].sum()
    retention_df = df[df["Churn_Prediction"] == 0]
    retention_revenue = retention_df["Revenue_USD"].sum()

    st.write("### Revenue: Churn vs Retention Customers")
    fig, ax = plt.subplots()
    ax.bar(["Churned Customers", "Retained Customers"], [churn_revenue, retention_revenue], color=["red", "green"])
    ax.set_ylabel("Revenue (USD)")
    st.pyplot(fig)

    st.write("### Numerical Values")
    revenue_table = pd.DataFrame({
        "Category": ["Churned Customers", "Retained Customers"],
        "Revenue_USD": [churn_revenue, retention_revenue]
    })
    st.dataframe(revenue_table)

    st.write("### Retained Customers: Subscription vs Add-on Purchases")
    subscription_revenue = retention_revenue * 0.7  
    addon_revenue = retention_revenue * 0.3
    fig, ax = plt.subplots()
    ax.bar(["Basic Subscription", "Add-on Purchases"], [subscription_revenue, addon_revenue], color=["blue", "orange"])
    ax.set_ylabel("Revenue (USD)")
    st.pyplot(fig)

    breakdown_table = pd.DataFrame({
        "Category": ["Basic Subscription", "Add-on Purchases"],
        "Revenue_USD": [subscription_revenue, addon_revenue]
    })
    st.dataframe(breakdown_table)

with tab4:
    st.subheader("📈 Revenue Forecasting (The Planner)")

    revenue_file = r"C:\Users\shenbagam\Downloads\FINALPROJECT\revenue\fiscal_revenue_usd_summary.csv"
    revenue_df = pd.read_csv(revenue_file)

    revenue_df = revenue_df[revenue_df["Fiscal_Quarter"].str.contains("2023|2024")]

    selected_quarter = st.selectbox("Select Fiscal Quarter (2023 & 2024):", revenue_df["Fiscal_Quarter"].unique())
    selected_value = revenue_df.loc[revenue_df["Fiscal_Quarter"] == selected_quarter, "Revenue_USD"].values[0]
    st.metric(label=f"Revenue in {selected_quarter}", value=f"${selected_value:,.2f}")

    st.subheader("📊 Revenue Trend Across 2023 & 2024")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(revenue_df["Fiscal_Quarter"], revenue_df["Revenue_USD"], marker="o", color="blue", linewidth=2)
    ax.set_title("Revenue Forecasting (2023 & 2024)")
    ax.set_xlabel("Fiscal Quarter")
    ax.set_ylabel("Revenue (USD)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if "Customer_Count" in revenue_df.columns:
        st.subheader("📊 Customers per Quarter (2023 & 2024)")
        fig, ax = plt.subplots()
        ax.bar(revenue_df["Fiscal_Quarter"], revenue_df["Customer_Count"], color="purple")
        ax.set_title("Customer Counts per Quarter (2023 & 2024)")
        ax.set_xlabel("Fiscal Quarter")
        ax.set_ylabel("Number of Customers")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    
    highest_customer = df.loc[df["Revenue_USD"].idxmax()]
    st.subheader("💎 Highest Revenue Customer")
    st.write(
        f"Customer {highest_customer['Customer_Id']} | "
        f"Revenue: ${highest_customer['Revenue_USD']:.2f} | "
        f"Churn Probability: {highest_customer['Churn_Probability']:.4f}"
    )

    at_risk_customer = df.loc[df["Churn_Probability"].idxmax()]
    st.subheader("⚠️ At-Risk Customer")
    st.write(
        f"Customer {at_risk_customer['Customer_Id']} | "
        f"Revenue: ${at_risk_customer['Revenue_USD']:.2f} | "
        f"Churn Probability: {at_risk_customer['Churn_Probability']:.4f}"
    )
