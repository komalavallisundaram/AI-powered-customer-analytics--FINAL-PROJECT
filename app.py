import streamlit as st
import pandas as pd
import json
import os
import google.genai as genai   
import plotly.express as px
import time

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_chat(prompt, model="models/gemini-2.5-flash"):
    """Call Gemini model with retry logic."""
    for attempt in range(3):  
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg:
                wait = (attempt + 1) * 5
                time.sleep(wait)
                continue
            if "429" in error_msg:
                return "⚠️ Gemini quota exceeded. Please wait ~15 seconds and try again, or upgrade your plan."
            return f"⚠️ Error contacting Gemini: {error_msg}"
    return "⚠️ Gemini service unavailable after multiple retries."

fusion_path = "C:/Users/shenbagam/Downloads/finaalproject/Outputs/fusion_profiles.json"
forecast_path = "C:/Users/shenbagam/Downloads/finaalproject/Outputs/revenue_forecast.json"

if os.path.exists(fusion_path):
    with open(fusion_path, "r") as f:
        fusion_profiles = json.load(f)
    df = pd.DataFrame(fusion_profiles)
    df["Revenue_Value"] = df["Projected_Revenue_USD"].apply(
        lambda x: float(str(x).replace("$", "").replace(",", "")) if str(x).strip() != "" else 0
    )
    df["Customer"] = df["Customer"].apply(lambda x: str(x).replace("Customer_", "Customer "))
else:
    st.error("Fusion profiles file not found. Please run preprocess.py first.")
    st.stop()

if os.path.exists(forecast_path):
    with open(forecast_path, "r") as f:
        revenue_forecast = json.load(f)
else:
    quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023",
                "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
    historical_revenue = [1696, 9932, 10485, 14535]
    predicted_revenue = [10058, 22624, 23644, 24664]

    revenue_forecast = []
    for i, q in enumerate(quarters):
        if i < len(historical_revenue):
            revenue_forecast.append({"Quarter": q, "Revenue": historical_revenue[i], "Type": "Historical"})
        else:
            revenue_forecast.append({"Quarter": q, "Revenue": predicted_revenue[i - len(historical_revenue)], "Type": "Forecast"})

    os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
    with open(forecast_path, "w") as f:
        json.dump(revenue_forecast, f, indent=4)

rev_df = pd.DataFrame(revenue_forecast)

st.set_page_config(page_title="Generative AI Insights", layout="wide")
st.title("💡 Generative AI Insights Dashboard")
st.caption("Transforming structured customer and financial data into actionable recommendations.")

tab1, tab2, tab3 = st.tabs([
    "🔍 Individual Customer Insights",
    "📊 Risk-Level Group Insights",
    "💰 Revenue Forecasting (The Planner)"
])

with tab1:
    st.subheader("Customer-Level Insight")
    selected_id = st.selectbox("Select Customer ID", df["Customer_ID"].unique())

    if selected_id:
        row = df[df["Customer_ID"] == selected_id].iloc[0]
        name, risk, churn_prob, revenue, segment = (
            row["Customer"], row["Risk_Level"], row["Churn_Probability"],
            row["Projected_Revenue_USD"], row["Segment"]
        )

        rev_value = float(str(revenue).replace("$", "").replace(",", ""))
        revenue_for_prompt = f"${rev_value:,.0f}"

        prompt = f"""
        Customer: {name} ({segment})
        Risk Level: {risk}, Churn Probability: {churn_prob}, Projected Revenue: {revenue_for_prompt}.
        Provide 3 sections: Executive Summary, Recommended Action, Offer.
        """

        insight_text = gemini_chat(prompt)

        st.markdown("### Strategic Output")
        st.write(insight_text)

        st.markdown("### Customer Metrics (Pie Chart)")
        pie_data = pd.DataFrame({
            "Metric": ["Churn Probability (%)", "Projected Revenue ($)", "Risk Level Score"],
            "Value": [
                float(str(churn_prob).replace("%", "")),
                rev_value,
                1 if risk == "High Risk" else (0.5 if risk == "Moderate Risk" else 0.2)
            ]
        })
        fig = px.pie(pie_data, names="Metric", values="Value", title=f"Customer {selected_id} Metrics")
        st.plotly_chart(fig)


with tab2:
    st.subheader("Group-Level Insight by Risk Category")
    risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
    selected_risk = st.selectbox("Select Risk Level", risk_levels)

    filtered_df = df[df["Risk_Level"] == selected_risk].copy()
    filtered_df["Churn_Prob_Value"] = filtered_df["Churn_Probability"].apply(lambda x: float(str(x).replace("%", "")))

    st.dataframe(filtered_df[["Customer_ID", "Customer", "Churn_Probability", "Revenue_Value", "Segment"]])

    avg_churn = filtered_df["Churn_Prob_Value"].mean()
    total_revenue = filtered_df["Revenue_Value"].sum()

    st.markdown(f"**Average Churn Probability:** {avg_churn:.1f}%")
    st.markdown(f"**Total Projected Revenue:** ${total_revenue:,.0f}")

    st.markdown("### Top 10 High Probable Churn Customers (Bar Chart)")
    top10 = filtered_df.sort_values(by="Churn_Prob_Value", ascending=False).head(10)
    fig_bar = px.bar(top10, x="Customer", y="Churn_Prob_Value",
                     hover_data=["Revenue_Value"],
                     title=f"Top 10 {selected_risk} Customers - Churn Probability")
    st.plotly_chart(fig_bar)

with tab3:
    st.subheader("Quarter-Wise Revenue Forecasting (The Planner)")

    quarters = rev_df["Quarter"].unique()
    selected_quarter = st.selectbox("Select Quarter", quarters)

    row = rev_df[rev_df["Quarter"] == selected_quarter].iloc[0]
    revenue_value, data_type = row["Revenue"], row["Type"]

    st.markdown(f"### {selected_quarter} Revenue Overview")
    st.markdown(f"**Type:** {data_type}")
    st.metric(label=f"Expected Revenue for {selected_quarter}", value=f"${revenue_value:,.0f}")

    prompt_revenue = f"""
    Quarter: {selected_quarter}, Type: {data_type}, Expected Revenue: ${revenue_value:,.0f}.
    Provide a 2–3 sentence executive summary highlighting financial outlook and planning recommendations.
    """

    revenue_insight = gemini_chat(prompt_revenue)

    st.success("💡 Quarterly Revenue Insight")
    st.write(revenue_insight)

    st.line_chart(rev_df.set_index("Quarter")["Revenue"])
