import streamlit as st
import pandas as pd
import json
from openai import OpenAI

client = OpenAI()

# -------------------------------
# Load Fusion Profiles
# -------------------------------
fusion_path = "C:/Users/shenbagam/Downloads/finaalproject/Outputs/fusion_profiles.json"
with open(fusion_path, "r") as f:
    fusion_profiles = json.load(f)

df = pd.DataFrame(fusion_profiles)

# -------------------------------
# Data Cleaning
# -------------------------------
df["Revenue_Value"] = df["Projected_Revenue_USD"].apply(
    lambda x: max(float(str(x).replace("$", "").replace(",", "")), 0)
)
df["Customer"] = df["Customer"].apply(lambda x: str(x).replace("Customer_", "Customer "))

# -------------------------------
# Dashboard Title
# -------------------------------
st.set_page_config(page_title="Generative AI Insights", layout="wide")
st.title("💡 The Voice of Data: Generative AI Insights")
st.caption("Transforming structured customer data into actionable recommendations.")

# ============================================================
# DROPDOWN INPUT
# ============================================================
selected_id = st.selectbox("Select Customer ID", df["Customer_ID"].unique())

if selected_id:
    customer_row = df[df["Customer_ID"] == selected_id].iloc[0]
    name = customer_row["Customer"]
    risk = customer_row["Risk_Level"]
    churn_prob = customer_row["Churn_Probability"]
    revenue = customer_row["Projected_Revenue_USD"]
    segment = customer_row["Segment"]

    # ============================================================
    # Revenue Pre-processing for Prompt
    # ============================================================
    rev_value = float(str(revenue).replace("$", "").replace(",", ""))
    if rev_value <= 0:
        revenue_for_prompt = "Minimal projected revenue impact"
    else:
        revenue_for_prompt = f"${rev_value:,.0f}"

    # ============================================================
    # LLM PROMPT ENGINEERING
    # ============================================================
    prompt = f"""
    You are a senior business strategy consultant preparing insights for an executive dashboard.
    Analyze the following customer profile and generate a polished, professional summary.

    Customer: {name}
    Customer ID: {selected_id}
    Risk Level: {risk}
    Churn Probability: {churn_prob}
    Projected Revenue: {revenue_for_prompt}
    Segment: {segment}

    Provide output in three clearly labeled sections:
    1. Executive Summary – concise, boardroom-ready observation.
    2. Recommended Action – actionable recommendation for account managers or leadership.
    3. Offer – tailored incentive or engagement plan to maximize retention and value.

    Tone: Formal, executive, and strategic. Keep each section 2–3 sentences maximum.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a business analyst generating customer insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.4
        )
        insight_text = response.choices[0].message.content
    except Exception:
        insight_text = f"""
        Executive Summary: {name} ({segment}) shows {risk} with churn probability {churn_prob}.
        Recommended Action: Maintain proactive engagement and monitor satisfaction.
        Offer: Provide personalized loyalty incentives ({revenue_for_prompt}).
        """

    # ============================================================
    # Post-process GPT output to bold section headers and add icons
    # ============================================================
    def format_insight_text(text):
        replacements = {
            "Executive Summary:": "📊 <b>Executive Summary:</b>",
            "Recommended Action:": "🛡️ <b>Recommended Action:</b>",
            "Offer:": "💰 <b>Offer:</b>"
        }
        for key, val in replacements.items():
            text = text.replace(key, val)
        return text

    insight_text = format_insight_text(insight_text)

    # Split into sections line by line
    sections = insight_text.split("\n")
    sections = [s.strip() for s in sections if s.strip()]

    # ============================================================
    # DISPLAY STRATEGIC OUTPUT IN SMALLER BOX
    # ============================================================
    st.markdown(
        """
        <div style="
            border:2px solid #0078D7;
            border-radius:10px;
            padding:15px;
            margin-top:20px;
            background-color:#ffffff;
            box-shadow:0 2px 4px rgba(0,0,0,0.1);
            width:70%;
        ">
            <h4 style="color:#0078D7; text-align:center;">Strategic Output</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    for section in sections:
        st.markdown(
            f"""
            <div style="
                border-left:3px solid #0078D7;
                padding:8px;
                margin:8px 0;
                font-size:14px;
                line-height:1.4;
                background-color:#f9f9f9;
                width:50%;
            ">
                {section}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # DISPLAY STRUCTURED METRICS CARD DIRECTLY BELOW
    # ============================================================
    if rev_value <= 0:
        revenue_display = "Minimal projected revenue impact"
    else:
        revenue_display = f"${rev_value:,.0f}"

    risk_color = "#D83B01" if "High" in risk else "#107C10" if "Low" in risk else "#FFB900"

    st.markdown(
        f"""
        <div style="
            border:2px solid #0078D7;
            border-radius:10px;
            padding:15px;
            margin-top:10px;
            background-color:#f9f9f9;
            width:50%;
        ">
            <p><b>Customer:</b> {name}</p>
            <p><b>Risk Level:</b> <span style="color:{risk_color};">{risk}</span> ({churn_prob})</p>
            <p><b>Projected Revenue:</b> {revenue_display}</p>
            <p><b>Segment:</b> {segment}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
