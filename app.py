#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
import gradio as gr

from xgboost import XGBClassifier


# In[2]:


# Load trained model
model = joblib.load("risk_model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Load feature names
feature_columns = joblib.load("feature_columns.pkl")

# Load columns that require scaling
scale_columns = joblib.load("scale_columns.pkl")

print("✅ Model and preprocessing objects loaded.")


# In[3]:


# ---------- Helper Functions ----------

def get_age_group(age):
    if age < 30:
        return "Young (<30)"
    elif age <= 45:
        return "Middle (31-45)"
    elif age <= 60:
        return "Senior (46-60)"
    else:
        return "Elder (60+)"


def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# In[4]:


# ---------- Encoding Dictionaries ----------

grade_map = {
    "Junior": 0,
    "Mid": 1,
    "Senior": 2
}

payment_behavior_map = {
    "Consistent": 0,
    "Occasionally Late": 1,
    "Frequently Late": 2
}

bmi_category_map = {
    "Underweight": 0,
    "Normal": 1,
    "Overweight": 2,
    "Obese": 3
}

age_group_map = {
    "Young (<30)": 0,
    "Middle (31-45)": 1,
    "Senior (46-60)": 2,
    "Elder (60+)": 3
}


# In[5]:


print(feature_columns)


# In[6]:


import numpy as np
import pandas as pd

def preprocess_input(
    age,
    gender,
    bmi,
    smoker,
    dependents,
    tenure_months,
    occupation,
    region,
    grade_level,
    marital_status,
    payment_method,
    payment_behavior,
    product_applied,
    policy_status,
    monthly_income,
    premium,
    policy_age_days,
    claim_frequency
):

    # ----------------------------
    # Derived Features
    # ----------------------------

    age_group = get_age_group(age)
    bmi_category = get_bmi_category(bmi)

    gender_enc = 1 if gender == "Female" else 0
    smoker_enc = 1 if smoker == "Yes" else 0

    late_payment_flag = int(payment_behavior == "Frequently Late")
    high_claim_flag = int(claim_frequency >= 3)

    premium_burden = (
        premium / monthly_income
        if monthly_income > 0 else 0
    )

    # ----------------------------
    # Log Transformations
    # ----------------------------

    log_income = np.log1p(monthly_income)

    log_premium = np.log1p(premium)

    log_burden = np.log1p(premium_burden)

    log_policy_age = np.log1p(policy_age_days)

    # ----------------------------
    # Ordinal Encoding
    # ----------------------------

    grade_enc = grade_map[grade_level]

    payment_behavior_enc = payment_behavior_map[payment_behavior]

    bmi_category_enc = bmi_category_map[bmi_category]

    age_group_enc = age_group_map[age_group]

    # ----------------------------
    # Build DataFrame
    # ----------------------------

    df = pd.DataFrame({

        "Age":[age],
        "BMI":[bmi],
        "Dependents":[dependents],
        "Tenure_Months":[tenure_months],
        "Claim_Frequency":[claim_frequency],

        "Late_Payment_Flag":[late_payment_flag],
        "High_Claim_Flag":[high_claim_flag],
        "Is_Smoker":[smoker_enc],
        "Gender_Enc":[gender_enc],

        "Grade_Level_Enc":[grade_enc],
        "Payment_Behavior_Enc":[payment_behavior_enc],
        "BMI_Category_Enc":[bmi_category_enc],
        "Age_Group_Enc":[age_group_enc],

        "Occupation":[occupation],
        "Region":[region],
        "Product_Applied":[product_applied],
        "Marital_Status":[marital_status],
        "Payment_Method":[payment_method],
        "Policy_Status":[policy_status],

        "Log_Monthly_Income_GHS":[log_income],
        "Log_Premium_GHS":[log_premium],
        "Log_Premium_Burden":[log_burden],
        "Log_Policy_Age_Days":[log_policy_age]

    })

    # ----------------------------
    # One-Hot Encoding
    # ----------------------------

    df = pd.get_dummies(
        df,
        columns=[
            "Occupation",
            "Region",
            "Product_Applied",
            "Marital_Status",
            "Payment_Method",
            "Policy_Status"
        ],
        drop_first=True
    )

    # ----------------------------
    # Align columns

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # ----------------------------
    # Scale

    df[scale_columns] = scaler.transform(df[scale_columns])

    return df


# In[7]:


def predict_model(df):

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0]

    confidence = probability.max() * 100

    risk_category = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return risk_category, confidence


# In[8]:


def explain_prediction(
    smoker,
    bmi,
    claim_frequency,
    payment_behavior
):

    reasons = []

    if smoker == "Yes":
        reasons.append("Customer is a smoker.")

    if bmi >= 30:
        reasons.append("BMI falls within the obese category.")

    if claim_frequency >= 3:
        reasons.append("High claim frequency detected.")

    if payment_behavior == "Frequently Late":
        reasons.append("Premium payments are frequently late.")

    if not reasons:
        reasons.append("No major risk indicators identified.")

    return "\n".join(f"• {r}" for r in reasons)


# In[9]:


with gr.Blocks(
    title="Insurance Risk Prediction System",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown(
        """
        # 🏥 Insurance Risk Prediction System

        Predict an insurance customer's risk category using an XGBoost Machine Learning model.
        """
    )

    with gr.Row():

        # -------------------------------------------------
        # LEFT COLUMN
        # -------------------------------------------------

        with gr.Column():

            gr.Markdown("## 👤 Personal Information")

            age = gr.Number(label="Age", value=30)

            gender = gr.Dropdown(
                ["Male", "Female"],
                label="Gender"
            )

            dependents = gr.Slider(
                0,
                10,
                value=1,
                step=1,
                label="Dependents"
            )

            occupation = gr.Dropdown(
                [
                    "Civil Engineer",
                    "Construction Worker",
                    "Cybersecurity Specialist",
                    "Data Analyst",
                    "Driver",
                    "Medical Doctor",
                    "Nurse",
                    "Software Engineer",
                    "Teacher"
                ],
                label="Occupation"
            )

            marital_status = gr.Dropdown(
                ["Single", "Married", "Divorced"],
                label="Marital Status"
            )

            region = gr.Dropdown(
                [
                    "Accra",
                    "Cape Coast",
                    "Ho",
                    "Koforidua",
                    "Kumasi",
                    "Sunyani",
                    "Takoradi",
                    "Tamale"
                ],
                label="Region"
            )

            grade_level = gr.Dropdown(
                ["Junior", "Mid", "Senior"],
                label="Grade Level"
            )

        # -------------------------------------------------
        # RIGHT COLUMN
        # -------------------------------------------------

        with gr.Column():

            gr.Markdown("## ❤️ Health & Insurance")

            bmi = gr.Number(label="BMI")

            smoker = gr.Dropdown(
                ["Yes", "No"],
                label="Smoker"
            )

            tenure_months = gr.Number(
                label="Tenure (Months)"
            )

            policy_age_days = gr.Number(
                label="Policy Age (Days)"
            )

            monthly_income = gr.Number(
                label="Monthly Income (GHS)"
            )

            premium = gr.Number(
                label="Premium (GHS)"
            )

            claim_frequency = gr.Number(
                label="Claim Frequency"
            )

            payment_behavior = gr.Dropdown(
                [
                    "Consistent",
                    "Occasionally Late",
                    "Frequently Late"
                ],
                label="Payment Behaviour"
            )

            payment_method = gr.Dropdown(
                [
                    "Bank Transfer",
                    "MoMo",
                    "Payroll Deduction"
                ],
                label="Payment Method"
            )

            product_applied = gr.Dropdown(
                [
                    "Basic Care Plan",
                    "Mekakrawa",
                    "Pru Wealth Plan",
                    "Prudent Life Plan",
                    "Prudential Travel Insurance Plan",
                    "Ultimate Premier Farewell Plan"
                ],
                label="Product Applied"
            )

            policy_status = gr.Dropdown(
                [
                    "Active",
                    "Cancelled",
                    "Lapsed",
                    "Pending Renewal",
                    "Renewed"
                ],
                label="Policy Status"
            )


# In[10]:


def predict_button(
    age,
    gender,
    bmi,
    smoker,
    dependents,
    tenure_months,
    occupation,
    region,
    grade_level,
    marital_status,
    payment_method,
    payment_behavior,
    product_applied,
    policy_status,
    monthly_income,
    premium,
    policy_age_days,
    claim_frequency
):

    # Preprocess
    df = preprocess_input(
        age,
        gender,
        bmi,
        smoker,
        dependents,
        tenure_months,
        occupation,
        region,
        grade_level,
        marital_status,
        payment_method,
        payment_behavior,
        product_applied,
        policy_status,
        monthly_income,
        premium,
        policy_age_days,
        claim_frequency
    )

    # Predict
    risk_category, confidence = predict_model(df)

    # Explain
    explanation = explain_prediction(
        smoker,
        bmi,
        claim_frequency,
        payment_behavior
    )

    # Recommendation
    if risk_category == "Low":
        recommendation = "✅ Customer presents a low insurance risk."

    elif risk_category == "Medium":
        recommendation = (
            "🟡 Moderate risk. Continue monitoring claims and payment behaviour."
        )

    elif risk_category == "High":
        recommendation = (
            "🟠 High risk. Consider premium adjustment and closer monitoring."
        )

    else:
        recommendation = (
            "🔴 Critical risk. Review underwriting conditions before policy approval."
        )

    return (
        risk_category,
        f"{confidence:.2f}%",
        explanation,
        recommendation
    )


# In[11]:


gr.Markdown("---")

predict = gr.Button(
    "🔍 Predict Risk",
    variant="primary",
    size="lg"
)

gr.Markdown("## 📊 Prediction Results")

risk_output = gr.Textbox(
    label="Risk Category"
)

confidence_output = gr.Textbox(
    label="Prediction Confidence"
)

explanation_output = gr.Textbox(
    label="Risk Drivers",
    lines=6
)

recommendation_output = gr.Textbox(
    label="Recommendation",
    lines=3
)

    # =====================================================
    # RIGHT COLUMN
    # =====================================================

with gr.Column():

        gr.Markdown("## ❤️ Health & Insurance Information")

        bmi = gr.Number(label="BMI", value=24.5)

        smoker = gr.Dropdown(
            ["No", "Yes"],
            label="Smoker"
        )

        tenure_months = gr.Number(
            label="Tenure (Months)",
            value=12
        )

        policy_age_days = gr.Number(
            label="Policy Age (Days)",
            value=365
        )

        monthly_income = gr.Number(
            label="Monthly Income (GHS)",
            value=5000
        )

        premium = gr.Number(
            label="Premium (GHS)",
            value=500
        )

        claim_frequency = gr.Number(
            label="Claim Frequency",
            value=1
        )

        payment_behavior = gr.Dropdown(
            [
                "Consistent",
                "Occasionally Late",
                "Frequently Late"
            ],
            label="Payment Behaviour"
        )

        payment_method = gr.Dropdown(
            [
                "Bank Transfer",
                "MoMo",
                "Payroll Deduction"
            ],
            label="Payment Method"
        )

        product_applied = gr.Dropdown(
            [
                "Dignity Farewell Plan",
                "Mekakrawa",
                "Pru Wealth Plan",
                "Prudent Life Plan",
                "Prudential Travel Insurance Plan",
                "Ultimate Premier Farewell Plan"
            ],
            label="Product Applied"
        )

        policy_status = gr.Dropdown(
            [
                "Active",
                "Cancelled",
                "Lapsed",
                "Pending Renewal",
                "Renewed"
            ],
            label="Policy Status"
        )


# In[ ]:


import gradio as gr

# ============================================================
# WRAPPER FUNCTION
# ============================================================

def run_prediction(
    age,
    gender,
    bmi,
    smoker,
    dependents,
    tenure_months,
    occupation,
    region,
    grade_level,
    marital_status,
    payment_method,
    payment_behavior,
    product_applied,
    policy_status,
    monthly_income,
    premium,
    policy_age_days,
    claim_frequency
):

    # Preprocess
    df = preprocess_input(
        age,
        gender,
        bmi,
        smoker,
        dependents,
        tenure_months,
        occupation,
        region,
        grade_level,
        marital_status,
        payment_method,
        payment_behavior,
        product_applied,
        policy_status,
        monthly_income,
        premium,
        policy_age_days,
        claim_frequency
    )

    # Prediction
    risk_category, confidence = predict_model(df)

    # Explanation
    drivers = explain_prediction(
        smoker,
        bmi,
        claim_frequency,
        payment_behavior
    )

    # Recommendation
    recommendation = get_recommendation(risk_category)

    return (
        get_risk_colour(risk_category),
        f"{confidence:.2f} %",
        drivers,
        recommendation
    )


# ============================================================
# BUILD APP
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Insurance Risk Prediction System"
) as app:

    gr.Markdown(
        """
        # 🏥 Insurance Risk Prediction System

        ### Machine Learning Decision Support System

        Predict the insurance risk category of a customer using a trained XGBoost model.
        """
    )

    with gr.Row():

        # =====================================================
        # LEFT COLUMN
        # =====================================================

        with gr.Column():

            gr.Markdown("## 👤 Customer Information")

            age = gr.Number(label="Age", value=30)

            gender = gr.Dropdown(
                ["Male", "Female"],
                label="Gender"
            )

            dependents = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                value=1,
                label="Dependents"
            )

            occupation = gr.Dropdown(
                [
                    "Accountant",
                    "Civil Engineer",
                    "Construction Worker",
                    "Cybersecurity Specialist",
                    "Data Analyst",
                    "Driver",
                    "Medical Doctor",
                    "Nurse",
                    "Software Engineer",
                    "Teacher"
                ],
                label="Occupation"
            )

            region = gr.Dropdown(
                [
                    "Accra",
                    "Cape Coast",
                    "Ho",
                    "Koforidua",
                    "Kumasi",
                    "Sunyani",
                    "Takoradi",
                    "Tamale"
                ],
                label="Region"
            )

            grade_level = gr.Dropdown(
                ["Junior", "Mid", "Senior"],
                label="Grade Level"
            )

            marital_status = gr.Dropdown(
                [
                    "Single",
                    "Married",
                    "Divorced"
                ],
                label="Marital Status"
            )

            gr.Markdown("---")

            gr.Markdown("## 📊 Prediction Results")

            with gr.Row():

                risk_output = gr.Textbox(
                    label="Risk Category",
                    interactive=False
                )

            confidence_output = gr.Textbox(
            label="Confidence",
            interactive=False
            )

        drivers_output = gr.Textbox(
            label="Risk Drivers",
            lines=6,
            interactive=False
            )

        recommendation_output = gr.Textbox(
            label="Recommendation",
            lines=3,
            interactive=False
            )
        with gr.Row():

            predict_btn =gr.Button(
                "🔍 Predict Risk",
                variant="primary",
                size="lg"
            )

            clear_btn = gr.ClearButton(
                value="🔁 Clear",
                components = [
                    age,
                    gender,
                    bmi,
                    smoker,
                    dependents,
                    tenure_months,
                    occupation,
                    region,
                    grade_level,
                    marital_status,
                    payment_method,
                    payment_behavior,
                    product_applied,
                    policy_status,
                    monthly_income,
                    premium,
                    policy_age_days,
                    claim_frequency,
                    risk_output,
                    confidence_output,
                    drivers_output,
                    recommendation_output
                    ]
                )
            predict_btn.click(
                fn=run_prediction,
                inputs=[
                    age,
                    gender,
                    bmi,
                    smoker,
                    dependents,
                    tenure_months,
                    occupation,
                    region,
                    grade_level,
                    marital_status,
                    payment_method,
                    payment_behavior,
                    product_applied,
                    policy_status,
                    monthly_income,
                    premium,
                    policy_age_days,
                    claim_frequency
                    ],
                outputs=[
                    risk_output,
                    confidence_output,
                    drivers_output,
                    recommendation_output
                    ]
                )


