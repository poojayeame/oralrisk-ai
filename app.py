import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="OralRisk AI", page_icon="🦷", layout="centered")

@st.cache_resource
def load_model():
    model    = joblib.load('oralrisk_model.pkl')
    features = joblib.load('oralrisk_features.pkl')
    return model, features

model, features = load_model()

st.markdown("## 🦷 OralRisk AI")
st.markdown("**Diabetes Risk Prediction from Oral Health + Clinical Indicators**")
st.caption("Built on NHANES 2017-2018 · 4,600 patients · XGBoost AUC: 0.93")
st.caption("*By Pooja Podduturu — MS Health Data Analytics, UNT*")
st.divider()

with st.form("form"):
    st.markdown("#### 👤 Demographics & Vitals")
    c1,c2,c3 = st.columns(3)
    with c1: age    = st.number_input("Age", 20, 90, 45)
    with c2: gender = st.selectbox("Gender", ["Male","Female"])
    with c3: bmi    = st.number_input("BMI", 15.0, 60.0, 27.0)

    c4,c5,c6 = st.columns(3)
    with c4: waist = st.number_input("Waist (cm)", 50.0, 160.0, 90.0)
    with c5: sbp   = st.number_input("Systolic BP", 80, 200, 120)
    with c6: dbp   = st.number_input("Diastolic BP", 50, 130, 80)

    st.markdown("#### 🩺 Lab Values")
    c7,c8 = st.columns(2)
    with c7: hdl   = st.number_input("HDL Cholesterol", 20.0, 120.0, 50.0)
    with c8: chol  = st.number_input("Total Cholesterol", 100.0, 400.0, 200.0)

    st.markdown("#### 🚬 Lifestyle")
    c9,c10,c11 = st.columns(3)
    with c9:  smoker    = st.selectbox("Ever smoked?", ["No","Yes"])
    with c10: sedentary = st.number_input("Sedentary mins/day", 0, 1440, 360)
    with c11: income    = st.slider("Income Level (1-12)", 1, 12, 6)

    education = st.selectbox("Education", [
        "Less than 9th grade",
        "9-11th grade (Includes 12th grade with no diploma)",
        "High school graduate/GED or equivalent",
        "Some college or AA degree",
        "College graduate or above"], index=2)

    st.markdown("#### 🦷 Oral Health")
    c12,c13 = st.columns(2)
    with c12:
        perio = st.selectbox("Periodontal Severity", [
            "Healthy (no disease)","Mild periodontitis",
            "Moderate periodontitis","Severe periodontitis"])
        perio_val    = {"Healthy (no disease)":0,"Mild periodontitis":1,
                        "Moderate periodontitis":2,"Severe periodontitis":3}[perio]
        gum_bleeding = st.radio("Gum Bleeding?", ["No","Yes"], horizontal=True)
        dry_mouth    = st.radio("Dry Mouth?",     ["No","Yes"], horizontal=True)
    with c13:
        missing = st.slider("Missing Teeth", 0, 28, 2)
        visit   = st.slider("Years Since Dental Visit", 0.0, 15.0, 1.0, step=0.5)

    submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

if submitted:
    edu_map = {"Less than 9th grade":0,
               "9-11th grade (Includes 12th grade with no diploma)":1,
               "High school graduate/GED or equivalent":2,
               "Some college or AA degree":3,
               "College graduate or above":4}
    row = {
        "age":float(age), "gender":1.0 if gender=="Male" else 0.0,
        "bmi":float(bmi), "waist_cm":float(waist),
        "systolic_bp":float(sbp), "diastolic_bp":float(dbp),
        "hdl_cholesterol":float(hdl), "total_cholesterol":float(chol),
        "smoker":1.0 if smoker=="Yes" else 0.0,
        "income_level":float(income), "education":float(edu_map[education]),
        "sedentary_mins":float(sedentary),
        "perio_severity":float(perio_val),
        "gum_bleeding":1.0 if gum_bleeding=="Yes" else 0.0,
        "missing_teeth":float(missing),
        "years_since_dental_visit":float(visit),
        "dry_mouth":1.0 if dry_mouth=="Yes" else 0.0,
    }
    X = np.array([[row.get(f, 0.0) for f in features]], dtype=np.float64)
    prob = float(model.predict_proba(X)[0][1])
    pct  = round(prob * 100, 1)

    st.divider()
    if pct >= 50:
        st.error(f"🔴 HIGH RISK — Diabetes Risk Score: {pct}%")
    elif pct >= 25:
        st.warning(f"🟡 MODERATE RISK — Diabetes Risk Score: {pct}%")
    else:
        st.success(f"🟢 LOW RISK — Diabetes Risk Score: {pct}%")

    st.progress(prob)

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Periodontal", ["✅ Healthy","⚠️ Mild","⚠️ Moderate","🔴 Severe"][perio_val])
    with c2: st.metric("Missing Teeth", f"{missing}")
    with c3: st.metric("Dental Visit Gap", f"{visit} yrs")
    st.caption("⚠️ For educational purposes only.")

with st.sidebar:
    st.markdown("### 🔬 About")
    st.markdown("""Predicts Type 2 Diabetes risk using oral health + clinical data.
Most diabetes tools ignore the mouth. This one does not.""")
    st.markdown("### 📊 Model")
    st.markdown("""| Metric | Score |
|--------|-------|
| ROC-AUC | 0.93 |
| Recall | 90% |
| Model | XGBoost |""")
    st.markdown("**Pooja Podduturu**  \nMS Health Data Analytics · UNT")
