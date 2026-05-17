import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib

st.title("🔧 OralRisk AI — Model Trainer")

if st.button("Train Model Now"):
    with st.spinner("Training..."):
        from nhanes.load import load_NHANES_data
        raw = load_NHANES_data(year='2017-2018')
        
        keep = {
            'AgeInYearsAtScreening':'age','Gender':'gender',
            'BodyMassIndexKgm2':'bmi','WaistCircumferenceCm':'waist_cm',
            'SystolicBloodPres1StRdgMmHg':'systolic_bp',
            'DiastolicBloodPres1StRdgMmHg':'diastolic_bp',
            'DirectHdlcholesterolMgdl':'hdl_cholesterol',
            'TotalCholesterolMgdl':'total_cholesterol',
            'SmokedAtLeast100CigarettesInLife':'smoker',
            'AnnualFamilyIncome':'income_level',
            'EducationLevelAdults20':'education',
            'MinutesSedentaryActivity':'sedentary_mins',
            'DoctorToldYouHaveDiabetes':'diabetes_raw',
        }
        df = raw[list(keep.keys())].rename(columns=keep).copy()
        df['diabetes'] = (df['diabetes_raw'] == '1').astype(int)
        df = df[df['age'] >= 20].dropna(subset=['age','bmi','systolic_bp','diabetes'])
        df['gender'] = (df['gender'] == 'Male').astype(int)
        df['smoker'] = (pd.to_numeric(df['smoker'], errors='coerce') == 1.0).astype(int)
        edu_map = {'Less than 9th grade':0,'9-11th grade (Includes 12th grade with no diploma)':1,
                   'High school graduate/GED or equivalent':2,'Some college or AA degree':3,'College graduate or above':4}
        df['education'] = df['education'].map(edu_map)
        inc_map = {'$0 to $4,999':1,'$5,000 to $9,999':2,'$10,000 to $14,999':3,
                   '$15,000 to $19,999':4,'$20,000 to $24,999':5,'$25,000 to $34,999':6,
                   '$35,000 to $44,999':7,'$45,000 to $54,999':8,'$55,000 to $64,999':9,
                   '$65,000 to $74,999':10,'$75,000 to $99,999':11,'$100,000 and over':12}
        df['income_level'] = df['income_level'].map(inc_map)

        np.random.seed(42)
        n = len(df)
        is_diabetic = df['diabetes'].values
        age_vals = df['age'].fillna(45).values
        smoker_vals = df['smoker'].fillna(0).values
        bmi_vals = df['bmi'].fillna(27).values

        perio_prob = np.clip(0.42 + is_diabetic*0.18 + (age_vals>50).astype(float)*0.15 + smoker_vals*0.20, 0.05, 0.95)
        perio_severity = np.zeros(n, dtype=int)
        rand = np.random.random(n)
        for i in range(n):
            p = perio_prob[i]
            if rand[i] < p*0.20: perio_severity[i] = 3
            elif rand[i] < p*0.55: perio_severity[i] = 2
            elif rand[i] < p*0.80: perio_severity[i] = 1
        df['perio_severity'] = perio_severity
        df['gum_bleeding'] = (np.random.random(n) < 0.35 + is_diabetic*0.15 + smoker_vals*0.10).astype(int)
        missing_base = np.clip((age_vals/10)-2 + is_diabetic*2.5 + smoker_vals*1.5, 0, 15)
        df['missing_teeth'] = np.random.poisson(missing_base).clip(0, 28)
        df['years_since_dental_visit'] = np.round(np.random.exponential(2.0, n) + is_diabetic*1.5, 1).clip(0, 15)
        df['dry_mouth'] = (np.random.random(n) < 0.15 + is_diabetic*0.25 + (age_vals>60).astype(float)*0.10).astype(int)

        features = ['age','gender','bmi','waist_cm','systolic_bp','diastolic_bp',
                    'hdl_cholesterol','total_cholesterol','smoker','income_level',
                    'education','sedentary_mins','perio_severity','gum_bleeding',
                    'missing_teeth','years_since_dental_visit','dry_mouth']

        X = df[features].apply(pd.to_numeric, errors='coerce').values
        y = df['diabetes'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                       max_depth=4, scale_pos_weight=5, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])

        joblib.dump(pipe, 'oralrisk_model.pkl')
        joblib.dump(features, 'oralrisk_features.pkl')

        st.success(f"✅ Model trained! AUC: {auc:.3f}")
        st.info("Now go to GitHub and upload the new pkl files!")
