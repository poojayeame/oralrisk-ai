import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib, io

st.title("🔧 OralRisk AI — Model Trainer")

if st.button("Train Model Now"):
    with st.spinner("Training..."):
        df = pd.read_csv('oralrisk_dataset.csv')
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
        st.success(f"✅ Done! AUC: {auc:.3f}")

        buf1 = io.BytesIO()
        joblib.dump(pipe, buf1)
        buf1.seek(0)
        st.download_button("⬇️ Download oralrisk_model.pkl", buf1, "oralrisk_model.pkl")

        buf2 = io.BytesIO()
        joblib.dump(features, buf2)
        buf2.seek(0)
        st.download_button("⬇️ Download oralrisk_features.pkl", buf2, "oralrisk_features.pkl")
