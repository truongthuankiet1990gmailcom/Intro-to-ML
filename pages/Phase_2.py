import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
# print("Current Working Directory:", os.getcwd())
# print("File exists:", os.path.exists("Phase_2/xgb_model.joblib"))
    # Load models and preprocessing tools
xgb_model = joblib.load("Phase_2/xgb_model.joblib")
label_encoders = joblib.load('Phase_2/label_encoders.joblib')
numerical_imputer = joblib.load('Phase_2/numerical_imputer_phase2.joblib')
ordinal_encoders = joblib.load('Phase_2/ordinal_encoders.joblib')
freq_encoders = joblib.load('Phase_2/freq_encodings.joblib')
lgb_model = joblib.load('Phase_2/lgb_model.joblib')
voting_model = joblib.load('Phase_2/voting_model.joblib')
print("Label Encoders:", label_encoders.keys())
print("Ordinal Encoders:", ordinal_encoders.keys())
print("Frequency Encoders:", freq_encoders.keys())
def preprocess_time(data):
    # Convert time to datetime format
        data['feature_12'] = pd.to_datetime(data['feature_12'], format='%H:%M:%S')
        data['year'] = data['feature_12'].dt.year
        data['month'] = data['feature_12'].dt.month
        data['day'] = data['feature_12'].dt.day
        data['dow'] = data['feature_12'].dt.dayofweek
        data['week'] = data['feature_12'].dt.isocalendar().week

        data['month_sin'] = np.sin(2 * np.pi * (data['month'] - 1) / 12)
        data['month_cos'] = np.cos(2 * np.pi * (data['month'] - 1) / 12)
        data['day_sin'] = np.sin(2 * np.pi * (data['day'] - 1) / 31)
        data['day_cos'] = np.cos(2 * np.pi * (data['day'] - 1) / 31)
        data['group'] = (data['year'] - 2020) * 48 + data['month'] *4 + data['day'] // 7

        data = data.drop(columns=['feature_12'], axis=1)
        return data
    

st.title("Phase 2: Machine Learning Model Deployment")

# User input for features
st.sidebar.header("User Input Features")

# Input fields for features based on the notebook
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
feature_0 = st.sidebar.number_input("Feature 0", min_value=0.0)
feature_1 = st.sidebar.selectbox("Marital Status (Feature 1)", options=["Married", "Divorced", "Single"])
feature_2 = st.sidebar.number_input("Feature 2")
feature_3 = st.sidebar.selectbox("Educational Level (Feature 3)", options=["High School", "Bachelor's", "Master's", "PhD"])
feature_4 = st.sidebar.selectbox("Occupation Status (Feature 4)", options=["Employed", "Self-Employed", "Unemployed"])
feature_5 = st.sidebar.number_input("Feature 5")
feature_6 = st.sidebar.selectbox("Location Type (Feature 6)", options=["Urban", "Rural", "Suburban"])
feature_7 = st.sidebar.selectbox("Tier (Feature 7)", options=["Basic", "Comprehensive", "Premium"])
feature_8 = st.sidebar.number_input("Feature 8")
feature_9 = st.sidebar.number_input("Feature 9")
feature_10 = st.sidebar.number_input("Feature 10")
feature_11 = st.sidebar.number_input("Feature 11")
feature_12 = st.sidebar.time_input("Time (Feature 12)", value=pd.to_datetime("12:00").time())
feature_13 = st.sidebar.selectbox("Education Quality (Feature 13)", options=["Poor", "Average", "Good"])
feature_14 = st.sidebar.selectbox("Feature 14", options=["Yes", "No"])
feature_15 = st.sidebar.selectbox("Activity Frequency (Feature 15)", options=["Rarely", "Monthly", "Weekly", "Daily"])
feature_16 = st.sidebar.selectbox("Housing Type (Feature 16)", options=["House", "Apartment", "Condo"])

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "feature_0": [np.log1p(feature_0)],  # Log transformation
    "feature_1": [feature_1],
    "feature_2": [feature_2],
    "feature_3": [feature_3],
    "feature_4": [feature_4],
    "feature_5": [feature_5],
    "feature_6": [feature_6],
    "feature_7": [feature_7],
    "feature_8": [feature_8],
    "feature_9": [feature_9],
    "feature_10": [feature_10],
    "feature_11": [feature_11],
    "feature_12": [feature_12],  # Add time input
    "feature_13": [feature_13],
    "feature_14": [feature_14],
    "feature_15": [feature_15],
    "feature_16": [feature_16]
})
st.write("Input Data:")
st.write(input_data)
input_data = preprocess_time(input_data)

def adding_features(data):
    # Add engineered features
    data['feature_0_squared'] = data['feature_0'] ** 2
    data['Age_squared'] = data['Age'] ** 2
    data['feature_0_feature_5'] = data['feature_0'] * data['feature_5']
    data['feature_0_Age'] = data['feature_0'] * data['Age']
    data['group_Age'] = data['group'] * data['Age']
    
    # Add missing value indicators
    numerical_missing_vals = ['Age', 'feature_0', 'feature_2', 'feature_5', 'feature_8', 'feature_9', 'feature_10', 'feature_11']
    categorical_missing_vals = ['feature_1', 'feature_4', 'feature_13']

    for c in numerical_missing_vals + categorical_missing_vals:
        data[f"is_{c}_na"] = data[c].isna().astype(int)
    
    # Drop constant columns
    data = data.drop(columns=['is_feature_9_na', 'is_feature_11_na'])
    
    # Impute missing values
    data[numerical_missing_vals] = numerical_imputer.transform(data[numerical_missing_vals])
    data[categorical_missing_vals] = data[categorical_missing_vals].fillna('missing')
    
    # Define feature categories
    features_for_one_hot = ['Gender', 'feature_1', 'feature_4', 'feature_6', 'feature_16']
    features_for_label = ['feature_14']
    features_for_ordinal = ['feature_3', 'feature_7', 'feature_13', 'feature_15']

    # Create frequency encoding for categorical features
    for feature in features_for_one_hot + features_for_label + features_for_ordinal:
        if feature in freq_encoders:
            data[f"{feature}_freq"] = data[feature].map(freq_encoders[feature])
    
    # One-hot encoding with reindexing to match training columns
    expected_columns = {
        'Gender': ['Gender_Male', 'Gender_Female'],
        'feature_1': ['feature_1_Divorced', 'feature_1_Married', 'feature_1_Single'],
        'feature_4': ['feature_4_Employed', 'feature_4_Self-Employed', 'feature_4_Unemployed'],
        'feature_6': ['feature_6_Rural', 'feature_6_Suburban', 'feature_6_Urban'],
        'feature_16': ['feature_16_Apartment', 'feature_16_Condo', 'feature_16_House']
    }
    for feature in features_for_one_hot:
        one_hot_encoded = pd.get_dummies(data[feature], prefix=feature, drop_first=True)
        one_hot_encoded = one_hot_encoded.reindex(columns=expected_columns[feature], fill_value=0)
        data = pd.concat([data, one_hot_encoded], axis=1)
        data.drop(columns=[feature], inplace=True)
    
    # Label encode features
    for feature in features_for_label:
        data[feature] = label_encoders[feature].transform(data[[feature]])

    # Ordinal encode features
    for feature in features_for_ordinal:
        data[feature] = ordinal_encoders[feature].transform(data[[feature]])

    # Log transform specific features
    features_for_robust = ['feature_0', 'feature_0_squared', 'feature_0_feature_5', 'feature_0_Age', 'group_Age']
    for feature in features_for_robust:
        data[feature] = np.log1p(data[feature])
    
    # Safely drop low importance features
    low_importance_features = [
        'is_Age_na', 'Gender_Male', 'feature_4_Unknown', 'feature_16_Apartment',
        'feature_4_Employed', 'feature_6_Rural', 'Gender_Female', 'feature_6_Suburban',
        'feature_1_Married', 'feature_1_Single', 'feature_16_House', 'feature_14_freq',
        'is_feature_0_na', 'feature_4_Unemployed', 'feature_4_Self-Employed', 'feature_1_Unknown',
        'feature_6_Urban', 'feature_13_freq', 'feature_7_freq', 'is_feature_10_na',
        'Gender_freq', 'feature_1_Divorced', 'feature_16_Condo', 'feature_14', 'is_feature_13_na'
    ]
    columns_to_drop = [col for col in low_importance_features if col in data.columns]
    data = data.drop(columns=columns_to_drop, axis=1)
    
    return data
input_data = adding_features(input_data)
training_features = voting_model.feature_names_in_  # Extract feature names used during training
input_data = input_data.reindex(columns=training_features, fill_value=0)  # Add missing columns with default values
# Make predictions using the loaded models

xgb_prediction = xgb_model.predict(input_data)
lgb_prediction = lgb_model.predict(input_data)
voting_prediction = voting_model.predict(input_data)

xgb_prediction_og = np.expm1(xgb_prediction[0])  # Inverse log transformation
lgb_prediction_og = np.expm1(lgb_prediction[0])  # Inverse log transformation
voting_prediction_og = np.expm1(voting_prediction[0])  # Inverse log transformation

st.subheader("Predictions")
st.write("XGBoost Model Prediction:", xgb_prediction_og)
st.write("LightGBM Model Prediction:", lgb_prediction_og)
st.write("Voting Classifier Prediction:", voting_prediction_og)