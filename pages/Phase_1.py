import streamlit as st
import joblib
import numpy as np
import pandas as pd

# def run():
    
# Load models and preprocessing tools
decision_tree = joblib.load('Phase_1/decision_tree.joblib')
label_encoders = joblib.load('Phase_1/label_encoders.joblib')
numerical_imputer = joblib.load('Phase_1/numerical_imputer_phase1.joblib')
ordinal_encoders = joblib.load('Phase_1/ordinal_encoders.joblib')
rf_model = joblib.load('Phase_1/rf_model.joblib')
voting_model = joblib.load('Phase_1/voting_phase1.joblib')

numerical_missing_vals = ['Age', 'feature_0', 'feature_2', 'feature_5', 'feature_8', 'feature_9', 'feature_10', 'feature_11']
categorical_missing_vals = ['feature_1', 'feature_4', 'feature_13']

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

    data[numerical_missing_vals] = numerical_imputer.transform(data[numerical_missing_vals])
    data[categorical_missing_vals] = data[categorical_missing_vals].fillna('Unknown')

    return data

# Streamlit application title
st.title("Phase 1: Machine Learning Model Deployment")

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

# Prepare input data for prediction
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

input_data = preprocess_time(input_data)
features_for_one_hot = ['Gender', 'feature_1', 'feature_4', 'feature_6', 'feature_16']
features_for_label = ['feature_14']
features_for_ordinal = ['feature_3', 'feature_7', 'feature_13', 'feature_15']

ordinal_features = {
    'feature_3': ['High School', "Bachelor's", "Master's", 'PhD'],
    'feature_7': ['Basic', 'Comprehensive', 'Premium'],
    'feature_13': ['Poor', 'Average', 'Good'],
    'feature_15': ['Rarely', 'Monthly', 'Weekly', 'Daily'],
}

# Apply preprocessing steps
# One-hot encode categorical features
for feature in features_for_one_hot:
    one_hot_encoded = pd.get_dummies(input_data[feature], prefix=feature, drop_first=True)
    input_data = pd.concat([input_data, one_hot_encoded], axis=1)
    input_data.drop(columns=[feature], inplace=True)

# Label encode features
for feature in features_for_label:
    input_data[feature] = label_encoders[feature].transform(input_data[[feature]])

# Ordinal encode features
for feature in features_for_ordinal:
    input_data[feature] = ordinal_encoders[feature].transform(input_data[[feature]])

# Align input_data with the training feature names
training_features = voting_model.feature_names_in_  # Extract feature names used during training
input_data = input_data.reindex(columns=training_features, fill_value=0)  # Add missing columns with default values
# Display the input data
st.write("Input Data:")
st.write(input_data)

# Prediction logic
if st.button("Predict"):
    prediction = voting_model.predict(input_data)
    prediction_og = np.expm1(prediction[0])  # Inverse log transformation
    st.write("Prediction:", prediction_og)
    
    prediction_dt = decision_tree.predict(input_data)
    prediction_dt_og = np.expm1(prediction_dt[0])  # Inverse log transformation
    st.write("Decision Tree Prediction:", prediction_dt_og)
    
    prediction_rf = rf_model.predict(input_data)
    prediction_rf_og = np.expm1(prediction_rf[0])  # Inverse log transformation
    st.write("Random Forest Prediction:", prediction_rf_og)
    
    