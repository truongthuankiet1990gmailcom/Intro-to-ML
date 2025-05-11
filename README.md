# Intro-to-ML

## Project Overview
This repository contains the final project for the **Introduction to Machine Learning (CSC14005)** course at VNU-HCMUS, Department of Computer Science. The project follows a comprehensive machine learning development cycle, structured into three iterative phases to solve a real-world problem through data analysis and modeling.

- **Data Link**: [Google Drive](https://drive.google.com/drive/u/0/folders/1ZIhnBlzOlJX3Eyz02nSwIlsxLP1dUJPK)
- **Kaggle Competition**: [IntroML CLC HK2 2024-2025](https://www.kaggle.com/competitions/introml-clc-hk2-2425/)
- **Deployment Link**: [Streamlit App](https://truongthuankiet1990gmailcom-intro-to-ml-app-du1wh5.streamlit.app/)

---

# 1. 📌Description
This project implements an end-to-end machine learning solution, leveraging frameworks like Scikit-learn, PyTorch/TensorFlow, and XGBoost/LightGBM/CatBoost to address a classification or regression task defined by the Kaggle competition. The development process began with exploratory analysis in Jupyter Notebooks, followed by modularizing the pipeline into components for data ingestion, transformation, and model training. These components were then automated into training and prediction scripts, incorporating best practices such as virtual environments, exception handling, logging, and detailed documentation. The final step involved deploying the solution as a Streamlit web application, mirroring a professional data science workflow.

---

# 2. ⚙️Technologies and Tools
- **Languages/Frameworks**: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM, CatBoost)
- **Development**: Jupyter Notebook, Git, GitHub
- **Environment**: Anaconda, Visual Studio Code
- **Deployment**: Streamlit

---

# 3.🎯 Project Objective

## 3.1 What is the Project Problem?
The project tackles a challenge posed by the Kaggle competition, such as predicting a target variable (e.g., customer behavior, classification outcome, or numerical value) based on the provided dataset. The goal is to develop a model that provides actionable insights or predictions to address a specific real-world issue.

## 3.2 What is the Context?
The dataset, sourced from the competition, contains features relevant to the problem domain. Key performance indicators (e.g., accuracy, error metrics, or domain-specific metrics) especially in this contest `RMSLE` is **the main scoring criteria** guide the evaluation of the solution, emphasizing the importance of robust modeling to maximize performance.

## 3.3 Project Objectives
1. Conduct thorough exploratory data analysis to understand data patterns and prepare it for modeling.
2. Develop and train multiple machine learning models using diverse frameworks and optimize their hyperparameters.
3. Evaluate model performance using appropriate metrics (e.g., accuracy, RMSE, AUC-ROC) and refine the solution for deployment.

## 3.4 Project Benefits
1. Provides a predictive tool for decision-making.
2. Enhances understanding of data through visualization and analysis.
3. Offers a scalable deployment for practical use.

## 3.5 Conclusion
The deployed model generates probability scores or predictions, offering more nuanced insights than binary outcomes. This enables targeted strategies, such as optimizing model inputs or focusing on high-impact features, to improve overall performance.

---

# 4. Solution Pipeline

## General EDA
### Dataset overview: 
The dataset consists of 1,200,000 training entries and 800,000 test entries, with 21 columns in the training set (including the target) and 20 in the test set. 
### Categorical Features Analysis:
- **Key observations:**
    - `Gender`: Slightly higher proportion of males compared to females.
    - `Feature_1 (Marital Status)`: Balanced distribution across categories (Married, Divorced, Single).
    - `Feature_3 (Education Level)`: Higher representation of individuals with Bachelor's and Master's degrees.
    - `Feature_4 (Employment Status)`: Majority are employed, followed by self-employed and unemployed.
    - `Feature_6 (Location Type)`: Balanced distribution across Urban, Rural, and Suburban.
    - `Feature_7 (Tier)`: Relatively balanced distribution across Basic, Comprehensive, and Premium tiers.
    - `Feature_14 (Binary Feature)`: Imbalance with more "No" than "Yes".
    - `Feature_16 (Housing Type)`: Balanced distribution across House, Apartment, and Condo.

<img src = "Images/category.png">

- **Actions Taken:**
    - Identified strong predictors (e.g., Feature_3, Feature_7, Feature_13, Feature_15) based on clear ordinal trends with the target variable.
    - Planned encoding strategies:
        - **One-hot encoding:** Gender, Feature_1, Feature_4, Feature_6, Feature_16.
        - **Label encoding:** Feature_14.
        - **Ordinal encoding:** Feature_3, Feature_7, Feature_13, Feature_15.
        - **Frequency encoding:** All categorical features.

<img src = "Images/box_plot_category.png">

### Numerical Features Analysis
- **Key Observations:**
    - Outliers:
        - Detected in Feature_0 and Feature_8 using box plots.
        - Planned to apply log transformation to handle outliers.
        <img src = "Images/box_plot_numerical.png">

    - Distributions:
        - Most numerical features (e.g., Age, Feature_0, Feature_5) are normally distributed.
        - Target and Feature_0 are skewed, requiring log transformation.
    <img src = "Images/numerical.png">
    - Correlations:
        - Weak correlations between most features and the target variable.
        - Feature_10 has the strongest negative correlation with the target (-0.20).
        - Feature_8 has a very weak positive correlation with the target (0.05).
        - Minimal multicollinearity observed between features.
- **Actions Taken:**
    - Planned log transformation for skewed features (Target, Feature_0).

### Missing Values Analysis
- **Objective:** Handle missing values effectively without losing valuable information.
- **Key Observations:**
    - Missing values are present in both numerical and categorical features.
    - NAN values in some features (e.g., Feature_1, Feature_4) provide additional information and should not be naively imputed.
- **Numerical features vs nan:**
    <img src = "Images/age_vs_nan.png">
    <img src = "Images/feature0_vs_nan.png">
    <img src = "Images/feature2_vs_nan.png">

- **Categorical features vs nan:**
    <img src = "Images/feature1_vs_nan.png">
    <img src = "Images/feature4_vs_nan.png">

- **Actions Taken:**
    - **For numerical features:**
        - Added a new column to indicate missing values (e.g., is_feature_na).
        - Imputed missing values with reasonable estimates (e.g., mean or median).
    - **For categorical features:**
        - Added a new column to indicate missing values.
        - Imputed missing values with the most frequent category or a placeholder (e.g., "Unknown").
### Feature Engineering
- Extracted date, month, and year from time-related features (e.g., Feature_12).
- Applied `sine` and `cosine` transformations to capture cyclical patterns in time-related features.

### Correlation Analysis
- Weak correlations between most features and the target variable.
- Feature_10 has the strongest negative correlation with feature_0 (-0.20).
- Feature_8 has the strongest positive correlation with the target (0.05).
<img src = "Images/heatmap.png">




## Phase 1: Scikit-learn
### Step 1: Preprocessing

- First we create periodic features from feature_12.
- Fill missing values, if numerical, then fill by median values, if categorical, then fill by Unknown values.
- Encode categorical feature and log-transform target feature, the reason why we logged the target is that:
    - **Alignment with the Metric:** RMSLE measures errors in the log space (log(1+y)). By taking the mean in the log space and exponentiating back, you’re finding the value that minimizes the squared logarithmic differences, which is exactly what RMSLE evaluates.
    - **Relative Errors:** RMSLE cares about relative errors (e.g., a 10% error is the same whether the true value is 100 or 10,000). The log transformation converts relative errors into additive differences, and the mean in this space corresponds to the geometric mean in the original space, which aligns with RMSLE’s focus on relative errors.

#### Phase 2: Model Development
- **What I See**: Multiple models were built using Scikit-learn (e.g., Logistic Regression, Random Forest), PyTorch/TensorFlow (e.g., neural networks), and XGBoost/LightGBM/CatBoost (e.g., gradient boosting). Initial training showed varying performance, with tree-based models (e.g., LightGBM) outperforming linear models due to the dataset's complexity. Hyperparameter tuning revealed the importance of balancing bias and variance, and cross-validation ensured robust generalization.
- **Actions Taken**: Selected the best model (e.g., LightGBM), optimized hyperparameters using techniques like grid search or Bayesian optimization, and integrated feature selection (e.g., RFE) to refine the feature set.

#### Phase 3: Model Evaluation
- **What I See**: Model performance was assessed using metrics suited to the task (e.g., RMSE for regression, AUC-ROC for classification). The best model achieved a competitive score on the test set, with error analysis revealing areas for improvement (e.g., misclassifications or high variance). Interpretability tools (e.g., SHAP values) confirmed that engineered features and key variables aligned with EDA insights. The model was then deployed successfully via Streamlit.
- **Actions Taken**: Fine-tuned the model, validated results against the competition benchmark, and deployed the solution with a user-friendly interface.

---

### 5. Main Business Insights
- Insights will vary based on the dataset and competition task. For example, if predicting a numerical target, trends in feature distributions or correlations could be highlighted with visualizations (e.g., scatter plots, heatmaps).
- Analysis of key features and their impact on the target variable should be included, supported by graphs or tables.

---

### 6. Modeling
1. Preprocessing involved tailored transformations (e.g., one-hot encoding for linear models, ordinal encoding for tree-based models) and feature engineering to create relevant predictors.
2. Models were compared using cross-validation with metrics suited to the task (e.g., RMSE for regression, AUC-ROC for classification).
3. The best-performing model (e.g., LightGBM or XGBoost) underwent feature selection (e.g., RFE) and hyperparameter tuning (e.g., Bayesian optimization).
4. Final model performance was assessed with detailed metrics, validated against the test set, and interpreted using techniques like SHAP values to align with EDA findings.

---

### 7. Financial or Practical Results
- Quantify the model’s impact (e.g., error reduction, prediction accuracy) based on competition metrics. If applicable, estimate practical benefits (e.g., improved decision-making efficiency) without specific financial data.
- Conclude with the achievement of project objectives and the solution’s readiness for use.

---

### 8. Web App and Next Steps
- The Streamlit app allows users to input data and receive predictions, integrating the trained model and preprocessing pipeline.
- Future steps include optimizing deployment (e.g., cloud hosting on AWS) and enhancing the app with additional features like interactive visualizations.
- Include screenshots of the app’s homepage, prediction interface, and sample output.

---

### 9. Run This Project Locally

#### Prerequisites
- Python 3.11+
- pip
- Git

#### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>