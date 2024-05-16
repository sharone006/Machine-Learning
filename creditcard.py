import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

@st.cache
def load_data():
    data = pd.read_csv(creditcard.csv)
    return data

def preprocess_data(data):
    X = data.drop(columns=['Class'])
    y = data['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def handle_imbalance(X, y, strategy='over'):
    if strategy == 'over':
        ros = RandomOverSampler(sampling_strategy=1.0)
        X_res, y_res = ros.fit_resample(X, y)
    elif strategy == 'under':
        rus = RandomUnderSampler(sampling_strategy=1.0)
        X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res

def train_model(X_train, y_train, model_type='logistic'):
    if model_type == 'logistic':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

def main():
    st.title("Credit Card Fraud Detection")
    st.write("### Load and Preprocess Data")
    data_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if data_file is not None:
        data = load_data(data_file)
        st.write(data.head())
        
        X, y = preprocess_data(data)
        
        imbalance_strategy = st.selectbox("Select imbalance handling strategy", ["None", "Oversampling", "Undersampling"])
        
        if imbalance_strategy == "Oversampling":
            X, y = handle_imbalance(X, y, strategy='over')
        elif imbalance_strategy == "Undersampling":
            X, y = handle_imbalance(X, y, strategy='under')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_type = st.selectbox("Select model type", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
        
        if model_type == "Logistic Regression":
            model = train_model(X_train, y_train, model_type='logistic')
        elif model_type == "Random Forest":
            model = train_model(X_train, y_train, model_type='random_forest')
        elif model_type == "Gradient Boosting":
            model = train_model(X_train, y_train, model_type='gradient_boosting')
        
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
        
        st.write(f"### Model Evaluation Metrics")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")
        st.write(f"ROC-AUC: {roc_auc:.2f}")
        
        st.write("### Predict Fraud on New Data")
        input_data = {}
        for i, column in enumerate(data.columns[:-1]):  # Skip the 'Class' column
            input_data[column] = st.number_input(f"Enter {column}", value=0.0)
        
        input_df = pd.DataFrame([input_data])
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)
        prediction = model.predict(input_scaled)
        
        if st.button("Predict"):
            result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
            st.write(f"The transaction is: {result}")

if __name__ == '__main__':
    main()
