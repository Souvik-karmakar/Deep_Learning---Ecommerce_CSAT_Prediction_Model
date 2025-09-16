# ==========================================
# CSAT Prediction System - Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

# -----------------------
# Load Saved Artifacts
# -----------------------
@st.cache_resource
def load_model():
    return keras.models.load_model("csat_model.h5")

@st.cache_resource
def load_scaler():
    scaler_path ="C:/Users/admin/OneDrive/Desktop/Alma MS projects/Deep Learning/My_Project/E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model-main/Streamlit/scaler.pkl"
    return joblib.load(scaler_path)

@st.cache_resource
def load_feature_list():
    features_path = "C:/Users/admin/OneDrive/Desktop/Alma MS projects/Deep Learning/My_Project/E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model-main/Streamlit/features.pkl"
    return joblib.load(features_path)
    return joblib.load(features_path)

# -----------------------
# Data Preprocessing
# -----------------------
def preprocess_new_data(data, features, numerical_features):
    df_processed = pd.DataFrame()
    for col in features:
        if col not in data.columns:
            df_processed[col] = 0
        elif col in numerical_features:
            df_processed[col] = data[col]
        else:
            df_processed[col] = 1
    return df_processed

def rename_column(df, numerical_cols):
    for col in df.columns:
        if col not in numerical_cols:
            df.rename(columns={col: f"{col}_{df[col].iloc[0]}"}, inplace=True)
    return df

# -----------------------
# App Layout
# -----------------------
st.set_page_config(
    page_title="CSAT Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("📈 Customer Satisfaction Score Prediction")
st.markdown("""
Welcome to the **CSAT Prediction System**!  
Choose below whether to predict for a **single record** or for a **CSV file (batch prediction)**.
""")

# Tabs: Single Input vs CSV Upload
tab1, tab2 = st.tabs(["📝 Single Record Input", "📂 CSV Upload"])

# -----------------------
# SINGLE RECORD INPUT
# -----------------------
with tab1:
    st.sidebar.header("🔹 Input Features")
    channel_name = st.sidebar.text_input("Channel Name", "Online")
    category = st.sidebar.text_input("Category", "Electronics")
    sub_category = st.sidebar.text_input("Sub-category", "Mobile")
    order_date_time = st.sidebar.text_input("Order Date & Time (DD/MM/YYYY HH:MM)", "01/08/2023 14:30")
    issue_reported_at = st.sidebar.text_input("Issue Reported At (DD/MM/YYYY HH:MM)", "01/08/2023 15:00")
    issue_responded = st.sidebar.text_input("Issue Responded At (DD/MM/YYYY HH:MM)", "01/08/2023 15:20")
    customer_city = st.sidebar.text_input("Customer City", "Bengaluru")
    product_category = st.sidebar.text_input("Product Category", "Smartphone")
    item_price = st.sidebar.number_input("Item Price", min_value=0.0, step=1.0, value=1000.0)
    connected_handling_time = st.sidebar.number_input("Connected Handling Time (seconds)", min_value=0.0, step=1.0, value=300.0)
    agent_name = st.sidebar.text_input("Agent Name", "John Doe")
    supervisor = st.sidebar.text_input("Supervisor", "Jane Smith")
    manager = st.sidebar.text_input("Manager", "Alice")
    tenure_bucket = st.sidebar.text_input("Tenure Bucket", "0-6 months")
    agent_shift = st.sidebar.text_input("Agent Shift", "Morning")
    survey_response_date = st.sidebar.text_input("Survey Response Date (DD-MMM-YY)", "01-Aug-23")

    if st.button("🚀 Predict Single CSAT Score"):
        st.info("Processing your input and predicting...")

        # Prepare Input Data
        new_data = pd.DataFrame({
            'channel_name': [channel_name],
            'category': [category],
            'Sub-category': [sub_category],
            'order_date_time': [order_date_time],
            'Issue_reported at': [issue_reported_at],
            'issue_responded': [issue_responded],
            'Customer_City': [customer_city],
            'Product_category': [product_category],
            'Item_price': [item_price],
            'connected_handling_time': [connected_handling_time],
            'Agent_name': [agent_name],
            'Supervisor': [supervisor],
            'Manager': [manager],
            'Tenure Bucket': [tenure_bucket],
            'Agent Shift': [agent_shift],
            'Survey_response_Date': [survey_response_date]
        })

        # Feature Engineering
        new_data['Issue_reported at'] = pd.to_datetime(new_data['Issue_reported at'], format='%d/%m/%Y %H:%M')
        new_data['issue_responded'] = pd.to_datetime(new_data['issue_responded'], format='%d/%m/%Y %H:%M')
        new_data['Response_Time_seconds'] = (new_data['issue_responded'] - new_data['Issue_reported at']).dt.total_seconds()
        new_data['order_date_time'] = pd.to_datetime(new_data['order_date_time'], format='%d/%m/%Y %H:%M')
        new_data['day_number_order_date'] = new_data['order_date_time'].dt.day
        new_data['Survey_response_Date'] = pd.to_datetime(new_data['Survey_response_Date'], format='%d-%b-%y')
        new_data['day_number_response_date'] = new_data['Survey_response_Date'].dt.day
        new_data['weekday_num_response_date'] = new_data['Survey_response_Date'].dt.weekday + 1
        new_data = new_data.drop(columns=['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date'])

        numerical_features = ['Item_price', 'connected_handling_time', 'Response_Time_seconds',
                              'day_number_order_date', 'day_number_response_date',
                              'weekday_num_response_date']

        new_data = rename_column(new_data, numerical_features)

        scaler = load_scaler()
        sorted_features = load_feature_list()

        new_data_processed = preprocess_new_data(new_data, sorted_features, numerical_features)
        new_data_processed[numerical_features] = scaler.transform(new_data_processed[numerical_features])
        X_test_array = new_data_processed.values.astype(np.float32)

        # Predict
        keras_model = load_model()
        predictions = keras_model.predict(X_test_array)
        pred_classes = np.argmax(predictions, axis=1)

        st.success(f"✅ Predicted CSAT Score: **{int(pred_classes)+1}**")
        st.write("Prediction Probabilities:")
        st.dataframe(pd.DataFrame(predictions, columns=[f"CSAT_{i+1}" for i in range(predictions.shape[1])]))
        st.bar_chart(predictions.T, use_container_width=True)

# -----------------------
# CSV UPLOAD INPUT
# -----------------------
with tab2:
    uploaded_file = st.file_uploader("📂 Upload a CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Uploaded Data Preview")
        st.write(df.head())

        try:
            # Feature Engineering (same as above, apply to whole DF)
            df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], errors='coerce')
            df['issue_responded'] = pd.to_datetime(df['issue_responded'], errors='coerce')
            df['Response_Time_seconds'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds()
            df['order_date_time'] = pd.to_datetime(df['order_date_time'], errors='coerce')
            df['day_number_order_date'] = df['order_date_time'].dt.day
            df['Survey_response_Date'] = pd.to_datetime(df['Survey_response_Date'], errors='coerce')
            df['day_number_response_date'] = df['Survey_response_Date'].dt.day
            df['weekday_num_response_date'] = df['Survey_response_Date'].dt.weekday + 1
            df = df.drop(columns=['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date'])

            numerical_features = ['Item_price', 'connected_handling_time', 'Response_Time_seconds',
                                  'day_number_order_date', 'day_number_response_date',
                                  'weekday_num_response_date']

            df = rename_column(df, numerical_features)

            scaler = load_scaler()
            sorted_features = load_feature_list()

            df_processed = preprocess_new_data(df, sorted_features, numerical_features)
            df_processed[numerical_features] = scaler.transform(df_processed[numerical_features])
            X_test_array = df_processed.values.astype(np.float32)

            # Predict
            keras_model = load_model()
            predictions = keras_model.predict(X_test_array)
            pred_classes = np.argmax(predictions, axis=1)

            df["Predicted_CSAT"] = pred_classes + 1

            st.success("✅ Predictions completed!")
            st.write(df.head(10))

            # Download button
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_out,
                file_name="csat_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
