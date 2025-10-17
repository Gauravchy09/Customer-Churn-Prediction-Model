import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load model, scaler, and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="💡", layout="wide")
st.title("💡 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on their details.")

# --- Input Form ---
with st.form(key='churn_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 30)
        balance = st.number_input('Balance', min_value=0, value=10000)

    with col2:
        credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)
        estimated_salary = st.number_input('Estimated Salary', min_value=0, value=50000)
        tenure = st.slider('Tenure (years)', 0, 10, 3)
        num_of_products = st.slider('Number of Products', 1, 4, 1)

    with col3:
        has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
        is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

    submit_button = st.form_submit_button(label='Predict Churn')

# --- Prediction Logic ---
if submit_button:
    # Prepare input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform(pd.DataFrame({'Geography':[geography]})).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Ensure correct column order
    feature_order = ['CreditScore','Gender','Age','Tenure','Balance',
                     'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary',
                     'Geography_France','Geography_Germany','Geography_Spain']
    input_data = input_data[feature_order]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    pred_prob = model.predict(input_scaled)[0][0]

    # Display results
    st.markdown("### Prediction Result")
    st.metric(label="Churn Probability", value=f"{pred_prob:.2f}")
    if pred_prob > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
