import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

# Load the pre-trained model
with open("Standard_scaler.pkl", "rb") as file:
    std = pickle.load(file)

with open("LabelEncoder.pkl", "rb") as file:
    le = pickle.load(file)

with open("one_hot_encoder.pkl", "rb") as file:
    ohe = pickle.load(file)

model= tf.keras.models.load_model("regression_model.h5")

# Streamlit app title
st.title("Regression Model Prediction App")
# Input fields for user data


geography = st.selectbox("Geography",ohe.categories_[0].tolist())
gender= st.selectbox("Gender",le.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# Encode categorical features
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data, geo_encoded_df], axis=1)

scaled_features = std.transform(input_data)

prediction = model.predict(scaled_features)

st.write("Prediction Result:", prediction[0][0])