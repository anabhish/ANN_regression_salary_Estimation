import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

#load the training model

#model = tf.keras.models.load_model('model.keras')
model = tf.keras.models.load_model('regression_model.h5')

# load the encoders and scaler

with open('label_encoder_gender_reg.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo_reg.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler_reg.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit APP

st.title('Estimated Salary prediction')

#User input

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,102)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare the input data

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active_member],
    'Exited' : [exited]
})


# one-hot encode Geography
geo_df = pd.DataFrame([[geography]], columns=['Geography'])

geo_encoded = onehot_encoder_geo.transform(geo_df).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# combine
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)


#scale the input data

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
predicted_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary: {predicted_salary:.2f}')


