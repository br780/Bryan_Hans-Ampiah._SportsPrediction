#!/usr/bin/env python
# coding: utf-8

# In[30]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle


# In[31]:


gradient = 'C:/Users/DELL/Desktop/Jupyter files/New folder/model_GradientBoostingRegressor.pkl'
scaler = 'C:/Users/DELL/Desktop/Jupyter files/New folder/scaler.pkl'
imputer = 'C:/Users/DELL/Desktop/Jupyter files/New folder/imputer.pkl'
with open(gradient,'rb') as model_file, open(scaler,'rb') as scaler_file, open(imputer,'rb') as impute_file:
    model = pickle.load(model_file)
    scale = pickle.load(scaler_file)
    impute = pickle.load(impute_file)


# In[32]:


st.title("Fifa Player Rating Predictor")


# In[33]:


# List of features used in the model
input_variables = ['movement_reactions', 'potential', 'passing', 'wage_eur', 'value_eur', 'dribbling', 
             'age', 'height_cm', 'weight_kg', 'attacking_finishing', 'skill_curve']


# In[34]:


inputs = {}
for feature in input_variables:
    inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)
    


# In[35]:


if st.button("Predict"):
    # Convert input data to DataFrame
    df_inputs = pd.DataFrame([inputs])
    # Ensure all features are present
    for feature in input_var:
        if feature not in df_inputs.columns:
            df_inputs[feature] = 0

    # Preprocess the input data
    new_scale = scale.transform(df_inputs)
    new_impute = impute.transform(X_new_scaled)

    # Make prediction
    prediction = model.predict(new_impute)
    st.write(f"Predicted Overall Rating: {prediction[0]}")

