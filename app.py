import pickle
import streamlit as st
import numpy as np
import pandas as pd

file = open('bodyfat.pkl', 'rb')
rf = pickle.load(file)
file.close()

df = pd.read_csv("train.csv")

st.title('Body Fat Estimator')

name = st.text_input('Name Of The Person')

density = st.number_input('Density Of The Human Body (in cm3)')

abdomen = st.number_input('Abdomen Size Of Human Body (in cm)')

chest = st.number_input('Chest Size Of The Human Body (in cm)')

hip = st.number_input('Hip Size Of The Human Body (in cm)')

weight = st.number_input('Weight Of The Human Body (in kg)')

submit = st.button('Estimate Body Fat')

if submit:
    query = np.array([density, abdomen, chest, weight, hip])
    query = query.reshape(1, 5)

    prediction = int(rf.predict(query)[0])

    st.title(f'The Predicted Body Fat for {name} is {prediction}%')
