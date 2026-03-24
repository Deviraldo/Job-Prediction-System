import numpy as np
import pandas as pd
import streamlit as st
import pickle

lg = pickle.load(open('placement.pkl','rb'))

#Web App
st.title("Job Placement Prediction")

input_text = st.text_input("Enter all features")


if input_text:
    input_list = input_text.split(',')
    
    np_df = np.asarray(input_list,dtype=float )
    prediction = lg.predict(np_df.reshape(1,-1))

    if prediction[0] == 1:
        st.write("The person is placed")
    else :
        st.write("The person is not placed")