import numpy as np

import pickle
import streamlit as st
lg = pickle.load(open('placement.pkl','rb'))

#Web App
st.title("Job Placement Prediction Model") 

input_text = st.text_input("Enter all the features")


if input_text:
    input_list = input_text.split(',')

    np_df = np.asarray(input_list,dtype=float)
    prediction = lg.predict(np_df.reshape(1,-1))

    if prediction[0]==1:
        print("Person is placed")
    
    else :
        print("Person is not placed")
    
