import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import umap.umap_ as umap
import seaborn as sns
sns.set()
import streamlit as st

hru=pd.read_csv("hru.csv")

st.title('hi')

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ---- INPUT FOR THE RECRUITMENT EVALUATION ----
name=st.text_input('Name','Zhen Wang')
st.select_slider('Choose',[1,2,3,4,5])
age=st.number_input('Age',0,100)
sc=st.selectbox('sc',["S","C"])


def predict():
            row=np.array([,])
            prediction=model.predict(X)[0]
            
            if prediction ==1:
                        st.success('')
            else:
                        st.error('')
st.button('Predict',on_click=predict)




















