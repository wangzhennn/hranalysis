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
st.text_input('Name','Zhen Wang')
st.select_slider('Choose',[1,2,3,4,5])
st.number_input('Age',0,100)
