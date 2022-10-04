import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import umap.umap_ as umap
import seaborn as sns
sns.set()
import streamlit as st
import pickle
import itertools
import shap
from streamlit_shap import st_shap

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

# ---- READ PREPROCESSED MODEL ----
@st.experimental_singleton
def read_objects():
    model_xgb = pickle.load(open('model_xgb.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    ohe = pickle.load(open('ohe.pkl','rb'))
    shap_values = pickle.load(open('shap_values.pkl','rb'))
    cats = list(itertools.chain(*ohe.categories_))
    return model_xgb, scaler, ohe, cats, shap_values

model_xgb, scaler, ohe, cats, shap_values = read_objects()

# ---- DEFINE EXPLAINER ----
explainer = shap.TreeExplainer(model_xgb)

# ---- COLLECT INPUT FROM USER ----
age=st.number_input('Age',0,100)
dfh=st.number_input('Distance From Home To Company (km)',0,500)
edu=st.select_slider('Education Level',[1,2,3,4,5])
stock_option=st.select_slider('Expected Stock Option Level',[1,2,3,4,5])
training=st.number_input('Expected Training Times per yer',0,100)
num_companiesworked=st.number_input('Total number of companies you have worked for',0,100)
performance=st.select_slider('Expected Performance',[1,2,3,4,5])
job_involvement=st.select_slider('Expected Job Involvement',[1,2,3,4,5])
satisfaction_job=st.select_slider('Expected Job Satisfaction',[1,2,3,4,5])
wlb=st.select_slider('Expected Work Life Balance',[1,2,3,4,5])
buiness_travel=st.selectbox('buiness_travel',options=ohe.categories_[0])
department=st.selectbox('department',options=ohe.categories_[1])
edu_field=st.selectbox('edu_field',options=ohe.categories_[2])
gender=st.selectbox('gender',options=ohe.categories_[3])
job_role=st.selectbox('job_role',options=ohe.categories_[4])
marital=st.selectbox('marital',options=ohe.categories_[5])

# make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
if st.button('Predict! 🚀'):
    # make a DF for categories and transform with one-hot-encoder
    new_df_cat = pd.DataFrame({'buiness_travel':buiness_travel,
                               'department':department,
                               'edu_field':edu_field,
                               'gender':gender,
                               'job_role':job_role,
                               'marital':marital}, index=[0])
    new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = cats , index=[0])

    # make a DF for the numericals and standard scale
    new_df_num = pd.DataFrame({'age':age,
                               'dfh':dfh,
                               'edu':edu,
                               'stock_option':stock_option,
                               'training':training,
                               'num_companiesworked':num_companiesworked,
                               'year_working':year_working,
                               'performance':performance,
                               'job_involvement':job_involvement,
                               'satisfaction_job':satisfaction_job,
                               'wlb':wlb}, index=[0])
    new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
    
    #bring all columns together
    line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

    #run prediction for 1 new observation
    predicted_value = model_xgb.predict(line_to_pred)[0]

    #print out result to user
    st.metric(label="Predicted Income", value=f'{round(predicted_value)} ')
    
    #print SHAP explainer to user
    st.subheader(f'Wait, why {round(predicted_value)} kr? Explain, AI 🤖:')
    shap_value = explainer.shap_values(line_to_pred)
    st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=500)























