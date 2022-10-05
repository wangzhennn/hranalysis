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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

hru=pd.read_csv("hru.csv")

st.title('Predict Your Income in Company X')
st.set_page_config(page_title="HR Analysis",page_icon="🔍")

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
    ohe_pkl = pickle.load(open('ohe.pkl','rb'))
    shap_values = pickle.load(open('shap_values.pkl','rb'))
    cats = list(itertools.chain(*ohe_pkl.categories_))
    print(len(cats))
    return model_xgb, scaler, ohe_pkl, cats, shap_values

model_xgb, scaler, ohe_pkl, cats, shap_values = read_objects()


# ---- DEFINE EXPLAINER ----
explainer = shap.TreeExplainer(model_xgb)

# ---- COLLECT INPUT FROM USER ----
job_role=st.selectbox('Expected Job Position',options=ohe_pkl.categories_[2])
age=st.number_input('Age',0,100)
gender=st.selectbox('Gender',options=ohe_pkl.categories_[1])
marital=st.selectbox('Marital Situation',options=ohe_pkl.categories_[3])
edu=st.select_slider('Education Level',[1,2,3,4,5])
with st.expander("What's Education Level?"):
    st.markdown("""
    1-Below College / 2-College / 3-Bachelor / 4-Master / 5-Doctor
    """)
dfh=st.number_input('Distance From Home to Company (KM)',0,500)
num_companiesworked=st.number_input('Total Number of Companies You Have Been Worked',0,100)
years_working=st.number_input('Years of Working Before',0,100)
stock_option=st.select_slider('Expected Stock Option Level',[1,2,3,4,5])
training=st.number_input('Expected Training Times per year',0,100)

performance=st.select_slider('Expected Job Performance Level',[1,2,3,4,5])
job_involvement=st.select_slider('Expected Job Involvement Level',[1,2,3,4,5])
satisfaction_job=st.select_slider('Expected Job Satisfaction Level',[1,2,3,4,5])
wlb=st.select_slider('Expected Work Life Balance Level',[1,2,3,4,5])
business_travel=st.selectbox('Expected Business Travel Frequency',options=ohe_pkl.categories_[0])

# make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
if st.button('Calaulate'):
            # make a DF for the numericals and standard scale
            new_df_num = pd.DataFrame({'age':age,
                               'dfh':dfh,
                               'edu':edu,
                               'stock_option':stock_option,
                               'num_companiesworked':num_companiesworked,
                               'training':training,
                               'years_working':years_working,
                               'performance':performance,
                               'job_involvement':job_involvement,
                               'satisfaction_job':satisfaction_job,
                               'wlb':wlb}, index=[0])
            new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns =new_df_num.columns, index=[0])  
            
            new_df_cat = pd.DataFrame({'business_travel':business_travel,
                               'gender':gender,
                               'job_role':job_role,
                               'marital':marital}, index=[0])
            new_values_cat = pd.DataFrame(columns=['Non-Travel','Travel_Frequently','Travel_Rarely','Female','Male','Healthcare Representative','Human Resources','Laboratory Technician','Manager','Manufacturing Director','Research Director','Research Scientist','Sales Executive','Sales Representative','Divorced','Married','Single'],dtype='object')
            new_values_cat['Non-Travel'] = 0
            new_values_cat['Travel_Frequently'] = 0
            new_values_cat['Travel_Rarely'] = 0
            new_values_cat['Female'] = 0
            new_values_cat['Male'] = 0
            new_values_cat['Healthcare Representative'] = 0
            new_values_cat['Human Resources'] = 0
            new_values_cat['Laboratory Technician'] = 0
            new_values_cat['Manager'] = 0
            new_values_cat['Manufacturing Director'] = 0
            new_values_cat['Research Director'] = 0
            new_values_cat['Research Scientist'] = 0
            new_values_cat['Sales Executive'] = 0
            new_values_cat['Sales Representative'] = 0
            new_values_cat['Divorced'] = 0
            new_values_cat['Married'] = 0
            new_values_cat['Single'] = 0
            new_values_cat[business_travel] = 1
            new_values_cat[gender] = 1
            new_values_cat[job_role] = 1
            new_values_cat[marital] = 1

            line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)
            predicted_value = model_xgb.predict(line_to_pred)[0]
            st.metric(label="Predicted Income", value=f'{round(predicted_value)} Ruppes' )
           





















