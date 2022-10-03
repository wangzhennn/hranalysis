#pip install umap-learn
import umap.umap_ as umap

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt

import seaborn as sns
sns.set()

hru=pd.read_csv("hru.csv")

hru.head(5)

pd.crosstab(hru.job_level,hru.stock_option)

hru.info()

hru.salary_increase.unique()

"""# 2.recommender"""

hru.job_involvement.hist()

hru.job_role.unique()

import scipy.sparse as ss
import numpy as np
from sklearn.preprocessing import LabelEncoder

le_employee_id = LabelEncoder()
le_job_involvement = LabelEncoder()

hru['employee_id_a']=le_employee_id.fit_transform(hru['employee_id'])

hru['job_involvement_a']=le_job_involvement.fit_transform(hru['job_involvement'])

ones = np.ones(len(hru), np.uint32)

matrix = ss.coo_matrix((ones, (hru['employee_id_a'], hru['job_involvement_a'])))

matrix

a = matrix.todense()

a

np.where(matrix.todense()[1] == 1)

hru[hru['employee_id_a']==1]

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42)

matrix_employee_id_a = svd.fit_transform(matrix)

matrix_job_involvement_a = svd.fit_transform(matrix.T)

svd.explained_variance_ratio_.sum()

matrix_job_involvement_a

from sklearn.metrics.pairwise import cosine_distances

cosine_distance_matrix_job_involvement_a = cosine_distances(matrix_job_involvement_a)

cosine_distance_matrix_job_involvement_a.shape

le_job_involvement.transform([3])

def similar_job_involvement(job_involvement, n):
  ix = le_job_involvement.transform([job_involvement])[0]
  sim_job_involvement = le_job_involvement.inverse_transform(np.argsort(cosine_distance_matrix_job_involvement_a[ix,:])[:n])
  return sim_job_involvement

similar_job_involvement(2,3)

np.argsort(cosine_distance_matrix_job_involvement_a[2,:])[:3]

le_job_involvement.inverse_transform(np.argsort(cosine_distance_matrix_job_involvement_a[2,:])[:3])

hru.employee_id.head()

hru[hru.employee_id == 4405]

hru.employee_id

hru_employee_ids = hru[hru.employee_id == 4405]['job_involvement_a'].unique()
hru_vector_ji = np.mean(matrix_job_involvement_a[hru_employee_ids], axis=0)

closest_for_employee = cosine_distances(hru_vector_ji.reshape(1,3), matrix_job_involvement_a)

le_job_involvement.inverse_transform(np.argsort(closest_for_employee[0])[:10])

"""# 3.predict"""

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

selected_df = hru[['department','edu_field','gender','job_role','marital','age','dfh','edu','income','salary_increase','stock_option','training','years_promotion','wlb','satisfaction_job']]

selected_df

X = selected_df.iloc[:,:-1]

y = selected_df.satisfaction_job

from sklearn.preprocessing import OneHotEncoder

import itertools

ohe_X = OneHotEncoder(sparse=False)

X_ohe=ohe_X.fit_transform(X.iloc[:,3:5])

X.iloc[:,3:5]

list(itertools.chain(*ohe_X.categories_))

X_cat = pd.DataFrame(X_ohe, columns = list(itertools.chain(*ohe_X.categories_)))

X_cat

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

transformed_nummerical = scaler.fit_transform(X.iloc[:,5:])

X.iloc[:,5:] = transformed_nummerical

X.index = range(len(X))
X_cat.index = range(len(X_cat))

X_enc = pd.concat([X.iloc[:,5:], X_cat], axis=1)

X_enc

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_enc, y)

from sklearn.linear_model import LinearRegression

model_ols = LinearRegression()

model_ols.fit(X_train, y_train)

print('Model OLS' + ' ' + str(model_ols.score(X_train, y_train)))

y_pred_train = model_ols.predict(X_train)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_pred_train, squared=False)

from xgboost import XGBRegressor

model_xgb = XGBRegressor()

X_train.columns

X_train

model_xgb.fit(X_train, y_train)

print('Model XGB' + ' ' + str(model_xgb.score(X_train, y_train)))

y_pred_train = model_xgb.predict(X_train)

mean_squared_error(y_train, y_pred_train, squared=False)

feat_importances = pd.Series(model_xgb.feature_importances_, index=X_enc.columns)
feat_importances.nlargest(20).plot(kind='barh')

import shap

explainer = shap.TreeExplainer(model_xgb)

shap_values = explainer.shap_values(X_enc)

shap.summary_plot(shap_values, X_enc, plot_type="bar")

shap.summary_plot(shap_values, X_enc)

shap.dependence_plot("age", shap_values, X_enc)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[1,:], X_enc.iloc[1,:])

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[2,:], X_enc.iloc[2,:])

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[3,:], X_enc.iloc[3,:])

shap_values[1,:]

X.iloc[0,:]

"""# 4.clustering"""

hruc=hru.iloc[:,0:19]

hruc

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

hruc_scaled=scaler.fit_transform(hruc)

from sklearn.preprocessing import MinMaxScaler
scaler_min_max=MinMaxScaler()

hruc_minmax=scaler_min_max.fit_transform(hruc)

hruc

sns.displot(data=hruc,
            x="wlb",
            kind="kde")

sns.displot(data=pd.DataFrame(hruc_scaled, columns=hruc.columns), 
            x="wlb",
            kind="kde")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

data_reduced_pca = pca.fit_transform(hruc_scaled)

print(pca.components_)

pca.components_.shape

print(pca.explained_variance_ratio_)

sns.scatterplot(data_reduced_pca[:,0],data_reduced_pca[:,1])

vis_data = pd.DataFrame(data_reduced_pca)
vis_data['income'] = hru['income']
vis_data['age'] = hru['age']
vis_data['years_working'] = hru['years_working']
vis_data['years_company'] = hru['years_company']
vis_data['years_promotion'] = hru['years_promotion']
vis_data['years_manager'] = hru['years_manager']
vis_data['performance'] = hru['performance']
vis_data['salary_increase'] = hru['salary_increase']
vis_data.columns = ['x', 'y', 'income','age','years_working','years_company','years_promotion','years_manager','performance','salary_increase']

alt.Chart(vis_data).mark_circle(size=60).encode(
    x='x',
    y='y',
    tooltip=['income','age','years_working','years_company','years_promotion','years_manager','performance','salary_increase']
).interactive()

plt.figure(figsize=(18,2))
sns.heatmap(pd.DataFrame(pca.components_, columns=hruc.columns), annot=True)
