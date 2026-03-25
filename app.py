#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np 
import pandas as pd
import joblib as jb
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression


# In[120]:


df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")


# In[121]:


df.head()


# In[125]:


df.isnull().sum()


# In[124]:


df['addiction_level'].fillna(df['addiction_level'].mode()[0],inplace=True)


# In[126]:


df.duplicated().sum()


# In[127]:


df.dtypes


# In[128]:


x = df.drop(['addicted_label','transaction_id','user_id','notifications_per_day','app_opens_per_day'],axis = 1)
y = df['addicted_label']


# In[129]:


df['academic_work_impact'].value_counts()


# In[130]:


df['addiction_level'].value_counts()


# In[131]:


df['stress_level'].value_counts()


# In[132]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[133]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[134]:


numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy = 'mean')),
    ('scaler',StandardScaler())
])


# In[135]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy = 'most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown = 'ignore',sparse_output = False))
     ])


# In[136]:


preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)
])


# In[137]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


# In[138]:


model = Pipeline(steps= [
                 ('pre',preprocessor),('reg',LogisticRegression(max_iter = 1000))
                 ])


# In[139]:


model.fit(X_train,y_train)


# In[140]:


y_pred = model.predict(X_test)
print(f'{classification_report(y_test,y_pred)}')


# In[141]:


jb.dump(model,'LogisticRegression.pkl')


# In[156]:


load = jb.load('LogisticRegression.pkl')

st.title('Mobile Phone Usage Addiction Prediction')
age = st.number_input('age')
gender = st.selectbox('gender',['male','female','other'])
daily_screen_time_hours=st.number_input('daily_screen_time_hours')
social_media_hours = st.number_input('social_media_hours')

gaming_hours = st.number_input('gaming_hours')
work_study_hours = st.number_input('work_study_hours')
sleep_hours = st.number_input('sleep_hours')
weekend_screen_time = st.number_input('weekend_screen_time')
stress_level = st.selectbox('stress_level',['high','low','medium'])
academic_work_impact=st.selectbox('academic_work_impact',['no','yes'])
addiction_level = st.selectbox('addiction_level',['moderate','severe','mild'])
if st.button('predict'):
    data = pd.DataFrame({
                           
                            'age' :[age],                         
                         'gender' :[gender] ,                   
        'daily_screen_time_hours' :[daily_screen_time_hours], 
             'social_media_hours' :[social_media_hours] ,   
                   'gaming_hours' :[gaming_hours],         
               'work_study_hours' :[work_study_hours],        
                    'sleep_hours' :[sleep_hours],              
            'weekend_screen_time' :[weekend_screen_time],     
                   'stress_level' :[stress_level],              
           'academic_work_impact' :[academic_work_impact],    
                'addiction_level' :[addiction_level] 
    })
    prediction = load.predict(data)
    
    if prediction == 1:
        st.success('Addicted')
    else:
        st.success('Not Addicted')
    


# In[ ]:




