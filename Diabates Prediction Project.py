#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the Lib


import numpy as ny
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[5]:


# Data collection and analysis
# Pima diabetes Datasets

diabetes_dataset = pd.read_csv(r'C:\Users\hatim\Downloads\diabetes.csv')


# In[6]:


diabetes_dataset.head()


# In[8]:


# of Rows and columns in this dataset
diabetes_dataset.shape


# In[9]:


# Getting the statistical mmeasure
diabetes_dataset.describe()


# In[13]:


# diabetes data set measure 

diabetes_dataset['Outcome'].value_counts()


# In[14]:


# Seperating the data and labels

X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']


# In[15]:


print(X,Y)


# In[16]:


# Data standerdization

scaler = StandardScaler()


# In[17]:


scaler.fit(X)


# In[19]:


standardized_data = scaler.transform(X) # Scaler.fit the data on same range


# In[ ]:




