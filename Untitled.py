#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df= pd.read_csv('finalTrain.csv')


# In[4]:


df.head()


# In[5]:


pd.Categorical(df)


# In[6]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df=df.drop(labels=['ID','Delivery_person_ID'],axis=1)


# In[11]:


df


# In[12]:


X = df.drop(labels=['Time_taken (min)'],axis=1)
Y = df[['Time_taken (min)']]


# In[13]:


X.head()


# In[14]:


Y


# In[15]:


categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns


# In[30]:


categorical_cols


# In[34]:


Order_Date=df['Order_Date'].unique()
Time_Orderd=df['Time_Orderd'].unique()
Time_Order_picked  =df['Time_Order_picked'].unique()
Weather_conditions=df['Weather_conditions'].unique()
Road_traffic_density=df['Road_traffic_density'].unique()
Type_of_order=df['Type_of_order'].unique()
Festival=df['Festival'].unique()
City=df['City'].unique()
Type_of_vehicle=df['Type_of_vehicle'].unique()



# In[35]:


from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[36]:


## Numerical Pipeline
num_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())

    ]

)

# Categorigal Pipeline
cat_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('ordinalencoder',OrdinalEncoder(categories=[Order_Date, Time_Orderd, Time_Order_picked, Weather_conditions, Road_traffic_density, Type_of_order, Type_of_vehicle, Festival,City])),
    ('scaler',StandardScaler())
    ]

)

preprocessor=ColumnTransformer([
('num_pipeline',num_pipeline,numerical_cols),
('cat_pipeline',cat_pipeline,categorical_cols)
])


# In[ ]:


[Order_Date, Time_Orderd, Time_Order_picked, Weather_conditions, Road_traffic_density, Type_of_order, Type_of_vehicle, Festival,City]


# In[37]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=30)


# In[38]:


X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())


# In[39]:


X_train.head()


# In[40]:


## Model Training

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[41]:


regression=LinearRegression()
regression.fit(X_train,y_train)


# In[42]:


regression.coef_


# In[43]:


regression.intercept_


# In[44]:


import numpy as np
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[45]:


## Train multiple models

models={
    'LinearRegression':LinearRegression(),
    'Lasso':Lasso(),
    'Ridge':Ridge(),
    'Elasticnet':ElasticNet()
}
trained_model_list=[]
model_list=[]
r2_list=[]

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)

    #Make Predictions
    y_pred=model.predict(X_test)

    mae, rmse, r2_square=evaluate_model(y_test,y_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model Training Performance')
    print("RMSE:",rmse)
    print("MAE:",mae)
    print("R2 score",r2_square*100)

    r2_list.append(r2_square)
    
    print('='*35)
    print('\n')


# In[ ]:




