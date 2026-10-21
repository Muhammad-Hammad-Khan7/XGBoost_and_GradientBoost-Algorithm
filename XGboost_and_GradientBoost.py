#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# In[ ]:


df=pd.read_csv('/content/python machine learning model xgboost - electricity demand dataset.csv')

# In[ ]:


df.head()

# #Data preprocessing

# In[ ]:


df.shape

# In[ ]:


df.isnull().values.any()

# In[ ]:


df.isnull().sum()

# In[ ]:


df.info()

# #Filling Null VAlues

# In[ ]:


from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy="mean")
fill_col=["Demand","Humidity","Temperature", "dayofyear", "year", "month", "dayofweek", "hour"]

# In[ ]:


df[fill_col]= pd.DataFrame(imputer.fit_transform(df[fill_col]), columns=fill_col,  index=df.index)

# In[ ]:


df.head()

# In[ ]:


df.isnull().sum()

# In[ ]:


x= df.drop(["Demand", "Timestamp"], axis=1)
y= df["Demand"]

# In[ ]:


y.head()

# In[ ]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

# # Applied Gradient_Boosting

# In[ ]:


gb= GradientBoostingRegressor()
gb.fit(x_train, y_train)

# In[ ]:


y_pred= gb.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(mse)
print(rmse)
print(f"R-squared: {r2:.4f}")

# #Applied XGBoost

# In[ ]:


#XGBoost
from xgboost import XGBRegressor
xgb= XGBRegressor()

xgb.fit(x_train,y_train)
y_pred_xgb_train=xgb.predict(x_train)
y_pred_xgb_test=xgb.predict(x_test)

print("Score Train",r2_score(y_train,y_pred_xgb_train))
print("Score Test",r2_score(y_test,y_pred_xgb_test))

# In[1]:


y_pred= xgb.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(rmse = np.sqrt(mse))
print(f"R-squared: {r2:.4f}")
