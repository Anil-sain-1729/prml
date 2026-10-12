
import pandas as pd
import numpy as np

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor



df = pd.read_csv("data/india_housing_prices.csv")
df.head()

# ## Pre-Processing

# ### ------------ Dropping `ID` column since it's irrelevant ------------ 


df.drop('ID', axis=1, inplace=True)


# ### ------------ Dropping `Locality` column since it's doesn't containg meaningful address ------------ 


df.drop('Locality', axis=1, inplace=True)

# ### ------------ Using Ordinal Encoder to Encode Ordinal Values -------------



ordinal_columns = ['Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 'Facing', 'Security']

categories = [['Apartment', 'Independent House', 'Villa'],
              ['Unfurnished', 'Semi-furnished', 'Furnished'],
              ['Low', 'Medium', 'High'],
              ['South', 'East', 'West', 'North'],  # In India, west facing and north facing properties are more valuable
              ['No', 'Yes']]

encoder = OrdinalEncoder(categories=categories)
df[ordinal_columns] = encoder.fit_transform(df[ordinal_columns])


# ### ------------ Separating Categorical and Numerical columns --------------

categorical_col = [column for column in df.columns if df[column].dtype == 'object']
numerical_col = [column for column in df.columns if df[column].dtype != 'object']


for column in categorical_col:
    df[column] = df[column].str.lower()

# ---------------  Reducing **Number of rows** to speed up training due to low compute power -----------

# Randomly sample 40% of the data
df_sampled = df.sample(frac=0.4, random_state=1)

X = df_sampled.drop('Price_in_Lakhs', axis=1)  
y = df_sampled['Price_in_Lakhs']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ### ------------ Encoding Categorical Variables ------------
# - DictVectorizer automatically handles categorical columns

train_dicts = X_train.to_dict(orient='records') # using resampled data as our train feature
test_dicts = X_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_test = dv.transform(test_dicts)


# ## Training The Model
# - Helper function and list to evaluate models and store scores

def evaluate_model(y, y_pred):
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    return mae, mse, rmse, r2

# ## XGBoost

xgb = XGBRegressor(random_state=1, objective='reg:squarederror')
print("Training Model")

xgb = XGBRegressor(tree_method='hist', # remove it out if you dont have GPU
                   device='cuda', # remove it out if you dont have GPU
                   n_estimators=100,
                   max_depth=10,
                   learning_rate=0.1,
                   random_state=1, objective='reg:squarederror')

xgb.fit(X_train, y_train)


y_pred = xgb.predict(X_test)
mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)
mae, mse, rmse, r2 

# XGBoost clearly performed well here.
# Do note that for RMSE, MSE, and MAE metrics, the lower the value the better, For R2, higher values indicate better benchmark.

# Saving the model

with open('model.bin', 'wb') as file:
    pickle.dump((dv, xgb), file)


print("Model Saved")