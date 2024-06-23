#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np


# In[22]:


players = pd.read_csv('male_players(legacy).csv')
legacy = pd.read_csv('players_22-1.csv')


# In[23]:


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler


# In[24]:


def cleaning_data(dataset):
    missing = []
    less_missing = []
    for i in dataset.columns:
        if((dataset[i].isnull().sum())< (0.2*(dataset.shape[0]))):
            missing.append(i)
        else:
            less_missing.append(i)
    dataset = dataset[missing]
    numerical_data = dataset.select_dtypes(include=np.number)
    numerical_data.fillna(numerical_data.mean(), inplace= True)
    return numerical_data


# In[25]:


legacy = cleaning_data(legacy)
players = cleaning_data(players)


# In[26]:


def correlation(dataset):
    correlation_matrix = dataset.corr()
    print(correlation_matrix['overall'].sort_values(ascending=False))
    features = correlation_matrix.index[abs(correlation_matrix['overall']) > 0.3].tolist()
    features.remove('overall')
    
    return features


# In[27]:


features = correlation(players)
X = players[features]
y = players['overall']


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_scale)


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


# In[29]:


# Define the models
def training(X_train, y_train):
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42)
    }
    # Train models and perform cross-validation
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse_scores = np.sqrt(-cv_scores)
        print(f'{name} - Cross-Validation RMSE Scores: {cv_rmse_scores}')
        print(f'{name} - Average Cross-Validation RMSE: {cv_rmse_scores.mean()}')
        
    # Train the model on the full training data
        model.fit(X_train, y_train)
        print(f'{name} trained.')
        
        y_pred = model.predict(X_test)
        RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
        print(f'{name} Test RMSE: {RMSE}.')
        
    return models


# In[30]:


def hyperparameter_tuning(model):
    # Hyperparameter tuning using GridSearchCV
    if isinstance(model, RandomForestRegressor):
        param_grid = {
            'n_estimators': [50,100],
            'max_depth': [10, 20],
            'min_samples_split': [2]
        }
    elif isinstance(model, GradientBoostingRegressor):
        param_grid = {
            'n_estimators': [50,100],
            'max_depth': [10, 20],
            'learning_rate': [0.01,0.1]
        }
    elif isinstance(model, DecisionTreeRegressor):
        param_grid = {
            'max_depth': [10, 20],
            'min_samples_split': [2,5]
        }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    return grid_search.best_estimator_


# In[31]:


def evaluate(models, X_test, y_test):
    for name, model in models.items():
        model_tuned = hyperparameter_tuning(model)
        y_pred_best = model_tuned.predict(X_test)
        RMSE_best = np.sqrt(mean_squared_error(y_test,y_pred_best))
        print(f'{name} Best Model Test RMSE: {RMSE_best}.')


# In[32]:


def saving(model,scaler,imputer):
    import pickle as pkl
    pkl.dump(model, open("C:/Users/DELL/Desktop/Jupyter files/New folder/model_"  + model.__class__.__name__ + '.pkl', 'wb'))
    pkl.dump(scaler, open('C:/Users/DELL/Desktop/Jupyter files/New folder/scaler.pkl', 'wb'))
    pkl.dump(imputer, open('C:/Users/DELL/Desktop/Jupyter files/New folder/imputer.pkl', 'wb'))


# In[13]:


models = training(X_train,y_train)
evaluate(models, X_test, y_test)
for model in models.values():
    saving(model,scaler,imputer)


# In[33]:


for model in models.values():
    saving(model,scaler,imputer)


# In[34]:


# Testing with new dataset
common_features = [feature for feature in features if feature in legacy.columns]
X_new = legacy[common_features]
   
missing_features = list(set(features) - set(common_features))
for feature in missing_features:
    X_new[feature] = 0
    
X_new = X_new[features]
X_new_scaled = scaler.transform(X_new)
X_new_imputed = imputer.transform(X_new_scaled)
y_new = legacy['overall']

for name, model in models.items():
    y_pred_new = model.predict(X_new_imputed)
    RMSE_new = np.sqrt(mean_squared_error(y_new, y_pred_new))
    print(f'{name} New Data Test RMSE: {RMSE_new}.')


# In[ ]:




