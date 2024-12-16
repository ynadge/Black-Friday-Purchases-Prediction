# Importing dependencies
import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Reading data
dataset_train = pd.read_csv(r'./Data/train.csv')

#  Handling NaN values
mp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=999)
columns = ['Product_Category_2', 'Product_Category_3']
mp.fit(dataset_train[['Product_Category_2', 'Product_Category_3']])
dataset_train[['Product_Category_2', 'Product_Category_3']] = mp.transform(dataset_train[['Product_Category_2', 'Product_Category_3']])

# Label encoding
categorical_columns = ['Gender', 'Occupation', 'City_Category',
                       'Product_Category_1', 'Product_Category_2',
                       'Product_Category_3', 'Age', 'Stay_In_Current_City_Years']
for i in categorical_columns :
	le = LabelEncoder()
	dataset_train[i] = le.fit_transform(dataset_train[i])

# Splitting into features and target variable.
X = dataset_train.iloc[:, :-1]  # Select all columns except the last one
X = X.drop(['User_ID', 'Product_ID'], axis = 1)
y = dataset_train.iloc[:, -1]   # Select the last column

# Creating our model and fitting the training data.
rf = RandomForestRegressor(n_estimators=10, criterion='squared_error', max_depth=14, min_samples_split=10,
                                 min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features=None,
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, oob_score=False, n_jobs=None, random_state=0, verbose=0,
                                 warm_start=False)
rf.fit(X, y)

#
with open("rf_model.pkl", "wb") as f:
	pickle.dump(rf, f)