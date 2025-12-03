import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# load the data
heart_df = pd.read_csv('data/heart.csv')

# display the 5 first rows
print(heart_df.head(), '\n')

# split data into features and target
X = heart_df.drop('output', axis=1)
y = heart_df['output']

print('Features\n', X.head())

# ensure X and y have the same number of samples
X = X.iloc[:len(y)]

# identify numerical and categorical columns
numercial_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print('\nNumerical features:', numercial_features)
print('\nCategorical features:', categorical_features)
