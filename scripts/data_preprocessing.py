import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
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

print('\n Info')
print(X.info())

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# process the features
X_transformed = numerical_transformer.fit_transform(X)

X_transformed_df = pd.DataFrame(X_transformed, columns=numercial_features)
print('X transformed DataFrame')
print(X_transformed_df.head())
print('\nX transformed DataFrame description')
print(X_transformed_df.describe())

# Save the preprocessor object
joblib.dump(numerical_transformer, 'models/preprocessor.pkl')

# Split data into training and test sets after preprocessing
X_train, X_test, y_train, y_test = train_test_split(X_transformed_df, y, test_size=0.2, random_state=42)

# Save preprocessed data and split datasets to files
pd.to_pickle(X_transformed_df, 'models/X_transformed.pkl')
pd.to_pickle(y_test, 'models/y_test.pkl')

print("Data preprocessing completed.")
