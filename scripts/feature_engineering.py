"""
Feature engineering involves selecting and creating relevant features that can improve the model's performance
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

X_preprocessed = joblib.load('models/X_transformed.pkl')

data = pd.read_csv('data/heart.csv')

y = data['output']

# feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_preprocessed, y)

print('Features preprecessed\n', X_preprocessed)
print('\nFeatures selected\n', pd.DataFrame(X_selected))

# Save X_selected and selector to files
joblib.dump(X_selected, 'models/X_selected.pkl')

print("Feature engineering completed successfully.")
