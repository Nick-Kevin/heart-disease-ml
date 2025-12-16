"""
Model evaluation assesses the performance of
the trained model using various metrics
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = pd.read_csv('data/heart.csv')
y = data['output']
X_selected = joblib.load('models/X_selected.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

model = joblib.load('models/best_model.pkl')

# prediction
y_pred = model.predict(X_test)

# evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('\nConfusion matrix\n', confusion_matrix(y_test, y_pred))
print('\n', classification_report(y_test, y_pred))
