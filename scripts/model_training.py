"""
Model training involves selecting a suitable machine learning algorithm
and training it on the preprocessed and engineered features
"""

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

data = pd.read_csv('data/heart.csv')

y = data['output']

X_selected = joblib.load('models/X_selected.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

# define hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

# Hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# best model
best_model = grid_search.best_estimator_

# Save the best model to a file
joblib.dump(best_model, 'models/best_model.pkl')

print("Best model saved successfully.")
