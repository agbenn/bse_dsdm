from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

class RandomForestModel:
    def __init__(self, feature_columns, target_column, train_data, test_data, params=None):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.train_data = train_data
        self.test_data = test_data
        self.params = params

        if self.params:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        X_train = self.train_data[self.feature_columns]
        y_train = self.train_data[self.target_column]
        self.model.fit(X_train, y_train)

    def predict(self):
        X_test = self.test_data[self.feature_columns]
        y_test = self.test_data[self.target_column]

        predictions = self.model.predict(X_test)
        return predictions

    def get_accuracy(self, predictions, y_test):
        accuracy = accuracy_score(y_test, predictions)
        return accuracy


def perform_grid_search(model, param_grid):
    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=model.model, param_grid=param_grid, cv=5,
                               scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(model.train_data[model.feature_columns], model.train_data[model.target_column])

    # Print the best hyperparameters and corresponding accuracy
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    # Evaluate the model with the best hyperparameters on the test set
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(model.test_data[model.feature_columns])
    accuracy = model.get_accuracy(predictions, model.test_data[model.target_column])
    print("Test Accuracy with Best Hyperparameters:", accuracy)

    return best_model

