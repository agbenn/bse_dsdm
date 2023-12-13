import numpy as np

def random_search_cv(model, param_grid, X, y, num_iterations=10, cv=5, random_seed=None):
    """
    Perform Random Search Cross-Validation on the given model using the specified parameter grid.

    Parameters:
    - model: an instance of a machine learning model
    - param_grid: dictionary of hyperparameters and their possible values
    - X: input features
    - y: target labels
    - num_iterations: number of random combinations to try (default is 10)
    - cv: number of folds for cross-validation (default is 5)
    - random_seed: seed for reproducibility (default is None)

    Returns:
    - Dictionary containing the best hyperparameter combination and its corresponding mean cross-validated score
    """
    np.random.seed(random_seed) if random_seed is not None else None

    best_score = float('-inf')
    best_params = None

    for _ in range(num_iterations):
        params = {param: np.random.choice(values) for param, values in param_grid.items()}
        model.set_params(**params)

        scores = cross_validate(model, X, y, cv=cv)
        mean_score = np.mean(scores['test_score'])

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return {'best_params': best_params, 'best_score': best_score}

# Example usage
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# Example model and parameter grid
rf_model = RandomForestClassifier()
param_grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

# Example data
X = np.random.rand(100, 10)  # Example features
y = np.random.randint(0, 2, size=100)  # Example labels (binary classification)

# Perform Random Search CV
result = random_search_cv(rf_model, param_grid, X, y, num_iterations=10, cv=5, random_seed=42)

# Print the best hyperparameter combination and its corresponding mean cross-validated score
print("Best Hyperparameter Combination:", result['best_params'])
print("Best Mean Cross-Validated Score:", result['best_score'])
