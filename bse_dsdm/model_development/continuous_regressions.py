from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, RandomForestRegressor
from sklearn.model_selection import train_test_split




def get_fitted_polynomial_regressor(X,y, poly_degree=2, model="default"):
    """
    Creating polynomials from features in a linear regression can be beneficial in certain situations to capture nonlinear relationships between the features and the target variable. Here are a few reasons why polynomial features might be useful in linear regression:

    * Nonlinear Relationships: Linear regression assumes a linear relationship between the features and the target variable. However, in real-world scenarios, the relationship might not be strictly linear. By introducing polynomial features (e.g., squaring or cubing the original features), we can model and capture nonlinear patterns in the data more effectively.

    * Flexibility: Polynomial features provide more flexibility in representing complex relationships. By including higher-order terms in the regression model, we allow the model to fit curves, bends, and more intricate patterns in the data.

    * Improved Model Fit: In cases where a linear relationship alone does not adequately capture the underlying structure of the data, incorporating polynomial features can improve the model's fit. This can lead to better predictions and reduced errors.

    * Interaction Effects: Polynomial features can also help capture interaction effects between different features. For example, including interaction terms like x1 * x2 (where x1 and x2 are different features) allows the model to capture the combined effect of these features on the target variable.
    """
    poly_model = None

    if model == "default":
        poly_model = LinearRegression()
    elif model == "lasso":
        poly_model = Lasso()
    elif model == "ridge":
        poly_model = Ridge()

    poly_features = PolynomialFeatures(degree=poly_degree)
    X_poly = poly_features.fit_transform(X)
    poly_model.fit(X_poly, y)
    return poly_model


def get_fitted_random_forest_regressor(X,y):

    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model

def get_fitted_regressor(X,y, model="default"):
    
    if model == "default":
        poly_model = LinearRegression()
    elif model == "lasso":
        poly_model = Lasso()
    elif model == "ridge":
        poly_model = Ridge()

    poly_model.fit(X, y)
    return poly_model

