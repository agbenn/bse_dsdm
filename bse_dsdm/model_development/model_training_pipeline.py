from bse_dsdm.model_development.decision_tree import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_validate

def lightgbm_pipeline(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = fit_light_gbm(X_train, y_train)

    lgb_predictions = model.predict(X_test)

    lgb_mse = round(mean_squared_error(y_test, lgb_predictions), 2)

    lgb_cv_preds = cross_val_predict(model, X_train, y_train, cv=5)
    scoring_results = cross_validate(model, X_train, y_train, cv=5, scoring=['accuracy','f1','roc_auc'])
    lgb_mse = mean_squared_error(y_train, lgb_cv_preds)

    scoring_results['mse'] = lgb_mse

    print(scoring_results)

    return scoring_results

