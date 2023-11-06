import itertools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def get_combinations(list_of_params):
   combination = [] # empty list 
   for r in range(1, len(list_of_params) + 1):
      # to generate combination
      combination.extend(itertools.combinations(list_of_params, r))
   return combination

def find_optimal_model_params(model, possible_features, X, y, min_num_of_features=2):

    param_combinations = [i for i in get_combinations(possible_features) if len(i) > min_num_of_features]

    best_mse = None
    best_params = None

    #find optimal parameters
    for combo in param_combinations:

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
        
        model.fit(X_train, y_train)  # perform linear regression
        
        y_pred = model.predict(X_test)  # make predictions

        mse = mean_squared_error(y_test, y_pred)

        if best_mse == None: 
            best_mse = mse
            best_params = combo
        elif mse < best_mse: 
            best_mse = mse
            best_params = combo
    
    return best_mse, best_params


# TODO create a function to find optimal values for hyper parameters (e.g. n_neighbors) by taking min max median sort approach
# i.e. test min max median, find optimal window hi or lo then repeat


