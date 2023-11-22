
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

def impute_values(df, impute_type='constant', impute_constant=None, columns_to_impute=None, n_neighbors=2):
    """
    allows for value impution
    options: ffill, bfill, mean, knn
    """
    filled_df = None
    if columns_to_impute:
        df = df[columns_to_impute]

    if impute_type == 'constant' and impute_constant is not None:
        filled_df = df.fillna(value=impute_constant)
    elif impute_type == 'constant' and impute_constant is None: 
        print('must input a impute value constant')
        return 
    elif impute_type == 'ffill':
        filled_df = df.fillna(method='ffill')
    elif impute_type == 'bfill':
        filled_df = df.fillna(method='bfill')
    elif impute_type == 'mean':
        filled_df = df.fillna(df.mean())
    elif impute_type == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # Perform KNN imputation
        imputed_data = imputer.fit_transform(df)

        # Convert the imputed data back to DataFrame
        filled_df = pd.DataFrame(imputed_data, columns=df.columns)
    else:
        print('invalid method selection')
    
    return filled_df




