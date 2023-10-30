
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

def drop_complete_columns(df, threshold=80): 

    # Calculate the percentage of missing values in each column
    missing_percentages = df.isnull().sum() / len(df) * 100

    # Define the threshold for dropping columns (80% or more missing values)
    threshold = 80

    # Get the column names that exceed the threshold
    columns_to_drop = missing_percentages[missing_percentages >= threshold].index

    # Drop the columns from the DataFrame
    return df.drop(columns_to_drop, axis=1)


def impute_values(df, impute_type, impute_constant=None, columns_to_impute=None, n_neighbors=2):
    
    if columns_to_impute:
        df = df[columns_to_impute]

    if impute_type == 'constant' and impute_constant:
        imputed_df = df.fillna(impute_constant)
    elif impute_type == 'ffill':
        imputed_df = df.fillna(method='ffill')
    elif impute_type == 'bfill':
        imputed_df = df.fillna(method='bfill')
    elif impute_type == 'mean':
        imputed_df = df.fillna(df.mean())
    elif impute_type == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # Perform KNN imputation
        imputed_data = imputer.fit_transform(df)

        # Convert the imputed data back to DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns)


def impute_outliers_with_mean(data, column_name, outliers):
    # Calculate the mean excluding outliers
    filtered_data = data[~data.index.isin(outliers.index)]
    mean_without_outliers = filtered_data[column_name].mean()

    # Impute the mean to the outliers
    data.loc[outliers.index, column_name] = mean_without_outliers

    return data


