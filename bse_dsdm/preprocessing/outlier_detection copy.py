import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def drop_complete_columns(df, threshold=80): 
    '''
    '''
    # Calculate the percentage of missing values in each column
    missing_percentages = df.isnull().sum() / len(df) * 100

    # Define the threshold for dropping columns (80% or more missing values)
    threshold = 80

    # Get the column names that exceed the threshold
    columns_to_drop = missing_percentages[missing_percentages >= threshold].index

    # Drop the columns from the DataFrame
    return df.drop(columns_to_drop, axis=1)

def remove_outliers(df, removal_type="std", cov_contamination=0.3, std_threshold=3, iqr_multiplier=1.5, local_n_neighbors=2):
    """
        decrease iqr to increase outlier removal
        increase cov_contamination to increase outlier removal
        decrease std_threshold to increase outlier removal
        local_n_neighbors depends on the sample size
    """
    new_data = {}

    if removal_type == "std":
        return remove_outliers_std_deviation(df, std_threshold)
    elif removal_type == "iqr":
        return remove_outliers_iqr(df, iqr_multiplier)
    elif removal_type == "iso_forest":
        return remove_outliers_iso_forest(df, "auto")
    elif removal_type == "min_covariance":
        return remove_outliers_min_covariance_det(df, cov_contamination)
    elif removal_type == "local_outlier":
        return remove_outliers_local_outlier(df,local_n_neighbors)
    elif removal_type == "svm":
        return remove_outliers_one_class_svm(df)
    else:
        print("no method selected")



def remove_outliers_iqr(data, iqr_multiplier=1.5): 
    """
    Remove outliers outside of the interquartile range (IQR) in specified columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to consider for outlier removal
    - multiplier: multiplier for the IQR to determine the outlier threshold

    Returns:
    - DataFrame without outliers in the specified columns
    """

    for column in data.columns:
        # Calculate the IQR for the column
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the upper and lower bounds for outliers
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        # Remove outliers outside of the bounds
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    return data


def remove_outliers_std_deviation(data, threshold=3):
    '''
    method removes outliers using a standard deviation
    :param object column_of_data: a column or series of a dataframe
    :param int threshold: level at which outliers are trimmed by std dev
    '''

    for col in data.columns:
        mean_value = data[col].mean()
        std_value = data[col].std()

        lower_bound = mean_value - (std_value * threshold)
        upper_bound = mean_value + (std_value * threshold)

        data = data.loc[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def remove_outliers_iso_forest(data, contamination="auto"):
    '''
    Method removes outliers using an iso forest, ignoring NaN values, for each column individually.
    :param object column_of_data: a column or series of a dataframe
    :param float contamination: level at which outliers are trimmed (0-0.5)
    '''
    
    # Create a copy of the DataFrame to avoid modifying the original one
    result = data.copy()

    # Process each column separately
    for col in data.columns:
        # Separate NaN and non-NaN values for the column
        non_nan_data = data[col].dropna()
        nan_data = data[col][data[col].isna()]

        # Check if non_nan_data is not empty
        if not non_nan_data.empty:
            # Apply Isolation Forest only on non-NaN data
            iso = IsolationForest(contamination=contamination)
            yhat = iso.fit_predict(non_nan_data.values.reshape(-1, 1))

            # Select all rows that are not outliers
            mask = yhat != -1

            # Mark outliers as NaN
            non_nan_data[mask] = np.nan

            # Combine processed non-NaN data with original NaN values for the column
            combined_data = pd.concat([non_nan_data, nan_data]).sort_index()
            result[col] = combined_data

    return result

def remove_outliers_min_covariance_det(data, contamination=0.01):
    '''
    Removing outliers based on a Gaussian distribution for each column
    :param DataFrame data: a DataFrame
    :param float contamination: level at which outliers are trimmed (0-0.5)
    :return: DataFrame with outliers removed for each column
    '''
    cleaned_data = pd.DataFrame(index=data.index, columns=data.columns)

    for column in data.columns:
        column_data = data[column]
        non_nan_values = column_data.dropna().values.reshape(-1, 1)

        ee = EllipticEnvelope(contamination=contamination)
        yhat = ee.fit_predict(non_nan_values)

        # select all rows that are not outliers
        mask = yhat != -1
        cleaned_data[column] = np.nan
        cleaned_data.loc[mask, column] = column_data[mask]

    return cleaned_data

def remove_outliers_local_outlier(data, n_neighbors=2):
    '''
    Removing outliers based on a neighbor distance from density algorithm for each column
    :param DataFrame data: a DataFrame
    :param int n_neighbors: number of neighbors to consider
    :return: DataFrame with outliers removed for each column
    '''
    cleaned_data = pd.DataFrame(index=data.index, columns=data.columns)

    for column in data.columns:
        column_data = data[column]
        non_nan_values = column_data.dropna().values.reshape(-1, 1)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        yhat = lof.fit_predict(non_nan_values)

        # select all rows that are not outliers
        mask = yhat != -1
        cleaned_data[column] = np.nan
        cleaned_data.loc[mask, column] = column_data[mask]

    return cleaned_data

def remove_outliers_one_class_svm(data):
    '''
    removeing outliers based on an unsupervised support vector machine model
    :param object column_of_data: a column or series of a dataframe
    '''
    # identify outliers in the training dataset
    ee = OneClassSVM(nu=0.01)
    yhat = ee.fit_predict(data)
    # select all rows that are not outliers
    mask = yhat != -1
    return data[mask]

