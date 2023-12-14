import pandas as pd


"""
best_score, best_index = get_optimal_dataset(model, datasets, continuous_columns, target_col)

optimal_score_dict = {0:"No Change",1:"One Class SVM",2:"Local Outlier",3:"Minimum Covariance Determinant",4:"ISO Forest",5:"STD Deviation",6:"IQR Dataset"}

print("The optimal outlier method was " + str(optimal_score_dict[best_index]) + " with an " + str(accuracy_metric) + " score of " + str(best_score))
"""

def get_optimal_dataset(model, datasets, X_cols, y_col, scoring_metric="roc_auc", min_or_max_score="max"):
    """
    scoring_metric choices: roc_auc, recall, precision, accuracy, r2
    """
    best_score = None
    best_score_index = 0

    for dataset_index in range(0,len(datasets)):

        if min_or_max_score == "min":
            score = cross_val_score(model, datasets[dataset_index][X_cols], datasets[dataset_index][y_col], cv=5, scoring_metric=scoring_metric)
            if score == None:
                best_score = score
                best_score_index = dataset_index
            elif score <= best_score:
                best_score = score
                best_score_index = dataset_index
        else: 
            score = cross_val_score(model, datasets[dataset_index][X_cols], datasets[dataset_index][y_col], cv=5, scoring_metric=scoring_metric)
            if score == None:
                best_score = score
                best_score_index = dataset_index
            elif score >= best_score:
                best_score = score
                best_score_index = dataset_index

    return best_score, best_score_index

def convert_to_age(df, column_name, reference_year):

    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = reference_year - df[column_name].dt.year
    df.rename(columns={column_name: 'age'}, inplace=True)
    return df


def extract_string(df, column_name):

    df[column_name] = df[column_name].str.replace(', ', ',')
    df[column_name] = df[column_name].str.replace(' ', '_')
    df_encoded = df[column_name].str.get_dummies(sep=',')
    df = pd.concat([df, df_encoded], axis=1)
    return df


def remove_columns_with_na(df, threshold=80): 
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