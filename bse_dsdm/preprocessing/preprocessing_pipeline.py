from bse_dsdm.preprocessing.feature_scaling import *
from bse_dsdm.preprocessing.outlier_detection import *
from bse_dsdm.preprocessing.value_encoder import *
from bse_dsdm.preprocessing.value_imputer import *
from bse_dsdm.preprocessing.exploratory_analysis import *
from bse_dsdm.preprocessing.outlier_imputer import *
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression

# TODO add a retain datasets flag

def preprocessing_continuous_outlier_removal_pipeline(datasets, continuous_columns, retain_previous_datasets=False): 
    new_datasets = []
    if retain_previous_datasets:
        new_datasets = datasets
    
    for dataset in datasets:
        #TODO make sure any removal is valid across the entire dataset - use index of return to trim copy
        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = remove_outliers_one_class_svm(temp_dataset[continuous_columns])
        temp_dataset = temp_dataset.dropna(subset=continuous_columns)
        new_datasets.append(temp_dataset)

        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = remove_outliers_local_outlier(temp_dataset[continuous_columns])
        temp_dataset = temp_dataset.dropna(subset=continuous_columns)
        new_datasets.append(temp_dataset)

        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = remove_outliers_min_covariance_det(temp_dataset[continuous_columns])
        temp_dataset = temp_dataset.dropna(subset=continuous_columns)
        new_datasets.append(temp_dataset)

        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = remove_outliers_iso_forest(temp_dataset[continuous_columns])
        temp_dataset = temp_dataset.dropna(subset=continuous_columns)
        new_datasets.append(temp_dataset)

        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = remove_outliers_std_deviation(temp_dataset[continuous_columns])
        temp_dataset = temp_dataset.dropna(subset=continuous_columns)
        new_datasets.append(temp_dataset)

        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = remove_outliers_iqr(temp_dataset[continuous_columns])
        temp_dataset = temp_dataset.dropna(subset=continuous_columns)
        new_datasets.append(temp_dataset)
   
    return new_datasets

def preprocessing_continuous_impution_pipeline(datasets, continuous_columns, retain_previous_datasets=False):
    new_datasets = []
    if retain_previous_datasets:
        new_datasets = datasets

    for dataset in datasets:
        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = impute_values(temp_dataset[continuous_columns], impute_type='mean')
        new_datasets.append(temp_dataset)
        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = impute_values(temp_dataset[continuous_columns], impute_type='knn')
        new_datasets.append(temp_dataset)
    return new_datasets

def preprocessing_encode_categorical_pipeline(datasets, categorical_cols, retain_previous_datasets=False):
    new_datasets = []
    if retain_previous_datasets:
        new_datasets = datasets

    for dataset in datasets: 
        temp_dataset = dataset.copy()
        #return_val = encode_categorical_columns(temp_dataset[categorical_cols])
        encoded_df = encode_categorical_columns(temp_dataset[categorical_cols])
        encoded_df.reset_index(drop=True, inplace=True)
        temp_dataset.reset_index(drop=True, inplace=True)
        temp_dataset = pd.concat([temp_dataset,encoded_df], axis=1)
        new_datasets.append(temp_dataset)
    return new_datasets

def preprocessing_encode_ordinal_pipeline(datasets, ordinal_cols, retain_previous_datasets=False):
    new_datasets = []
    if retain_previous_datasets:
        new_datasets = datasets

    for dataset in datasets: 
        temp_dataset = dataset.copy()
        temp_dataset[ordinal_cols] = encode_ordinal_columns(temp_dataset[ordinal_cols])
        new_datasets.append(temp_dataset)
    return new_datasets

def preprocessing_outlier_imputer_pipeline(datasets, continuous_columns, retain_previous_datasets=False):
    new_datasets = []
    if retain_previous_datasets:
        new_datasets = datasets

    for dataset in datasets:
        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = impute_outliers_with_mean(temp_dataset[continuous_columns],outlier_detection_method='iso_forest')
        new_datasets.append(temp_dataset)
        temp_dataset = dataset.copy()
        temp_dataset[continuous_columns] = impute_outliers_with_mean(temp_dataset[continuous_columns],outlier_detection_method='svm')
        new_datasets.append(temp_dataset)
    return new_datasets

def preprocessing_feature_scaler(datasets, columns_to_scale, retain_previous_datasets=False):
    new_datasets = []
    if retain_previous_datasets:
        new_datasets = datasets
        
    value = 0
    for dataset in datasets:
        print('scaling dataset ' + str(value))
        value += 1
        temp_dataset = dataset.copy()
        temp_dataset[columns_to_scale] = scale_features(temp_dataset[columns_to_scale], scaling_method='z_score')
        new_datasets.append(temp_dataset)
        temp_dataset = dataset.copy()
        temp_dataset[columns_to_scale] = scale_features(temp_dataset[columns_to_scale], scaling_method='min_max_mean')
        new_datasets.append(temp_dataset)
        temp_dataset = dataset.copy()
        temp_dataset[columns_to_scale] = scale_features(temp_dataset[columns_to_scale], scaling_method='min_max')
        new_datasets.append(temp_dataset)
        temp_dataset = dataset.copy()
        temp_dataset[columns_to_scale] = scale_features(temp_dataset[columns_to_scale], scaling_method='iqr')
        new_datasets.append(temp_dataset)
    return new_datasets


