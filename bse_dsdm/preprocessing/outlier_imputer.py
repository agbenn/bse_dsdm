import pandas as pd
from bse_dsdm.preprocessing.outlier_detection import get_outlier_mask_iso_forest,\
get_outlier_mask_local_outliers, get_outlier_mask_min_covariance_det, get_outlier_mask_one_class_svm

def impute_outliers_with_mean(data, outlier_detection_method='iso_forest', cov_contamination=0.3, local_n_neighbors=2):

    outliers = None

    if outlier_detection_method == "iso_forest":
        outliers = get_outlier_mask_iso_forest(data)
    elif outlier_detection_method == "min_covariance":
        outliers = get_outlier_mask_local_outliers(data, cov_contamination)
    elif outlier_detection_method == "local_outlier":
        outliers = get_outlier_mask_min_covariance_det(data, local_n_neighbors)
    elif outlier_detection_method == "svm":
        outliers = get_outlier_mask_one_class_svm(data)

    # Calculate the mean excluding outliers
    filtered_data = data.loc[~outliers,:]
    mean_without_outliers = filtered_data.mean()

    # Impute the mean to the outliers
    data.loc[outliers,:] = mean_without_outliers

    return data

