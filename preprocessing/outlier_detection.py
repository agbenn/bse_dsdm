import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def detect_outliers_iqr(data): 
    # Detect outliers:

    # Calculate quartiles 25% and 75%
    q25, q75 = np.quantile(data, 0.25), np.quantile(data, 0.75)

    # calculate the IQR
    iqr = q75 - q25

    # calculate the outlier cutoff
    cut_off = iqr * 1.5

    # calculate the lower and upper bound value
    lower, upper = q25 - cut_off, q75 + cut_off

    # Calculate the number of records below and above lower and above bound value respectively
    outliers = [x for x in data if (x >= upper) | (x <= lower)]

    # Print basic information (can be removed)
    print('The IQR is',iqr)
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)

    return outliers


def detect_outliers_std_deviation(column_of_data, threshold=3):
    
    mean = np.mean(column_of_data)
    std = np.std(column_of_data)
    cutoff = threshold * std
    lower_bound = mean - cutoff
    upper_bound = mean + cutoff

    # Calculate the number of records below and above lower and above bound value respectively
    outliers = [x for x in column_of_data if (x >= upper_bound) | (x <= lower_bound)]


    # Print basic information (can be removed)
    print('The mean is',mean)
    print('The std is',std)
    print('The lower bound value is', lower_bound)
    print('The upper bound value is', upper_bound)
    print('The list of outliers are', outliers)

    return outliers


def detect_outliers_iso_forest(column_of_data):
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

def detect_outliers_min_covariance_det(column_of_data):
    ee = EllipticEnvelope(contamination=0.01)
    yhat = ee.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

def detect_outliers_local_outlier(column_of_data):
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

def detect_outliers_one_class_svm(column_of_data):
    # identify outliers in the training dataset
    ee = OneClassSVM(nu=0.01)
    yhat = ee.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

