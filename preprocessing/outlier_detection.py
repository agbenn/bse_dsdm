import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def remove_outliers_iqr(column_of_data): 
    '''
    method removes outliers using an interquartile range
    :param object column_of_data: a column or series of a dataframe
    '''

    # Calculate quartiles 25% and 75%
    q25, q75 = np.quantile(column_of_data, 0.25), np.quantile(column_of_data, 0.75)

    # calculate the IQR
    iqr = q75 - q25

    # calculate the outlier cutoff
    cut_off = iqr * 1.5

    # calculate the lower and upper bound value
    lower, upper = q25 - cut_off, q75 + cut_off

    # Calculate the number of records below and above lower and above bound value respectively
    column_of_data = [x for x in column_of_data if (x <= upper) | (x >= lower)]

    # Print basic information (can be removed)
    print('The IQR is',iqr)
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)

    return column_of_data


def remove_outliers_std_deviation(column_of_data, threshold=3):
    '''
    method removes outliers using a standard deviation
    :param object column_of_data: a column or series of a dataframe
    :param int threshold: level at which outliers are trimmed by std dev
    '''
    mean = np.mean(column_of_data)
    std = np.std(column_of_data)
    cutoff = threshold * std
    lower_bound = mean - cutoff
    upper_bound = mean + cutoff

    # Calculate the number of records below and above lower and above bound value respectively
    column_of_data = [x for x in column_of_data if (x <= upper_bound) | (x >= lower_bound)]

    # Print basic information (can be removed)
    print('The mean is',mean)
    print('The std is',std)
    print('The lower bound value is', lower_bound)
    print('The upper bound value is', upper_bound)

    return column_of_data


def remove_outliers_iso_forest(column_of_data, contamination=0.01):
    '''
    method removes outliers in using an iso forest 
    (splits data between max and min values and trims the furthest branches at a contamination level)
    :param object column_of_data: a column or series of a dataframe
    :param float contamination: level at which outliers are trimmed (0-0.5)
    '''
    iso = IsolationForest(contamination=contamination)
    yhat = iso.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

def remove_outliers_min_covariance_det(column_of_data, contamination=0.01):
    '''
    removeing outliers based on a Gaussian distribution
    :param object column_of_data: a column or series of a dataframe
    :param float contamination: level at which outliers are trimmed (0-0.5)
    '''
    ee = EllipticEnvelope(contamination=contamination)
    yhat = ee.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

def remove_outliers_local_outlier(column_of_data):
    '''
    removeing outliers based on a neighbor distance from density algorithm
    :param object column_of_data: a column or series of a dataframe
    '''
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

def remove_outliers_one_class_svm(column_of_data):
    '''
    removeing outliers based on an unsupervised support vector machine model
    :param object column_of_data: a column or series of a dataframe
    '''
    # identify outliers in the training dataset
    ee = OneClassSVM(nu=0.01)
    yhat = ee.fit_predict(column_of_data)
    # select all rows that are not outliers
    mask = yhat != -1
    return column_of_data[mask]

