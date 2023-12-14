import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
#import lightgbm as lgb
import numpy as np
import pandas as pd
from math import log, e
import seaborn as sns


#TODO create a function to auto output metrics in a report


def get_na_columns_test_train(test, train):
    train_na = train.isna().sum()
    test_na = test.isna().sum()
    nas_testrain = pd.concat([train, test], axis=1, keys=['Train', 'Test'])

    display(nas_testrain[nas_testrain.sum(axis=1) > 0])

def get_na_columns_test_train(df):
    df = df.isna().sum()

    display(df.sum() > 0)

def get_columns_by_type(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns

    #categorical_na_columns = [col for col in categorical_columns if df[col].isna().any()]
    #numerical_na_columns = [col for col in numerical_columns if df[col].isna().any()]

    return categorical_columns, numerical_columns

def get_na_columns_by_type(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns

    categorical_na_columns = [col for col in categorical_columns if df[col].isna().any()]
    numerical_na_columns = [col for col in numerical_columns if df[col].isna().any()]

    return categorical_na_columns, numerical_na_columns

