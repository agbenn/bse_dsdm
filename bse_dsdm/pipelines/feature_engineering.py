# TODO create 5 features for dataset

from bse_dsdm.preprocessing.value_econder import *
import pandas as pd

df_train = pd.read_csv('./train.csv')
df_sample = pd.read_csv('./sample.csv')
data = df_train.merge(df_sample, on='id', how='left')


def feature_encoding_categorical(data, column):
    data = encode_categorical_column('body_type')


def feature_encoding_ordinal(data, column, mapping):
    data = encode_ordinal_variable(data, columns=['work_rate'], mapping={'work_rate': {
                                   'Low/Low': 1, 'Low/Medium': 2, 'Low/High': 3, 'Medium/Low': 4, 'Medium/Medium': 5, 'Medium/High': 6, 'High/Low': 7, 'High/Medium': 8, 'High/High': 9}})


def feature_target_encoding(data, column):
    data['nationality_name'] = target_encode_column(data, 'nationality_name', 'height_cm', compute_type='mean')
