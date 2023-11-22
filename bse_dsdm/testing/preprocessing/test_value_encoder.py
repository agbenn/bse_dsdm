import pandas as pd
from bse_dsdm.preprocessing.value_encoder import *
import pytest

data = pd.DataFrame({'letter': ['a','b','c','d'],
                'color': ['red', 'blue', 'green', 'red']})

data1 = pd.DataFrame({'level1': ['lower','highest','highest','lowest'],
                'level': ['high', 'med', 'lo', 'lo']})

data2 = pd.DataFrame({'A': [1, 2, 3, 4],
    'B': [5, 8, 12, 15]})

data3 = pd.concat([data, data1, data2], axis=1)

@pytest.mark.unittest
def test_encode_categorical_variable():
    assert encode_categorical_column(data).shape[1] == 7

@pytest.mark.unittest   
def test_encode_ordinal_variable():
    assert encode_ordinal_variable(data1).shape[1] == 4

@pytest.mark.unittest
def test_target_encode_column():
    assert target_encode_column(data3, 'color', 'B', compute_type='mean').sum() == 40
    assert target_encode_column(data3, 'color', 'B', compute_type='count').sum() == 6
    assert target_encode_column(data3, 'color', 'B', compute_type='sum').sum() == 60

