import pytest
import pandas as pd
from bse_dsdm.preprocessing.value_imputer import *


@pytest.mark.unittest
def test_value_imputer(): 
    data = {'A': [1, 2, 3, None, 5, None, 75],
        'B': [5, 8, None, 15, 18, None, 45]}
    df = pd.DataFrame(data)

    print(impute_values(df, 'constant', impute_constant=0))
    assert impute_values(df, 'constant', impute_constant=0).isna().sum().sum() == 0
    assert impute_values(df, 'ffill').isna().sum().sum() == 0
    assert impute_values(df, 'bfill').isna().sum().sum() == 0
    assert impute_values(df, 'mean').isna().sum().sum() == 0
    assert impute_values(df, 'knn', n_neighbors=3).isna().sum().sum() == 0

