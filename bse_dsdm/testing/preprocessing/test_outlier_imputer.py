import pytest
import pandas as pd
from bse_dsdm.preprocessing.outlier_imputer import *

@pytest.mark.unittest
def test_outlier_imputer(): 
    data = {'A': [1, 2, 3, 3, 5, 2, 75],
        'B': [5, 8, 12, 15, 18, 5, 45]}
    data = pd.DataFrame(data)


    assert data.A.sum() == 91
    assert impute_outliers_with_mean(data).A.sum().astype(int) == 15
