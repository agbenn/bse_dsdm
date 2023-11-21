import pytest
import pandas as pd
from bse_dsdm.preprocessing.feature_scaling import *

@pytest.mark.unittest
def test_feature_scaling(): 
    data = {'A': [1, 2, 3, 2, 5, 5, 75],
        'B': [5, 8, 3, 15, 18, 6, 45]}
    data = pd.DataFrame(data)

    
    assert scale_features(data, 'z_score')['A'].sum().astype(int) == 0
    assert scale_features(data, 'min_max')['A'].sum().astype(int) == 1
    assert scale_features(data, 'min_max_mean')['A'].sum().astype(int) == 0
    assert scale_features(data, 'iqr')['A'].sum().astype(int) == 24
