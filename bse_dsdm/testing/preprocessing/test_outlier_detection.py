import pytest
from bse_dsdm.preprocessing.outlier_detection import *

@pytest.mark.unittest
def test_remove_outliers():
    data = {'A': [1, 2, 3, 4, 5, 20, 75],
        'B': [5, 8, 12, 15, 18, 25, 45]}
    df = pd.DataFrame(data)
    
    assert remove_outliers(df, removal_type="std", std_threshold=1).shape[0] == 4
    assert remove_outliers(df, removal_type="iqr").shape[0] == 6
    assert remove_outliers(df, removal_type="iso_forest").shape[0] == 5
    assert remove_outliers(df, removal_type="local_outlier", local_n_neighbors=2).shape[0] == 5
    assert remove_outliers(df, removal_type="min_covariance", cov_contamination=.3).shape[0] == 5
    assert remove_outliers(df, removal_type="local_outlier").shape[0] == 5
