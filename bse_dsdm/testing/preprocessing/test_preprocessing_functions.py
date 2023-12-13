from bse_dsdm.preprocessing.preprocessing_functions import *
import pandas as pd
import pytest

@pytest.mark.unittest
def test_convert_to_age():
    data5 = pd.DataFrame({'Birthday_date': ['1990-05-15', '1985-10-20', '2000-03-05', '1978-12-25'],
                          })
    reference_year = 2023
    result_df = convert_to_age(data5.copy(), 'Birthday_date', reference_year)
    assert 'age' in result_df.columns
    expected_age = [33, 38, 23, 45]
    assert result_df['age'].tolist() == expected_age
    assert 'Birthday_date' not in result_df.columns


@pytest.mark.unittest
def test_extract_string_dummies():
    data6 = {
        'original_column': ['strong, body', 'weak, body', 'medium', 'normal, body'],
    }
    df = pd.DataFrame(data6)
    column_name = 'original_column'
    result_df = extract_string(df, column_name)
    expected_columns = ['original_column', 'strong', 'weak', 'medium', 'normal', 'body']

    for col in expected_columns[1:]:
        assert col in result_df.columns

    assert result_df['strong'].tolist() == [1, 0, 0, 0]
    assert result_df['weak'].tolist() == [0, 1, 0, 0]
    assert result_df['medium'].tolist() == [0, 0, 1, 0]
    assert result_df['normal'].tolist() == [0, 0, 0, 1]
    assert result_df['body'].tolist() == [1, 1, 0, 1]