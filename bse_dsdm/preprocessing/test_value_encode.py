import pandas as pd
from datetime import datetime
from bse_dsdm.preprocessing.value_encoder import *
import pytest


data = pd.DataFrame({'letter': ['a', 'b', 'c', 'd'],
                     'color': ['red', 'blue', 'green', 'red']})

data1 = pd.DataFrame({'level1': ['lower', 'highest', 'highest', 'lowest'],
                      'level': ['high', 'med', 'lo', 'lo']})

data2 = pd.DataFrame({'A': [1, 2, 3, 4],
                      'B': [5, 8, 12, 15]})

data3 = pd.concat([data, data1, data2], axis=1)

data4 = pd.DataFrame({'birthday_date': ['1992-04-22', '2004-11-23', '2017-04-19', '1998-12-02'], })
reference_year = 2023


@pytest.mark.unittest
def test_encode_categorical_variable():
    assert encode_categorical_columns(data).shape[1] == 7


@pytest.mark.unittest
def test_encode_ordinal_variable():
    assert encode_ordinal_columns(data1).shape[1] == 4


@pytest.mark.unittest
def test_target_encode_column():
    assert target_encode_column(data3, 'color', 'B', compute_type='mean').sum() == 40
    assert target_encode_column(data3, 'color', 'B', compute_type='count').sum() == 6
    assert target_encode_column(data3, 'color', 'B', compute_type='sum').sum() == 60


def convert_to_age(df, column_name, reference_year):

    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = reference_year - df[column_name].dt.year
    df.rename(columns={column_name: 'age'}, inplace=True)
    return df


def test_convert_to_age():
    # Sample data
    data = {
        'DateOfBirth': ['1990-05-15', '1985-10-20', '2000-03-05', '1978-12-25'],
    }
    sample_data = pd.DataFrame(data)

    # Define the reference year for age calculation
    reference_year = 2023

    # Convert DateOfBirth to age
    result_df = convert_to_age(sample_data.copy(), 'DateOfBirth', reference_year)

    # Check if 'age' column exists in the resulting DataFrame
    assert 'age' in result_df.columns

    # Check if 'age' column has the correct values after conversion
    expected_age = [33, 38, 23, 45]  # Calculated manually based on the reference year and dates
    assert result_df['age'].tolist() == expected_age

    # Check for correct renaming of the column
    assert 'DateOfBirth' not in result_df.columns


'''@pytest.mark.unittest
def test_convert_to_age():
    data4 = pd.DataFrame({'birthday_date': ['1992-04-22', '2004-11-23', '2017-04-19', '1998-12-02'],})
    reference_year = 2023
    result_df = convert_to_age(data4.copy(), 'DateOfBirth', reference_year)
    assert 'age' in result_df.columns
    expected_age = [31 19, 6, 25]
    assert result_df['age'].tolist() == expected_age
    assert 'DateOfBirth' not in result_df.columns'''
