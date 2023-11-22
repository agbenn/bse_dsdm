import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def encode_categorical_column(data):
    """
    nominal or categorical i.e. blue red green
    """
    # Apply one-hot encoding
    encoder = OneHotEncoder()
    
    encoded_data = encoder.fit_transform(data).toarray()
    # Convert the encoded data back to a dataframe
    encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(data.columns))

    return pd.concat([encoded_data, data.drop(columns=data.columns)], axis=1)

        

def encode_ordinal_variable(data, columns=None, mapping=None): 
    """
    levels of category or ordinal i.e. low medium high
    """
    if mapping:
        data[columns] = data[columns].map(mapping)
    else:
        # Apply label encoding
        encoder = LabelEncoder()
        for col in data.columns:
            encoded_data = encoder.fit_transform(data[col])
            full_col_name = col + '_encoded'
            data[full_col_name] = encoded_data

    return data

def target_encode_column(df, categorical_groupby_value, compute_column, compute_type='mean'): 
    """ 
    useful when there is a correlation between the categorical variable and the other variable.
    i.e. mean GDP grouped by country

    """

    grouped_by_mapping = None

    # Calculating the 
    if compute_type == 'mean':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].mean()
    elif compute_type == 'sum':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].sum()
    elif compute_type == 'count':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].count()

    # Encoding the categorical variable 'group_by_value' using target encoding
    return df[categorical_groupby_value].map(grouped_by_mapping)

