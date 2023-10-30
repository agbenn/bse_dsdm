import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def encode_categorical_column():
    # Create a dataframe with a categorical variable
    df = pd.DataFrame({'id': [1,2,3,4],
                    'color': ['red', 'blue', 'green', 'red']})

    # Apply one-hot encoding
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[['color']]).toarray()

    # Convert the encoded data back to a dataframe
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out (['color']))


    # Concatenate the encoded dataframe with the 'id' column
    encoded_df = pd.concat([df['id'], encoded_df], axis=1)

    print('Original data:')
    display(df)
    print('\nEncoded data:')
    display(encoded_df)

def encode_ordinal_variable(column_of_data, mapping=None): 

    if mapping:
        column_of_data =column_of_data.map(mapping)
    else:
        # Apply label encoding
        encoder = LabelEncoder()
        column_of_data = encoder.fit_transform(column_of_data)

    return

def target_encode_column(df, categorical_groupby_value, compute_column, compute_type='mean'): 
    # useful when there is a correlation between the categorical variable and the other variable.

    grouped_by_mapping = None

    # Calculating the mean GDP for each country
    if compute_type == 'mean':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].mean()
    elif compute_type == 'sum':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].sum()
    elif compute_type == 'count':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].count()

    # Encoding the categorical variable 'group_by_value' using target encoding
    return df[categorical_groupby_value].map(grouped_by_mapping)

