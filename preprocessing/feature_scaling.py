from scipy import stats 

def standardize_with_z_score(column_of_data): 
    '''
    method standardizes a column of data using a z-score method
    :param object column_of_data: a column or series of a dataframe
    '''
    # Rescale the variables using z-score standardization
    return (column_of_data - column_of_data.mean()) / column_of_data.std()

def min_max_scale_with_mean(column_of_data):
    '''
    method standardizes a column of data using a min max denominator scaling with the distance to the mean in the numerator
    :param object column_of_data: a column or series of a dataframe
    '''
    column_of_data = (column_of_data-column_of_data.mean())/(max(column_of_data)-min(column_of_data))
    return column_of_data

def min_max_scale(column_of_data):
    '''
    method standardizes a column of data using min max scaling
    :param object column_of_data: a column or series of a dataframe
    '''
    min_value = min(column_of_data)
    max_value = max(column_of_data)
    normalized_data = []

    for value in column_of_data:
        normalized_value = (value - min_value) / (max_value - min_value)
        normalized_data.append(normalized_value)

    return normalized_data

def robust_scaling_with_iqr(column_of_data):
    '''
    method standardizes a column of data using an interquartile range for the divisor
    :param object column_of_data: a column or series of a dataframe
    '''
    IQR1 = stats.iqr(column_of_data, interpolation = 'midpoint') 
    column_of_data = (column_of_data-column_of_data.median())/IQR1
    return column_of_data