

def standardize_with_z_score(column_of_data): 
    # Rescale the variables using z-score standardization
    return (column_of_data - column_of_data.mean()) / column_of_data.std()

def min_max_scale(column_of_data):
    min_value = min(column_of_data)
    max_value = max(column_of_data)
    normalized_data = []

    for value in column_of_data:
        normalized_value = (value - min_value) / (max_value - min_value)
        normalized_data.append(normalized_value)

    return normalized_data