from bse_dsdm.preprocessing.preprocessing_pipeline import * 
from bse_dsdm.model_development.model_training_pipeline import *


train = pd.read_csv('../../data/position-prediction-for-football-players/train.csv')
print(train.columns)
train.describe().to_csv('training_data.csv')
print(train.isna().sum()>0)
print(train.corr())
train = drop_complete_columns(train, 80)
train = train.drop(columns=['player_traits'])

categorical_cols = ['club_name','league_name','nationality_name','preferred_foot','work_rate','body_type','real_face']
cols_not_useful = ['id','short_name','birthday_date','club_joined', 'club_contract_valid_until', 'position']
continuous_cols = [x for x in train.columns if x not in categorical_cols+cols_not_useful]
cols_to_scale = ((train[continuous_cols].max() - train[continuous_cols].min()) > (train[continuous_cols].median()))
cols_to_scale = cols_to_scale.loc[cols_to_scale == True].index

training_datasets = preprocessing_continuous_impution_pipeline([train], continuous_columns=continuous_cols)
training_datasets = preprocessing_continuous_outlier_removal_pipeline(training_datasets, continuous_columns=continuous_cols, retain_previous_datasets=True)
training_datasets = preprocessing_encode_categorical_pipeline(training_datasets, categorical_cols)
"""
training_datasets = preprocessing_outlier_imputer_pipeline(training_datasets, continuous_columns=continuous_cols, retain_previous_datasets=True)
training_datasets = preprocessing_feature_scaler(training_datasets, columns_to_scale=cols_to_scale)
"""

value = 0
for dataset in training_datasets:
    X = dataset.drop(columns='position')
    y = dataset['position']
    print('dataset ' + str(value))
    lightgbm_pipeline(X,y)
    file_name = 'test_' + str(value) + '.csv'
    dataset.to_csv(file_name)
    value += 1

    
"""
y = data['price']
X = data[['num_rooms', 'num_baths', 'square_meters']].fillna(method='bfill')
model = LinearRegression()
print(data.columns)
preprocessing_outlier_removal_pipeline(model,data, X.columns, y.columns)

value = 0
for each in training_datasets: 
    print(get_columns_with_na(each, 0))
    file_name = 'test_' + str(value) + '.csv'
    each.to_csv(file_name)
    value += 1
"""