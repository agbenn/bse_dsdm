from bse_dsdm.preprocessing.feature_scaling import *
from bse_dsdm.preprocessing.outlier_detection import *
from bse_dsdm.preprocessing.value_encoder import *
from bse_dsdm.preprocessing.value_imputer import *
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression

def preprocessing_continuous_outlier_removal_pipeline(model, data, continuous_columns, target_col, accuracy_metric="roc_auc"): 
    drop_complete_columns(data)
    
    data1 = remove_outliers_one_class_svm(data[continuous_columns + target_col])
    data2 = remove_outliers_local_outlier(data[continuous_columns + target_col])
    data3 = remove_outliers_min_covariance_det(data[continuous_columns + target_col])
    data4 = remove_outliers_iso_forest(data[continuous_columns + target_col])
    data5 = remove_outliers_std_deviation(data[continuous_columns + target_col])
    data6 = remove_outliers_iqr(data[continuous_columns + target_col])

    datasets = [data, data1, data2, data3, data4, data5, data6]

    best_score, best_index = get_optimal_dataset(model, datasets, continuous_columns, target_col)

    optimal_score_dict = {0:"No Change",1:"One Class SVM",2:"Local Outlier",3:"Minimum Covariance Determinant",4:"ISO Forest",5:"STD Deviation",6:"IQR Dataset"}

    print("The optimal outlier method was " + str(optimal_score_dict[best_index]) + " with an " + str(accuracy_metric) + " score of " + str(best_score))

    return datasets[best_index]


def preprocessing_impution_pipeline():
    cross_validation_scoring()
    
def encoding_pipeline():



def get_optimal_dataset(model, datasets, X_cols, y_col, scoring_metric="roc_auc", min_or_max_score="max"):
    """
    scoring_metric choices: roc_auc, recall, precision, accuracy, r2
    """
    best_score = None
    best_score_index = 0

    for dataset_index in range(0,len(datasets)):

        if min_or_max_score == "min":
            score = cross_val_score(model, datasets[dataset_index][X_cols], datasets[dataset_index][y_col], cv=5, scoring_metric=scoring_metric)
            if score == None:
                best_score = score
                best_score_index = dataset_index
            elif score <= best_score:
                best_score = score
                best_score_index = dataset_index
        else: 
            score = cross_val_score(model, datasets[dataset_index][X_cols], datasets[dataset_index][y_col], cv=5, scoring_metric=scoring_metric)
            if score == None:
                best_score = score
                best_score_index = dataset_index
            elif score >= best_score:
                best_score = score
                best_score_index = dataset_index

    return best_score, best_score_index

data = pd.read_csv('../data/train.csv')
y = data['price']
X = data[['num_rooms', 'num_baths', 'square_meters']].fillna(method='bfill')
model = LinearRegression()
print(data.columns)
preprocessing_outlier_removal_pipeline(model,data, X.columns, y.columns)