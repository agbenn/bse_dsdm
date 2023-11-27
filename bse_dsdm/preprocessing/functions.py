 """
best_score, best_index = get_optimal_dataset(model, datasets, continuous_columns, target_col)

optimal_score_dict = {0:"No Change",1:"One Class SVM",2:"Local Outlier",3:"Minimum Covariance Determinant",4:"ISO Forest",5:"STD Deviation",6:"IQR Dataset"}

print("The optimal outlier method was " + str(optimal_score_dict[best_index]) + " with an " + str(accuracy_metric) + " score of " + str(best_score))
"""

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