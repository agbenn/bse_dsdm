from bse_dsdm.preprocessing.feature_scaling import *
#from bse_dsdm.preprocessing.outlier_detection import *
from bse_dsdm.preprocessing.optimal_outlier_detection import *
from bse_dsdm.preprocessing.value_encoder import *
from bse_dsdm.preprocessing.value_imputer import *
from bse_dsdm.preprocessing.data_loader import *
from bse_dsdm.preprocessing.preprocessing_functions import *
from bse_dsdm.preprocessing.exploratory_analysis import *
from bse_dsdm.accuracy_testing.cross_validation import *
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV


data=MLDataLoader()
data.load_data("../data/train.csv")

data.data=remove_columns_with_na(data.data,80)

categorical_columns_na=get_columns_by_type(data.data)[0]
print(categorical_columns_na)
numerical_columns_na=get_columns_by_type(data.data)[1]
print(numerical_columns_na)


X = data.data[['num_rooms', 'num_baths', 'square_meters', 'year_built',
       'num_crimes']]
y = data.data['price']

remove_optimal_outliers(X,y,model_type='binary')



