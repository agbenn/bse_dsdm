import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class Split_data:

    def _init_(self, file_path, test_size, random_state):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.train_data = None
        self.test_data = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=self.random_state)
        self.train_data = train_data
        self.test_data = test_data

    def get_train_data(self):
        if self.train_data is None:
            raise ValueError("Train data not loaded.")
        return self.train_data

    def get_test_data(self):
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        return self.test_data
