import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class FeatureDictionary:
    def __init__(self, train_file, test_file, numeric_cols):
        self.train_file = train_file
        self.test_file = test_file
        self.numeric_cols = numeric_cols
        self.df_train = None
        self.df_test = None
        self.feature_to_encoder = {}
        self.feature_to_type = {}  # whether it's label or numeric
        self.columns = []

    def gen_feature_dict(self):
        if not self.train_file or not self.test_file:
            raise Exception("provide file for train and test sets")
        if not self.numeric_cols:
            raise Exception("provide which columns are numeric")

        self.df_train = pd.read_csv(self.train_file)
        self.df_test = pd.read_csv(self.test_file)
        df = pd.concat([self.df_train, self.df_test])

        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feature_to_type[col] = 'numeric'
            else:
                le = LabelEncoder()
                le.fit(df[col])
                self.feature_to_encoder[col] = le
                self.feature_to_type[col] = 'cat'
            self.columns.append(col)
        print(self.df_test)
        return self.df_train, self.df_test


class DataParser:
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict

    def parse_data(self, df):
        parsed_data = []
        for col in df.columns:
            if col in self.feature_dict.ignore_cols:
                continue
            if col in self.feature_dict.numeric_cols:
                parsed_data.append(np.array(df[col]))
            else:
                labels = self.feature_dict.feature_to_encoder[col].transform(df[col]).reshape(-1, 1)
                parsed_data.append(labels)
        return parsed_data