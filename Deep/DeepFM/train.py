import pandas as pd
import os

from Deep.DeepFM.preprocessor import FeatureDictionary, DataParser


def train():
    train_file = '../../Data/safe-driver/test.csv'
    test_file = '../../Data/safe-driver/test.csv'
    numeric_cols = ['ps_reg_03', 'ps_car_12','ps_car_13','ps_car_14', 'ps_car_15']
    feature_dict = FeatureDictionary(train_file, test_file, numeric_cols)

    df_train, df_test = feature_dict.gen_feature_dict()
    data_parser = DataParser(feature_dict)
    train_parsed = data_parser.parse_data(df_train)
    test_parsed = data_parser.parse_data(df_test)

    print("train_parsed:")

    # deep_fm_ = DeepFM()
    # deep_fm_.buid()
    # deep_fm_.train()


def evaluate():
    pass



path2=os.path.abspath('../../Data/safe-driver/test.csv')
print(path2)
pd.read_csv('../../Data/safe-driver/test.csv')
train()
