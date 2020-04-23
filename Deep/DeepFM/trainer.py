from deep_fm import DeepFM
from preprocessor import FeatureDictionary, DataParser


def train():
    train_file = "train.csv"
    test_file = "test.csv"
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