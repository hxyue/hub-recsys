import os

from Util.databuilder import DataBuilder
from Classic.CollaborativeFiltering.ItemBasedCF import Itemcf

path = "../Data/ml-1m/ratings.dat"
data = DataBuilder(path)


def test_itemcf():
    data_builder.eval(Itemcf())

data_builder = DataBuilder(path, just_test_one=True)
test_itemcf()


