#coding:utf-8
import random
import pickle
import numpy as np

random.seed(1234)

with open('data/reviews.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

with open('data/meta.pkl', 'rb') as f:
    meta_df = pickle.load(f)
    meta_df = meta_df[['asin', 'categories']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
    #获取改商品的最小子类别


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


asin_map, asin_key = build_map(meta_df, 'asin') #商品id OneHot
cate_map, cate_key = build_map(meta_df, 'categories') #商品类别 oneHot
revi_map, revi_key = build_map(reviews_df, 'reviewerID') #useid oneHot

user_count, item_count, cate_count, example_count = \
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)

reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]


cate_list = list(meta_df['categories'])
cate_list = np.array(cate_list, dtype=np.int32)


with open('data/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)