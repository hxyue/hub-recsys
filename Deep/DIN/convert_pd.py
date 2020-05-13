#coding:utf-8
import pickle
import pandas as pd

#------ 数据集说明 ------
"""
亚马逊数据集包含产品评论和产品原始数据，用作基准数据集。 
我们对名为Electronics的子集进行实验，其中包含192,403个用户，63,001个商品，
801个类别和1,689,188个样本。 
此数据集中的用户行为很丰富，每个用户和商品都有超过5条评论。 
特征包括goods_id，cate_id，用户评论goods_id_list和cate_id_list。
用户的所有行为都是（b1，b2，...，bk，... ，bn），
任务是通过利用前k个评论商品来预测第（k + 1）个评论的商品。 
训练数据集是用每个用户的k = 1,2，...，n-2生成的。 
在测试集中，我们预测最后一个给出第一个n - 1评论商品。

reviews_Electronics数据集
reviewerID	评论者id，例如[A2SUAM1J3GNN3B]
asin	产品的id，例如[0000013714]
reviewerName	评论者昵称
helpful	评论的有用性评级，例如2/3
reviewText	评论文本
overall	产品的评级
summary	评论摘要
unixReviewTime	审核时间（unix时间）
reviewTime	审核时间（原始）
"""

"""
meta_Electronics数据集
asin	产品的ID
imUrl	产品图片地址
title	产品名称
categories	产品所属的类别列表
description	产品描述
related  产品的相关操作
price   产品价格
salesRank   产品售卖等级
brand   产品品牌
"""


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


reviews_df = to_df('../../Data/meta_electronics/reviews_Electronics_5.json')
print(reviews_df.columns)
with open('../../Data/meta_electronics/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../../Data/meta_electronics/meta_Electronics.json')
print(meta_df)
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('../../Data/meta_electronics/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)