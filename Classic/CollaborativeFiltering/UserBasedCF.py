import pandas as pd
from math import sqrt
import numpy as np

"""
user-based CF 
关注点，必须要使用用户都有评价的数据
基本步骤如下所示：
基于物品计算用户相似度
取与用户相似度最高的五个用户的物品进行推荐
"""

def cosine_sim(x, y, norm=False):
    """
    计算两个向量x和y的余弦相似度
    :param x: 向量x
    :param y: 向量y
    :param norm: 是否进行归一化
    :return:
    """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

def euclidean_sim(x, y):
    """
    :param x:
    :param y:
    :param norm:
    :return: 欧氏距离衡量相似度 = 1/(1+距离)
    """
    assert len(x) == len(y), "len(x) != len(y)"
    distance = sum(np.array([pow((x[i] - y[i]),2) for i in range(len(x))]))
    return 1 / (1 + sqrt(distance))
    # 这里返回值越小，相似度越大, 为1表示完全不相关


##计算两用户之间的Pearson相关系数
def pearson_sim(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_info = all_dic[user1]
    user2_info = all_dic[user2]
    # 找到两位用户都评论过的电影
    cross_movies = []
    for movieid in user1_info.keys():
        if movieid in user2_info.keys():
            cross_movies.append((movieid, user1_info[movieid], user2_info[movieid]))
    cross_movies = pd.DataFrame(cross_movies, columns=['movieid', 'u1_rating', 'u2_rating'])
    cross_movies = cross_movies.apply(lambda x: x.astype(float))
    corr_value = cross_movies['u1_rating'].corr(cross_movies['u2_rating'])
    return corr_value


# 计算某个用户和其他用户的相似度
def top_simliar(user_id, top_num=10):
    res = []
    for user_id in all_dic.keys():

    for userid in all_dic.keys():
        if not userid == userID:
            euc = Euclidean(userid, userID)
            res.append((userid, euc))
    res.sort(key=lambda val: val[1])
    return res[:5]

# 将相似用户的兴趣推荐给该用户
def count_list(list_name):  # 用来统计列表中元素出现的个数
    count_list = {}
    for item in set(list_name):
        count_list[item] = list_name.count(item)
    return count_list


def predict(user_id, top_nums = 10, evaluation="cosine"):
    similar_users = top10_simliar(userID)
    recmd = []
    for i in range(5):
        user_info = similar_users[i][0]
        for movieid in all_dic[user_info].keys():
            if movieid not in all_dic[userID].keys():
                recmd.append(movieid)
    recmd_all = count_list(recmd)
    recmd_top10 = dict(sorted(recmd_all.items(), key=lambda item: item[1], reverse=True)[:10])
    # 这一写法是根据字典中值的大小，对字典进行排序，排完之后是列表形式，还需要再转化为字典
    return recmd_top10.keys()

