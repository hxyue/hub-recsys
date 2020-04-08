import pandas as pd
from math import sqrt
import sys
import os

PROJECT_PATH = "/Users/didi/Documents/HM/hub-recsys"

def dataRead(PROJECT_PATH):
    movies = pd.read_csv(PROJECT_PATH + '/data/ml-1m/movies.dat', sep='::',
                         names=['MovieID', 'Title', 'Genres'], engine='python')
    ratings = pd.read_csv(PROJECT_PATH + '/data/ml-1m/ratings.dat', sep='::',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
    data = ratings.merge(movies, on='MovieID', how='left')
    data = data[['UserID', 'Rating', 'MovieID', 'Title']].sort_values('UserID')
    userItemMatrix = {}
    for UserID, group in (data.groupby('UserID')):
        for i in range()
        print(UserID)
        print(list(group['Rating']))
        print(list(group['MovieID']))
    return data


dataRead(PROJECT_PATH)

def parser(data):
    """
    :param data:
    :return: user-item matrix
    """

'''
为了得到用户兴趣表，横轴为movie_id，纵轴为user_id
先做成 {user_id_1: {movie_1: rating, 
                   movie_2: rating,
                   ......
                  }
       user_id_2: {......}
       ......
      }
这一字典形式
'''
all_dic = {}
file = open('./ml-latest-small/movie_ratings.csv', 'r', encoding='UTF-8')
for line in file.readlines()[1:2000]:  # 如果从0开始的话，会把head列标题也读进来，这个读出来是以逗号分隔的
    line = line.strip().split(',')
    if line[0] not in all_dic.keys():
        all_dic[line[0]] = {line[2]: line[1]}
    else:
        all_dic[line[0]][line[2]] = line[1]

'''计算任何两位用户之间的相似度，
由于每位用户评论的电影不完全一样，
所以先要找到两位用户共同评论过的电影
然后计算两者之间的欧式距离，算出两者之间的相似度，
取与用户相似度最高的五个用户的兴趣推荐给该用户
'''


def Euclidean(user1, user2):  # 返回的是相似度 = 1/(1+距离)
    user1_data = all_dic[user1]
    user2_data = all_dic[user2]
    distance = 0
    for key in user1_data.keys():
        if key in user2_data.keys():
            distance += pow((float(user1_data[key]) - float(user2_data[key])), 2)  # pow(x,y)表示x的y次方，这里是(x-y)^2
    return 1 / (1 + sqrt(distance))  # 这里返回值越小，相似度越大, 为1表示完全不相关


# 计算某个用户和其他用户的相似度
def top10_simliar(userID):
    res = []
    for userid in all_dic.keys():
        if not userid == userID:
            euc = Euclidean(userid, userID)
            res.append((userid, euc))
    res.sort(key=lambda val: val[1])
    return res[:5]


# print(top10_simliar('1'))

# 将相似用户的兴趣推荐给该用户
def count_list(list_name):  # 用来统计列表中元素出现的个数
    count_list = {}
    for item in set(list_name):
        count_list[item] = list_name.count(item)
    return count_list


def recommend(userID):
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


# 至此得到了包含10部推荐电影的movie_id的列表

'''
但有时我们会碰到因为两个用户之间数据由于数据膨胀，一方数据大，一方数据小，但是两者成明显的线性关系
我们引入Pearson相关系数来衡量两个变量之间的线性相关性。
注意：通过Pearson系数得到用户的相似度和通过欧式距离得到结果可能不一样
'''


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


'''
找到和每个用户相关系数在阈值以上的用户，并将他们的电影推荐给该用户
'''


def recommend_v2(userID, thres, n_movies):
    rec_list = []
    for user in all_dic.keys():
        r = pearson_sim(userID, user)
        if r > thres:
            for movieid in all_dic[user].keys():
                if movieid not in all_dic[userID].keys():
                    rec_list.append(movieid)
    recmd_all = count_list(rec_list)
    recmd_top10 = dict(sorted(recmd_all.items(), key=lambda item: item[1], reverse=True)[:n_movies])
    return recmd_top10.keys()


# 找到和用户1相关性在0.8以上的前10个用户
recommend_v2('1', 0.8, 10)
