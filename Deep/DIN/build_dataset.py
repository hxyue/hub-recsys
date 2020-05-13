#coding:utf-8
import random
import pickle

random.seed(1234)

with open('data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)


def gen_neg():
  neg = pos_list[0]
  while neg in pos_list:
    neg = random.randint(0, item_count-1)
  return neg

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  neg_list = [gen_neg() for i in range(len(pos_list))] #获取用户没有记录的商品作为负例

  for i in range(1, len(pos_list)):# range(1,5) [1,2,3,4]
    hist = pos_list[:i] #[:1] 取的是list的长度
    if i != len(pos_list) - 1: #迭代的最后一列
      train_set.append((reviewerID, hist, pos_list[i], 1))
      train_set.append((reviewerID, hist, neg_list[i], 0))
    else:
      label = (pos_list[i], neg_list[i])
      test_set.append((reviewerID, hist, label))

"""
用户的所有行为都是（b1，b2，...，bk，... ，bn），
任务是通过利用前k个评论商品来预测第（k + 1）个评论的商品。 
训练数据集是用每个用户的k = 1,2，...，n-2生成的。 
在测试集中，我们预测最后一个给出第一个n - 1评论商品。
"""

random.shuffle(train_set)
random.shuffle(test_set)
print len(test_set)
print len(train_set)
print user_count
# assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])
# 由于小样本数据，并非每个用户都具备5条以上评价数据，所以失效。

with open('dataset.pkl', 'wb') as f:
  print train_set
  print test_set
  print cate_list
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
  #cate_list 按照商品id排序对应的类目信息