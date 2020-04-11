from tensorflow.keras import layers
import tensorflow as tf


from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


print(tf.__version__)
print(tf.keras.__version__)


# pylint: disable=protected-access
_concat = rnn_cell_impl._concat
_like_rnncell = rnn_cell_impl._like_rnncell
# pylint: enable=protected-access



class ActivationUnit(tf.keras.Model):
    def __init__(self):
        #super(ActivationUnit, self).__init__()
        self.act_dense_layer = layers.Dense(3, activation='relu', name='act_dense_layer')
        self.act_dense_out = layers.Dense(1, activation='sigmoid', name='act_dense_out')
        layers.LSTM(32)

    def call(self, inputs):
        '''
        在数据处理环节处理好，每次输入的len是不变的，但可以有不同len的输入,不足max_len的补0
        target_embed:[batch,1,embed_size]
        user_embed:[batch,max_len,embed_size]
        '''
        target_embed = inputs[0]
        user_embed = inputs[1]
        embed_size, max_len = user_embed.shape[2], user_embed.shape[1]

        target_embed = tf.tile(target_embed, [1, max_len])
        target_embed = tf.reshape(target_embed, [-1, max_len, embed_size])

        act_input = tf.concat([target_embed, target_embed - user_embed, user_embed], axis=2)
        act_input = self.act_dense_layer(act_input)
        act_out = self.act_dense_out(act_input)

        return act_out

layer = ActivationUnit()
layer([tf.ones([2, 3]), tf.ones([2, 3, 3])])
layer.summary()


def weighted_sum(weight, inputs):
    '''
    weight:[batch,max_len,1]
    inputs:[batch,max_len,embed_size]
    out:[batch,1]
    '''
    return tf.reduce_sum(weight * inputs, 1)


class DIN(tf.keras.Model):
    def __init__(self):
        super(DIN, self).__init__()
        self.activationUnit = ActivationUnit()
        self.dense_layer1 = layers.Dense(8, activation='relu', name='dense_layer1')
        self.dense_layer2 = layers.Dense(3, activation='relu', name='dense_layer2')
        self.dense_out = layers.Dense(1, activation='sigmoid', name='dense_out')

    def call(self, inputs):
        '''
        user_goods_embed:[batch,max_len,embed_size]
        user_shops_embed:[batch,max_len,embed_size]
        target_good_embed:[batch,embed_size]
        target_shop_embed:[batch,embed_size]
        target_other_embed:[batch,embed_size]
        '''
        assert (len(inputs) == 5)
        user_goods_embed = inputs[0]
        user_shops_embed = inputs[1]
        target_good_embed = inputs[2]
        target_shop_embed = inputs[3]
        target_other_embed = inputs[4]

        target_goods_weight = self.activationUnit([target_good_embed, user_goods_embed])
        sum_goods_embed = weighted_sum(target_goods_weight, user_goods_embed)
        target_shops_weight = self.activationUnit([target_shop_embed, user_shops_embed])
        sum_shops_embed = weighted_sum(target_shops_weight, user_shops_embed)

        out = tf.concat([sum_goods_embed, sum_shops_embed, target_good_embed, target_shop_embed, target_other_embed],
                        axis=1)
        print('dense input:', out.shape)
        out = self.dense_layer1(out)
        out = self.dense_layer2(out)
        out = self.dense_out(out)
        return out



class DIEN(tf.keras.Model):
    def __init__(self):
        super(DIEN, self).__init__()


# RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
layer = DIN()
layer([tf.ones([2, 3, 10]), tf.ones([2, 3, 10]), tf.ones([2, 10]), tf.ones([2, 10]), tf.ones([2, 10])])
layer.summary(line_length=100)