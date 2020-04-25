from tensorflow.keras import layers, Sequential, Input, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

print(tf.__version__)
print(tf.keras.__version__)




class EmbeddingGRU(tf.keras.Model):
    def __init__(self, n_uid, n_mid, n_cat, embedding_dim, hidden_size, attention_size, inputs, max_length):
        #self.mid_his_pad = pad_sequences(self.mid_his_input, maxlen=self.max_length, padding='post')
        #self.cat_his_pad = pad_sequences(self.cat_his_input, maxlen=self.max_length, padding='post')
        self.n_uid = n_uid
        self.n_mid = 1000
        self.n_cat = 1000
        self.embedding_dim = 40
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.uids = inputs[0]
        self.mids = inputs[1]
        self.cats = 100
        self.mid_mask = inputs[5]
        self.target = inputs[6]
        self.max_length = 10

    def embeddingGRU(self):
        self.mid_his_input = Input((self.max_length,), dtype='int32', name='mid_his_input')
        self.cat_his_input = Input((self.max_length,), dtype='int32', name='cat_his_input')
        self.uid_input = Input((self.n_uid,), dtype='int32', name='uid_input')
        self.mid_input = Input((self.n_mid,), dtype='int32', name='mid_input')
        self.cat_input = Input((self.n_cat,), dtype='int32', name='cat_input')

        # -- embedding --
        mid_embedding = layers.Embedding(self.n_mid, self.embedding_dim)
        cat_embedding = layers.Embedding(self.n_cat, self.embedding_dim)
        uid_input_embedding = layers.Embedding(self.n_cat, self.embedding_dim)(self.uid_input)
        mid_his_input_embedding = mid_embedding(self.mid_his_input)
        cat_his_input_embedding = cat_embedding(self.cat_his_input)
        cat_input_embedding = cat_embedding(self.cat_input)
        mid_input_embedding = mid_embedding(self.mid_input)
        print(cat_his_input_embedding.shape)

        item_his_eb = tf.concat([mid_his_input_embedding, cat_his_input_embedding], 2)
        print(item_his_eb.shape)
        #item_eb = tf.concat([mid_embedding(self.mid_input), cat_embedding(self.cat_input)], 1)


        # -- 兴趣抽象层次 --

        #whole_sequence_output, final_state = layers.GRU(self.hidden_size, return_sequences=True, return_state=True)(item_his_eb) #(batch_size, timesteps, units)






        model = Model(inputs=([self.mid_his_input, self.cat_his_input]), outputs=item_his_eb)


        '''
        rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                 att_scores=tf.expand_dims(alphas, -1),
                                                 sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                 scope="gru2")

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)
        '''
        return model


'''
    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    # Attention layer
    (self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
     softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
    def din_fcn_attention(self, query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                          return_alphas=False, forCnn=False):
        if isinstance(facts, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            facts = tf.concat(facts, 2)
        if len(facts.get_shape().as_list()) == 2:
            facts = tf.expand_dims(facts, 1)

        if time_major:
            # (T,B,D) => (B,T,D)
            facts = tf.array_ops.transpose(facts, [1, 0, 2])
        # Trainable parameters
        mask = tf.equal(mask, tf.ones_like(mask))
        facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
        querry_size = query.get_shape().as_list()[-1]
        query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
        query = prelu(query)
        queries = tf.tile(query, [1, tf.shape(facts)[1]])
        queries = tf.reshape(queries, tf.shape(facts))
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
        # Mask
        # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        if not forCnn:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Scale
        # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

        # Activation
        if softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        if mode == 'SUM':
            output = tf.matmul(scores, facts)  # [B, 1, H]
            # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
        if return_alphas:
            return output, scores
        return output




    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input = tf.concat([h_states, click_seq], -1)
        noclick_input = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input, stag=stag)[:, :, 0]
        click_loss_ = -tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, input, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=input, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat



'''


embedding_gru = EmbeddingGRU(200, 100, 20, 10, 5, 10, [[1],[2],[3],[4],[5],[6],[7],[8]], 4)
model = embedding_gru.DIN()
model.summary()

