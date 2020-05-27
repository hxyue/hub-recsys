from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, Embedding, Input, Embedding, LSTM, Lambda, Flatten
import tensorflow as tf
from .layers import Transformer, BiLSTM, AttentionSequencePoolingLayer
import numpy as np


class DSIN(Layer):

    def __init__(self, dnn_feature_columns, sess_feature_list, sess_max_count=5, bias_encoding=False,
                 att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='sigmoid',
                 dnn_dropout=0,
                 dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, init_std=0.0001, seed=1024, task='binary',
                 **kwargs):
        self._dnn_feature_columns = dnn_feature_columns
        self._sess_feature_list = sess_feature_list
        self._sess_max_count = sess_max_count
        self._bias_encoding = bias_encoding
        self._att_embedding_size = att_embedding_size
        self._att_head_num = att_head_num
        self._dnn_hidden_units = dnn_hidden_units
        self._dnn_activation = dnn_activation
        self._dnn_dropout = dnn_dropout
        self._dnn_use_bn = dnn_use_bn
        self._l2_reg_dnn = l2_reg_dnn
        self._l2_reg_embedding = l2_reg_embedding
        self._init_std = init_std
        self._seed = seed
        self._task = task

        super(DSIN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="embeddings")
        super(DSIN, self).build(input_shape)


    def call(self, inputs):

        uid, gender, item_id, cate_id, score, sess_item_id, sess_number = inputs
        cat_embedding_size = 100
        item_embedding_size = 100

        item_input_embedding = Embedding(self.n_item, self.embedding_dim)(item_id)
        cate_input_embedding = Embedding(self.n_cate, self.embedding_dim)(cate_id)
        uid_input_embedding = Embedding(self.n_uid, self.embedding_dim)(uid)
        gender_input_embedding = Embedding(self.n_gender, self.embedding_dim)(gender)
        dnn_input_emb = tf.concat(
            [uid_input_embedding, gender_input_embedding, cate_input_embedding, item_input_embedding], 2)
        query_emb = tf.concat(
            [cate_input_embedding, item_input_embedding], 2)

        user_sess_length = Input(shape=(1,), name='sess_length')

        # --------------*interest_division_layer*--------------
        sess_hist_embedding = []
        for sess_item_id_one in sess_item_id:
            sess_item_emb = Embedding(self.n_item, self.embedding_dim, input_length=10)(sess_item_id_one)
            sess_hist_embedding.append(sess_item_emb)

        transformer_layer = Transformer(self._att_embedding_size, self._att_head_num, dropout_rate=0,
                                        use_layer_norm=False, use_positional_encoding=(not self._bias_encoding),
                                        seed=self._seed, supports_masking=True, blinding=True)

        # --------------*interest_extractor_layer*--------------
        tr_out = []
        for i in range(self._sess_max_count):
            tr_out.append(transformer_layer([sess_hist_embedding[i], sess_hist_embedding[i]]))
        sess_fea = tf.concat(tr_out, 1)


        bilstm_outputs = BiLSTM(cat_embedding_size+item_embedding_size, layers=2, res_layers=0, dropout_rate=0.2)(sess_fea)

        interest_attention = Flatten()(
            AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True, supports_masking=False)(
                [query_emb, sess_fea, user_sess_length]))

        bilstm_attention = Flatten()(
            AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True)(
                [query_emb, bilstm_outputs, user_sess_length]))

        dnn_input_emb = tf.concat(
            [dnn_input_emb, interest_attention, bilstm_attention], 2)
        output = Dense(self._dnn_hidden_units, activation=self._dnn_activation, kernel_regularizer=self._l2_reg_dnn)(
            dnn_input_emb)
        output = Dense(1, use_bias=False, activation=None)(output)
        return output


if __name__ == "__main__":
    from tensorflow.keras.models import Model
    uid = Input(shape=(max_seq_len,), name='encoder_inputs')
    gender = Input(shape=(max_seq_len,), name='decoder_inputs')
    item_id = Input(shape=(max_seq_len,), name='decoder_inputs')
    cate_id = Input(shape=(max_seq_len,), name='decoder_inputs')
    score = Input(shape=(max_seq_len,), name='decoder_inputs')
    sess_item_id = Input(shape=(max_seq_len,), name='decoder_inputs')
    sess_number = Input(shape=(max_seq_len,), name='decoder_inputs')
    outputs = DSIN(vocab_size, model_dim)([encoder_inputs, decoder_inputs])
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    model.summary()
