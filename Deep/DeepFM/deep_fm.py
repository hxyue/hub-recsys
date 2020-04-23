from tensorflow import keras
from keras.layers import Embedding, Flatten, Dense, Activation, add, multiply, concatenate, \
    BatchNormalization, Dropout, Lambda, Layer
from collections import deque
embedding_size = 8  # embedding layer size
seed = 0.8
dropout_rate = 0.8
batch_size = 256
epochs = 100
lr=0.0005
decay=0.1
model_path = ''


class NumericLayer(Layer):
    def __init__(self, output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        return keras.backend.dot(x, self.kernel)  # TODO:

    def get_config(self):
        config = super().get_config()
        config['output_dim'] = self.output_dim
        return config


class DeepFM:
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict
        self.model = None

    def build_model(self):

        columns = self.feature_dict.columns

        # first order
        inputs = []
        inputs_numeric = deque([])
        embedding_layers = deque([])
        fm_layers = []
        for col in columns:
            col_type = self.feature_dict.feature_to_type[col]

            if col_type == "numeric":
                input_ = keras.Input(shape=(1, ), dtype='float', name='input_%s' % col)
                inputs.append(input_)
                input_ = BatchNormalization(input_)
                inputs_numeric.append(input_)
                flatten_layer = NumericLayer(output_dim=1)(input_)
                fm_layers.append(flatten_layer)
            else:
                input_ = keras.Input(shape=(1, ), dtype='int32', name='input_%s' % col)
                inputs.append(input_)
                count_categories = len(self.feature_dict.feature_to_encoder[col].classes_)
                embed = Embedding(count_categories,
                                  1,
                                  input_length=1,
                                  name='linear_%s' % col,
                                  embeddings_regularizer=keras.regularizers.l2)(input_)
                embedding_layers.append(embed)
                flatten_layer = Flatten(embed)
                fm_layers.append(flatten_layer)

        first_order_layer = add(fm_layers)
        first_order_layer = BatchNormalization(first_order_layer)
        first_order_layer = Dropout(dropout_rate, seed=seed)(first_order_layer)

        # second order
        flatten_layers = []
        for col in columns:
            col_type = self.feature_dict.feature_to_type[col]
            if col_type == "numeric":
                input_ = NumericLayer(output_dim=embedding_size)(inputs_numeric.popleft())
                flatten_layers.append(input_)
            else:
                embed = embedding_layers.popleft()
                flatten_layer = Flatten()(embed)
                flatten_layers.append(flatten_layer)

        summed_features_embed = add(flatten_layers)
        summed_features_embed_square = multiply([summed_features_embed. summed_features_embed])
        squared_features_embed = []
        for layer in flatten_layers:
            squared_features_embed.append(multiply([layer, layer]))
        summed_squared_features_embed = add(squared_features_embed)
        subtract_layer = Lambda(lambda x: x[0] - x[1], output_shape=lambda shapes: shapes[0])
        second_order_layer = subtract_layer(
            [summed_features_embed_square, summed_squared_features_embed])
        second_order_layer = Lambda(lambda x: x * .5)(second_order_layer)
        second_order_layer = Dropout(dropout_rate, seed=seed)(second_order_layer)

        # deep layer
        deep_layer = concatenate(flatten_layers)
        deep_layer = Dense(32)(deep_layer)
        deep_layer = Activation('relu', name='output_1')(deep_layer)
        deep_layer = Dropout(rate=dropout_rate, seed=seed)(deep_layer)
        deep_layer = Dense(32)(deep_layer)
        deep_layer = Activation('relu', name='output_2')(deep_layer)
        deep_layer = Dropout(rate=dropout_rate, seed=seed)(deep_layer)
        concat_input = concatenate([first_order_layer, second_order_layer, deep_layer], axis=1)
        output = Dense(1, activation='sigmoid', name='main_output')(concat_input)
        self.model = keras.Model(inputs=inputs, outputs=output)

        optimizer = keras.optimizers.Adam(lr=lr, decay=0.1)
        auc = keras.metrics.AUC()
        accuracy = keras.metrics.binary_accuracy()
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc, accuracy])

    def train(self, x, y, dev_x, dev_y):
        self.build_model()
        self.model.fit(x, y, batch_size=batch_size, validation_data=(dev_x, dev_y), epochs=epochs)

    def inference(self, x):
        predictions = self.model.predict(x)
        return predictions

    def evaluate(self, x, y):
        from sklearn.metrics import roc_auc_score, log_loss
        predictions = self.inference(x)
        print(roc_auc_score(y, predictions))
        print(log_loss(y, predictions))

    def export(self):
        self.model.save_model(model_path)