# coding:utf-8

from tensorflow import keras
from tensorflow.keras import layers

num_words = 30000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
print(x_train[0])
print(y_test)
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.Masking(mask_value=0.0, name="mask_padding"),
        layers.GRU(32, return_sequences=True),
        layers.GRU(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model


model = lstm_model()
model.summary()

#history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)

