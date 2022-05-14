from keras_preprocessing import sequence
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

number_of_words = 20000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

def format_input(input):
    new_input = np.zeros((len(input), number_of_words))
    for i, row in enumerate(input):
        for j in row:
            new_input[i][j] = 1.
    return new_input

X_train = format_input(X_train)
X_test = format_input(X_test)


model = Sequential([
    layers.Dense(units=16, activation='relu'),
    layers.Dense(units=16, activation='relu'),
    layers.Dense(units=1, activation='relu')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64)

test_loss, test_acurracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_acurracy))