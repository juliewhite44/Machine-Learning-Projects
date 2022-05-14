import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

batch_size = 64
IMG_SIZE = (100, 100)

train = './archive/fruits-360_dataset/fruits-360/Training'
test = './archive/fruits-360_dataset/fruits-360/Test'

train_data = tf.keras.utils.image_dataset_from_directory(
  train,
  shuffle=True,
  seed=123,
  image_size=IMG_SIZE,
  batch_size=batch_size)

test_data = tf.keras.utils.image_dataset_from_directory(
  test,
  seed=123,
  image_size=IMG_SIZE,
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

IMG_SHAPE = IMG_SIZE + (3,)

tensorflow_model = tf.keras.applications.MobileNetV3Large(
    input_shape=IMG_SHAPE, alpha=1.0, minimalistic=False, include_top=False,
    weights='imagenet', pooling='max',
    dropout_rate=0.2, classifier_activation='softmax',
    include_preprocessing=True
)

tensorflow_model.trainable = False

prediction_layer = tf.keras.layers.Dense(131)

inputs = tf.keras.Input(shape=IMG_SHAPE)
outputs = prediction_layer(tensorflow_model(inputs, training=False))
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 2

loss, acc = model.evaluate(test_data)

print("initial loss: ", loss)
print("initial accuracy: ", acc)

model.fit(train_data,
                    epochs=initial_epochs)

print("n = ", len(tensorflow_model.layers))

loss, acc = model.evaluate(test_data)

print("loss after fit: ", loss)
print("accuracy after fit: ", acc)

tensorflow_model.trainable = True

fine_tune_at = 245
for layer in tensorflow_model.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()

fine_tune_epochs = 3
total_epochs =  initial_epochs + fine_tune_epochs

model.fit(train_data,
                         epochs=total_epochs,
                         initial_epoch=initial_epochs)

loss, acc = model.evaluate(test_data)

print("loss after fine tuning: ", loss)
print("accuracy after fine tuning: ", acc)



